#!/usr/bin/env python3
# ruff: noqa: N806
"""OCR bake-off over sampled pages of the 1890 Pierce medical adviser.

    python ocr_bench.py sample    # render pages -> out/pages/, write out/manifest.json
    python ocr_bench.py run       # run every engine -> out/tokens.csv, out/runs.csv
    python ocr_bench.py report    # -> out/report.md
    python ocr_bench.py demo      # self-check

Ground truth is optional: drop a corrected transcript at out/gt/<page_id>.txt and
`report` starts emitting CER/WER for that page. `run` seeds those files from the
best-confidence engine so labelling is correction, not typing.
"""
import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = ROOT / "data/raw/pierce-peoples-common-sense-medical-adviser-1890.pdf"
PDF = DEFAULT_PDF
OUT = Path(__file__).resolve().parent / "out"
PAGES, GT = OUT / "pages", OUT / "gt"
DPI = 300  # OCR resolution
SCAN_DPI = 150  # feature-scan resolution (buckets only, never OCR'd)
SCAN_STRIDE = 13  # every Nth page gets feature-scanned
N_PER_BUCKET = 3
# Measured, not assumed. See report.md §"Why these buckets" — foxing and
# multi-column turned out not to be discriminating axes in this book; ink fade,
# figure text-wrap and dense small type are.
BUCKETS = [
    "clean",
    "faint_ink",
    "figure_adjacent",
    "dense_small_type",
    "front_matter",
    "multi_column",
    "blank_plate",
]


# ---------------------------------------------------------------- page sampling


def render(page, dpi=DPI):
    pix = page.get_pixmap(dpi=dpi)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)


def features(page, img):
    """Per-page degradation/layout measurements. Resolution-independent."""
    h, w = img.shape[:2]
    words = page.get_text("words")  # x0,y0,x1,y1,word,block,line,word_no
    sx, sy = w / page.rect.width, h / page.rect.height

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = binv > 0
    bg = float(np.median(gray[~ink])) if (~ink).any() else 220.0

    # ink_depth: how dark the strokes sit below paper. Low => faded/light print.
    # p10 not p25: less swayed by antialiased stroke edges, so the ranking across
    # pages holds at any render DPI (the absolute value shifts, the order does not).
    ink_depth = bg - float(np.percentile(gray[ink], 10)) if ink.any() else 0.0

    # orphan_ink: ink outside every text-layer word box => plates, figures, rules.
    covered = np.zeros((h, w), bool)
    for x0, y0, x1, y1, *_ in words:
        covered[int(y0 * sy) : int(y1 * sy) + 1, int(x0 * sx) : int(x1 * sx) + 1] = True
    orphan_ink = float((ink & ~covered).sum()) / max(ink.sum(), 1)

    # span: fraction of words straddling the vertical centreline. Full-measure
    # single-column text puts ~1 word per line across it (~0.05-0.10); a true
    # two-column page puts ~none (~0).
    xc = page.rect.width / 2
    span = sum(1 for x0, _, x1, *_ in words if x0 < xc < x1) / max(len(words), 1)
    both_sides = (
        sum(1 for x0, _, x1, *_ in words if x1 < xc) > 20
        and sum(1 for x0, _, x1, *_ in words if x0 > xc) > 20
    )

    return {
        "n_words": len(words),
        "ink_depth": round(ink_depth, 1),
        "orphan_ink": round(orphan_ink, 4),
        "span": round(span, 4),
        "two_sided": both_sides,
    }


def bucket_of(i, f, faint_cut, dense_cut):
    if f["n_words"] < 20:
        return "blank_plate"
    if i < 12:
        return "front_matter"  # title/preface: display faces, letterspaced caps
    if f["orphan_ink"] > 0.35:
        return "figure_adjacent"  # text wraps a plate in a narrow ragged measure
    if f["span"] < 0.02 and f["two_sided"]:
        return "multi_column"
    if f["ink_depth"] < faint_cut:
        return "faint_ink"
    if f["n_words"] > dense_cut:
        return "dense_small_type"
    return "clean"


def cmd_sample():
    """Scan the book at low DPI, bucket by percentile, render the picks at 300 DPI."""
    PAGES.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF)
    idx = sorted(set(list(range(0, 14)) + list(range(0, doc.page_count, SCAN_STRIDE))))
    print(f"feature-scanning {len(idx)} pages at {SCAN_DPI} DPI...")
    feats = {}
    for n, i in enumerate(idx):
        feats[i] = features(doc[i], render(doc[i], SCAN_DPI))
        if n % 20 == 0:
            print(f"  {n}/{len(idx)}")

    body = [f for i, f in feats.items() if f["n_words"] >= 20 and i >= 12]
    faint_cut = float(np.percentile([f["ink_depth"] for f in body], 15))
    dense_cut = float(np.percentile([f["n_words"] for f in body], 85))
    print(f"cuts: faint ink_depth<{faint_cut:.1f}, dense n_words>{dense_cut:.0f}")

    picked, counts = [], defaultdict(int)
    for i, f in feats.items():
        b = bucket_of(i, f, faint_cut, dense_cut)
        if counts[b] >= N_PER_BUCKET:
            continue
        counts[b] += 1
        page_id = f"p{i + 1:04d}"
        cv2.imwrite(str(PAGES / f"{page_id}.png"), render(doc[i]))
        picked.append({"page_id": page_id, "pdf_index": i, "bucket": b, **f})
        print(f"{page_id}  {b:17s} {f}")

    (OUT / "manifest.json").write_text(json.dumps(picked, indent=2, default=str))
    empty = [b for b in BUCKETS if not counts[b]]
    print(f"\n{len(picked)} pages -> {OUT/'manifest.json'}")
    if empty:
        print(f"NO PAGES FOUND for: {', '.join(empty)} (not present in this book at this stride)")
    print("hand-edit buckets in manifest.json if you disagree, then `run`")


# ---------------------------------------------------------------------- engines
# Each engine: ctx -> list of (text, conf|None, x0, y0, x1, y1) in image pixels.
# Raise anything on failure; the runner records the message instead of skipping.


def eng_embedded(ctx):
    """The Internet Archive OCR layer already baked into the PDF. Zero-install baseline."""
    page = ctx["page"]
    sx = ctx["img"].shape[1] / page.rect.width
    sy = ctx["img"].shape[0] / page.rect.height
    return [
        (w[4], None, w[0] * sx, w[1] * sy, w[2] * sx, w[3] * sy) for w in page.get_text("words")
    ]


def eng_tesseract(ctx):
    import pytesseract

    d = pytesseract.image_to_data(ctx["img"], lang="eng", output_type=pytesseract.Output.DICT)
    out = []
    for i, t in enumerate(d["text"]):
        if not t.strip() or float(d["conf"][i]) < 0:
            continue
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        out.append((t, float(d["conf"][i]) / 100.0, x, y, x + w, y + h))
    return out


def eng_paddleocr(ctx):
    from paddleocr import PaddleOCR

    ocr = _cache("paddle", lambda: PaddleOCR(lang="en"))
    res = ocr.predict(ctx["img_path"])[0]
    return [
        (
            t,
            float(s),
            *map(
                float,
                (
                    min(p[0] for p in b),
                    min(p[1] for p in b),
                    max(p[0] for p in b),
                    max(p[1] for p in b),
                ),
            ),
        )
        for t, s, b in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"], strict=True)
    ]


def eng_doctr(ctx):
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor

    m = _cache("doctr", lambda: ocr_predictor(pretrained=True))
    res = m(DocumentFile.from_images(ctx["img_path"]))
    h, w = ctx["img"].shape[:2]
    out = []
    for block in res.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                (a, b), (c, d) = word.geometry
                out.append((word.value, float(word.confidence), a * w, b * h, c * w, d * h))
    return out


def eng_kraken(ctx):
    """Kraken warm-started from CATMuS-Print (Zenodo 10592716). Model path via
    KRAKEN_MODEL env or out/models/catmus-print.mlmodel."""
    import os

    from kraken import binarization, blla, rpred
    from kraken.lib import models
    from PIL import Image

    mp = os.environ.get("KRAKEN_MODEL", str(OUT / "models" / "catmus-print.mlmodel"))
    net = _cache("kraken:" + mp, lambda: models.load_any(mp))
    im = Image.open(ctx["img_path"]).convert("RGB")
    seg = blla.segment(binarization.nlbin(im.convert("L")).convert("1"))
    out = []
    for rec in rpred.rpred(net, im, seg):
        x = [p[0] for p in rec.line] if hasattr(rec, "line") else [0, 0]
        y = [p[1] for p in rec.line] if hasattr(rec, "line") else [0, 0]
        for tok in rec.prediction.split():
            out.append(
                (
                    tok,
                    float(np.mean(rec.confidences)) if rec.confidences else None,
                    min(x),
                    min(y),
                    max(x),
                    max(y),
                )
            )
    return out


def eng_calamari(ctx):
    import os

    ckpt = os.environ.get("CALAMARI_MODEL")
    if not ckpt:
        raise RuntimeError(
            "set CALAMARI_MODEL to a .ckpt.json (calamari needs line images, "
            "not full pages -- feed it kraken/tesseract line crops)"
        )
    raise RuntimeError("calamari is line-level only; wire it after a line segmenter is chosen")


_CACHE = {}


def _cache(k, build):
    if k not in _CACHE:
        _CACHE[k] = build()
    return _CACHE[k]


ENGINES = {
    "embedded": eng_embedded,
    "tesseract5": eng_tesseract,
    "paddleocr": eng_paddleocr,
    "doctr": eng_doctr,
    "kraken_catmus": eng_kraken,
    "calamari": eng_calamari,
}


# -------------------------------------------------------------------- the runner


def cmd_run(only=None):
    manifest = json.loads((OUT / "manifest.json").read_text())
    doc = fitz.open(PDF)
    engines = {k: v for k, v in ENGINES.items() if not only or k in only}

    tok_f = open(OUT / "tokens.csv", "w", newline="")
    tok = csv.writer(tok_f)
    tok.writerow(["page_id", "bucket", "engine", "tok_idx", "text", "conf", "x0", "y0", "x1", "y1"])
    runs = []
    for row in manifest:
        img_path = PAGES / f"{row['page_id']}.png"
        ctx = {
            "img": cv2.imread(str(img_path)),
            "img_path": str(img_path),
            "page": doc[row["pdf_index"]],
        }
        for name, fn in engines.items():
            t0 = time.time()
            try:
                toks = fn(ctx)
                status = "ok"
            except Exception as e:
                toks, status = [], f"{type(e).__name__}: {e}".replace("\n", " ")[:200]
            dt = time.time() - t0
            for i, (text, conf, x0, y0, x1, y1) in enumerate(toks):
                tok.writerow(
                    [
                        row["page_id"],
                        row["bucket"],
                        name,
                        i,
                        text,
                        "" if conf is None else round(conf, 4),
                        *(round(v, 1) for v in (x0, y0, x1, y1)),
                    ]
                )
            confs = [c for _, c, *_ in toks if c is not None]
            runs.append(
                {
                    "page_id": row["page_id"],
                    "bucket": row["bucket"],
                    "engine": name,
                    "status": status,
                    "seconds": round(dt, 2),
                    "n_tokens": len(toks),
                    "mean_conf": round(float(np.mean(confs)), 4) if confs else "",
                }
            )
            print(
                f"{row['page_id']} {name:15s} {status[:60]:60s} " f"{len(toks):5d} tok {dt:6.2f}s"
            )
    tok_f.close()
    with open(OUT / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)
    _seed_gt(manifest)
    print(f"\n-> {OUT/'tokens.csv'}, {OUT/'runs.csv'}")


def _seed_gt(manifest):
    """Write GT starter files from the highest-mean-confidence working engine."""
    GT.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    t = pd.read_csv(OUT / "tokens.csv", keep_default_na=False)
    for row in manifest:
        pid = row["page_id"]
        if (GT / f"{pid}.txt").exists():
            continue
        sub = t[(t.page_id == pid) & (t.engine != "embedded")]
        if sub.empty:
            continue
        best = sub.groupby("engine").size().idxmax()
        text = " ".join(sub[sub.engine == best].text.astype(str))
        (GT / f"{pid}.seed-{best}.txt").write_text(text)


# ------------------------------------------------------------------ scoring/report


def levenshtein(a, b):
    """Edit distance over any sequence. Two-row DP."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def cer_wer(hyp, ref):
    ref, hyp = norm(ref), norm(hyp)
    if not ref:
        return None, None
    return (
        levenshtein(hyp, ref) / len(ref),
        levenshtein(hyp.split(), ref.split()) / len(ref.split()),
    )


def cmd_report():
    import pandas as pd

    t = pd.read_csv(OUT / "tokens.csv", keep_default_na=False)
    r = pd.read_csv(OUT / "runs.csv", keep_default_na=False)
    L = [
        "# OCR bake-off — Pierce 1890",
        "",
        f"{r.page_id.nunique()} pages x {r.engine.nunique()} engines, {DPI} DPI.",
        "",
    ]

    L += ["## Engine status", "", "| engine | ok pages | failed | note |", "|---|---|---|---|"]
    for e, g in r.groupby("engine"):
        bad = g[g.status != "ok"]
        L.append(
            f"| {e} | {(g.status=='ok').sum()} | {len(bad)} | "
            f"{bad.status.iloc[0] if len(bad) else ''} |"
        )

    L += [
        "",
        "## Throughput & yield (mean per page, ok runs only)",
        "",
        "| engine | s/page | tokens/page | mean conf |",
        "|---|---|---|---|",
    ]
    ok = r[r.status == "ok"]
    for e, g in ok.groupby("engine"):
        mc = pd.to_numeric(g.mean_conf, errors="coerce").mean()
        L.append(
            f"| {e} | {g.seconds.mean():.2f} | {g.n_tokens.mean():.0f} | "
            f"{'' if pd.isna(mc) else f'{mc:.3f}'} |"
        )

    L += [
        "",
        "## Confidence by page-type bucket",
        "",
        "| engine | bucket | n tok | mean | p10 | frac conf<0.7 |",
        "|---|---|---|---|---|---|",
    ]
    c = t[t.conf != ""].copy()
    c["conf"] = c.conf.astype(float)
    for (e, b), g in c.groupby(["engine", "bucket"]):
        L.append(
            f"| {e} | {b} | {len(g)} | {g.conf.mean():.3f} | "
            f"{g.conf.quantile(.1):.3f} | {(g.conf<0.7).mean():.2%} |"
        )

    L += [
        "",
        "## Low-confidence tokens (token-local garbling suspects)",
        "",
        "| engine | bucket | page | conf | token |",
        "|---|---|---|---|---|",
    ]
    for _, x in c[c.conf < 0.5].nsmallest(30, "conf").iterrows():
        L.append(f"| {x.engine} | {x.bucket} | {x.page_id} | {x.conf:.2f} | `{x.text}` |")

    # accuracy only where a human transcript exists
    gts = sorted(GT.glob("[!.]*.txt"))
    gts = [p for p in gts if ".seed-" not in p.name]
    L += ["", "## Accuracy vs ground truth", ""]
    if not gts:
        L += [
            f"No ground truth yet. Correct a `{GT}/<page_id>.seed-*.txt` file, "
            f"save it as `{GT}/<page_id>.txt`, re-run `report`.",
            "",
        ]
    else:
        L += ["| engine | pages | CER | WER |", "|---|---|---|---|"]
        acc = defaultdict(list)
        for p in gts:
            pid, ref = p.stem, p.read_text()
            for e, g in t[t.page_id == pid].groupby("engine"):
                cw = cer_wer(" ".join(g.text.astype(str)), ref)
                if cw[0] is not None:
                    acc[e].append(cw)
        for e, v in sorted(acc.items(), key=lambda kv: np.mean([x[0] for x in kv[1]])):
            L.append(
                f"| {e} | {len(v)} | {np.mean([x[0] for x in v]):.4f} | "
                f"{np.mean([x[1] for x in v]):.4f} |"
            )

    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def cmd_demo():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein(["a", "b"], ["a", "c"]) == 1
    cer, wer = cer_wer("Smart weed", "Smart-weed")
    assert abs(cer - 0.1) < 1e-9 and wer == 2.0, (cer, wer)  # 1 ref word, sub+insert
    assert cer_wer("x", "")[0] is None
    # Buckets come from percentile cuts, so what must survive a DPI change is the
    # *ranking* of pages, not the absolute feature value.
    doc = fitz.open(PDF)
    pgs = [73, 233, 868, 179, 499]  # figure-wrap, faint, dense, faint, faintest
    lo = [features(doc[i], render(doc[i], 150)) for i in pgs]
    hi = [features(doc[i], render(doc[i], 300)) for i in pgs]
    assert [f["n_words"] for f in lo] == [f["n_words"] for f in hi]
    for k, sep in (("ink_depth", 10), ("orphan_ink", 0.05), ("span", 0.01)):
        for a in range(len(pgs)):
            for b in range(len(pgs)):
                if lo[a][k] - lo[b][k] > sep:  # clearly ordered at 150 DPI
                    assert hi[a][k] > hi[b][k], (k, pgs[a], pgs[b], lo, hi)
    assert bucket_of(73, lo[0], 100, 900) == "figure_adjacent", lo[0]
    assert bucket_of(233, lo[1], 200, 900) == "faint_ink", lo[1]
    assert bucket_of(868, lo[2], 0, 400) == "dense_small_type", lo[2]
    print("ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("sample", "run", "report", "demo"), nargs="?", default="report"
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    args, remaining = parser.parse_known_args()
    PDF = args.pdf
    OUT.mkdir(parents=True, exist_ok=True)
    {
        "sample": cmd_sample,
        "run": lambda: cmd_run(remaining or None),
        "report": cmd_report,
        "demo": cmd_demo,
    }[args.command]()
