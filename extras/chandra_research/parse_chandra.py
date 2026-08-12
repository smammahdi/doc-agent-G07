"""Local step 3: Chandra's output -> our figure schema, then compare against what we have.

    python3 parse_chandra.py --selftest        # no data needed
    python3 parse_chandra.py output/           # after unzipping the Kaggle artifact

Chandra reports `bbox` in **pixels of the image it rendered**, at whatever DPI it chose,
which is almost certainly not our 300. Boxes are therefore normalised by the page size
Chandra used before being written out; `bbox_img` is then recomputed at our DPI so the
result is directly comparable to figextract and Document AI rows.

If the page size cannot be determined, the row is still written with `bbox_norm: null` and
counted in the report rather than dropped or guessed at.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FUSED = ROOT / "extras/layout_research/output/fused/figures.jsonl"
FIGURE_LABELS = ("image", "figure", "picture", "chart", "diagram")


def load_manifest(sample_dir):
    p = Path(sample_dir) / "manifest.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("sample_index_to_book_page", {})


def chunks_of(meta):
    """Chandra's metadata has shifted shape across versions; accept the plausible ones."""
    if isinstance(meta, list):
        return meta
    for key in ("chunks", "blocks", "pages", "results"):
        v = meta.get(key)
        if isinstance(v, list):
            # a list of pages, each holding its own chunks
            if v and isinstance(v[0], dict) and any(k in v[0] for k in ("chunks", "blocks")):
                out = []
                for i, page in enumerate(v, 1):
                    for c in page.get("chunks") or page.get("blocks") or []:
                        c = dict(c)
                        c.setdefault("page", page.get("page", i))
                        out.append(c)
                return out
            return v
    return []


def page_size(chunk, meta):
    """-> (w, h) in Chandra's pixel space, or None if it never told us."""
    for src in (chunk, meta if isinstance(meta, dict) else {}):
        for wk, hk in (
            ("page_width", "page_height"),
            ("width", "height"),
            ("image_width", "image_height"),
        ):
            w, h = src.get(wk), src.get(hk)
            if isinstance(w, (int, float)) and isinstance(h, (int, float)) and w and h:
                return float(w), float(h)
    return None


def is_figure(label):
    return any(h in (label or "").lower() for h in FIGURE_LABELS)


def to_rows(meta, index_to_page, dpi=300, page_px=None):
    """-> rows in figextract's schema. `page_px` maps book page id -> (w,h) at our dpi."""
    rows = []
    seq = Counter()
    for c in chunks_of(meta):
        label = c.get("label") or c.get("type") or ""
        if not is_figure(label):
            continue
        bbox = c.get("bbox") or c.get("boundingBox")
        if not bbox or len(bbox) != 4:
            continue
        page_no = str(c.get("page", 1))
        page_id = index_to_page.get(page_no, f"sample{page_no}")
        if isinstance(page_id, int):
            page_id = f"p{page_id:04d}"

        size = page_size(c, meta)
        bn = None
        if size:
            w, h = size
            bn = [
                round(bbox[0] / w, 5),
                round(bbox[1] / h, 5),
                round(bbox[2] / w, 5),
                round(bbox[3] / h, 5),
            ]
        seq[page_id] += 1
        row = {
            "page_id": page_id,
            "cls": label,
            "score": None,
            "bbox_chandra_px": list(bbox),
            "bbox_norm": bn,
            "caption_printed": None,
            "caption_generated": (c.get("content") or "").strip()[:2000] or None,
            "caption_source": "chandra:metadata",
            "detector": "chandra2",
            "impl": "chandra-ocr/hf",
            "figure_id": f"chandra_{page_id}_{seq[page_id]:02d}",
        }
        if bn and page_px and page_id in page_px:
            w, h = page_px[page_id]
            row["bbox_img"] = [int(bn[0] * w), int(bn[1] * h), int(bn[2] * w), int(bn[3] * h)]
            row["dpi"] = dpi
        rows.append(row)
    return rows


def book_page_sizes(page_ids, dpi=300, pdf=None):
    """Our render sizes, so Chandra boxes can be expressed in the same pixel space."""
    import fitz

    source = (
        Path(pdf) if pdf else ROOT / "data/raw/pierce-peoples-common-sense-medical-adviser-1890.pdf"
    )
    if not source.is_file():
        return {}
    doc = fitz.open(source)
    m = fitz.Matrix(dpi / 72, dpi / 72)
    out = {}
    for pid in page_ids:
        if not (pid.startswith("p") and pid[1:].isdigit()):
            continue
        i = int(pid[1:]) - 1
        if 0 <= i < doc.page_count:
            r = (doc[i].rect * m).irect
            out[pid] = (r.width, r.height)
    doc.close()
    return out


def compare_to_fused(rows):
    if not FUSED.exists():
        return "fused output not found; skipped comparison"
    fu = {}
    for line in FUSED.open():
        if line.strip():
            r = json.loads(line)
            fu.setdefault(r["page_id"], []).append(r)
    lines = ["| page | chandra | fused | fused vote tiers |", "|---|---|---|---|"]
    for pid in sorted({r["page_id"] for r in rows}):
        f = fu.get(pid, [])
        lines.append(
            f"| {pid} | {sum(1 for r in rows if r['page_id'] == pid)} | "
            f"{len(f)} | {','.join(str(x['votes']) for x in f) or '-'} |"
        )
    return "\n".join(lines)


def selftest():
    meta = {
        "chunks": [
            {
                "page": 1,
                "label": "Figure",
                "bbox": [100, 200, 500, 800],
                "page_width": 1000,
                "page_height": 2000,
                "content": "<p>A vertebra</p>",
            },
            {
                "page": 1,
                "label": "Text",
                "bbox": [0, 0, 10, 10],
                "page_width": 1000,
                "page_height": 2000,
                "content": "body",
            },
            {"page": 2, "label": "Image", "bbox": [10, 20, 30, 40], "content": "no size given"},
        ]
    }
    rows = to_rows(meta, {"1": 74, "2": 21}, page_px={"p0074": (1385, 2213)})
    assert len(rows) == 2, rows  # Text is not a figure
    assert rows[0]["page_id"] == "p0074"
    assert rows[0]["bbox_norm"] == [0.1, 0.1, 0.5, 0.4]
    assert rows[0]["bbox_img"] == [138, 221, 692, 885], rows[0]["bbox_img"]
    assert rows[0]["caption_generated"] == "<p>A vertebra</p>"
    assert rows[0]["caption_printed"] is None  # never merge generated with printed
    assert rows[1]["bbox_norm"] is None  # unknown page size -> null, not a guess
    # nested page-list shape must flatten too
    assert len(chunks_of({"pages": [{"page": 1, "chunks": [{"label": "Figure"}]}]})) == 1
    print("ok (chandra parse: label filter, normalisation, page mapping, missing size)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", help="unzipped Kaggle output")
    ap.add_argument("--sample", default=str(HERE / "sample"))
    ap.add_argument("--pdf")
    ap.add_argument("--out", type=Path, default=HERE / "figures.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.out_dir:
        return selftest()

    metas = sorted(Path(a.out_dir).rglob("*_metadata.json"))
    if not metas:
        raise SystemExit(f"no *_metadata.json under {a.out_dir}")
    idx = load_manifest(a.sample)
    rows = []
    for m in metas:
        meta = json.loads(m.read_text())
        found = to_rows(meta, idx)
        pids = [r["page_id"] for r in found]
        found = to_rows(meta, idx, page_px=book_page_sizes(set(pids), pdf=a.pdf))
        rows += found
        print(f"{m.name}: {len(chunks_of(meta))} chunks, {len(found)} figure blocks")

    out = a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n{len(rows)} figure rows -> {out}")
    print(f"labels seen: {Counter(r['cls'] for r in rows)}")
    print(f"rows without geometry: {sum(1 for r in rows if r['bbox_norm'] is None)}")
    print("\n" + compare_to_fused(rows))


if __name__ == "__main__":
    main()
