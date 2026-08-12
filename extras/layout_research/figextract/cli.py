"""Command line entry point.

python -m figextract probe
python -m figextract run --detector ppdoclayout_v3
python -m figextract run --detector doclayout_yolo --backend official --pages 21,74
python -m figextract run --all
python -m figextract compare
python -m figextract contact --detector ppdoclayout_v3
python -m figextract selftest
"""

import argparse
import sys
import traceback

from . import compare as cmp_mod
from . import fuse as fuse_mod
from .config import OUTPUT, find_pdf, run_dir
from .detectors import NAMES, build
from .pipeline import extract


def cmd_probe(a):
    import shutil
    from pathlib import Path

    from .config import DOCLAYOUT_ONNX, DOCLAYOUT_PT, EYNOLLAH_MODELS, PPV3_DIR

    print(f"python {sys.version.split()[0]}")
    for mod in (
        "numpy",
        "cv2",
        "fitz",
        "onnxruntime",
        "torch",
        "transformers",
        "pytesseract",
        "doclayout_yolo",
        "eynollah",
    ):
        try:
            m = __import__(mod)
            print(f"  {mod:16s} {getattr(m, '__version__', 'ok')}")
        except Exception as e:
            print(f"  {mod:16s} MISSING ({type(e).__name__})")
    print(f"tesseract binary : {shutil.which('tesseract') or 'MISSING'}")
    try:
        print(f"pdf              : {find_pdf(a.pdf)}")
    except FileNotFoundError as e:
        print(f"pdf              : {e}")
    for label, p in (
        ("doclayout .onnx", DOCLAYOUT_ONNX),
        ("doclayout .pt", DOCLAYOUT_PT),
        ("ppv3 weights", PPV3_DIR),
        ("eynollah models", EYNOLLAH_MODELS),
    ):
        print(f"{label:17s}: {'OK  ' if Path(p).exists() else 'MISSING'} {p}")
    print(f"output           : {OUTPUT}")


def cmd_run(a):
    pdf = find_pdf(a.pdf)
    names = NAMES if a.all else [a.detector]
    failed = []
    for n in names:
        print(f"\n=== {n} ===")
        try:
            det = build(
                n, backend=a.backend, weights=a.weights, conf=a.conf, models=a.models
            ).load()
            extract(
                det,
                pdf,
                out_base=a.out,
                pages=a.pages,
                limit=a.limit,
                dpi=a.dpi,
                overwrite=a.overwrite,
                quiet=a.quiet,
            )
        except Exception as e:
            # never silently skip a detector -- an unrun model must be visible as unrun
            failed.append((n, f"{type(e).__name__}: {e}"))
            print(f"FAILED {n}: {type(e).__name__}: {e}")
            if a.traceback:
                traceback.print_exc()
    if failed:
        print("\nfailed detectors:")
        for n, msg in failed:
            print(f"  {n}: {msg.splitlines()[0]}")
        if not a.all:
            sys.exit(1)


def cmd_contact(a):
    import json

    import cv2
    import numpy as np

    out = run_dir(a.detector, a.out)
    rows = [json.loads(line) for line in (out / "figures.jsonl").open() if line.strip()]
    if not rows:
        sys.exit("no figures; run first")
    rows = rows[: a.limit] if a.limit else rows
    cell, cols = 260, 6
    grid = np.full((cell * ((len(rows) + cols - 1) // cols), cell * cols, 3), 255, np.uint8)
    for n, r in enumerate(rows):
        im = cv2.imread(str(out / r["crop_path"]))
        if im is None:
            continue
        s = min((cell - 34) / im.shape[1], (cell - 34) / im.shape[0])
        im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))))
        y, x = (n // cols) * cell, (n % cols) * cell
        grid[y + 28 : y + 28 + im.shape[0], x + 4 : x + 4 + im.shape[1]] = im
        cv2.putText(
            grid,
            f"{r['page_id']} {r['fig_label'] or '-'}",
            (x + 4, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 160),
            1,
            cv2.LINE_AA,
        )
    p = out / "contact_sheet.png"
    cv2.imwrite(str(p), grid)
    print(f"{len(rows)} crops -> {p}")


def cmd_selftest(a):
    from .captions import parse
    from .geometry import Detection, iou, match_nearest, merge_boxes

    assert abs(iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-9
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 50 / 150) < 1e-9

    assert merge_boxes([(0, 0, 10, 10), (9, 0, 20, 10)], 0) == [(0, 0, 20, 10)]
    assert len(merge_boxes([(0, 0, 10, 10), (90, 90, 99, 99)], 2)) == 2

    fig = Detection("figure", 0.9, 100, 200, 400, 500)
    below = Detection("figure_caption", 0.9, 100, 510, 400, 540)
    above = Detection("figure_caption", 0.9, 100, 100, 400, 130)
    assert match_nearest(fig, [below, above], 400) is below  # nearer wins
    assert match_nearest(fig, [Detection("c", 0.9, 900, 510, 990, 540)], 400) is None

    assert parse("Fig. 45.", "View of the pulmonary circulation")[0] == "Fig. 45"
    assert parse("F1g, 18", "")[0] == "Fig. 18"  # OCR-garbled label
    assert parse("", "")[0] is None
    assert parse("Fig. 3. The human skeleton", "")[1] == "The human skeleton"
    assert (
        parse("Fig. 9.", "a\nLower end of the thigh-bone", whole_caption=True)[1]
        == "a Lower end of the thigh-bone"
    )

    # fusion: overlapping boxes from different detectors collapse to one voted cluster,
    # and each field is taken from the first detector in precedence order that has it
    from .fuse import cluster, fuse_cluster

    def _row(det, box, **kw):
        r = dict(
            detector=det,
            bbox_norm=box,
            bbox_img=[int(v * 1000) for v in box],
            score=0.9,
            figure_id="f01",
            crop_path="crops/f01.png",
            fig_label=None,
            caption_printed=None,
        )
        r.update(kw)
        return r

    rows = [
        _row("orphan_ink", [0.1, 0.1, 0.5, 0.5], caption_printed="a caption"),
        _row("doclayout_yolo", [0.11, 0.11, 0.51, 0.51], fig_label="Fig. 7"),
        _row("ppdoclayout_v3", [0.8, 0.8, 0.9, 0.9]),
    ]
    groups = cluster(rows, 0.5)
    assert sorted(len(g) for g in groups) == [1, 2]
    pair = next(g for g in groups if len(g) == 2)
    fused = fuse_cluster(pair, "p0074", 1)
    assert fused["votes"] == 2 and fused["voters"] == ["doclayout_yolo", "orphan_ink"]
    assert fused["confidence"] == "medium"
    assert fused["fig_label"] == "Fig. 7" and fused["fig_label_from"] == "doclayout_yolo"
    assert fused["caption_printed"] == "a caption"  # taken from the other voter
    assert fused["source_detector"] == "doclayout_yolo"  # higher precedence than orphan
    assert fused["crop_path"] == "../doclayout_yolo/crops/f01.png"

    # end-to-end on the known lung plate, with whatever detector is available
    import json

    from .pdf import open_book, render

    doc = open_book(find_pdf(a.pdf))
    img = render(doc[73])
    det = build("orphan_ink").load()
    figs, _, _ = det.split(det.detect(doc[73], img))
    assert figs, "orphan_ink found no figure on p74"
    assert max(f.area for f in figs) > 0.05 * img.shape[0] * img.shape[1]
    # coords must be JSON-serializable: cv2 hands back numpy int32, which json refuses,
    # and that only surfaces at write time -- i.e. hours into a full-book run
    json.dumps([list(f.box) for f in figs])
    print(f"ok (orphan_ink: {len(figs)} box(es) on p74)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="figextract")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--pdf")
        p.add_argument("--out", help=f"output base (default {OUTPUT})")

    p = sub.add_parser("probe", help="what this machine has")
    common(p)
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("run", help="extract figures")
    common(p)
    p.add_argument("--detector", default="ppdoclayout_v3", choices=NAMES)
    p.add_argument("--all", action="store_true", help="run every detector in turn")
    p.add_argument("--backend", choices=["auto", "official", "onnx"], default="auto")
    p.add_argument("--weights"), p.add_argument("--models")
    p.add_argument("--conf", type=float)
    p.add_argument("--pages", help="1-indexed, comma separated")
    p.add_argument("--limit", type=int)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--traceback", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("contact", help="visual QA sheet")
    common(p)
    p.add_argument("--detector", default="ppdoclayout_v3", choices=NAMES + ["fused"])
    p.add_argument("--limit", type=int)
    p.set_defaults(fn=cmd_contact)

    p = sub.add_parser("compare", help="cross-detector report")
    common(p)
    p.add_argument("--iou", type=float, default=0.5)
    p.set_defaults(fn=cmp_mod.cmd_compare)

    p = sub.add_parser("fuse", help="consensus-fuse the detector runs")
    common(p)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument(
        "--min-votes",
        type=int,
        default=1,
        help="drop clusters below this many detectors (default 1: keep all)",
    )
    p.set_defaults(fn=fuse_mod.cmd_fuse)

    p = sub.add_parser("selftest")
    common(p)
    p.set_defaults(fn=cmd_selftest)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
