"""python3 -m docai probe            # auth, processor types, no spend
python3 -m docai smoke            # 10 chosen pages  (~$0.10)
python3 -m docai sweep            # all 1,034 pages  (~$10.34)
python3 -m docai compare          # vs the fused consensus
"""

import argparse

from . import compare as cmp_mod
from . import run as run_mod
from .config import find_pdf, settings

# Ten pages picked so the smoke test can fail in every way that matters, not just succeed:
#   plates the three detectors agree on, pages only one detector found, a blank front-matter
#   page that produced a false positive, and two pages whose Fig. label is disputed.
SMOKE_PAGES = [
    21,
    32,
    36,
    74,  # 3-vote plates (incl. the p74 lung plate)
    2,  # blank front matter -> orphan_ink false positive
    5,
    84,
    261,  # found by exactly one detector each
    138,
    312,
]  # disputed Fig. labels between detectors


def cmd_probe(a):
    from . import client as cl

    s = settings()
    print(f"project  : {s['project']}")
    print(f"location : {s['location']}")
    print(f"creds    : {s['credentials'] or '(ADC default)'}")
    print(f"pdf      : {find_pdf(a.pdf)}")
    c = cl.make_client(s["location"])
    parent = c.common_location_path(s["project"], s["location"])
    types = sorted(t.type_ for t in c.fetch_processor_types(parent=parent).processor_types)
    print(f"processor types visible: {len(types)}")
    for t in types:
        if "LAYOUT" in t or "OCR" in t:
            print(f"  {t}")
    procs = list(c.list_processors(parent=parent).processors)
    print(f"existing processors: {len(procs)}")
    for p in procs:
        print(f"  {p.display_name}  {p.type_}  {p.state.name}  {p.name.split('/')[-1]}")


def cmd_enable(a):
    from . import client as cl

    s = settings()
    print(f"enabling documentai.googleapis.com on {s['project']} ...")
    print(cl.enable_api(s["project"]))


def cmd_smoke(a):
    pages = [int(x) for x in a.pages.split(",")] if a.pages else SMOKE_PAGES
    idx = [p - 1 for p in pages]
    print(f"{len(idx)} pages, ~${len(idx) * 0.01:.2f} at $10/1000 pages")
    run_mod.process_pages(find_pdf(a.pdf), idx, settings(), quiet=a.quiet, create=not a.no_create)


def cmd_sweep(a):
    import fitz

    pdf = find_pdf(a.pdf)
    n = fitz.open(pdf).page_count
    idx = list(range(n if not a.limit else min(a.limit, n)))
    print(f"{len(idx)} pages, ~${len(idx) * 0.01:.2f} at $10/1000 pages")
    if not a.yes:
        raise SystemExit("this spends money: re-run with --yes to confirm")
    run_mod.process_pages(pdf, idx, settings(), quiet=a.quiet, create=not a.no_create)


def cmd_ocrtest(a):
    """Does OCR_PROCESSOR return the per-word boxes the highlight NFR needs?"""
    from . import ocr

    pages = [int(x) for x in a.pages.split(",")] if a.pages else [74, 21]
    print(f"{len(pages)} pages via Document OCR")
    ocr.run(find_pdf(a.pdf), [p - 1 for p in pages], settings(), quiet=a.quiet, save_raw=True)


def cmd_ocrsweep(a):
    """Document OCR over the whole book -- per-word boxes for every page."""
    import fitz

    from . import ocr

    pdf = find_pdf(a.pdf)
    n = fitz.open(pdf).page_count
    idx = list(range(min(a.limit, n) if a.limit else n))
    print(f"{len(idx)} pages, ~${len(idx) * 0.0015:.2f} at $1.50/1000 pages")
    if not a.yes:
        raise SystemExit("this spends money: re-run with --yes to confirm")
    ocr.run(pdf, idx, settings(), quiet=a.quiet)


def cmd_compare(a):
    cmp_mod.report(thr=a.iou)


def cmd_selftest(a):
    """Exercise the parsing logic on stub blocks -- no network, no spend.

    The response shape is the part most likely to break silently (nested blocks, a block
    with no geometry, normalized -> pixel conversion), so that is what gets pinned here.
    """
    from types import SimpleNamespace as N

    from . import parse as ps

    def vert(x, y):
        return N(x=x, y=y)

    def block(type_, text, box, children=(), block_id="b"):
        bb = (
            None
            if box is None
            else N(
                normalized_vertices=[
                    vert(box[0], box[1]),
                    vert(box[2], box[1]),
                    vert(box[2], box[3]),
                    vert(box[0], box[3]),
                ]
            )
        )
        return N(
            block_id=block_id,
            bounding_box=bb,
            page_span=N(page_start=1, page_end=1),
            text_block=N(type=type_, text=text, blocks=list(children)),
            table_block=None,
            list_block=None,
        )

    inner = block("figure", "A vertebra of the neck", [0.1, 0.2, 0.5, 0.6], block_id="in")
    outer = block("paragraph", "body text", [0.0, 0.0, 1.0, 1.0], [inner], block_id="out")
    doc = N(document_layout=N(blocks=[outer]), pages=[N(tokens=[1, 2, 3])])

    assert len(list(ps.flatten(doc.document_layout.blocks))) == 2, "nested block missed"
    assert ps.is_figure("figure") and ps.is_figure("IMAGE") and not ps.is_figure("paragraph")
    assert ps.block_bbox_norm(inner) == [0.1, 0.2, 0.5, 0.6]
    assert ps.block_bbox_norm(block("figure", "", None)) is None  # no geometry -> None

    sv = ps.survey(doc)
    assert sv["token_count"] == 3 and sv["blocks_with_bbox"] == 2

    rows = ps.figure_rows(doc, ["p0074"], {0: (1000, 2000)})
    assert len(rows) == 1, "only the figure block should become a row"
    r = rows[0]
    assert r["bbox_img"] == [100, 400, 500, 1200], r["bbox_img"]  # norm -> px at size
    assert r["page_id"] == "p0074" and r["cls"] == "figure"
    # the printed/generated caption split is load-bearing downstream -- never merge them
    assert r["caption_printed"] is None
    assert r["caption_generated"] == "A vertebra of the neck"

    # a block whose page_span points past the submitted slice must be dropped, not crash
    assert ps.figure_rows(doc, [], {}) == []
    print("ok (parse: nesting, geometry, page mapping, caption split)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="docai")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--pdf")
        p.add_argument("--quiet", action="store_true")
        p.add_argument(
            "--no-create",
            action="store_true",
            help="fail instead of creating a processor in the cloud project",
        )

    p = sub.add_parser("probe", help="auth + processor inventory, no spend")
    p.add_argument("--pdf")
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("enable", help="turn on the Document AI API for the project")
    p.set_defaults(fn=cmd_enable)

    p = sub.add_parser("smoke", help="the 10 chosen pages")
    common(p)
    p.add_argument("--pages", help="1-indexed, comma separated; overrides the default 10")
    p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("sweep", help="the whole book")
    common(p)
    p.add_argument("--limit", type=int)
    p.add_argument("--yes", action="store_true", help="confirm the spend")
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("ocrtest", help="Document OCR: do we get per-word boxes?")
    common(p)
    p.add_argument("--pages", help="1-indexed, comma separated (default 74,21)")
    p.set_defaults(fn=cmd_ocrtest)

    p = sub.add_parser("ocrsweep", help="Document OCR over the whole book")
    common(p)
    p.add_argument("--limit", type=int)
    p.add_argument("--yes", action="store_true", help="confirm the spend")
    p.set_defaults(fn=cmd_ocrsweep)

    p = sub.add_parser("compare", help="vs fused consensus")
    p.add_argument("--iou", type=float, default=0.5)
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("selftest", help="offline: parsing logic only")
    p.set_defaults(fn=cmd_selftest)

    a = ap.parse_args(argv)
    a.fn(a)
