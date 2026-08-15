"""Document OCR processor — the geometry Layout Parser does not return.

Layout Parser gave us reading-order text and nothing else: its DocumentLayoutBlock carries
`page_span` only (no bounding box in the v1beta3 proto), and it returned `pages: []`, so no
tokens either. This module checks whether OCR_PROCESSOR supplies what both of our hard
requirements need: per-word boxes in image coordinates.
"""

import json
import time

from google.api_core import exceptions as gexc
from google.cloud import documentai_v1beta3 as documentai

from . import client as cl
from . import pdfio
from .config import DPI, ONLINE_MAX_PAGES, out_dir


def process(client, name, pdf_bytes):
    req = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf"),
    )
    return client.process_document(request=req).document


def poly_to_norm(poly):
    verts = list(getattr(poly, "normalized_vertices", []) or [])
    if not verts:
        return None
    xs, ys = [v.x for v in verts], [v.y for v in verts]
    return [round(min(xs), 5), round(min(ys), 5), round(max(xs), 5), round(max(ys), 5)]


def text_of(document, layout):
    """Resolve a layout's text_anchor against document.text."""
    out = []
    for seg in getattr(getattr(layout, "text_anchor", None), "text_segments", []) or []:
        out.append(document.text[int(seg.start_index) : int(seg.end_index)])
    return "".join(out).strip()


def survey(document):
    """Per page: how many tokens, and do they carry usable geometry?"""
    rows = []
    for i, p in enumerate(getattr(document, "pages", []) or []):
        toks = list(getattr(p, "tokens", []) or [])
        boxed = sum(1 for t in toks if poly_to_norm(t.layout.bounding_poly))
        rows.append(
            {
                "slice_page": i,
                "tokens": len(toks),
                "tokens_with_bbox": boxed,
                "lines": len(getattr(p, "lines", []) or []),
                "blocks": len(getattr(p, "blocks", []) or []),
            }
        )
    return rows


def word_rows(document, page_ids, sizes):
    """-> one row per word, with the pixel box a highlight overlay needs."""
    rows = []
    for i, p in enumerate(getattr(document, "pages", []) or []):
        if i >= len(page_ids):
            continue
        w, h = sizes[i]
        for t in getattr(p, "tokens", []) or []:
            bn = poly_to_norm(t.layout.bounding_poly)
            if bn is None:
                continue
            txt = text_of(document, t.layout)
            if not txt:
                continue
            rows.append(
                {
                    "page_id": page_ids[i],
                    "text": txt,
                    "bbox_norm": bn,
                    "bbox_img": [int(bn[0] * w), int(bn[1] * h), int(bn[2] * w), int(bn[3] * h)],
                    "confidence": round(float(t.layout.confidence), 4),
                    "engine": "docai_ocr",
                }
            )
    return rows


def run(
    pdf_path, page_indices, settings, dpi=DPI, quiet=False, chunk=ONLINE_MAX_PAGES, save_raw=False
):
    """OCR the given 0-based pages in <=15-page slices. Resumable.

    Pages already recorded in pages_done.txt are skipped, so a killed run costs one slice.
    Raw responses are off by default: they run ~1.8 MB per page, which is 1.8 GB over the
    book for data we already distilled into words.jsonl.
    """
    out = out_dir("ocr")
    words = out / "words.jsonl"
    marker = out / "pages_done.txt"
    done = set(marker.read_text().split()) if marker.exists() else set()
    todo = [i for i in page_indices if f"p{i + 1:04d}" not in done]
    if not todo:
        print(f"nothing to do: {len(page_indices)} pages already in {words}")
        return []

    c = cl.make_client(settings["location"])
    name = cl.get_or_create(
        c, settings["project"], settings["location"], display_name="figure-gt-ocr", type_match="ocr"
    )
    if not quiet:
        print(
            f"processor: {name}\n{len(todo)} pages, "
            f"~${len(todo) * 0.0015:.2f} at $1.50/1000 pages"
        )

    total, t0, stats = 0, time.time(), []
    with words.open("a") as wf, marker.open("a") as mf:
        for start in range(0, len(todo), chunk):
            part = todo[start : start + chunk]
            page_ids = [f"p{i + 1:04d}" for i in part]
            try:
                doc = process(c, name, pdfio.subset_bytes(pdf_path, part))
            except gexc.GoogleAPICallError as e:
                print(f"FAILED {page_ids[0]}..{page_ids[-1]}: {type(e).__name__}: {e}")
                continue

            if save_raw:
                (out / "raw").mkdir(exist_ok=True)
                (out / "raw" / f"{page_ids[0]}_{page_ids[-1]}.json").write_text(
                    json.dumps(documentai.Document.to_dict(doc), indent=1)[:4_000_000]
                )

            sizes = pdfio.page_sizes_px(pdf_path, part, dpi)
            rows = word_rows(doc, page_ids, sizes)
            for r in rows:
                wf.write(json.dumps(r) + "\n")
            wf.flush()
            mf.write("\n".join(page_ids) + "\n")
            mf.flush()
            stats.extend(survey(doc))
            total += len(rows)
            if not quiet:
                print(
                    f"  {page_ids[0]}..{page_ids[-1]}: {len(rows)} words "
                    f"({total} total, {time.time() - t0:.0f}s)"
                )

    unboxed = sum(s["tokens"] - s["tokens_with_bbox"] for s in stats)
    print(
        f"\n{total} word boxes over {len(todo)} pages in {(time.time()-t0)/60:.1f} min; "
        f"tokens missing a box: {unboxed}\n-> {words}"
    )
    return stats
