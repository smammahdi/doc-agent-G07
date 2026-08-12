"""Drive the API over the book in 15-page slices.

No GCS, no batch job. Batch processing would mean creating a bucket, uploading a 65 MB
book, polling an LRO and cleaning up storage afterwards -- for 69 sync calls that cost the
same $10 per 1,000 pages. If throughput ever matters, swap this for batch_process_documents.
ponytail: sequential online calls, ~69 for the full book; move to batch only if it drags.
"""

import json
import time
from collections import Counter

from google.api_core import exceptions as gexc

from . import client as cl
from . import parse as ps
from . import pdfio
from .config import DPI, ONLINE_MAX_PAGES, out_dir


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def process_pages(
    pdf_path,
    page_indices,
    settings,
    name=None,
    dpi=DPI,
    quiet=False,
    out_name="layout_parser",
    create=True,
    save_raw=True,
):
    """Run the given 0-based page indices through Layout Parser. Resumable.

    Every page already present in figures.jsonl is skipped, so an interrupted sweep costs
    one slice, and a re-run costs nothing.
    """
    out = out_dir(out_name)
    jsonl = out / "figures.jsonl"
    done = set()
    if jsonl.exists():
        done = {json.loads(line)["page_id"] for line in jsonl.open() if line.strip()}
    surveyed = out / "survey.json"

    todo = [i for i in page_indices if f"p{i + 1:04d}" not in done]
    if not todo:
        print(f"nothing to do: all {len(page_indices)} pages already in {jsonl}")
        return jsonl

    c = cl.make_client(settings["location"])
    name = name or cl.get_or_create(
        c,
        settings["project"],
        settings["location"],
        processor_id=settings["processor_id"],
        create=create,
    )
    if not quiet:
        print(f"processor: {name}")

    surveys, n_fig, t0 = [], 0, time.time()
    with jsonl.open("a") as f:
        for part in chunks(todo, ONLINE_MAX_PAGES):
            page_ids = [f"p{i + 1:04d}" for i in part]
            pdf_bytes = pdfio.subset_bytes(pdf_path, part)
            try:
                doc = cl.process_pdf(c, name, pdf_bytes)
            except gexc.GoogleAPICallError as e:
                # a failed slice must be visible, not silently absent from the output
                print(f"FAILED {page_ids[0]}..{page_ids[-1]}: {type(e).__name__}: {e}")
                continue

            # Save the raw response BEFORE parsing it. A parse bug must never cost a second
            # paid call to recover from -- the answer is already on disk.
            if save_raw:
                (out / "raw").mkdir(exist_ok=True)
                (out / "raw" / f"{page_ids[0]}_{page_ids[-1]}.json").write_text(
                    json.dumps(_doc_to_dict(doc), indent=1)[:4_000_000]
                )

            imgs = {k: pdfio.render(pdf_path, i, dpi) for k, i in enumerate(part)}
            sizes = {k: (im.shape[1], im.shape[0]) for k, im in imgs.items()}
            sv = ps.survey(doc)
            sv["pages"] = f"{page_ids[0]}..{page_ids[-1]}"
            surveys.append(sv)

            rows = ps.figure_rows(doc, page_ids, sizes)
            for r in rows:
                k, j = r.pop("slice_index"), r.pop("_seq")
                fid = f"docai_{r['page_id']}_{j:02d}"
                (out / "crops").mkdir(exist_ok=True)
                img = imgs[k]
                dark, std = pdfio.ink(img, r["bbox_img"])
                pdfio.save_crop(img, r["bbox_img"], out / "crops" / f"{fid}.png")
                r.update(
                    figure_id=fid,
                    pdf_index=part[k],
                    dpi=dpi,
                    crop_path=f"crops/{fid}.png",
                    ink_frac=dark,
                    ink_std=std,
                    is_blank=bool(dark < 0.01 and std < 25),
                )
                f.write(json.dumps(r) + "\n")
                f.flush()
                n_fig += 1
                if not quiet:
                    print(
                        f"  {r['page_id']} {fid} {r['cls']:12s} "
                        f"{(r['caption_generated'] or '')[:60]}"
                    )

            if not quiet:
                print(
                    f"{sv['pages']}: {len(rows)} figure blocks | "
                    f"types={sv['block_types']} | tokens={sv['token_count']}"
                )

    merged = Counter()
    for s in surveys:
        merged.update(s["block_types"])
    surveyed.write_text(
        json.dumps({"slices": surveys, "block_types_total": dict(merged)}, indent=1)
    )
    dt = time.time() - t0
    print(
        f"\n{n_fig} figure blocks over {len(todo)} pages in {dt/60:.1f} min "
        f"(${len(todo) * 0.01:.2f} at $10/1000 pages) -> {jsonl}"
    )
    return jsonl


def _doc_to_dict(doc):
    """Proto -> plain dict, tolerating both proto-plus and raw protobuf objects."""
    try:
        from google.cloud import documentai_v1beta3 as documentai

        return documentai.Document.to_dict(doc)
    except Exception:
        from google.protobuf.json_format import MessageToDict

        return MessageToDict(getattr(doc, "_pb", doc))
