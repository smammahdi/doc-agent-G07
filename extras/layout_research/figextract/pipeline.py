"""Detect figures, bind their label and caption, write crops + provenance JSONL."""

import json
import time

import cv2

from . import captions as cap
from .config import DPI, run_dir
from .geometry import match_nearest
from .pdf import open_book, page_ids, render

LABEL_GAP = 0.12  # a caption/label must sit within this fraction of page height


def extract(
    detector, pdf_path, out_base=None, pages=None, limit=None, dpi=DPI, overwrite=False, quiet=False
):
    """-> path to figures.jsonl. Resumable: re-running skips pages already recorded."""
    out = run_dir(detector.name, out_base)
    jsonl = out / "figures.jsonl"
    ocr, ocr_kind = cap.make_ocr()

    done = set()
    if jsonl.exists() and not overwrite:
        done = {json.loads(line)["page_id"] for line in jsonl.open() if line.strip()}
        if done and not quiet:
            print(f"resuming: {len(done)} pages already done")

    doc = open_book(pdf_path)
    t0, n_fig, n_pages = time.time(), 0, 0
    with jsonl.open("a" if done else "w") as f:
        for i, page_id in page_ids(doc, pages, limit):
            if page_id in done:
                continue
            n_pages += 1
            page = doc[i]
            img = render(page, dpi)
            h, w = img.shape[:2]
            figs, caps, lbls = detector.split(detector.detect(page, img))

            for j, fg in enumerate(figs, 1):
                row = _one(detector, img, fg, caps, lbls, ocr, ocr_kind, h, w)
                fid = f"fig_{i + 1:04d}_{j:02d}"
                cv2.imwrite(str(out / "crops" / f"{fid}.png"), img[fg.y0 : fg.y1, fg.x0 : fg.x1])
                row.update(
                    figure_id=fid,
                    page_id=page_id,
                    pdf_index=i,
                    dpi=dpi,
                    crop_path=f"crops/{fid}.png",
                    detector=detector.name,
                    impl=getattr(detector, "impl", detector.name),
                )
                f.write(json.dumps(row) + "\n")
                f.flush()  # resumable: a killed run loses at most one page
                n_fig += 1
                if not quiet:
                    print(
                        f"{page_id} {fid} {fg.cls:12s} {fg.score:.2f} "
                        f"{str(row['fig_label'] or '-'):9s} {row['caption_printed'][:46]}"
                    )

    dt = time.time() - t0
    if not quiet:
        print(
            f"\n{n_fig} figures over {n_pages} pages in {dt/60:.1f} min "
            f"({dt/max(n_pages,1):.2f}s/page) -> {jsonl}"
        )
    return jsonl


def _ink(img, box):
    """(dark fraction, contrast) for a crop, with an ABSOLUTE threshold.

    Otsu is useless here: on a blank page it adapts and binarises paper grain into
    'ink', so every blank crop looks full of content. A fixed cutoff does not.
    """
    import numpy as np

    crop = img[box[1] : box[3], box[0] : box[2]]
    if crop.size == 0:
        return 0.0, 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return round(float((g < 128).mean()), 4), round(float(np.std(g)), 1)


def _one(detector, img, fg, caps, lbls, ocr, ocr_kind, h, w):
    """Bind one figure to its caption and label. Matches are exclusive."""
    gap = LABEL_GAP * h
    cap_box = match_nearest(fg, caps, gap)
    if cap_box is not None:
        caps.remove(cap_box)  # else one caption serves every figure
    lbl_box = match_nearest(fg, lbls, gap)
    if lbl_box is not None:
        lbls.remove(lbl_box)

    # The `Fig. N.` label lives inside the top of the figure box, so read that strip
    # whether or not the detector also gave us an explicit label region.
    label_text = cap.read_label_strip(img, fg.box, ocr)
    if lbl_box is not None:
        label_text = cap.read_region(img, lbl_box, ocr, psm=7) + "\n" + label_text
    label_text += "\n" + cap.read_band(img, fg.box, ocr, above=True)

    if cap_box is not None:
        caption_text, whole, src = cap.read_region(img, cap_box, ocr), True, "caption_box"
    else:
        caption_text, whole, src = cap.read_band(img, fg.box, ocr, above=False), False, "band"

    fig_label, caption = cap.parse(label_text, caption_text, whole_caption=whole)
    dark, std = _ink(img, fg.box)
    return {
        "cls": fg.cls,
        "score": round(fg.score, 4),
        "ink_frac": dark,
        "ink_std": std,
        # blank front-matter paper detected as a figure. Flagged, never silently dropped:
        # an audit trail that hides its own false positives is not an audit trail.
        "is_blank": bool(dark < 0.01 and std < 25),
        "bbox_img": [fg.x0, fg.y0, fg.x1, fg.y1],
        "bbox_norm": [
            round(fg.x0 / w, 5),
            round(fg.y0 / h, 5),
            round(fg.x1 / w, 5),
            round(fg.y1 / h, 5),
        ],
        "caption_bbox_img": list(cap_box.box) if cap_box else None,
        "label_bbox_img": list(lbl_box.box) if lbl_box else None,
        "fig_label": fig_label,
        "caption_printed": caption,
        "caption_generated": None,  # reserved for the VLM pass; never merged
        "caption_source": f"{ocr_kind}:{src}",
    }
