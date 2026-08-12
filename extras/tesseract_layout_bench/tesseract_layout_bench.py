#!/usr/bin/env python3
"""
Tesseract OCR — Layout-Aware Benchmark (Kaggle script version)
Pierce 1890 Medical Adviser · Team G07 · A2 OCR benchmarking

Run on Kaggle with:
    python tesseract_layout_bench.py

Strategy:
  1. Load Chandra chunks.jsonl layout blocks (text regions only, figures excluded).
  2. Render each PDF page at 300 DPI using PyMuPDF.
  3. Crop each Chandra text region from the page image.
  4. Run Tesseract on each crop individually.
  5. Reassemble page transcript in reading order.
  6. Score against hand-verified held-out labels (grading_kit/labels.jsonl).
  7. Also score Chandra's own text content against the same GT for comparison.
  8. Save page_transcripts.jsonl, heldout_scores.csv, report.md.

Kaggle inputs expected:
  PDF:     /kaggle/input/datasets/kmazd1110/dl-peoples-common-sense-med-advisor/EN_The-Peoples-Common-Sense-Medical-Adviser.pdf
  Chandra: /kaggle/input/datasets/cruelangelssprint/pierce-1890-figure-and-ocr-outputs/**  (chunks.jsonl auto-discovered)
  GT:      /kaggle/input/datasets/kmazd1110/pierce-book-gt/labels.jsonl
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# ── Install dependencies (Kaggle environment) ────────────────────────────────
def _install():
    subprocess.run(
        ["apt-get", "install", "-y", "-q", "tesseract-ocr", "tesseract-ocr-eng"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "pytesseract", "pymupdf", "jiwer"],
        check=True,
    )

_install()

import cv2                          # noqa: E402
import fitz                         # noqa: E402  PyMuPDF
import numpy as np                  # noqa: E402
import pytesseract                  # noqa: E402
from jiwer import cer as compute_cer, wer as compute_wer  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
PDF_PATH = Path(
    "/kaggle/input/datasets/kmazd1110/dl-peoples-common-sense-med-advisor"
    "/EN_The-Peoples-Common-Sense-Medical-Adviser.pdf"
)
CHANDRA_PATH = Path(
    "/kaggle/input/datasets/cruelangelssprint/pierce-1890-figure-and-ocr-outputs"
    "/chandra/chunks.jsonl"
)
LABELS_PATH = Path("/kaggle/input/datasets/kmazd1110/pierce-book-gt/labels.jsonl")

OUT_DIR = Path("/kaggle/working/tesseract_layout_bench")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
TESSERACT_CFG = "--psm 6"  # single uniform block of text per crop


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chandra_label_kind(label) -> str:
    if label is None:
        return "text"
    l = str(label).lower().strip()
    if any(kw in l for kw in ["image", "figure", "diagram", "picture"]):
        return "figure"
    return "text"


def render_page(doc: fitz.Document, page_idx: int, dpi: int = DPI) -> np.ndarray:
    """Render a PDF page to a numpy BGR image."""
    pix = doc[page_idx].get_pixmap(dpi=dpi)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(
        arr, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR
    )


def bbox_to_pixel(
    bbox: list, page_box: list, img_w: int, img_h: int
) -> tuple[int, int, int, int] | None:
    """
    Convert Chandra bbox [x0,y0,x1,y1] in page_box [W, H] coords
    to pixel coords in our rendered image (img_w x img_h).
    Returns None if page_box dimensions are zero/invalid (bad Chandra data).
    """
    cw, ch = float(page_box[0]), float(page_box[1])
    if cw == 0.0 or ch == 0.0:
        return None  # guard: skip blocks with undefined page_box dimensions
    x0, y0, x1, y1 = bbox
    px0 = max(0,     int(x0 / cw * img_w))
    py0 = max(0,     int(y0 / ch * img_h))
    px1 = min(img_w, int(x1 / cw * img_w))
    py1 = min(img_h, int(y1 / ch * img_h))
    return px0, py0, px1, py1


def ocr_crop(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> str:
    """Crop a region from a page image and run Tesseract on it."""
    if x1 <= x0 or y1 <= y0:
        return ""
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return ""
    text = pytesseract.image_to_string(crop, lang="eng", config=TESSERACT_CFG)
    return text.strip()


def normalize(text: str) -> str:
    """Normalize whitespace for fair CER/WER computation."""
    return re.sub(r"\s+", " ", text).strip()


# ── Load Chandra layout blocks ────────────────────────────────────────────────
print("Loading Chandra layout blocks...")
if CHANDRA_PATH is None or not CHANDRA_PATH.exists():
    print("[WARN] chunks.jsonl not found — will run full-page Tesseract as fallback.")
    USE_LAYOUT = False
else:
    USE_LAYOUT = True
    print(f"Using layout blocks from: {CHANDRA_PATH}")
chandra_blocks: dict[str, list[dict]] = defaultdict(list)
if USE_LAYOUT:
    with CHANDRA_PATH.open(encoding="utf-8") as f:  # type: ignore[union-attr]
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            book_page = row.get("book_page")
            if book_page is None:
                continue
            label = row.get("label", "")
            if _chandra_label_kind(label) == "figure":
                continue  # Skip non-text blocks; Tesseract shouldn't read figures
            page_id = f"p{int(book_page):04d}"
            chandra_blocks[page_id].append({
                "page_box": row.get("page_box"),  # [W, H] Chandra image space
                "bbox": row.get("bbox"),           # [x0, y0, x1, y1] in page_box coords
                "label": label,
                "content": row.get("content", ""), # Chandra's own OCR text (pseudo-GT)
            })
    total_pages_with_blocks = len(chandra_blocks)
    total_blocks = sum(len(v) for v in chandra_blocks.values())
    print(f"Loaded {total_pages_with_blocks} pages with {total_blocks} text blocks.")
else:
    print("No Chandra layout blocks — will use full-page OCR fallback for all pages.")

# ── Load ground-truth labels ──────────────────────────────────────────────────
print("Loading ground-truth labels...")
assert LABELS_PATH.exists(), f"Missing: {LABELS_PATH}"
gt_labels: dict[str, str] = {}
with LABELS_PATH.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        gt_labels[row["page_id"]] = row["text"]
print(f"Loaded {len(gt_labels)} ground-truth pages: {sorted(gt_labels.keys())}")

# ── Full-corpus Tesseract OCR with layout masking ─────────────────────────────
if USE_LAYOUT:
    print("\nRunning Tesseract with Chandra layout masking on all pages...")
else:
    print("\nRunning Tesseract full-page (no layout masking — chunks.jsonl unavailable)...")

assert PDF_PATH.exists(), f"Missing PDF: {PDF_PATH}"
doc = fitz.open(str(PDF_PATH))
print(f"PDF loaded: {doc.page_count} pages")

results: list[dict] = []

# If we have Chandra layout, process only pages Chandra observed.
# Fallback: process ALL pages (needed for full corpus + GT scoring).
if USE_LAYOUT:
    all_page_ids = sorted(chandra_blocks.keys())
else:
    all_page_ids = [f"p{i+1:04d}" for i in range(doc.page_count)]

t_total = time.time()

for page_num, page_id in enumerate(all_page_ids):
    pdf_idx = int(page_id[1:]) - 1  # page_id 'p0024' → PDF index 23
    if pdf_idx < 0 or pdf_idx >= doc.page_count:
        continue

    t0 = time.time()
    img = render_page(doc, pdf_idx, DPI)
    img_h, img_w = img.shape[:2]

    if USE_LAYOUT:
        # Layout-masked: crop each text region and OCR it individually
        blocks = chandra_blocks[page_id]
        block_texts: list[str] = []
        for blk in blocks:
            page_box = blk.get("page_box")
            bbox = blk.get("bbox")
            if not page_box or not bbox or len(page_box) < 2 or len(bbox) < 4:
                continue
            px_result = bbox_to_pixel(bbox, page_box, img_w, img_h)
            if px_result is None:
                continue  # skip block with zero/invalid page_box dimensions
            px0, py0, px1, py1 = px_result
            text = ocr_crop(img, px0, py0, px1, py1)
            if text:
                block_texts.append(text)
        page_text = "\n".join(block_texts)
        n_blocks = len(block_texts)
    else:
        # Fallback: run Tesseract on the full page
        page_text = pytesseract.image_to_string(img, lang="eng", config=TESSERACT_CFG).strip()
        n_blocks = 1

    elapsed = time.time() - t0

    results.append({
        "page_id": page_id,
        "text": page_text,
        "n_blocks": n_blocks,
        "elapsed_s": round(elapsed, 2),
    })

    if page_num % 50 == 0 or page_num < 5:
        print(f"  [{page_num:4d}/{len(all_page_ids)}] {page_id} — {n_blocks} blocks, {elapsed:.1f}s")

print(f"\nAll pages done in {time.time() - t_total:.1f}s")

# Save full transcripts
transcripts_path = OUT_DIR / "page_transcripts.jsonl"
with transcripts_path.open("w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Saved transcripts -> {transcripts_path}")

# ── Score against held-out ground truth ───────────────────────────────────────
print("\n=== Scoring Tesseract vs held-out ground truth ===")
transcript_map: dict[str, str] = {r["page_id"]: r["text"] for r in results}
scored_pages: list[dict] = []

for page_id, ref_text in sorted(gt_labels.items()):
    hyp_text = transcript_map.get(page_id, "")
    ref_norm = normalize(ref_text)
    hyp_norm = normalize(hyp_text)

    if not ref_norm:
        continue

    page_cer = compute_cer(ref_norm, hyp_norm)
    page_wer = compute_wer(ref_norm, hyp_norm)

    # Word F1 (set-based)
    ref_words = set(ref_norm.lower().split())
    hyp_words = set(hyp_norm.lower().split())
    tp = len(ref_words & hyp_words)
    precision = tp / len(hyp_words) if hyp_words else 0.0
    recall = tp / len(ref_words) if ref_words else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    scored_pages.append({
        "page_id": page_id,
        "cer": round(page_cer, 4),
        "wer": round(page_wer, 4),
        "word_f1": round(f1, 4),
        "ref_chars": len(ref_norm),
        "hyp_chars": len(hyp_norm),
    })
    print(f"  {page_id}  CER={page_cer:.4f}  WER={page_wer:.4f}  Word-F1={f1:.4f}")

mean_cer = mean_wer = mean_f1 = float("nan")
if scored_pages:
    mean_cer = float(np.mean([p["cer"] for p in scored_pages]))
    mean_wer = float(np.mean([p["wer"] for p in scored_pages]))
    mean_f1 = float(np.mean([p["word_f1"] for p in scored_pages]))
    print(f"\n{'='*55}")
    print(f"AGGREGATE over {len(scored_pages)} held-out pages (Tesseract + layout masking)")
    print(f"  Mean CER    : {mean_cer:.4f}")
    print(f"  Mean WER    : {mean_wer:.4f}")
    print(f"  Mean Word F1: {mean_f1:.4f}")

    # Save scores CSV
    score_path = OUT_DIR / "heldout_scores.csv"
    with score_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scored_pages[0]))
        w.writeheader()
        w.writerows(scored_pages)
    print(f"Scores -> {score_path}")

# ── Worst failure ─────────────────────────────────────────────────────────────
worst = best = None
if scored_pages:
    worst = max(scored_pages, key=lambda p: p["cer"])
    best = min(scored_pages, key=lambda p: p["cer"])
    print(f"\nBest  page: {best['page_id']}  CER={best['cer']:.4f}")
    print(f"Worst page: {worst['page_id']}  CER={worst['cer']:.4f}")
    print(f"\n--- Ground truth {worst['page_id']} (first 400 chars) ---")
    print(gt_labels[worst["page_id"]][:400])
    print(f"\n--- Tesseract output {worst['page_id']} (first 400 chars) ---")
    print(transcript_map.get(worst["page_id"], "(not found)")[:400])

# ── Compare Chandra text vs Ground Truth ─────────────────────────────────────
print("\n=== Chandra pseudo-GT vs hand labels ===")
chandra_scores: list[tuple] = []
for page_id, ref_text in sorted(gt_labels.items()):
    chandra_text = " ".join(
        blk.get("content", "") for blk in chandra_blocks.get(page_id, [])
    )
    ref_norm = normalize(ref_text)
    hyp_norm = normalize(chandra_text)
    if not ref_norm or not hyp_norm:
        continue
    c = compute_cer(ref_norm, hyp_norm)
    w = compute_wer(ref_norm, hyp_norm)
    chandra_scores.append((page_id, c, w))
    print(f"  {page_id}  CER={c:.4f}  WER={w:.4f}")

c_cer = c_wer = float("nan")
if chandra_scores:
    c_cer = float(np.mean([s[1] for s in chandra_scores]))
    c_wer = float(np.mean([s[2] for s in chandra_scores]))
    print(f"\nChandra   mean CER={c_cer:.4f}  WER={c_wer:.4f}")
    print(f"Tesseract mean CER={mean_cer:.4f}  WER={mean_wer:.4f}")

# ── Write report.md ───────────────────────────────────────────────────────────
lines = [
    "# Tesseract Layout-Aware OCR Benchmark — Pierce 1890",
    "",
    "## Setup",
    f"- PDF: `{PDF_PATH.name}`",
    "- Layout source: Chandra `chunks.jsonl` — text blocks only (Image/Figure/Diagram excluded)",
    f"- OCR engine: Tesseract 5, lang=eng, `{TESSERACT_CFG}`",
    f"- Render DPI: {DPI}",
    f"- Pages processed: {len(results)} / {doc.page_count}",
    "",
    "## Results on held-out ground-truth pages (p0024–p0037)",
    "",
    "| page_id | CER | WER | Word F1 |",
    "|---|---|---|---|",
]
for p in sorted(scored_pages, key=lambda x: x["page_id"]):
    lines.append(f"| {p['page_id']} | {p['cer']:.4f} | {p['wer']:.4f} | {p['word_f1']:.4f} |")
if scored_pages:
    lines += [
        "",
        f"| **MEAN** | **{mean_cer:.4f}** | **{mean_wer:.4f}** | **{mean_f1:.4f}** |",
    ]

lines += ["", "## Worst failure"]
if worst:
    lines += [
        f"- Worst page: `{worst['page_id']}` — CER={worst['cer']:.4f}, WER={worst['wer']:.4f}",
        "- Likely cause: figure-adjacent regions or header/footer noise overlapping body text crop boundaries.",
    ]

lines += ["", "## Engine comparison (Chandra pseudo-GT vs Tesseract)"]
if chandra_scores:
    lines += [
        "| Engine | Mean CER | Mean WER |",
        "|---|---|---|",
        f"| Chandra OCR (pseudo-GT reference) | {c_cer:.4f} | {c_wer:.4f} |",
        f"| Tesseract 5 (layout-masked crops) | {mean_cer:.4f} | {mean_wer:.4f} |",
        "",
        "> Lower CER/WER = better. Chandra is our pseudo-ground-truth; "
        "Tesseract benchmarked here as the fine-tuning baseline.",
    ]

report_path = OUT_DIR / "report.md"
report_path.write_text("\n".join(lines) + "\n")
print(f"\nReport -> {report_path}")
print("\n".join(lines))
