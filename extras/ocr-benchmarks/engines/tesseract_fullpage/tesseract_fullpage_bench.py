#!/usr/bin/env python3
"""
Tesseract OCR — Direct Full-Page Benchmark (No Layout Detections Used)
Pierce 1890 Medical Adviser · Team G07 · A2 Baseline Benchmarking

Run on Kaggle:
    python tesseract_fullpage_bench.py

Strategy:
  1. Load ground truth labels from labels.jsonl (test set pages p0024 - p0047).
  2. Load each page image directly (un-cropped full page from heldout_pages or rendered PDF).
  3. Run Tesseract directly on the entire page image using --psm 3 (Automatic Page Segmentation).
  4. Compute CER, WER, and Word F1 scores against Ground Truth.
  5. Save tesseract_fullpage_results.jsonl, tesseract_fullpage_scores.csv, report.md.

Kaggle dataset inputs expected:
  - Labels: /kaggle/input/datasets/kmazd1110/gt-ocr-dl-dataset/ocr-gt-labels/labels.jsonl
  - Images: /kaggle/input/datasets/kmazd1110/gt-ocr-dl-dataset/ocr-gt-labels/heldout_pages/ (or PDF fallback)
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ── 1. Install Dependencies (Kaggle Environment) ─────────────────────────────
def _install():
    if os.path.exists("/kaggle"):
        try:
            subprocess.run(
                ["apt-get", "update", "-y", "-q"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["apt-get", "install", "-y", "-q", "tesseract-ocr", "tesseract-ocr-eng"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "pytesseract", "pymupdf", "pillow", "jiwer"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"[WARN] Dependency setup note: {e}")

_install()

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image

try:
    from jiwer import cer as compute_cer, wer as compute_wer
except ImportError:
    def levenshtein_distance(ref_seq, hyp_seq):
        n, m = len(ref_seq), len(hyp_seq)
        if n == 0: return m
        if m == 0: return n
        dp = list(range(m + 1))
        for i in range(1, n + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, m + 1):
                temp = dp[j]
                if ref_seq[i - 1] == hyp_seq[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[m]

    def compute_cer(ref, hyp):
        dist = levenshtein_distance(list(ref), list(hyp))
        return dist / float(len(ref)) if ref else 0.0

    def compute_wer(ref, hyp):
        ref_w, hyp_w = ref.split(), hyp.split()
        dist = levenshtein_distance(ref_w, hyp_w)
        return dist / float(len(ref_w)) if ref_w else 0.0

# ── 2. Configure Paths ────────────────────────────────────────────────────────
KAGGLE_LABELS_PATH = Path("/kaggle/input/datasets/kmazd1110/gt-ocr-dl-dataset/ocr-gt-labels/labels.jsonl")
LOCAL_LABELS_PATH = Path("grading_kit/labels.jsonl")

KAGGLE_IMAGES_DIR = Path("/kaggle/input/datasets/kmazd1110/gt-ocr-dl-dataset/ocr-gt-labels/heldout_pages")
LOCAL_IMAGES_DIR = Path("grading_kit/heldout_pages")

KAGGLE_PDF_PATH = Path("/kaggle/input/datasets/kmazd1110/dl-peoples-common-sense-med-advisor/EN_The-Peoples-Common-Sense-Medical-Adviser.pdf")
LOCAL_PDF_PATH = Path("data/raw/pierce-peoples-common-sense-medical-adviser-1890.pdf")

# Output Paths
if os.path.exists("/kaggle"):
    OUT_DIR = Path("/kaggle/working")
    OUT_RESULTS_JSONL = Path("/kaggle/working/tesseract_fullpage_results.jsonl")
    OUT_SCORES_CSV = Path("/kaggle/working/tesseract_fullpage_scores.csv")
    OUT_REPORT_MD = Path("/kaggle/working/report.md")
else:
    OUT_DIR = Path("extras/tesseract_fullpage_bench/output")
    OUT_RESULTS_JSONL = Path("extras/results/tesseract_fullpage_results.jsonl")
    OUT_SCORES_CSV = Path("extras/results/tesseract_fullpage_scores.csv")
    OUT_REPORT_MD = Path("extras/results/report.md")

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)

# Resolve Ground Truth Labels Path
if KAGGLE_LABELS_PATH.exists():
    LABELS_PATH = KAGGLE_LABELS_PATH
elif LOCAL_LABELS_PATH.exists():
    LABELS_PATH = LOCAL_LABELS_PATH
else:
    found = list(Path(".").rglob("labels.jsonl")) + list(Path("/kaggle").rglob("labels.jsonl"))
    LABELS_PATH = found[0] if found else KAGGLE_LABELS_PATH

IMAGES_DIR = KAGGLE_IMAGES_DIR if KAGGLE_IMAGES_DIR.exists() else (LOCAL_IMAGES_DIR if LOCAL_IMAGES_DIR.exists() else None)
PDF_PATH = KAGGLE_PDF_PATH if KAGGLE_PDF_PATH.exists() else (LOCAL_PDF_PATH if LOCAL_PDF_PATH.exists() else None)

print("=" * 80)
print(f"DIRECT FULL-PAGE TESSERACT BENCHMARK (NO LAYOUTS)")
print(f"GROUND TRUTH LABELS PATH: {LABELS_PATH}")
print(f"HELDOUT IMAGES DIR    : {IMAGES_DIR}")
print(f"PDF PATH              : {PDF_PATH}")
print("=" * 80)

# ── 3. Helper Functions ───────────────────────────────────────────────────────
def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("æ", "ae").replace("œ", "oe").replace("ﬁ", "fi").replace("ﬂ", "fl")
    return text

def compute_word_f1(ref: str, hyp: str) -> float:
    ref_w = set(normalize(ref).lower().split())
    hyp_w = set(normalize(hyp).lower().split())
    tp = len(ref_w & hyp_w)
    prec = tp / len(hyp_w) if hyp_w else 0.0
    rec = tp / len(ref_w) if ref_w else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

# ── 4. Load Ground Truth Labels ──────────────────────────────────────────────
print("Loading Ground Truth labels...")
gt_labels: dict[str, str] = {}
with LABELS_PATH.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            gt_labels[row["page_id"]] = row["text"]

test_page_ids = sorted(gt_labels.keys())
print(f"Found {len(test_page_ids)} test set pages: {test_page_ids}")

# Open PDF doc if needed
pdf_doc = fitz.open(str(PDF_PATH)) if (PDF_PATH and PDF_PATH.exists()) else None

# ── 5. Run Direct Full-Page Tesseract OCR (--psm 3) ───────────────────────────
print("\nRunning Direct Full-Page Tesseract OCR (--psm 3)...")
start_time = time.time()
results = []
scored_results = []

for pid in test_page_ids:
    page_start = time.time()
    book_page_num = int(pid.replace("p", ""))
    
    # Load un-cropped full page image
    img_pil = None
    if IMAGES_DIR and IMAGES_DIR.exists():
        for ext in [".jpg", ".png", ".jpeg"]:
            img_file = IMAGES_DIR / f"{pid}{ext}"
            if img_file.exists():
                img_pil = Image.open(img_file).convert("RGB")
                break
                
    if img_pil is None and pdf_doc is not None:
        pix = pdf_doc[book_page_num - 1].get_pixmap(dpi=300)
        img_pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
    if img_pil is None:
        print(f"[ERROR] Could not load image or PDF for page {pid}. Skipping.")
        continue
        
    # Run Tesseract directly on the entire un-cropped page image
    # --psm 3: Fully automatic page segmentation, but no OSD (default)
    tess_full_text = pytesseract.image_to_string(img_pil, lang="eng", config="--psm 3").strip()
    elapsed = time.time() - page_start
    
    results.append({
        "page_id": pid,
        "text": tess_full_text,
        "elapsed_s": round(elapsed, 2)
    })
    
    # Compute evaluation metrics against GT
    ref_norm = normalize(gt_labels[pid])
    hyp_norm = normalize(tess_full_text)
    
    cer = compute_cer(ref_norm, hyp_norm)
    wer = compute_wer(ref_norm, hyp_norm)
    f1 = compute_word_f1(ref_norm, hyp_norm)
    
    scored_results.append({
        "page_id": pid,
        "cer": cer,
        "wer": wer,
        "f1": f1,
        "gt_chars": len(ref_norm),
        "hyp_chars": len(hyp_norm),
        "elapsed_s": round(elapsed, 2)
    })
    print(f"  {pid:7s}: CER={cer:8.4f} | WER={wer:8.4f} | Word F1={f1:8.4f} | GT Chars={len(ref_norm):5d} | Hyp Chars={len(hyp_norm):5d}")

total_time = time.time() - start_time
print(f"\nCompleted Direct Full-Page Tesseract OCR on {len(results)} pages in {total_time:.2f} seconds.")

# ── 6. Save Transcripts Output ────────────────────────────────────────────────
with OUT_RESULTS_JSONL.open("w", encoding="utf-8") as f:
    for row in results:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Saved transcripts to: {OUT_RESULTS_JSONL}")

# ── 7. Print & Save Benchmark Evaluation Report ─────────────────────────────
print("\n" + "=" * 80)
print(f"{'PAGE':7s} | {'CER':8s} | {'WER':8s} | {'WORD F1':8s} | GT CHARS | HYP CHARS")
print("=" * 80)

for s in scored_results:
    print(f"{s['page_id']:7s} | {s['cer']:8.4f} | {s['wer']:8.4f} | {s['f1']:8.4f} | {s['gt_chars']:8d} | {s['hyp_chars']:9d}")

mean_cer = sum(s["cer"] for s in scored_results) / len(scored_results) if scored_results else 0.0
mean_wer = sum(s["wer"] for s in scored_results) / len(scored_results) if scored_results else 0.0
mean_f1 = sum(s["f1"] for s in scored_results) / len(scored_results) if scored_results else 0.0

print("=" * 80)
print(f"Direct Full-Page Tesseract MEAN: CER = {mean_cer:.4f} ({mean_cer*100:.2f}%) | WER = {mean_wer:.4f} ({mean_wer*100:.2f}%) | Word F1 = {mean_f1:.4f} ({mean_f1*100:.2f}%)")
print("=" * 80)

# Save CSV
with OUT_SCORES_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["page_id", "cer", "wer", "f1", "gt_chars", "hyp_chars", "elapsed_s"])
    writer.writeheader()
    writer.writerows(scored_results)
print(f"Saved CSV scores to: {OUT_SCORES_CSV}")

# Save Markdown Report
report_md = f"""# Direct Full-Page Tesseract Benchmark Report (No Layouts)

**Corpus**: *The People's Common Sense Medical Adviser* (1890, R. V. Pierce)  
**Layout Engine**: None (Direct Full-Page Image)  
**OCR Engine**: Tesseract 5 (`--psm 3`)  
**Evaluation Set**: {len(scored_results)} Test Pages  

## Overall Benchmark Summary

| Metric | Score | Percentage |
|---|---|---|
| **Mean Character Error Rate (CER)** | `{mean_cer:.4f}` | **{mean_cer*100:.2f}%** |
| **Mean Word Error Rate (WER)** | `{mean_wer:.4f}` | **{mean_wer*100:.2f}%** |
| **Mean Word F1 Score** | `{mean_f1:.4f}` | **{mean_f1*100:.2f}%** |

Saved predictions to `{OUT_RESULTS_JSONL.name}` for downstream comparison.
"""
OUT_REPORT_MD.write_text(report_md, encoding="utf-8")
print(f"Saved Report MD to: {OUT_REPORT_MD}")
