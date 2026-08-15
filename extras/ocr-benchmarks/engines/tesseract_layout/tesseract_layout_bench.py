#!/usr/bin/env python3
"""
Tesseract OCR — PP-DocLayoutV3 Layout-Aware Benchmark
Pierce 1890 Medical Adviser · Team G07 · A2 OCR Benchmarking

Run on Kaggle:
    python tesseract_layout_bench.py

Kaggle dataset inputs:
  - Layout:  /kaggle/input/datasets/kmazd1110/ocr-layout-dataset/ocr-layout-dataset/ppdoclayout-v3/detections.jsonl
  - Labels:  /kaggle/input/datasets/kmazd1110/gt-ocr-dl-dataset/ocr-gt-labels/labels.jsonl
  - Images:  /kaggle/input/datasets/kmazd1110/gt-ocr-dl-dataset/ocr-gt-labels/heldout_pages/ (or PDF fallback)
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

# ── 1. Install System & Python Dependencies ──────────────────────────────────
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
            print(f"[WARN] Dependency setup warning: {e}")

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
KAGGLE_DET_PATH = Path("/kaggle/input/datasets/kmazd1110/ocr-layout-dataset/ocr-layout-dataset/ppdoclayout-v3/detections.jsonl")
LOCAL_DET_PATH = Path("extras/layout-benchmarks/outputs/ppdoclayout-v3/detections.jsonl")

KAGGLE_LABELS_PATH = Path("/kaggle/input/datasets/kmazd1110/gt-ocr-dl-dataset/ocr-gt-labels/labels.jsonl")
LOCAL_LABELS_PATH = Path("grading_kit/labels.jsonl")

KAGGLE_IMAGES_DIR = Path("/kaggle/input/datasets/kmazd1110/gt-ocr-dl-dataset/ocr-gt-labels/heldout_pages")
LOCAL_IMAGES_DIR = Path("grading_kit/heldout_pages")

KAGGLE_PDF_PATH = Path("/kaggle/input/datasets/kmazd1110/dl-peoples-common-sense-med-advisor/EN_The-Peoples-Common-Sense-Medical-Adviser.pdf")
LOCAL_PDF_PATH = Path("data/raw/pierce-peoples-common-sense-medical-adviser-1890.pdf")

# Output Paths
if os.path.exists("/kaggle"):
    OUT_DIR = Path("/kaggle/working/tesseract_ppdoclayout_v3")
    OUT_RESULTS_JSONL = Path("/kaggle/working/tesseract_ppdoclayout_v3_results.jsonl")
    OUT_SCORES_CSV = Path("/kaggle/working/tesseract_ppdoclayout_v3_scores.csv")
    OUT_REPORT_MD = Path("/kaggle/working/report.md")
else:
    OUT_DIR = Path("extras/tesseract_layout_bench/output")
    OUT_RESULTS_JSONL = Path("extras/results/tesseract_ppdoclayout_v3_results.jsonl")
    OUT_SCORES_CSV = Path("extras/results/tesseract_ppdoclayout_v3_scores.csv")
    OUT_REPORT_MD = Path("extras/results/report.md")

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)

# Discover actual layout detection path
if KAGGLE_DET_PATH.exists():
    DET_PATH = KAGGLE_DET_PATH
elif LOCAL_DET_PATH.exists():
    DET_PATH = LOCAL_DET_PATH
else:
    # Auto discover
    found = list(Path(".").rglob("*ppdoclayout-v3/detections.jsonl")) + list(Path("/kaggle").rglob("*ppdoclayout-v3/detections.jsonl"))
    DET_PATH = found[0] if found else KAGGLE_DET_PATH

# Discover actual labels path
if KAGGLE_LABELS_PATH.exists():
    LABELS_PATH = KAGGLE_LABELS_PATH
elif LOCAL_LABELS_PATH.exists():
    LABELS_PATH = LOCAL_LABELS_PATH
else:
    found = list(Path(".").rglob("labels.jsonl")) + list(Path("/kaggle").rglob("labels.jsonl"))
    LABELS_PATH = found[0] if found else KAGGLE_LABELS_PATH

# Discover actual images directory or PDF
IMAGES_DIR = KAGGLE_IMAGES_DIR if KAGGLE_IMAGES_DIR.exists() else (LOCAL_IMAGES_DIR if LOCAL_IMAGES_DIR.exists() else None)
PDF_PATH = KAGGLE_PDF_PATH if KAGGLE_PDF_PATH.exists() else (LOCAL_PDF_PATH if LOCAL_PDF_PATH.exists() else None)

print("=" * 80)
print(f"LAYOUT DETECTIONS PATH : {DET_PATH}")
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

# ── 5. Load PP-DocLayoutV3 Detections for Test Set Pages ───────────────────────
print("Loading PP-DocLayoutV3 layout detections...")
det_blocks: dict[str, list[dict]] = defaultdict(list)
if DET_PATH.exists():
    with DET_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                pid = row.get("page_id")
                if pid in gt_labels:
                    det_blocks[pid].append(row)
    print(f"Loaded layout detections for {len(det_blocks)} test set pages.")
else:
    print(f"[WARN] Layout file {DET_PATH} not found!")

# Open PDF doc if needed
pdf_doc = fitz.open(str(PDF_PATH)) if (PDF_PATH and PDF_PATH.exists()) else None

# ── 6. Run Tesseract OCR on Test Set Pages ────────────────────────────────────
print("\nRunning Tesseract OCR on PP-DocLayoutV3 text crops...")
start_time = time.time()
results = []
scored_results = []

for pid in test_page_ids:
    page_start = time.time()
    book_page_num = int(pid.replace("p", ""))
    
    # Load page image (from image file or rendered PDF)
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
        
    img_w, img_h = img_pil.size
    blocks = det_blocks.get(pid, [])
    
    # Separate text blocks vs figure blocks
    text_blocks = [b for b in blocks if not b.get("is_figure", False)]
    # Sort top-to-bottom y0, then left-to-right x0
    text_blocks.sort(key=lambda b: (b.get("bbox_norm", [0,0,0,0])[1], b.get("bbox_norm", [0,0,0,0])[0]))
    
    transcripts = []
    for b in text_blocks:
        norm = b.get("bbox_norm", [0, 0, 0, 0])
        px0 = max(0, int(norm[0] * img_w))
        py0 = max(0, int(norm[1] * img_h))
        px1 = min(img_w, int(norm[2] * img_w))
        py1 = min(img_h, int(norm[3] * img_h))
        
        if px1 <= px0 or py1 <= py0:
            continue
            
        crop = img_pil.crop((px0, py0, px1, py1))
        tess_text = pytesseract.image_to_string(crop, lang="eng", config="--psm 6").strip()
        if tess_text:
            transcripts.append(tess_text)
            
    full_transcript = "\n\n".join(transcripts)
    elapsed = time.time() - page_start
    
    results.append({
        "page_id": pid,
        "text": full_transcript,
        "n_blocks": len(text_blocks),
        "elapsed_s": round(elapsed, 2)
    })
    
    # Compute evaluation metrics against GT
    ref_norm = normalize(gt_labels[pid])
    hyp_norm = normalize(full_transcript)
    
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
        "n_blocks": len(text_blocks),
        "elapsed_s": round(elapsed, 2)
    })

total_time = time.time() - start_time
print(f"\nCompleted Tesseract OCR on {len(results)} pages in {total_time:.2f} seconds.")

# ── 7. Save Transcripts Output ────────────────────────────────────────────────
with OUT_RESULTS_JSONL.open("w", encoding="utf-8") as f:
    for row in results:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Saved transcripts to: {OUT_RESULTS_JSONL}")

# ── 8. Print & Save Benchmark Evaluation Report ─────────────────────────────
print("\n" + "=" * 90)
print(f"{'PAGE':7s} | {'CER':8s} | {'WER':8s} | {'WORD F1':8s} | {'BLOCKS':6s} | GT CHARS | HYP CHARS")
print("=" * 90)

for s in scored_results:
    print(f"{s['page_id']:7s} | {s['cer']:8.4f} | {s['wer']:8.4f} | {s['f1']:8.4f} | {s['n_blocks']:6d} | {s['gt_chars']:8d} | {s['hyp_chars']:9d}")

mean_cer = sum(s["cer"] for s in scored_results) / len(scored_results) if scored_results else 0.0
mean_wer = sum(s["wer"] for s in scored_results) / len(scored_results) if scored_results else 0.0
mean_f1 = sum(s["f1"] for s in scored_results) / len(scored_results) if scored_results else 0.0

print("=" * 90)
print(f"PP-DocLayoutV3 + Tesseract MEAN: CER = {mean_cer:.4f} ({mean_cer*100:.2f}%) | WER = {mean_wer:.4f} ({mean_wer*100:.2f}%) | Word F1 = {mean_f1:.4f} ({mean_f1*100:.2f}%)")
print("=" * 90)

# Save CSV
with OUT_SCORES_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["page_id", "cer", "wer", "f1", "gt_chars", "hyp_chars", "n_blocks", "elapsed_s"])
    writer.writeheader()
    writer.writerows(scored_results)
print(f"Saved CSV scores to: {OUT_SCORES_CSV}")

# Save Markdown Report
report_md = f"""# Tesseract + PP-DocLayoutV3 Layout OCR Benchmark Report

**Corpus**: *The People's Common Sense Medical Adviser* (1890, R. V. Pierce)  
**Layout Engine**: PP-DocLayoutV3 (`detections.jsonl`)  
**OCR Engine**: Tesseract 5 (`--psm 6`)  
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
