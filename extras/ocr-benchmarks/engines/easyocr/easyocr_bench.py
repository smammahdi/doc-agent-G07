#!/usr/bin/env python3
"""
EasyOCR Benchmark: Layout-Aware vs. Direct Full-Page (Optimized)
Pierce 1890 Medical Adviser · Team G07 · A2 OCR Benchmarking

Runs:
  1. EasyOCR + PP-DocLayoutV3 (Layout-Aware Crop OCR)
  2. EasyOCR Direct Full-Page (Optimized GPU batching & canvas scaling)
  3. Side-by-Side Comparison & Benchmark Report

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
                [sys.executable, "-m", "pip", "install", "-q", "easyocr", "pymupdf", "pillow", "jiwer"],
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
import torch
import easyocr
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
    OUT_DIR = Path("/kaggle/working")
    OUT_LAYOUT_JSONL = Path("/kaggle/working/easyocr_layout_results.jsonl")
    OUT_FULLPAGE_JSONL = Path("/kaggle/working/easyocr_fullpage_results.jsonl")
    OUT_SCORES_CSV = Path("/kaggle/working/easyocr_comparison_scores.csv")
    OUT_REPORT_MD = Path("/kaggle/working/easyocr_benchmark_report.md")
else:
    OUT_DIR = Path("extras/easyocr_bench/output")
    OUT_LAYOUT_JSONL = Path("extras/results/easyocr_layout_results.jsonl")
    OUT_FULLPAGE_JSONL = Path("extras/results/easyocr_fullpage_results.jsonl")
    OUT_SCORES_CSV = Path("extras/results/easyocr_comparison_scores.csv")
    OUT_REPORT_MD = Path("extras/results/easyocr_benchmark_report.md")

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_LAYOUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

# Resolve paths
if KAGGLE_DET_PATH.exists():
    DET_PATH = KAGGLE_DET_PATH
elif LOCAL_DET_PATH.exists():
    DET_PATH = LOCAL_DET_PATH
else:
    found = list(Path(".").rglob("*ppdoclayout-v3/detections.jsonl")) + list(Path("/kaggle").rglob("*ppdoclayout-v3/detections.jsonl"))
    DET_PATH = found[0] if found else KAGGLE_DET_PATH

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
print(f"EASYOCR BENCHMARK (LAYOUT-AWARE VS FULL-PAGE)")
print(f"CUDA Available        : {torch.cuda.is_available()}")
print(f"LAYOUT DETECTIONS PATH: {DET_PATH}")
print(f"GROUND TRUTH PATH     : {LABELS_PATH}")
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

# Ensure p0041 and p0043 reflect printed text rather than synthetic AI alt-text
if "p0041" in gt_labels and "A detailed black and white" in gt_labels["p0041"]:
    gt_labels["p0041"] = "33\n\nTHE MUSCLES.\n\nA representation of the superficial layer of muscles on the anterior portion of the body."
if "p0043" in gt_labels and "A detailed black and white" in gt_labels["p0043"]:
    gt_labels["p0043"] = "35\n\nTHE MUSCLES.\n\nA representation of the superficial layer of muscles on the posterior portion of the body."

test_page_ids = sorted(gt_labels.keys())
print(f"Loaded {len(test_page_ids)} test pages: {test_page_ids}")

# ── 5. Load PP-DocLayoutV3 Detections ─────────────────────────────────────────
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

# ── 6. Initialize EasyOCR Reader ─────────────────────────────────────────────
print("Initializing EasyOCR Reader (English)...")
gpu_flag = torch.cuda.is_available()
reader = easyocr.Reader(["en"], gpu=gpu_flag)
print(f"EasyOCR initialized successfully (GPU={gpu_flag}).")

pdf_doc = fitz.open(str(PDF_PATH)) if (PDF_PATH and PDF_PATH.exists()) else None

def get_page_image(pid: str) -> Image.Image | None:
    book_page_num = int(pid.replace("p", ""))
    if IMAGES_DIR and IMAGES_DIR.exists():
        for ext in [".jpg", ".png", ".jpeg"]:
            img_file = IMAGES_DIR / f"{pid}{ext}"
            if img_file.exists():
                return Image.open(img_file).convert("RGB")
    if pdf_doc is not None:
        pix = pdf_doc[book_page_num - 1].get_pixmap(dpi=300)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return None

# ── 7. Run Mode 1: Layout-Aware EasyOCR (PP-DocLayoutV3 Crops) ─────────────────
print("\n" + "=" * 80)
print("RUNNING MODE 1: LAYOUT-AWARE EASYOCR (PP-DOCLAYOUT-V3 CROPS)")
print("=" * 80)
layout_results = []
layout_scores = []
start_time = time.time()

for pid in test_page_ids:
    page_start = time.time()
    img_pil = get_page_image(pid)
    if img_pil is None:
        print(f"[ERROR] Could not load image for {pid}. Skipping.")
        continue
    
    img_w, img_h = img_pil.size
    blocks = det_blocks.get(pid, [])
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
            
        crop_pil = img_pil.crop((px0, py0, px1, py1))
        crop_np = np.array(crop_pil)
        
        # EasyOCR on crop region with GPU batching
        lines = reader.readtext(crop_np, batch_size=16, detail=0, paragraph=True)
        crop_text = "\n".join(lines).strip()
        if crop_text:
            transcripts.append(crop_text)
            
    full_transcript = "\n\n".join(transcripts)
    elapsed = time.time() - page_start
    
    layout_results.append({
        "page_id": pid,
        "text": full_transcript,
        "n_blocks": len(text_blocks),
        "elapsed_s": round(elapsed, 2)
    })
    
    ref_norm = normalize(gt_labels[pid])
    hyp_norm = normalize(full_transcript)
    
    cer = compute_cer(ref_norm, hyp_norm)
    wer = compute_wer(ref_norm, hyp_norm)
    f1 = compute_word_f1(ref_norm, hyp_norm)
    
    layout_scores.append({
        "page_id": pid,
        "cer": cer,
        "wer": wer,
        "f1": f1,
        "gt_chars": len(ref_norm),
        "hyp_chars": len(hyp_norm),
        "elapsed_s": round(elapsed, 2)
    })
    print(f"  [Layout] {pid:7s}: CER={cer:8.4f} | WER={wer:8.4f} | Word F1={f1:8.4f} | Time={elapsed:5.2f}s")

layout_total_time = time.time() - start_time
print(f"Completed Layout-Aware EasyOCR in {layout_total_time:.2f}s.")

# Save layout results
with OUT_LAYOUT_JSONL.open("w", encoding="utf-8") as f:
    for row in layout_results:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Saved layout predictions to: {OUT_LAYOUT_JSONL}")

# ── 8. Run Mode 2: Direct Full-Page EasyOCR (Optimized) ────────────────────────
print("\n" + "=" * 80)
print("RUNNING MODE 2: DIRECT FULL-PAGE EASYOCR (UN-CROPPED, OPTIMIZED)")
print("=" * 80)
fullpage_results = []
fullpage_scores = []
start_time = time.time()

for pid in test_page_ids:
    print(f"  -> Processing {pid}...", end=" ", flush=True)
    page_start = time.time()
    img_pil = get_page_image(pid)
    if img_pil is None:
        print("Skipped (image missing)")
        continue
        
    img_np = np.array(img_pil)
    
    # Fast Full-Page OCR using GPU batching & canvas scaling
    # detail=0 and paragraph=False avoids the slow O(N^2) CPU distance clustering stall
    lines = reader.readtext(
        img_np,
        batch_size=16,
        canvas_size=1600,
        mag_ratio=1.0,
        detail=0,
        paragraph=False
    )
    fullpage_transcript = "\n".join(lines).strip()
    elapsed = time.time() - page_start
    
    fullpage_results.append({
        "page_id": pid,
        "text": fullpage_transcript,
        "elapsed_s": round(elapsed, 2)
    })
    
    ref_norm = normalize(gt_labels[pid])
    hyp_norm = normalize(fullpage_transcript)
    
    cer = compute_cer(ref_norm, hyp_norm)
    wer = compute_wer(ref_norm, hyp_norm)
    f1 = compute_word_f1(ref_norm, hyp_norm)
    
    fullpage_scores.append({
        "page_id": pid,
        "cer": cer,
        "wer": wer,
        "f1": f1,
        "gt_chars": len(ref_norm),
        "hyp_chars": len(hyp_norm),
        "elapsed_s": round(elapsed, 2)
    })
    print(f"Done in {elapsed:5.2f}s | CER={cer:8.4f} | WER={wer:8.4f} | Word F1={f1:8.4f}")

fullpage_total_time = time.time() - start_time
print(f"\nCompleted Direct Full-Page EasyOCR in {fullpage_total_time:.2f}s.")

# Save fullpage results
with OUT_FULLPAGE_JSONL.open("w", encoding="utf-8") as f:
    for row in fullpage_results:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Saved fullpage predictions to: {OUT_FULLPAGE_JSONL}")

# ── 9. Final Side-by-Side Comparison & Reporting ────────────────────────────
print("\n" + "=" * 110)
print(f"{'PAGE':7s} | {'FULLPAGE CER':14s} | {'LAYOUT CER':12s} | {'FULLPAGE F1':14s} | {'LAYOUT F1':12s} | WINNER (BY F1)")
print("=" * 110)

comparison_rows = []
for i, pid in enumerate(test_page_ids):
    fp = fullpage_scores[i]
    lay = layout_scores[i]
    
    winner = "🟢 LAYOUT-AWARE" if lay["f1"] > fp["f1"] else ("🔵 FULL-PAGE" if fp["f1"] > lay["f1"] else "⚪ TIE")
    diff = lay["f1"] - fp["f1"]
    
    comparison_rows.append({
        "page_id": pid,
        "fullpage_cer": round(fp["cer"], 4),
        "layout_cer": round(lay["cer"], 4),
        "fullpage_wer": round(fp["wer"], 4),
        "layout_wer": round(lay["wer"], 4),
        "fullpage_f1": round(fp["f1"], 4),
        "layout_f1": round(lay["f1"], 4),
        "f1_delta": round(diff, 4),
        "winner": winner
    })
    print(f"{pid:7s} | {fp['cer']:14.4f} | {lay['cer']:12.4f} | {fp['f1']:14.4f} | {lay['f1']:12.4f} | {winner:16s} ({diff:+.4f})")

fp_mean_cer = sum(s["cer"] for s in fullpage_scores) / len(fullpage_scores) if fullpage_scores else 0.0
fp_mean_wer = sum(s["wer"] for s in fullpage_scores) / len(fullpage_scores) if fullpage_scores else 0.0
fp_mean_f1 = sum(s["f1"] for s in fullpage_scores) / len(fullpage_scores) if fullpage_scores else 0.0

lay_mean_cer = sum(s["cer"] for s in layout_scores) / len(layout_scores) if layout_scores else 0.0
lay_mean_wer = sum(s["wer"] for s in layout_scores) / len(layout_scores) if layout_scores else 0.0
lay_mean_f1 = sum(s["f1"] for s in layout_scores) / len(layout_scores) if layout_scores else 0.0

print("=" * 110)
print(f"DIRECT FULL-PAGE EASYOCR MEAN : CER = {fp_mean_cer:.4f} ({fp_mean_cer*100:.2f}%) | WER = {fp_mean_wer:.4f} ({fp_mean_wer*100:.2f}%) | Word F1 = {fp_mean_f1:.4f} ({fp_mean_f1*100:.2f}%)")
print(f"PP-DOCLAYOUT EASYOCR MEAN     : CER = {lay_mean_cer:.4f} ({lay_mean_cer*100:.2f}%) | WER = {lay_mean_wer:.4f} ({lay_mean_wer*100:.2f}%) | Word F1 = {lay_mean_f1:.4f} ({lay_mean_f1*100:.2f}%)")
print("=" * 110)

# Save Comparison CSV
with OUT_SCORES_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "page_id", "fullpage_cer", "layout_cer", "fullpage_wer", "layout_wer", "fullpage_f1", "layout_f1", "f1_delta", "winner"
    ])
    writer.writeheader()
    writer.writerows(comparison_rows)
print(f"Saved comparison CSV to: {OUT_SCORES_CSV}")

# Save Markdown Report
report_md = f"""# EasyOCR Benchmark Report: Layout-Aware vs. Direct Full-Page

**Corpus**: *The People's Common Sense Medical Adviser* (1890, R. V. Pierce)  
**Layout Engine**: PP-DocLayoutV3 (`ppdoclayout-v3/detections.jsonl`)  
**OCR Engine**: EasyOCR (Deep Learning CRAFT + CRNN)  
**Evaluation Set**: {len(test_page_ids)} Test Pages  

## Overall Benchmark Summary

| Strategy | Mean CER ⬇️ | Mean WER ⬇️ | **Mean Word F1 Score ⬆️** |
|---|---|---|---|
| **Direct Full-Page EasyOCR** (Un-cropped) | `{fp_mean_cer:.4f}` ({fp_mean_cer*100:.2f}%) | `{fp_mean_wer:.4f}` ({fp_mean_wer*100:.2f}%) | **`{fp_mean_f1:.4f}` ({fp_mean_f1*100:.2f}%)** |
| **PP-DocLayoutV3 + EasyOCR** (Layout-Aware) | `{lay_mean_cer:.4f}` ({lay_mean_cer*100:.2f}%) | `{lay_mean_wer:.4f}` ({lay_mean_wer*100:.2f}%) | **`{lay_mean_f1:.4f}` ({lay_mean_f1*100:.2f}%)** |
| **Net Impact of Layout Cropping** | **`{lay_mean_cer - fp_mean_cer:+.4f}`** | **`{lay_mean_wer - fp_mean_wer:+.4f}`** | **`{lay_mean_f1 - fp_mean_f1:+.4f}`** |

## Per-Page Breakdown

| Page ID | Full-Page CER | Layout CER | Full-Page Word F1 | Layout Word F1 | Winner |
|---|---|---|---|---|---|
"""
for r in comparison_rows:
    report_md += f"| **`{r['page_id']}`** | `{r['fullpage_cer']:.4f}` | `{r['layout_cer']:.4f}` | `{r['fullpage_f1']:.4f}` | `{r['layout_f1']:.4f}` | {r['winner']} |\n"

OUT_REPORT_MD.write_text(report_md, encoding="utf-8")
print(f"Saved Markdown Report to: {OUT_REPORT_MD}")
