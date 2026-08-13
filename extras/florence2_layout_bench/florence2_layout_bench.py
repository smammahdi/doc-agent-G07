#!/usr/bin/env python3
"""
Florence-2 Layout-Aware OCR Benchmark (Isolated Environment Script)
Pierce 1890 Medical Adviser · Team G07 · A2 OCR benchmarking
"""

import csv
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import cv2
import fitz  # PyMuPDF
import numpy as np
import torch
from PIL import Image
from jiwer import cer as compute_cer, wer as compute_wer
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports

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

OUT_DIR = Path("/kaggle/working/florence2_layout_bench")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "microsoft/Florence-2-base"
OCR_TASK = "<OCR>"
MAX_TOKENS = 512
DPI = 300

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Bypass optional flash_attn import requirement ──────────────────────────────
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

print(f"Loading {MODEL_ID} on {DEVICE}...")
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)

model.eval()
print(f"Model loaded in {time.time()-t0:.1f}s | dtype={next(model.parameters()).dtype}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def _chandra_label_kind(label) -> str:
    TEXT_LABELS = {
        "text", "section-header", "caption",
        "footnote", "list-group", "table",
    }
    if label is None:
        return "skip"
    return "text" if str(label).lower().strip() in TEXT_LABELS else "skip"

def _strip_html(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html).strip()

def render_page(doc: fitz.Document, page_idx: int, dpi: int = DPI) -> np.ndarray:
    pix = doc[page_idx].get_pixmap(dpi=dpi)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)

def bbox_to_pixel(bbox: list, page_box: list, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    pb_x0, pb_y0, pb_x1, pb_y1 = (float(v) for v in page_box)
    cw = pb_x1 - pb_x0
    ch = pb_y1 - pb_y0
    if cw <= 0.0 or ch <= 0.0:
        return None
    x0, y0, x1, y1 = (float(v) for v in bbox)
    px0 = max(0, int((x0 - pb_x0) / cw * img_w))
    py0 = max(0, int((y0 - pb_y0) / ch * img_h))
    px1 = min(img_w, int((x1 - pb_x0) / cw * img_w))
    py1 = min(img_h, int((y1 - pb_y0) / ch * img_h))
    return px0, py0, px1, py1

@torch.inference_mode()
def florence_ocr_crop(img_bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> str:
    if x1 <= x0 or y1 <= y0:
        return ""
    crop_bgr = img_bgr[y0:y1, x0:x1]
    if crop_bgr.size == 0:
        return ""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    inputs = processor(
        text=OCR_TASK,
        images=pil_img,
        return_tensors="pt",
    ).to(DEVICE, dtype=torch.float16 if DEVICE == "cuda" else torch.float32)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=MAX_TOKENS,
        num_beams=3,
        do_sample=False,
    )
    raw = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        raw, task=OCR_TASK, image_size=(pil_img.width, pil_img.height)
    )
    return str(parsed.get(OCR_TASK, "")).strip()

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

# ── Main processing ───────────────────────────────────────────────────────────
def main():
    print("Loading Chandra layout blocks...")
    chandra_blocks: dict[str, list[dict]] = defaultdict(list)
    skipped_label = 0
    with CHANDRA_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            book_page = row.get("book_page")
            if book_page is None:
                continue
            label = row.get("label", "")
            if _chandra_label_kind(label) == "skip":
                skipped_label += 1
                continue
            page_id = f"p{int(book_page):04d}"
            chandra_blocks[page_id].append({
                "page_box": row.get("page_box"),
                "bbox": row.get("bbox"),
                "label": label,
                "content": _strip_html(row.get("content", "")),
            })
    print(f"Loaded {sum(len(v) for v in chandra_blocks.values())} text blocks across {len(chandra_blocks)} pages.")

    print("Loading GT labels...")
    gt_labels = {}
    with LABELS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            gt_labels[row["page_id"]] = row["text"]

    doc = fitz.open(str(PDF_PATH))
    all_page_ids = sorted(chandra_blocks.keys())
    results = []
    t_total = time.time()

    transcripts_path = OUT_DIR / "page_transcripts.jsonl"
    out_f = transcripts_path.open("w", encoding="utf-8")

    for page_num, page_id in enumerate(all_page_ids):
        pdf_idx = int(page_id[1:]) - 1
        if pdf_idx < 0 or pdf_idx >= doc.page_count:
            continue
        t0 = time.time()
        img = render_page(doc, pdf_idx, DPI)
        img_h, img_w = img.shape[:2]

        block_texts = []
        for blk in chandra_blocks[page_id]:
            px_res = bbox_to_pixel(blk.get("bbox"), blk.get("page_box"), img_w, img_h)
            if px_res is None:
                continue
            px0, py0, px1, py1 = px_res
            text = florence_ocr_crop(img, px0, py0, px1, py1)
            if text:
                block_texts.append(text)

        page_text = "\n".join(block_texts)
        elapsed = time.time() - t0

        row = {"page_id": page_id, "text": page_text, "n_blocks": len(block_texts), "elapsed_s": round(elapsed, 2)}
        results.append(row)
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if page_num % 50 == 0 or page_num < 3:
            print(f"  [{page_num+1:4d}/{len(all_page_ids)}] {page_id} — {len(block_texts)} blocks, {elapsed:.1f}s")

    out_f.close()
    print(f"OCR finished in {(time.time()-t_total)/60:.1f} min.")

    # ── Scoring ───────────────────────────────────────────────────────────────
    transcript_map = {r["page_id"]: r["text"] for r in results}
    scored_pages = []
    for page_id, ref_text in sorted(gt_labels.items()):
        hyp_text = transcript_map.get(page_id, "")
        ref_norm = normalize(ref_text)
        hyp_norm = normalize(hyp_text)
        if not ref_norm:
            continue
        c = compute_cer(ref_norm, hyp_norm)
        w = compute_wer(ref_norm, hyp_norm)
        ref_words = set(ref_norm.lower().split())
        hyp_words = set(hyp_norm.lower().split())
        tp = len(ref_words & hyp_words)
        prec = tp / len(hyp_words) if hyp_words else 0.0
        rec = tp / len(ref_words) if ref_words else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        scored_pages.append({"page_id": page_id, "cer": round(c, 4), "wer": round(w, 4), "word_f1": round(f1, 4)})

    mean_cer = float(np.mean([p["cer"] for p in scored_pages]))
    mean_wer = float(np.mean([p["wer"] for p in scored_pages]))
    mean_f1 = float(np.mean([p["word_f1"] for p in scored_pages]))

    score_path = OUT_DIR / "heldout_scores.csv"
    with score_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scored_pages[0]))
        writer.writeheader()
        writer.writerows(scored_pages)

    lines = [
        "# Florence-2 Layout-Aware OCR Benchmark",
        f"- Mean CER    : {mean_cer:.4f}",
        f"- Mean WER    : {mean_wer:.4f}",
        f"- Mean Word F1: {mean_f1:.4f}",
    ]
    report_path = OUT_DIR / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Results saved. Mean CER: {mean_cer:.4f}, Mean WER: {mean_wer:.4f}")

if __name__ == "__main__":
    main()
