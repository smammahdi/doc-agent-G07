# OCR Benchmarks Workspace (`extras/ocr-benchmarks/`)

This directory contains the Kaggle benchmark runners, modular OCR pipelines, saved predictions, and evaluation reports for optical character recognition on the historical medical document (*People's Common Sense Medical Adviser*, Pierce, 1890).

---

## Directory Structure

```
extras/ocr-benchmarks/
├── README.md                           # Master OCR benchmark documentation
├── engines/                            # Standalone OCR engine harnesses & runners:
│   ├── chandra/                        # Chandra OCR parsing and manifest tools
│   ├── easyocr/                        # EasyOCR runner and notebook
│   ├── florence2_layout/               # Florence-2-base runner and notebook
│   ├── mineru/                         # MinerU parser benchmark
│   ├── paddleocr_vl/                   # PaddleOCR runner and notebook
│   ├── tesseract_fullpage/             # Tesseract 5.x raw full-page baseline
│   ├── tesseract_layout/               # Tesseract 5.x + PP-DocLayoutV3 pipeline
│   ├── ocr_research/                   # Google Cloud Document AI research
│   └── modular_suite/                  # Multi-engine comparative runner scripts (compare-results.py)
├── notebooks/                          # Interactive evaluation notebooks (Qwen, Heldout)
├── outputs/                            # Standardized OCR predictions by scope:
│   ├── heldout/                        # 22-page test set predictions (DeepSeek, EasyOCR, Florence-2, GLM-OCR, PaddleOCR, Tesseract, TrOCR, Qwen)
│   └── full-book/                      # 1,034-page book extractions (Chandra chunks.jsonl, Document AI, MinerU, Qwen)
└── reports/                            # Quantitative CER/WER/Word-F1 comparison reports & JSON outputs
```

---

## Input Contract & Ground Truth

Each runner evaluates the 22 text-bearing held-out pages (`p0024` through `p0047`, excluding full-page woodcut engravings `p0041` and `p0043`):
- Page images: `grading_kit/heldout_pages/p0024.jpg`–`p0047.jpg`
- Ground-truth reference labels: `grading_kit/labels.jsonl`
- Pre-extracted layout detections: `extras/layout-benchmarks/outputs/ppdoclayout-v3/detections.jsonl`

---

## Reusable Comparison Scorer

To run the unified evaluation across saved OCR predictions:

```bash
python extras/ocr-benchmarks/engines/modular_suite/compare-results.py \
  --labels grading_kit/labels.jsonl \
  --exclude-page p0041 \
  --exclude-page p0043 \
  --engine "Chandra=extras/ocr-benchmarks/outputs/full-book/chandra/chunks.jsonl" \
  --engine "MinerU full-page=extras/ocr-benchmarks/outputs/full-book/mineru/full-page/pages.jsonl" \
  --engine "GLM-OCR full-page=extras/ocr-benchmarks/outputs/heldout/glm/full-page/pages.jsonl" \
  --engine "PaddleOCR layout=extras/ocr-benchmarks/outputs/heldout/paddleocr/ppdoclayout-v3/pages.jsonl" \
  --engine "Tesseract layout=extras/ocr-benchmarks/outputs/heldout/tesseract/layout_results.jsonl" \
  --engine "Qwen3.5=extras/ocr-benchmarks/outputs/heldout/qwen/pages.jsonl" \
  --json extras/ocr-benchmarks/reports/output_reports/ocr-benchmark-comparison-22-pages.json \
  --markdown extras/ocr-benchmarks/reports/output_reports/ocr-benchmark-comparison-22-pages.md
```
