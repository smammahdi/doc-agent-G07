# OCR benchmarks

This directory contains the editable Kaggle OCR benchmark runners maintained
for the evaluation workflow:

- `deepseek-ocr.py` — DeepSeek-OCR-2 (`deepseek-ai/DeepSeek-OCR-2`) on the committed held-out pages;
- `glm-ocr.py` — GLM-OCR (`zai-org/GLM-OCR`) on the committed held-out pages;
- `mineru-ocr.py` — MinerU2.5-Pro (`opendatalab/MinerU2.5-Pro-2604-1.2B`) on full pages and PP-DocLayoutV3 regions;
- `paddle-ocr.py` — PaddleOCR 3.7.0 (`PP-OCRv6_medium_det` + `PP-OCRv6_medium_rec`) on the committed held-out pages;
- `trocr.py` — TrOCR (`microsoft/trocr-large-printed`) line recognition on full pages and PP-DocLayoutV3 regions.
- `compare-results.py` — reusable scorer for saved page JSONL, Chandra block output, and the raw Qwen readable chunk export.

All runners are standalone Python scripts formatted as Jupytext percent notebooks. Their generated text, metrics, archives, caches, and model checkpoints belong in Kaggle working storage, not in Git.

## Input Contract

Each runner evaluates the same 24 committed held-out Pierce pages (`p0024` through `p0047`):
- Page images: `grading_kit/heldout_pages/p0024.jpg`–`p0047.jpg`
- Ground-truth reference labels: `grading_kit/labels.jsonl`
- Pre-extracted layout detections: `extras/output/ppdoclayout-v3/detections.jsonl`

## Execution Modes

Every runner evaluates two modes:
1. `full-page`: feeds the full rendered page image directly to the OCR engine.
2. `ppdoclayout-v3`: crops the existing non-figure bounding boxes from the committed `PP-DocLayoutV3` detections and feeds each crop to the engine.

No layout detector is re-executed by these runners.

## Output Archive

Each runner writes one downloadable ZIP archive to `/kaggle/working/<engine>-ocr-benchmark.zip` structured as:

```text
<engine>-ocr-benchmark.zip
├── full-page/
│   ├── pages.jsonl
│   ├── regions.jsonl
│   └── metrics.json
├── ppdoclayout-v3/
│   ├── pages.jsonl
│   ├── regions.jsonl
│   └── metrics.json
└── comparison.json
```

Both modes report character error rate (CER), word error rate (WER), and word-level F1 score against the reference labels. Primary metrics use Unicode NFKC normalization, case-folding, and punctuation stripping for fair cross-model comparison; exact raw-text metrics are also retained.

## Jupytext Conversion

To convert any `.py` runner into an editable `.ipynb` notebook for Kaggle:

```bash
jupytext --to notebook extras/ocr-benchmarks/<runner>.py
```

To convert all benchmark runners at once:

```bash
jupytext --to notebook extras/ocr-benchmarks/*.py
```

## Reusing the comparison scorer

Pass one or more saved JSONL sources as `NAME=PATH`.  The scorer applies the
same normalization and reports both macro and micro CER/WER plus multiset
Word-F1 for every source:

```bash
python extras/ocr-benchmarks/compare-results.py \
  --labels grading_kit/labels.jsonl \
  --engine "Chandra=extras/output/chandra/chunks.jsonl" \
  --engine "MinerU=extras/output/mineru-ocr-full-book/full-page/pages.jsonl" \
  --engine "Tesseract=extras/tesseract_layout_bench/result/tesseract_ppdoclayout_v3_results.jsonl" \
  --exclude-page p0041 \
  --exclude-page p0043 \
  --json /tmp/ocr-comparison.json \
  --markdown /tmp/ocr-comparison.md
```

Qwen's committed `extras/ocr_results/qwen3.5-ocr.txt` is a readable chunk
export rather than page JSONL. Add it with `--qwen-raw`; the scorer groups
chunks by page and removes repeated chunk overlap before applying the same
metrics:

```bash
python extras/ocr-benchmarks/compare-results.py \
  --labels grading_kit/labels.jsonl \
  --engine "Chandra=extras/output/chandra/chunks.jsonl" \
  --qwen-raw "Qwen3.5 raw chunks=extras/ocr_results/qwen3.5-ocr.txt" \
  --exclude-page p0041 \
  --exclude-page p0043 \
  --json /tmp/ocr-comparison-with-qwen.json \
  --markdown /tmp/ocr-comparison-with-qwen.md
```
