# OCR benchmarks

This folder contains the three editable Kaggle OCR runners maintained for the
current experiment:

- `deepseek-ocr.py` — DeepSeek-OCR-2 on the committed evaluation pages;
- `paddle-ocr.py` — PaddleOCR PP-OCRv6 on the committed evaluation pages;
- `trocr.py` — TrOCR over existing DocLayout-YOLO text regions.
- `glm-ocr.py` — the official `zai-org/GLM-OCR` checkpoint on the same pages.
- `mineru-ocr.py` — MinerU2.5-Pro on full pages and PP-DocLayoutV3 regions.

All runners are Python files. Their generated text, metrics, archives, caches,
and model weights belong in Kaggle working storage, not in Git. Results should
only be added back after the page labels and measurements have been reviewed.

Each runner writes one archive to `/kaggle/working/`:

```text
<engine>-ocr-benchmark.zip
├── full-page/{pages.jsonl,regions.jsonl,metrics.json}
├── ppdoclayout-v3/{pages.jsonl,regions.jsonl,metrics.json}
└── comparison.json
```

Both modes process the same 24 pages and report CER, WER, and word-F1 from the
saved `pages.jsonl` text. The PP-DocLayoutV3 boxes are the committed layout
input; no layout model is rerun by these OCR runners.
