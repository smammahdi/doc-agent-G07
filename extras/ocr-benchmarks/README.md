# OCR benchmarks

This folder contains the three editable Kaggle OCR runners maintained for the
current experiment:

- `deepseek-ocr.py` — DeepSeek-OCR-2 on the committed evaluation pages;
- `paddle-ocr.py` — PaddleOCR PP-OCRv6 on the committed evaluation pages;
- `trocr.py` — TrOCR over existing DocLayout-YOLO text regions.

All runners are Python files. Their generated text, metrics, archives, caches,
and model weights belong in Kaggle working storage, not in Git. Results should
only be added back after the page labels and measurements have been reviewed.
