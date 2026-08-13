# OCR research

This directory preserves the OCR experiments behind
`src/doc_agent/vision/ocr.py`. The production path defaults to local Tesseract;
these tools are for engine comparison and for regenerating the offline Google
Document AI reference sidecar.

## Local OCR bake-off

`ocr_bench.py` samples real Pierce pages by measured page characteristics,
runs OCR engines, and stores word text, confidence, and pixel boxes. Missing
engines are recorded as failures rather than silently skipped.

```bash
bash scripts/get_data.sh
python extras/ocr_research/ocr_bench.py demo
python extras/ocr_research/ocr_bench.py sample
python extras/ocr_research/ocr_bench.py run
python extras/ocr_research/ocr_bench.py report
```

The benchmark may seed correction files, but a seed is not ground truth. CER,
WER, or word F1 is valid only after a person corrects the selected real pages.

## Document AI reference generation

`docai/` contains the original paid-API probe and full-book word-box exporter.
The runtime never calls Google. It only reads a locally supplied
`words.jsonl` when `ocr.mode: document_ai_reference` is selected.

Document AI commands require the `google-cloud-documentai` package and
credentials supplied by environment variables or an untracked `.env.vertex`.
No credential belongs in Git. Commands that sweep the corpus require `--yes`
because they spend money.

```bash
PYTHONPATH=extras/ocr_research python -m docai probe
PYTHONPATH=extras/ocr_research python -m docai ocrtest --pages 34,74
PYTHONPATH=extras/ocr_research python -m docai ocrsweep --yes
```

The generated `words.jsonl` contains `page_id`, `text`, normalized and pixel
boxes, confidence, and engine provenance. It is a strong silver reference and
a useful transcription seed, but cloud confidence alone is not human ground
truth. Generated outputs remain ignored by Git and should be shared through a
dataset release, not committed to the code repository.

## OCR benchmark runners

The active Kaggle-facing OCR runners now live in
`extras/ocr-benchmarks/`. Generated OCR results are deliberately not stored in
Git until they are reviewed against manually verified labels.
