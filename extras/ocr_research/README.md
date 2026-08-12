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

## Full-book TrOCR outputs from existing layout regions

`run_layout_trocr.py` reads existing layout records and never runs a layout
model. It saves TrOCR output for `chandra` regions (six missing pages remain
`layout_missing`) and for the selected `doclayout_yolo` regions (all 1,034
pages). Both runs use the same `microsoft/trocr-base-printed` checkpoint and
line-crop recognition settings.

```bash
python extras/ocr_research/run_layout_trocr.py \
  --layout chandra \
  --layout-path /path/to/chandra/chunks.jsonl \
  --source-pdf /path/to/pierce-1890.pdf \
  --output extras/ocr_research/results/trocr_base_printed/chandra \
  --cache-dir /path/to/local/trocr-pages
```

The runner checkpoints `regions.jsonl` before its page row in `pages.jsonl`
and resumes completed pages without duplicate records. Full generated results,
page renders, line crops, and model caches are local experiment artifacts; do
not treat these outputs as OCR accuracy until they are compared with manually
verified transcriptions.

### GPU execution

The full-book run is prepared as a private Kaggle kernel:
[`g07-t4-full-book-trocr`](https://www.kaggle.com/code/cruelangelssprint/g07-t4-full-book-trocr).
It attaches the Pierce PDF dataset and the existing layout-artifact dataset,
clones the `a2/trocr-layout-comparison` branch, and runs the same exporter
twice on CUDA. The kernel writes independent `chandra/` and
`doclayout_yolo/` directories, each containing only `pages.jsonl`,
`regions.jsonl`, and `summary.json`, plus a downloadable ZIP. No layout model
is rerun, and no Chandra text is used as OCR input.

The two input datasets are:

- [`kmazd1110/dl-peoples-common-sense-med-advisor`](https://www.kaggle.com/datasets/kmazd1110/dl-peoples-common-sense-med-advisor)
  for the 1,034-page Pierce PDF;
- [`cruelangelssprint/pierce-1890-figure-and-ocr-outputs`](https://www.kaggle.com/datasets/cruelangelssprint/pierce-1890-figure-and-ocr-outputs)
  for `chandra/chunks.jsonl` and the page-complete DocLayout-YOLO detections.

The kernel must finish with 1,034 page rows for the DLY run and 1,028
`complete` plus six `layout_missing` page rows for the Chandra run before its
outputs are copied into the repository results directory.
