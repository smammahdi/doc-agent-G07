# Curated research outputs

This is the repository's small, model-organized artifact release. It contains
machine-readable outputs that are useful for reproducing the layout work;
it is separate from `extras/ocr_research/`, which contains code and notebooks.

These files are research evidence, not automatically accepted ground truth:

- `chandra/chunks.jsonl` — 8,544 block records on 1,028 observed Pierce pages.
  Chandra geometry is used as a provisional layout reference; its block text
  is not fed into OCR runs.
- `document-ai/words.jsonl` — 419,565 Document AI word boxes from 1,016
  word-bearing pages. This is a cloud/silver reference and is not hand-checked
  ground truth.
- `orphan-ink/`, `doclayout-yolo/`, and `ppdoclayout-v3/` — page-complete
  150-DPI layout reruns. Each contains `pages.jsonl`, `detections.jsonl`,
  `summary.json`, and `run.log`.
- `ppdoclayout-plus-l/` and `picodet-s/` — full-book detector records and their matching
  summaries/evaluations from the same provisional-reference benchmark. These
  older runner outputs do not use the page-commit schema, so they are kept
  separate rather than presented as identical to the three canonical reruns.
- `layout-pdfs/` — eight 24-page visual evidence PDFs for held-out pages
  `p0024`–`p0047`, covering Chandra, projection, orphan-ink, DocLayout-YOLO,
  PP-DocLayoutV3, PP-DocLayout-plus-L, PicoDet-S, and Document AI word boxes.
  These are for manual inspection; Chandra and Document AI remain provisional
  references, not hand-checked ground truth.
- `../output_reports/` — saved two-mode held-out OCR outputs for DeepSeek-OCR-2,
  GLM-OCR, TrOCR large-printed, and the historical MinerU 2604 run. PaddleOCR
  is excluded until its benchmark is rerun correctly.
- `mineru-ocr-full-book/` — complete 1,034-page MinerU 2605 outputs in full-page
  and PP-DocLayoutV3-region modes, with metrics limited to the 24 labelled
  held-out pages.

Machine-readable outputs are JSON/JSONL/text; visual evidence is under
`layout-pdfs/`. All copied files are intentionally kept below normal GitHub
file-size limits. Raw PDFs, rendered pages, crops, model checkpoints, virtual
environments, private Kaggle outputs, and credentials remain outside Git. The
source PDF and sidecars are described in the research READMEs and can be
regenerated or obtained from the referenced Kaggle/Internet Archive inputs.

## Layout output schema

`pages.jsonl` has one ordered page record, including explicit empty pages.
`detections.jsonl` has pixel and normalized boxes plus model class/score.
`summary.json` records page counts, runtime, settings, and provisional Chandra
agreement metrics. `run.log` records page/checkpoint progress.

## Provenance

The copied artifacts came from the real Pierce 1890 PDF and existing local
Chandra/Document AI/layout runs. No synthetic rows were generated for this
release. The exact run settings and limitations remain in
the research READMEs and `CHANGELOG.md`.
