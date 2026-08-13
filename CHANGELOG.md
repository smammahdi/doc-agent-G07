# Changelog

This changelog follows the verified history of `main`. Entries are newest first
and link to the commits that introduced each change. A2 has not been tagged as a
release yet.

## Unreleased

No A2 release tag has been created yet.

### Kaggle PaddleOCR and DeepSeek-OCR runner

- Added `extras/ocr_research/kaggle-paddle-deepseek-ocr.py` and its notebook.
  They save independent PaddleOCR and DeepSeek-OCR text from existing Chandra
  or DocLayout-YOLO regions, with a real page-34 smoke by default and an
  explicit `--pages all` full-book mode.
- The runner checkpoints `regions.jsonl` before each page record, reuses page
  renders and crops, records OCR errors without losing page progress, and
  leaves the production OCR path and layout models unchanged.
- No Kaggle model run or OCR-quality claim is recorded in this commit; the
  notebook is prepared for a Tesla T4 execution.
- The notebook gives each layout its own output and cache roots, preventing a
  Chandra run from reusing DocLayout-YOLO checkpoints (or vice versa).

### Dedicated full-book PaddleOCR notebook

- Added `extras/ocr_research/kaggle-paddleocr-full-book.ipynb` and its Python
  wrapper for a focused Tesla T4 run over all 1,034 pages using the existing
  DocLayout-YOLO regions. It installs `PaddleOCR==3.7.0` with
  `paddlepaddle-gpu==3.3.1`, verifies a real page-34 smoke first, resumes
  page-by-page, and packages only the PaddleOCR results for Kaggle download.
  No layout model is rerun and no OCR-quality claim is made by the notebook.

### Curated model outputs

- Added `extras/output/` as the shared, model-organized release directory.
  It contains the real Chandra `chunks.jsonl`, Document AI `words.jsonl`,
  page-complete orphan-ink/DocLayout-YOLO/PP-DocLayoutV3 reruns, PP-DocLayout-
  plus-L and PicoDet-S benchmark records, and the saved TrOCR outputs.
- The output README records schemas and evidence boundaries. These are
  inspectable research outputs, not human ground truth; raw scans, weights,
  crops, credentials, and environments remain outside Git.
- Added `extras/output/layout-pdfs/` with eight 24-page visual evidence PDFs
  for held-out pages `p0024`–`p0047`: Chandra, projection, orphan-ink,
  DocLayout-YOLO, PP-DocLayoutV3, PP-DocLayout-plus-L, PicoDet-S, and
  Document AI word boxes. PaddleOCR is intentionally not listed here because
  it is an OCR engine producing text-line boxes, not an independent semantic
  layout detector. Chandra and Document AI remain provisional references.

### TrOCR outputs from existing layouts

- Added an optional lazy `microsoft/trocr-base-printed` reader and a resumable
  research runner that saves real TrOCR transcriptions from existing Chandra
  reference regions and the selected DocLayout-YOLO regions. No OCR score is
  claimed until manually verified labels are repaired.
- Added a private Kaggle GPU entrypoint for the same exporter over the full
  Pierce book. It keeps Chandra and DocLayout-YOLO outputs separate, does not
  rerun either layout model, and requires 1,028 observed Chandra pages plus
  six layout-missing pages and all 1,034 DLY pages before acceptance.
- The first two submissions were rejected by Kaggle's P100 runtime because its
  installed PyTorch build does not support sm_60. The active submission uses
  the explicit `NvidiaTeslaT4` accelerator and aborts before OCR on older GPUs.
- Verified and committed the complete Chandra TrOCR export in
  `extras/ocr_research/results/trocr_base_printed/chandra/`: 1,034 ordered
  page records, 1,028 completed pages, six explicit layout-missing pages
  (`p0002`, `p0003`, `p0004`, `p0006`, `p1031`, `p1033`), and 8,191
  OCR regions across four Tesla T4 shards (commit `8208bf9`). This is saved
  OCR output from the existing Chandra regions; it is not a quality score or
  a claim that Chandra is human ground truth.
- The four Chandra shards used the same `microsoft/trocr-base-printed`
  checkpoint and 300-DPI crop settings. Rendered pages, line crops, model
  weights, and shard archives remain outside Git.
- Verified the third real DocLayout-YOLO shard on Tesla T4: pages `517–775`
  are complete as a separate canonical output, with 259 ordered page records
  and 1,909 OCR regions. The corrected final T4 shard is now included in the
  complete export below.
- Verified and committed the complete DocLayout-YOLO TrOCR export in
  `extras/ocr_research/results/trocr_base_printed/doclayout_yolo/`: 1,034
  ordered completed page records, 7,108 OCR regions, 29 pages with no
  recognized text, and 11,666.899 seconds across four Tesla T4 shards
  (commit `b1f4d87`). This is saved OCR output from existing DLY regions, not
  a layout or OCR accuracy claim.
- A preliminary held-out diagnostic over the 14 manually transcribed pages
  `p0024`–`p0037` gives macro CER/WER/word-F1 of `0.2964/0.4071/0.7130`
  for Chandra-fed TrOCR and `0.1977/0.3167/0.7979` for DLY-fed TrOCR. The
  transcription file currently contains literal multiline JSON strings and
  is not valid one-record-per-line JSONL, so these numbers are provisional
  analysis rather than a passing grading gate. No full-book OCR error is
  reported without manually verified reference text.

### DeepSeek-OCR research preparation

- Added an isolated exporter for the official `deepseek-ai/DeepSeek-OCR`
  custom Transformers interface. It saves raw text from existing Chandra or
  DocLayout-YOLO crops only; it does not change the runtime OCR default or
  rerun layout detection. Added a Kaggle launcher for a bounded real-page
  smoke (default `p0034`, existing DLY regions); no DeepSeek output or
  accuracy claim exists yet.
- Completed the bounded DeepSeek smoke on real Pierce page `p0034` using the
  existing DocLayout-YOLO regions: 1 page, 5 regions, non-empty text from all
  5, CUDA/Tesla T4, 24.854 seconds. This is only an exporter feasibility
  result; no OCR quality or full-book DeepSeek claim is made.

### Layout artifact package

- Prepared the external Kaggle package
  [`cruelangelssprint/pierce-1890-figure-and-ocr-outputs`](https://www.kaggle.com/datasets/cruelangelssprint/pierce-1890-figure-and-ocr-outputs)
  version 3, including the Chandra reference blocks, existing detector
  outputs, and page-complete 150-DPI rerun records.

### Held-out OCR work

- Added 14 Pierce held-out page images (`p0024`–`p0037`), accompanying manual
  transcription text, OCR metric helpers/tests, a dependency lockfile, and
  Bandit configuration
  ([`f311891`](https://github.com/smammahdi/doc-agent-G07/commit/f311891),
  Mehemud Azad).
- Validation is still pending for that increment: the transcription file must
  be converted to valid one-record-per-line JSONL, and the OCR tests must report
  real OCR failures instead of substituting reference text or simulated noise.

## 2026-08-12 — Research integration and layout evaluation

### Layout implementation and evidence

- Added runtime-selectable orphan-ink and DocLayout-YOLO layout modes while
  retaining projection as the default fallback
  ([`95fb7ab`](https://github.com/smammahdi/doc-agent-G07/commit/95fb7ab)).
- Recorded the full-book comparison of orphan ink, DocLayout-YOLO,
  PP-DocLayoutV3, PP-DocLayout-plus-L, and PicoDet-S against the provisional
  Chandra reference
  ([`fb8968c`](https://github.com/smammahdi/doc-agent-G07/commit/fb8968c)).
- Refreshed the design record with results from the canonical 1,034-page reruns
  ([`23248aa`](https://github.com/smammahdi/doc-agent-G07/commit/23248aa)).
- Merged the complete A2 knowledge-base foundation into `main`
  ([`7e1a11b`](https://github.com/smammahdi/doc-agent-G07/commit/7e1a11b)).

### Team research workspace

- Mehemud Azad added the implementation plan and Kaggle OCR/held-out scoring
  notebooks
  ([`f12ec8b`](https://github.com/smammahdi/doc-agent-G07/commit/f12ec8b)).
- Added the reusable figure-extraction package and layout-comparison runner,
  including orphan ink, DocLayout-YOLO, PP-DocLayoutV3, fusion, caption, and
  geometry utilities
  ([`bc1b3f3`](https://github.com/smammahdi/doc-agent-G07/commit/bc1b3f3)).
- Added the OCR bake-off and Document AI research clients, parsers, comparison,
  and provenance utilities
  ([`b6ea525`](https://github.com/smammahdi/doc-agent-G07/commit/b6ea525)).
- Added the executed Chandra research notebook and direct-output parsing/sample
  utilities; offline packaging/setup code was deliberately excluded
  ([`5c12077`](https://github.com/smammahdi/doc-agent-G07/commit/5c12077)).
- Added the `extras/` index explaining the boundary between research utilities,
  runtime code, and untracked generated artifacts
  ([`0c747af`](https://github.com/smammahdi/doc-agent-G07/commit/0c747af)).

## 2026-08-10 — A2 knowledge-base foundation

### Corpus and reproducibility

- Expanded `.gitignore` to protect local credentials, environments, model
  weights, generated outputs, caches, and raw corpus files
  ([`ecc1bc7`](https://github.com/smammahdi/doc-agent-G07/commit/ecc1bc7)).
- Declared the G07 project axes and the Pierce 1890 corpus in the task,
  provenance, and grading manifest
  ([`a08460c`](https://github.com/smammahdi/doc-agent-G07/commit/a08460c)).
- Implemented the verified Internet Archive corpus downloader with size and
  checksum validation
  ([`d6709b3`](https://github.com/smammahdi/doc-agent-G07/commit/d6709b3)).
- Implemented deterministic PyMuPDF page rendering with bounded runs, stable
  page identifiers, and generated-output isolation
  ([`c79e250`](https://github.com/smammahdi/doc-agent-G07/commit/c79e250),
  merged by [`a5127b4`](https://github.com/smammahdi/doc-agent-G07/commit/a5127b4)).

### Pipeline stages

- Established an explicit identity preprocessing baseline that validates and
  preserves the fixed `Page` contract
  ([`64ab2ce`](https://github.com/smammahdi/doc-agent-G07/commit/64ab2ce)).
- Implemented projection-based layout detection and the Tesseract OCR baseline
  for real Pierce page images
  ([`afa7d19`](https://github.com/smammahdi/doc-agent-G07/commit/afa7d19)).
- Implemented overlapping chunk construction, sentence-transformer embedding,
  FAISS persistence/loading, and the index build entrypoint
  ([`81933b3`](https://github.com/smammahdi/doc-agent-G07/commit/81933b3)).
- Added optional Chandra block ingestion at the layout seam with explicit
  missing-page policies
  ([`9ed1ee3`](https://github.com/smammahdi/doc-agent-G07/commit/9ed1ee3)).
- Added optional offline Document AI word ingestion at the OCR seam, including
  overlap deduplication and punctuation-aware text assembly
  ([`d30518c`](https://github.com/smammahdi/doc-agent-G07/commit/d30518c)).

### Design and evidence

- Replaced the starter design and diagram placeholders with the initial
  corpus-specific A2 design record
  ([`18a4ce4`](https://github.com/smammahdi/doc-agent-G07/commit/18a4ce4)).
- Updated the design record, EDA, knowledge-base demo, and pipeline report to
  enforce an evidence-first workflow
  ([`6ef50b9`](https://github.com/smammahdi/doc-agent-G07/commit/6ef50b9)).
- Executed all-page EDA over the 1,034-page Pierce corpus and saved the measured
  notebook outputs
  ([`6b964ce`](https://github.com/smammahdi/doc-agent-G07/commit/6b964ce)).
- Updated the pipeline diagram to distinguish implemented runtime stages,
  offline reference sources, and pending index/retrieval evidence
  ([`45c3fc3`](https://github.com/smammahdi/doc-agent-G07/commit/45c3fc3)).

## 0.1.0 — Starter skeleton — 2026-08-10

- Imported the course-provided starter repository
  ([`4214f05`](https://github.com/smammahdi/doc-agent-G07/commit/4214f05)).
