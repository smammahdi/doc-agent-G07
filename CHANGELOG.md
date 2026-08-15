# Changelog

This changelog follows the verified history of `main`. Entries are newest first
and link to the commits that introduced each change. A2 has not been tagged as a
release yet.

## Unreleased

No A2 release tag has been created yet.

### Verified Stage-4 Knowledge Base Implementation & Durable Evidence

- Implemented the Stage 4 winning configuration in core starter modules (`src/doc_agent/index/`):
  - `chunk.py`: Explicit fixed 128-word chunking with 16-word step overlap (`fixed_128_16`).
  - `embed.py`: Qwen3-Embedding-0.6B support (1024 dimensions) with official query instruction prefix, last-token pooling, and L2 normalisation.
  - `store.py`: FAISS `IndexFlatIP` persistence and exact inner-product (cosine similarity) loading.
  - `configs/config.yaml`: Updated `embed.model` to `Qwen/Qwen3-Embedding-0.6B`, `embed.dim: 1024`, `index.type: faiss:flat_ip`, `index.chunk_words: 128`, `index.overlap: 16`.
- Built and populated `data/processed/index` with the verified 3,830 production chunks, 1024-dim FAISS FlatIP index, and metadata.
- Added comprehensive unit tests in `tests/test_index.py` covering chunk boundaries, invalid settings, Qwen query/document prefixes, normalization, FlatIP build/load, and fail-closed metadata checks.
- Executed `notebooks/kb_demo.ipynb` with live grounding against the production index:
  - Part 1: OCR held-out metrics (24 pages: Macro Word-F1 = 0.9592, Micro CER = 0.1334, Micro WER = 0.1840).
  - Part 2: Verified index statistics (3,830 chunks, 1024-d, FlatIP, 1,034 pages / 409,102 words).
  - Part 3: Live retrieval demo with `q_test_02` (perspiration functions on `p0078`), honest hydrotherapy multi-page failure diagnosis (`q_multi_test_02` on `p0373`), and out-of-corpus abstention evaluation (`q_neg_06` rejected below $\tau = 0.55$).
- Created external A2 Form evidence sheet (`development/a2-form-evidence.md`) and unedited Gemini conversation transcript (`development/transcript_drafts/gemini-transcript-draft.txt`).

- Updated `extras/ocr-benchmarks/deepseek-ocr.py` with inference mode, bounded generation (`max_new_tokens=4096`, deterministic greedy decoding), full-page internal tiling vs pre-cropped non-figure region mode (`crop_mode=False`), and metric-only grounding markup normalization.

### MinerU2.5-Pro OCR benchmark runner

- Added `extras/ocr-benchmarks/mineru-ocr.py` for the same 24-page, two-mode
  benchmark used by the other OCR runners: direct full-page parsing and
  PP-DocLayoutV3 non-figure crops.
- The runner uses the official `opendatalab/MinerU2.5-Pro-2604-1.2B` model and
  `mineru-vl-utils` Transformers client on an ordinary internet-enabled Kaggle
  T4, then writes CER, WER, word-F1, structured MinerU blocks, and one
  downloadable `mineru-ocr-benchmark.zip`.
- This commit adds the runner only. It does not record a successful model run
  or claim OCR quality before the generated archive is verified.

### Kaggle PaddleOCR and DeepSeek-OCR runner

- Added `extras/ocr_research/kaggle-deepseek-ocr-heldout.py` as a standalone
  24-page DeepSeek-OCR check. It installs only the official Transformers-side
  dependencies, clones `main` for the committed images/labels, and writes
  per-page text plus CER/WER/word-F1 metrics to a downloadable ZIP.
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

### Held-out TrOCR notebook (24 pages)

- Added `extras/ocr_research/kaggle-trocr-heldout.ipynb` for a complete
  held-out run over `p0024`--`p0047` (24 pages), rather than the older 14-page
  diagnostic slice.
- The notebook feeds existing DocLayout-YOLO non-figure regions to
  `microsoft/trocr-large-printed` as line crops. It does not rerun layout and
  does not use Chandra or Document AI text as OCR input.
- It writes one JSON record per page, then reports CER, WER, and word-F1 for
  all 24 pages and for a fair text-only subset excluding `p0041` and `p0043`,
  whose labels contain image descriptions.
- `trocr-large-printed` is the largest official Microsoft printed TrOCR
  checkpoint, not a claim of current OCR state of the art. Run it on a Tesla
  T4 with Internet enabled; the notebook writes a downloadable ZIP under
  `/kaggle/working/`.
- The existing held-out DeepSeek result is from
  `deepseek-ai/DeepSeek-OCR`. The newer official `DeepSeek-OCR-2` has not been
  run yet and remains a separate follow-up experiment; its results must not be
  compared as if they came from the same checkpoint.

### Held-out PaddleOCR benchmark

- Replaced the over-scoped full-book entrypoint with the small
  `extras/ocr_research/kaggle-paddleocr-heldout.ipynb` benchmark. It runs the
  current PaddleOCR 3.7 PP-OCRv6 pipeline on the 24 committed held-out pages
  and reports per-page CER, WER, and word-F1 against the existing labels. This
  is an OCR check only; no layout model is rerun and no full-book claim is
  made.
- Added Kaggle CUDA-library alignment for PaddleX's indirect PyTorch import:
  NCCL 2.27.5, NVJITLINK 12.8.93, and NVTX 12.8.90. This prevents the
  `ncclCommShrink` import failure seen with Kaggle's older preinstalled CUDA
  libraries.

### Curated model outputs

- Added `extras/ocr-benchmarks/outputs/` and `extras/layout-benchmarks/outputs/` as
  the shared, model-organized release directories. They contain the real Chandra
  `chunks.jsonl`, Document AI `words.jsonl`, page-complete orphan-ink/DocLayout-YOLO/
  PP-DocLayoutV3 reruns, PP-DocLayout-plus-L and PicoDet-S benchmark records, and
  the saved TrOCR outputs.
- The output documentation records schemas and evidence boundaries. These are
  inspectable research outputs, not human ground truth; raw scans, weights,
  crops, credentials, and environments remain outside Git.
- Added `extras/layout-benchmarks/outputs/heldout-visualizations/` with eight 24-page
  visual evidence PDFs for held-out pages `p0024`–`p0047`: Chandra, projection,
  orphan-ink, DocLayout-YOLO, PP-DocLayoutV3, PP-DocLayout-plus-L, PicoDet-S, and
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
