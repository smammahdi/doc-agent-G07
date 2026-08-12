# Changelog

This changelog follows the verified history of `main`. Entries are newest first
and link to the commits that introduced each change. A2 has not been tagged as a
release yet.

## Unreleased

No A2 release tag has been created yet.

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
