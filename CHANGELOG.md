# Changelog

## Unreleased — A2 knowledge-base work (2026-08-12)

This entry records the current A2 work in progress. It is intentionally not an
A2 completion claim; the form, held-out labels, full index run, and retrieval
evidence still have to be completed before the `a2-submit` tag.

### Added

- Declared the Pierce 1890 scanned-book corpus and the G07 project axes in the
  task, provenance, manifest, configuration, and design records.
- Implemented the A2 pipeline foundation in the named starter modules:
  deterministic PDF page loading, identity preprocessing, layout regions,
  Tesseract OCR, optional offline reference-text adapters, chunking, embedding,
  and FAISS index persistence.
- Added all-page EDA for the 1,034-page Pierce source and an explicit pipeline
  diagram showing which stages are implemented and which remain pending.
- Recorded the full-book layout comparison and rerun summaries for the
  projection/orphan-ink heuristic, DocLayout-YOLO, PP-DocLayoutV3,
  PP-DocLayout-plus-L, and PicoDet-S. The comparison uses Chandra as a
  provisional reference only; it is not human ground truth.
- Preserved the team's reproducible research tools under `extras/` for layout,
  OCR/Document AI, and Chandra experiments. These tools are research material,
  not query-time dependencies, and their private generated outputs, model
  weights, credentials, and raw corpus remain outside the repository.

### Evidence boundary

- The current design record reports real bounded/full-book research runs, but
  does not claim OCR accuracy, layout accuracy, full index coverage, or retrieval
  quality without hand labels and a completed notebook run.
- Document AI and Chandra outputs are reference/pre-annotation sources. They
  must not be presented as the A2 OCR oracle; the held-out labels must be
  manually corrected from the scanned pages.

### Still required for `a2-submit`

- Add the real held-out page images and hand-corrected `grading_kit/labels.jsonl`.
- Complete and run `notebooks/kb_demo.ipynb` with OCR quality, index statistics,
  one real query/top chunk/page check, and one honest failure case.
- Run the reproducible full index (or leave a verified build script that does
  so) and record its actual chunk, dimension, page, and word counts.
- Fill `forms/A2_form.docx`, add one unedited AI transcript per member, and
  verify every number in the form against code/notebook output.
- Resolve the clean-clone CI/reproducibility gate, then create and push the
  `a2-submit` tag from `main`.

## 0.1.0 — Starter skeleton
