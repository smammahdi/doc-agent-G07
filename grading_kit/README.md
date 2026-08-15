# grading_kit/ — the one folder that makes this project reproducible and gradable.

- **manifest.yaml** — the single entry point: the three axes (domain, data speciality, primary NFR) + pointers to the corpus,
  the held-out slice, and the build/run/eval commands.
- **heldout_pages/** — page-images set aside, never OCR-trained on.
- **labels.jsonl** — ground-truth transcriptions for the held-out pages (the oracle:
  OCR is scored against them, and fresh grading questions are authored from them).
- **tasks.jsonl** — the canonical mixed-modality evaluation set. Every task has an
  `image_paths` list; it is empty for text-only tasks and contains repository-relative
  held-out scan paths for image-grounded tasks.

A grader (or you) opens ONLY `manifest.yaml`; it points to everything else. The build/run
scripts and the eval tasks are named there, not copied here, so they never go stale.
