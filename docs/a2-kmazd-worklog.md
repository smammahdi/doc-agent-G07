# A2 — K. M. Mehemud Azad (2105014): findings & work plan

Companion to `docs/a2-plan.md` (the repo survey — left as written). This file records what I
own for A2, the faculty rules we discovered that are not written in the repo, the open
speciality question, and my execution order. Local working notes; not for Mahdi's branch.

---

## 1. Faculty rules that exist only outside the repo

These bind us but appear in no handbook file — record them here so they survive.

1. **Image-QA condition (corpus approval).** Sir approved Pierce with: *"Ok. Use it. But you
   need to answer questions from images also. Store detailed image descriptions separately."*
   → Figure handling is mandatory for G07. Descriptions live as separate stored artifacts
   (sidecar + figure-chunks in the index), never merged into OCR text. Citations for image
   answers must show the actual figure crop (a VLM caption can hallucinate; the crop is what a
   human verifies in <30 s per our NFR target). Captioner must be open-weight and pinned.
2. **OCR ruling.** Proprietary models (Gemini, Document AI) may be used for **offline label
   generation / pre-annotation only** — never at inference, and not to generate indexed
   content. Cited in our A1 form ("per the OCR ruling"); the A1 answers and Mahdi's A2 rules
   both already respect it.

Written rules, for reference: corpus floors and OCR policy in
`handbook/03-Project-Specification.md` (Part B, ~lines 41–67); the A2 checklist in
`handbook/05-Codebase-Guide.md` (~line 333); catalogs in
`handbook/06-Group-Assignment-Workbook.xlsx` (sheets: Domain, Data Speciality, NFR); CI-enforced
regulation in `STRUCTURE.md`. Corpus compliance is settled: 1,034 pages / 354,367 words vs the
≥300 / ≥60k floors, IA page-image scan, public domain, domain registered to G07 at A1.

## 2. Speciality slug — decision to raise with the team

Our declared `dirty-ocr` maps to catalog **#13 "Dirty-OCR-provided"** = "clean noisy machine
text (**no clean images**)" — a corpus handed over as garbled text, OCR skipped. Not our
project: we have real scans and produce our own OCR. Two honest candidates:

- **#8 Old / reformed orthography** → E3 (script-tuned OCR in `vision/ocr.py` + normalisation
  in `index/chunk.py`). Matches the A1 form's evidence: long-s, œ/æ ligatures, "8mart-weed",
  "Iiife-root". The A1 EDA rules out the E1/enhancement route (binarisation neutral; errors
  token-local on clean pages).
- **#9 Table / figure-heavy** → E2 (layout detection / structure extraction in
  `vision/layout.py`). Newly live because the image-QA condition (§1) makes figure work
  compulsory anyway — claiming #9 makes the mandatory work also the graded speciality item.

Either is defensible; `dirty-ocr` as written is not. Needs Mahdi + likely sir sign-off since
the axes were declared at A1 (`configs/task.yaml`, `grading_kit/manifest.yaml`).

## 3. My work packages (Mahdi's brief, 2026-08-12)

Base: `a2/pierce-kb-foundation` @ `45c3fc3`. A2 scope only; real Pierce pages only; no
synthetic labels/metrics; DocAI = pre-annotation only; don't commit PDF/images/weights/
sidecars/credentials; edit existing stubs, no new Python modules; one logical unit per commit;
own git identity; report owned files before editing; tests/ruff/black/mypy at the end.

**Package 1 — held-out OCR evidence** (branch `a2/heldout-ocr`)
Own: `grading_kit/heldout_pages/`, `grading_kit/labels.jsonl`, OCR-eval-related stubs
(`tests/test_ocr.py` as needed). Select representative real pages (body text, figure, front/
zero-word, landscape if present) → pre-annotate from Mahdi's DocAI output → **manually verify
every page against the scan** → score Tesseract/reference CER/WER/word-F1 with exact page IDs
→ record ≥1 real failure case → labels parse + provenance matches manifest.

**Package 2 — index + demo + reproducibility gate** (branch `a2/index-demo-ci`)
Own: `index/chunk.py`, `index/embed.py`, `index/store.py`, `scripts/build_index.sh`,
`notebooks/kb_demo.ipynb`; dependency/CI files in a **separate** commit. Full corpus →
256/32 chunks → MiniLM-384 → FAISS HNSW → reload check → record pages/chunks/vectors/dim/
params/source-hash/model info → kb_demo with one real query verified by eye + one real
retrieval failure. CI follow-up: missing `uv.lock` (CI runs `uv sync --frozen` → red on every
push) and Bandit B101 on `hooks.py:24` assert (hooks.py is FIXED → fix via Bandit config, not
the file). No `|| true`. Report before committing a generated lockfile (edit-only boundary).

## 4. Execution order

Dependencies drive the order: Package 1 needs rendered page images (corpus + loader), and its
manual transcription can run while the long full-corpus OCR pass runs.

- **A. Groundwork (shared, first):** venv on Python 3.11 (`/opt/homebrew/bin/python3.11` —
  default 3.14 has no torch/faiss wheels), `uv sync`, generate `uv.lock` +
  `requirements.lock`, `brew install tesseract`, `bash scripts/get_data.sh` (65 MB,
  hash-verified), then a `page_limit: 5` smoke run end-to-end.
- **B. Full-corpus build (long, unattended):** render 1,034 pages @300 DPI, projection layout,
  Tesseract OCR, chunk/embed/index. Record all build statistics.
- **C. Held-out slice (parallel with B):** pick pages by strata from the EDA findings; get the
  DocAI sidecar from Mahdi; generate pre-annotation drafts; **Mehemud hand-verifies each page
  against the scan** (the one step that must be human); keep pages out of all tuning.
- **D. OCR evaluation:** CER/WER/word-F1 vs the hand-checked labels, exact page IDs, failure
  case write-up. Reproducible command.
- **E. CI repair (separate small commit):** lockfile + Bandit config.
- **F. Demo + evidence:** execute `kb_demo.ipynb` with a real query (verified by eye), then
  hand the measured numbers to whoever updates `design_choices.md` / `pipeline_diagram.md`.

What I (Claude) drive: A, B, D, E, F, the pre-annotation tooling in C, and a fast side-by-side
review flow so the manual verification in C is quick. What only Mehemud does: the actual
label verification, team communications, and sign-off on each commit.

## 5. Pings to send the team (short, today)

1. **Mahdi:** are both packages mine, or is one Jonayed's? (His brief contains two "Your A2
   ownership" blocks.)
2. **Mahdi:** send the Document AI words sidecar (needed for pre-annotation).
3. **Mahdi + sir:** speciality relabel — #8/E3 vs #9/E2 (§2).
4. **Mahdi:** repo URL check — brief says `doc-agent-G07.git`, remote is `doc-agent-7.git`.
5. **Mahdi:** heads-up that fixing CI requires committing a generated `uv.lock` (his
   edit-only boundary asks for a report first).

## 6. Housekeeping

- Attribution = commit identity, not branch name. Mine is
  `Mehemud Azad <2105014@ugrad.cse.buet.ac.bd>` — this email must be added to my GitHub
  account or the graded per-member attribution misses my commits.
- Save AI-chat transcripts for `transcripts/2105014.txt` (one per member per milestone).
- The A1 form's licence link names IA item `…00pierrich`; `scripts/get_data.sh` pins
  `…00pier`. Two IA copies of the same book — A2 provenance should name the pinned one.
- Drop the A1 form's "add modern data from web scraping" idea — conflicts with the fixed
  grounded task, our groundedness ≥ 0.9 target, and the no-live-web-search policy.
