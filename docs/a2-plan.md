# A2 — where we are and what's next

Working branch: `a2/pierce-kb-foundation` (9 commits ahead of `main`, 0 behind — a clean
fast-forward). It already contains everything on `a2/measured-design`, so that branch is
redundant and can be deleted once this one merges.

This document is a read of the repository as it stands, checked against the A2 checklist in
`handbook/05-Codebase-Guide.md` (§ "A2 — Build the knowledge base") and the carry-over rule in
`SUBMISSION.md`. It records what the code actually does, what is missing, and a proposed order
of work. It does not claim any result the repo cannot reproduce.

---

## 1. What is built

All of these are real implementations on this branch, not stubs. None of them has been run
against the full corpus on this machine.

| Stage | File | What it does now |
|---|---|---|
| 1 Ingest | `src/doc_agent/ingest/loader.py` | PyMuPDF renders the Pierce PDF page-by-page to RGB JPEG at 300 DPI / q80. Atomic writes, one-indexed `pNNNN` IDs, render settings encoded in the output path so a setting change can't reuse stale files. `page_limit` supports bounded smoke runs. |
| 1 Preprocess | `src/doc_agent/ingest/preprocess.py` | Identity only. Validates pages, writes nothing, returns them unchanged. `mode` other than `identity` is rejected. |
| 2 Layout | `src/doc_agent/vision/layout.py` | Three modes: `projection` (OpenCV Otsu + row-run grouping, full-page fallback), `full_page`, and `chandra` (reads an offline `chunks.jsonl`, normalises each row against its own `page_box`, clamps to the image). Chandra labels map onto the fixed `text/heading/table/figure` kinds. Missing Chandra pages follow explicit `missing_pages` behaviour. |
| 3 OCR | `src/doc_agent/vision/ocr.py` | `tesseract` baseline via pytesseract on the region crop, plus an offline `document_ai_reference` mode that assigns each precomputed word to the smallest containing Region by centre point and joins closing punctuation. Batch path deduplicates overlapping Regions deterministically. |
| 4 Chunk | `src/doc_agent/index/chunk.py` | Whitespace tokenisation, 256 tokens with 32 overlap, deterministic IDs, page provenance preserved. |
| 4 Embed | `src/doc_agent/index/embed.py` | Lazy `all-MiniLM-L6-v2` load, shape and finiteness checks, refuses to emit placeholder vectors. |
| 4 Store | `src/doc_agent/index/store.py` | FAISS `IndexHNSWFlat` build + load, atomic `chunks.jsonl`, `metadata.json`, and a load-time check that metadata / vector count / chunk count agree. |
| — | `scripts/build_index.sh` | One command that runs the whole thing. |
| — | `reports/pipeline_diagram.md` | Mermaid diagram plus an explicit evidence boundary. |
| — | `configs/design_choices.md` | The 8-facet table across all ten stages. |

The code quality is good — consistent validation, atomic writes, no silent fallbacks. The gap
is not code, it is **evidence**: nothing here has produced a measured number yet.

---

## 2. Gaps against the A2 checklist

### 2a. A2 core (items 1–11)

| # | Item | State |
|---|---|---|
| 1–8 | loader → preprocess → layout → OCR → chunk → embed → store → `build_index.sh` | ✅ implemented |
| 9 | `notebooks/kb_demo.ipynb` — OCR quality + one retrieval example | ❌ **0 of 3 code cells executed.** Also needs a working retrieval to show. |
| 10 | `reports/pipeline_diagram.md` | ✅ done |
| 11 | `configs/design_choices.md` | ✅ done |

**The index has never actually been built.** That is the single biggest hole — item 9, the
design table's own "pending" notes, and the reproducibility gate all depend on it.

### 2b. Data-speciality enhancement — currently nothing

The A2 checklist has a mandatory "▲ Data-speciality (implement only your condition)" section.
Our declared speciality in `configs/task.yaml` and `grading_kit/manifest.yaml` is `dirty-ocr`.
No speciality enhancement is implemented in any candidate home:

- **E1** (degraded scans / bleed-through) → `ingest/preprocess.py` + `ingest/enhance.py` — preprocess is identity, enhance is disabled and unimplemented.
- **E3** (font-typography diversity / old-orthography) → `vision/ocr.py` + `index/chunk.py` — stock Tesseract, no orthography normalisation.
- **E26** (dirty-OCR-provided: skip OCR, clean the given text) → `index/chunk.py` — chunking does no cleaning at all.

This needs a decision (see §5) and then an implementation. It is a graded A2 item.

### 2c. NFR enhancement

- **E5** — scalable ANN index (HNSW/IVF) → `index/store.py`: ✅ satisfied by the FAISS HNSW store.
- **E4** — paragraph/semantic chunking: not done, and not required if we claim E5.

Our NFR is `explainable`. Page provenance survives into every `Chunk` via `page_ids`, so the
"100% page citations" target is structurally supported. The verifiable-in-under-30-seconds half
of the target has no measurement yet.

### 2d. A1 carry-over — graded at A2

`SUBMISSION.md` is explicit that the A1 artifacts are graded at A2 for this cohort.

| Item | State |
|---|---|
| `configs/task.yaml` | ✅ |
| `data/provenance.md` + corpus | ✅ (corpus fetched by script, correctly not committed) |
| `notebooks/eda.ipynb` | ✅ 2 of 2 code cells executed |
| `grading_kit/manifest.yaml` | ✅ |
| `grading_kit/heldout_pages/` | ❌ **empty** — README only |
| `grading_kit/labels.jsonl` | ❌ **one `REPLACE ME` placeholder line** |
| `data/validate.py → validate()` | ❌ `NotImplementedError` |
| `data/versioning.py → snapshot()` | ❌ `NotImplementedError` |
| `logging_conf.py → register()` | ❌ `get_logger()` is done; the tracing handler raises |
| `transcripts/<student number>.txt` | ❌ missing — directory has only a README |

The held-out slice is the blocker for any honest OCR number. Without transcribed pages there is
no CER/WER, and `design_choices.md` already says so.

### 2e. Tests and reproducibility

- 12 placeholder tests are `@pytest.mark.skip` across 8 files, including `test_ingest.py`,
  `test_ocr.py`, and `test_smoke.py`. The codebase guide marks these "IMPLEMENT — CI runs these".
- `requirements.lock` is still the generated-file stub with no pins. The reproducibility gate in
  `STRUCTURE.md` reads this file, so it must be produced by
  `uv pip compile pyproject.toml -o requirements.lock`.

---

## 3. Local environment blockers

Everything in §2 that needs a real run is gated on these four. All are fixable.

1. **No `.venv`.** `make setup` runs `uv sync --frozen`, which will fail against the stub lockfile.
2. **Python version.** Default `python3` here is 3.14. `torch>=2.2,<2.4` and `faiss-cpu>=1.8,<1.9`
   publish no wheels for 3.13/3.14. Python 3.11.14 is installed at `/opt/homebrew/bin/python3.11`
   and matches `requires-python`; the venv must be pinned to it.
3. **`tesseract` is not installed.** The OCR baseline shells out to it via pytesseract.
   Needs `brew install tesseract`.
4. **Corpus not downloaded.** `scripts/get_data.sh` fetches ~65 MB from the Internet Archive and
   verifies size + SHA-256. Safe and idempotent; just hasn't been run here.

Note that the full dependency set (torch, transformers, gradio, wandb) is several GB.

---

## 4. Proposed order of work

Sequenced so each phase unblocks the next.

### Phase 1 — Make the pipeline runnable (blocks everything else)
1. Create the venv against Python 3.11, generate a real `requirements.lock`, commit it.
2. `brew install tesseract`; record the version in the design record.
3. Run `scripts/get_data.sh` to fetch and verify the corpus.
4. Smoke-run the full pipeline with `ingest.page_limit: 5` to prove the wiring end-to-end.

### Phase 2 — Build the index for real
5. Full-corpus run: ingest → layout → OCR → chunk → embed → FAISS. This is the long one;
   Tesseract over 1,034 pages of regions is the bottleneck.
6. Record real build statistics — page count, region count, chunk count, index size, wall time —
   and fold them into `design_choices.md` and `pipeline_diagram.md`, replacing the "pending" notes.

### Phase 3 — Held-out slice and a real OCR number
7. Choose held-out pages (spread across body text, figure pages, and tables), copy the images
   into `grading_kit/heldout_pages/`.
8. Hand-transcribe them into `grading_kit/labels.jsonl`. This is manual and slow — it is the
   long pole, and it is what makes every OCR claim honest.
9. Score Tesseract CER/WER against those labels. Report the number even if it is bad; a measured
   bad number is worth more here than an unmeasured good one.

### Phase 4 — Data-speciality enhancement
10. Settle the E-item question in §5, then implement it in its designated home and measure it
    against the same held-out slice — before-and-after on the same pages.

### Phase 5 — A1 carry-over and gates
11. Implement `data/validate.py → validate()` (≥300 pages, ≥60k words, split by document) and
    `data/versioning.py → snapshot()` (hash + corpus version id).
12. Implement `logging_conf.register()` so the seams emit `TraceStep` lines to `traces/run.jsonl`.
13. Replace the placeholder tests for ingest and OCR with real unit tests.
14. Add the per-member `transcripts/<student number>.txt` files.

### Phase 6 — Demo, docs, submission
15. Execute `notebooks/kb_demo.ipynb` end-to-end with real outputs committed.
16. Fill `forms/A2_form.docx`.
17. Final pass over `design_choices.md` so every "pending" line is either a measured number or an
    explicit, justified deferral. Then tag `a2-submit`.

**Note on item 15:** kb_demo must show "OCR quality and one retrieval example". A retrieval
example needs `retrieval/retriever.py → retrieve()`, which our design record currently defers to
A3 and which the checklist also lists under A3. The minimum honest resolution is a direct FAISS
query against the built index inside the notebook — demonstrating the index retrieves — without
claiming the Stage 5 `Retriever` is implemented. Worth confirming against the A2 form's wording.

---

## 5. Open decisions

These need the team's call; they change what we build.

1. **Which data-speciality E-item do we owe?** We declared `dirty-ocr`, but the catalog's E26 is
   "dirty-OCR-*provided* — skip OCR, clean the given text", and we are producing our own OCR from
   scans. Our `task.yaml` comment says the cause is "period typography and uneven ink", which
   reads closer to E1 (degraded scans → preprocess/enhance) or E3 (typography/old-orthography →
   ocr + chunk). The honest options are to reframe the speciality in `task.yaml`/`manifest.yaml`,
   or to build the E-item that matches what the corpus actually does to us. Either way, one has
   to be implemented and measured.

2. **Does the identity preprocess stand?** Checklist item 2 says "deskew, denoise, binarize,
   augment (baseline)". We deliberately shipped identity and documented why. That is defensible
   as a measured choice, but only once we can show a comparison on the held-out slice — otherwise
   it reads as a missing item rather than a decision.

3. **Do we fine-tune OCR?** Checklist item 4 says "fine-tune a pretrained OCR model"; we run
   stock Tesseract with `ocr.finetune: false`. Same shape of problem as #2 — a deliberate
   deferral needs a number behind it.

4. **Where does the full-corpus run happen?** If the 1,034-page Tesseract pass is too slow on
   this machine, we should agree on a shared compute environment now, because Phase 2 gates
   Phases 3, 4, and 6.

5. **Chandra and Document AI outputs.** Both are real and already measured, but neither is
   committed or publicly fetchable, so a grader cannot reproduce them. Decide whether they stay
   as documented-but-unreproducible comparison evidence, or come out of the graded narrative.

---

## 6. Merge hygiene

`main` is untouched. When this branch is ready, open a PR the same way `a2/ingest-loader` went in,
and delete `a2/measured-design` — it is fully contained in this branch.
