# Implementation Plan: Milestone A2 (Package 1: Held-out OCR & Package 2: Index, Demo, CI)

This plan details the implementation strategy for K. M. Mehemud Azad (student 2105014) on Team G07's capstone repository (`doc-agent`). It covers both owned packages:
1. **Package 1 — Held-out OCR Evidence** (branch: `a2/heldout-ocr`)
2. **Package 2 — Index + Demo + Reproducibility Gate** (branch: `a2/index-demo-ci`)

All work is now based on `7e1a11b` (tip of `origin/main`, which merges Mahdi's foundation work up to commit `23248aa`), preserving existing pipeline architecture, contract signatures, starter rules, and author identity (`Mehemud Azad <2105014@ugrad.cse.buet.ac.bd>`).

---

## User Review Required

> [!IMPORTANT]
> **Base Commit & Branch Setup**:
> `origin/main` is now updated to commit `7e1a11b` ("Merge A2 knowledge-base foundation into main").
> We will:
> 1. Fetch and pull `origin/main` (`7e1a11b`) to update local `main`.
> 2. Create/reset `a2/heldout-ocr` directly from `origin/main` (`7e1a11b`).
> 3. Create/reset `a2/index-demo-ci` directly from `origin/main` (`7e1a11b`).
> 4. Exclude stray large files from early `kmazd` (`A1_form.docx`, uncorrected raw OCR dumps).

> [!NOTE]
> **Data Speciality Alignment**:
> Declared `data_speciality: "dirty-ocr"` in `task.yaml` / `manifest.yaml` represents text delivered already garbled. Candidate options under team/faculty discussion are catalog #8 (`old/reformed-orthography`) or #9 (`table/figure-heavy`). Our code will maintain full compatibility while awaiting final team sign-off.

> [!NOTE]
> **Faculty Condition on Figure QA**:
> Image descriptions must be stored separately as sidecar chunks/artifacts and figure crops shown with citations. OCR text and figure captions will remain strictly unmerged.

---

## Open Questions

> [!IMPORTANT]
> 1. Should `a2/index-demo-ci` be merged into `main` after review, or kept as a standalone topic branch until Mahdi integrates A2 for submission?
> 2. Are there specific page IDs preferred for the 5-10 held-out evaluation pages in `grading_kit/heldout_pages/` (e.g. pages containing medical diagrams, multi-column tables, or archaic typography)?

---

## Proposed Changes

### Branch: `a2/heldout-ocr` (Package 1)

#### [NEW] [heldout_pages](file:///Users/mehemudazad/Desktop/DL_project/doc-agent-7/grading_kit/heldout_pages)
- Populate `grading_kit/heldout_pages/` with 5-10 real, rendered JPEG page scans from R. V. Pierce's "The People's Common Sense Medical Adviser" (e.g., `p0005.jpg`, `p0012.jpg`, `p0025.jpg`, `p0050.jpg`, `p0100.jpg`).

#### [MODIFY] [labels.jsonl](file:///Users/mehemudazad/Desktop/DL_project/doc-agent-7/grading_kit/labels.jsonl)
- Replace placeholder line with hand-verified, high-precision ground truth transcriptions for each held-out page.
- Format: `{"page_id": "p0005", "text": "<exact ground truth text>"}`.

#### [MODIFY] [test_ocr.py](file:///Users/mehemudazad/Desktop/DL_project/doc-agent-7/tests/test_ocr.py)
- Implement unit tests evaluating OCR transcription performance against `grading_kit/labels.jsonl`.
- Compute CER (Character Error Rate), WER (Word Error Rate), and Word F1 score.
- Assert exact page ID alignment between images, manifest, and ground truth labels.
- Record and document at least 1 real OCR failure case (e.g., archaic font, ligature, narrow column, or noise).

---

### Branch: `a2/index-demo-ci` (Package 2)

#### [MODIFY] [chunk.py](file:///Users/mehemudazad/Desktop/DL_project/doc-agent-7/src/doc_agent/index/chunk.py)
- Implement `split(chunks: list[Chunk], cfg: dict) -> list[Chunk]`.
- Re-chunk text according to `cfg['index']['chunk_tokens']` (e.g., 256) and `cfg['index']['overlap']` (e.g., 32), preserving doc and page provenance.

#### [MODIFY] [embed.py](file:///Users/mehemudazad/Desktop/DL_project/doc-agent-7/src/doc_agent/index/embed.py)
- Implement `encode(chunks: list[Chunk], cfg: dict)` using `SentenceTransformer` with `cfg['embed']['model']` (e.g. `sentence-transformers/all-MiniLM-L6-v2`).
- Return numpy array or torch tensor of embeddings matching chunk indices.

#### [MODIFY] [store.py](file:///Users/mehemudazad/Desktop/DL_project/doc-agent-7/src/doc_agent/index/store.py)
- Implement `build(chunks, vectors, cfg: dict) -> None`: Create FAISS HNSW/L2 vector index and serialize index + metadata payload to `data/processed/index/`.
- Implement `load(cfg: dict)`: Load serialized FAISS index and chunk lookup table from disk.

#### [MODIFY] [build_index.sh](file:///Users/mehemudazad/Desktop/DL_project/doc-agent-7/scripts/build_index.sh)
- Complete bash script to run the end-to-end knowledge base construction pipeline over the full Pierce corpus.

#### [MODIFY] [kb_demo.ipynb](file:///Users/mehemudazad/Desktop/DL_project/doc-agent-7/notebooks/kb_demo.ipynb)
- Create executed demonstration notebook loading the saved FAISS vector store.
- Perform 1 verified real medical domain retrieval query and display retrieved text with page citations.

#### [NEW] [.bandit.yaml](file:///Users/mehemudazad/Desktop/DL_project/doc-agent-7/.bandit.yaml) & [uv.lock](file:///Users/mehemudazad/Desktop/DL_project/doc-agent-7/uv.lock)
- Generate synced `uv.lock` ensuring `uv sync --frozen --extra dev` succeeds in CI.
- Configure `.bandit.yaml` to exclude `src/doc_agent/hooks.py` (which contains necessary `assert` statements in uneditable pre-commit hook architecture).

---

## Verification Plan

### Automated Tests
```bash
# 1. Verify environment and lockfile reproducibility
uv sync --frozen --extra dev

# 2. Code formatting and linting gates
uv run ruff check .
uv run black --check .
uv run mypy src

# 3. Security audit gate
uv run bandit -c .bandit.yaml -r src/

# 4. Test suite execution
uv run pytest tests/test_structure.py tests/test_contracts.py tests/test_tools.py tests/test_ocr.py
```

### Manual Verification
1. Run `bash scripts/build_index.sh` and verify vector index artifact generation under `data/processed/index/`.
2. Open and inspect `notebooks/kb_demo.ipynb` output to confirm top-k chunk retrieval quality and page citations against the source scan.
