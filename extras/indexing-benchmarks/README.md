# Stage 4 Indexing, Chunking, & Vector Search Benchmarks

This directory contains the reproducible experimental suite to evaluate and justify all design choices for **Stage 4 (Chunking, Embedding Models, and FAISS Vector Indexing)** on the *People's Common Sense Medical Adviser* (Pierce, 1890).

---

## 1. Directory Structure

```
extras/indexing-benchmarks/
├── README.md               # Overview, candidate rationale, and published model cards
├── data/
│   ├── canonical-pages.jsonl   # Full 1,034-page normalized text corpus (364,824 words)
│   └── retrieval-queries.jsonl # 60 verified whole-book retrieval queries with exact character spans
├── code/                   # Reusable, modular Python package
│   ├── __init__.py
│   ├── corpus.py           # Corpus loader & 6-page missing policy handler
│   ├── queries.py          # Query schema validation & dev/test/out-of-corpus splitting
│   ├── chunking.py         # 5 chunking strategies (fixed, paragraph-header, parent-child)
│   ├── models.py           # Model adapters with official query/document prefixes
│   ├── evaluation.py       # Metrics, span containment, and 1,000-sample bootstrap 95% CIs
│   ├── faiss_benchmark.py  # Inner Product Flat, HNSW, and IVFFlat latency/agreement suite
│   └── runner.py           # Master 5x5 factorial benchmark orchestrator
├── notebooks/
│   ├── 01-run-benchmark.ipynb    # Thin execution notebook for Kaggle/local runs
│   └── 02-review-results.ipynb   # Visual inspection & failure analysis
├── results/                # Output directory for validated JSON reports and per-query logs
└── bundles/                # Packaged Kaggle input archives (`indexing-benchmark-data.zip`)
```

---

## 2. Embedding Model Shortlist & Published Benchmark Rationale

External benchmarks (such as MTEB, BEIR, and published model cards) justify why these five distinct architectures were shortlisted. However, **the local Pierce historical medical benchmark selects the final winner** to optimize for our primary Non-Functional Requirement (**Explainability** — 100% human-verifiable citations under 30 seconds).

| # | Model Identifier | Canonical ID | Parameters & Dim | Official Model Card | Published Benchmark Rationale |
|---|---|---|:---:|:---:|---|
| **1** | `sentence-transformers/all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` | 22.7M / 384-d | [HuggingFace Hub](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | **Lightweight Baseline**: Ultra-fast latency (~14,200 sentences/sec) and minimal memory footprint (<100MB), serving as the standard efficiency reference. |
| **2** | `BAAI/bge-small-en-v1.5` | `bge-small-en-v1.5` | 33.4M / 384-d | [HuggingFace Hub](https://huggingface.co/BAAI/bge-small-en-v1.5) | **Compact English Retrieval Leader**: Top-tier performance on MTEB retrieval leaderboard within the sub-50M parameter class; requires official task instruction prefix. |
| **3** | `BAAI/bge-m3` | `bge-m3` | 568M / 1024-d | [HuggingFace Hub](https://huggingface.co/BAAI/bge-m3) | **Universal Long-Document Candidate**: Supports multi-granularity dense matching and up to 8,192 input tokens with robust cross-domain generalization. |
| **4** | `nomic-ai/nomic-embed-text-v1.5` | `nomic-embed-text-v1.5` | 137M / 768-d | [HuggingFace Hub](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | **Long-Context / MRL Candidate**: 8,192 token window with Matryoshka Representation Learning (MRL), utilizing asymmetric `search_query` / `search_document` prefixes. |
| **5** | `Qwen/Qwen3-Embedding-0.6B` | `Qwen3-Embedding-0.6B` | 590M / 1024-d | [HuggingFace Hub](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | **Modern Instruction-Aware Candidate**: High semantic capacity with instruction conditioning and last-token pooling tailored for specialized domain retrieval. |

---

## 3. Chunking Strategies Evaluated

All chunking strategies operate on exact whitespace word boundaries:
1. **`fixed_128_16`**: Small window (128 words, 16 overlap, step 112 words).
2. **`fixed_256_32`**: Baseline window (256 words, 32 overlap, step 224 words).
3. **`fixed_512_64`**: Large window (512 words, 64 overlap, step 448 words).
4. **`paragraph_header_aware`**: Structural segmentation respecting `\n\n` paragraph boundaries and uppercase section headers (target $\le 300$ words).
5. **`parent_child_128_512`**: Small child chunks (128 words) indexed in FAISS; upon retrieval, parent text context (512 words) is returned.

---

## 4. FAISS Vector Search Architectures

All indexes use **L2-normalized embeddings with `METRIC_INNER_PRODUCT`** to compute exact cosine similarity:
- **`IndexFlatIP`**: Exact brute-force baseline ($O(N)$).
- **`IndexHNSWFlat`**: Hierarchical Navigable Small World graph ($M=32$, $efConstruction=64$, $efSearch=32$).
- **`IndexIVFFlat`**: Inverted Voronoi File clustering ($nlist=64$, $nprobe=8$).

---

## 5. Execution Workflow

1. **Local Run**:
   ```bash
   python3 -c "from code.runner import run_stage4_benchmark; from pathlib import Path; run_stage4_benchmark(Path('results'))"
   ```
2. **Kaggle Offline Run**:
   - Attach `bundles/indexing-benchmark-data.zip` and the offline model wheels.
   - Run `notebooks/01-run-benchmark.ipynb`.
   - Inspect outputs in `notebooks/02-review-results.ipynb`.
