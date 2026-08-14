# Stage 4 Indexing Experiments: Chunking, Embedding, & FAISS Benchmarks

This directory contains the experimental framework to evaluate and justify all design choices for **Stage 4 (Chunking, Embedding, and Vector Indexing)**.

## Experiments Covered

### 1. Chunking Strategies
- **Fixed-Window Token Chunking**:
  - Small: `128` tokens / `16` overlap
  - Baseline: `256` tokens / `32` overlap (starter default)
  - Large: `512` tokens / `64` overlap
- **Recursive Semantic Chunking**:
  - Respects Markdown section headings (`#`, `##`), paragraph breaks (`\n\n`), and sentence boundaries (`.?!\n`), ensuring medical recipes and anatomical descriptions are not sliced mid-sentence.

### 2. Embedding Models
- `sentence-transformers/all-MiniLM-L6-v2` (384-d, 22M parameters)
- `BAAI/bge-small-en-v1.5` (384-d, 33M parameters)
- `BAAI/bge-base-en-v1.5` (768-d, 110M parameters)
- `sentence-transformers/all-mpnet-base-v2` (768-d, 110M parameters)
- `Alibaba-NLP/gte-Qwen2-1.5B-instruct` (1536-d, 1.5B parameters)

### 3. Vector Index Architectures (FAISS)
- **`IndexFlatIP`**: Exact Inner Product / Cosine Similarity baseline (100% recall, zero approximation error, ultra-fast for corpora $\le 10,000$ chunks).
- **`IndexHNSWFlat`**: Graph-based Approximate Nearest Neighbor ($M=16, 32$; $efSearch=64$).
- **`IndexIVFFlat`**: Voronoi cell cluster indexing ($nlist=16, 32$; $nprobe=8$).
- **`IndexIVFPQ`**: Inverted File with 8-bit Product Quantization byte-level compression.

---

## How to Run on Kaggle (Offline)

1. Attach the `embedding-indexing-offline-assets` dataset to your Kaggle notebook.
2. Attach the Pierce OCR layout output bundle.
3. Turn **Internet: OFF**.
4. Run [`indexing-benchmark.py`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/extras/indexing-experiments/indexing-benchmark.py).
5. Download the resulting `indexing-benchmark-outputs.zip` containing `indexing_comparison_results.json`.
6. Use the metrics to populate the **Stage 4** row of [`configs/design_choices.md`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/configs/design_choices.md).
