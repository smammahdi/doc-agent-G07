# Embedding Models & Indexing Tools — Offline Asset Builder

This directory contains the Kaggle asset builder script to package multiple sentence embedding models (MiniLM, BGE, GTE-Qwen2, MPNet) and vector indexing dependencies for 100% network-free execution.

## Packaged Models

| Model ID | Short Name | Type | Dims | Size | Description |
|:---|:---|:---:|:---:|:---:|:---|
| `sentence-transformers/all-MiniLM-L6-v2` | `all-minilm-l6-v2` | Text Embed | 384 | ~90 MB | Lightweight 22M param fast dense baseline (starter default) |
| `BAAI/bge-small-en-v1.5` | `bge-small-en-v1-5` | Text Embed | 384 | ~133 MB | SOTA compact 33M param MTEB retrieval champion |
| `Qwen/Qwen3-Embedding-0.6B` | `qwen3-embedding-0-6b` | Text Embed | 1024 | ~1.2 GB | SOTA 0.6B compact Qwen3 instruction-aware text embedder |
| `Qwen/Qwen3-Embedding-4B` | `qwen3-embedding-4b` | Text Embed | 2560 | ~8.2 GB | High-capacity 4.0B param Qwen3 dense text embedder |
| `Qwen/Qwen3-Reranker-0.6B` | `qwen3-reranker-0-6b` | Reranker | — | ~1.2 GB | Fast 0.6B param Qwen3 instruction-tuned cross-encoder reranker |
| `Qwen/Qwen3-VL-Embedding-2B` | `qwen3-vl-embedding-2b` | Multimodal Embed | 1536 | ~4.2 GB | 2.0B Qwen3 Vision-Language multimodal embedder for text + figures |
| `BAAI/bge-m3` | `bge-m3` | Text Embed | 1024 | ~2.2 GB | Universal SOTA model (dense + sparse + multi-vector) |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | `ms-marco-minilm-l6-v2` | Reranker | — | ~90 MB | 22M param baseline cross-encoder reranker |

## Packaged Python Wheels
- `sentence-transformers`
- `transformers`
- `faiss-cpu`
- `tiktoken`
- `safetensors`
- `accelerate`
- `einops`
- `rank-bm25` (for hybrid dense + sparse retrieval)
- `langchain-text-splitters` (for recursive & semantic chunking)

## How to Build on Kaggle

1. Create a new Kaggle notebook.
2. Configure settings:
   - **Accelerator**: None (CPU)
   - **Internet**: **ON**
3. Paste the contents of [`package-embedding-models.py`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/development/offline_asset_builders/embedding-indexing/package-embedding-models.py) into a notebook cell and run it.
4. When finished, create a new private Kaggle Dataset from the notebook's output:
   - Name: `embedding-indexing-offline-assets`
5. Attach the resulting dataset to your offline indexing and retrieval benchmark notebooks with **Internet: OFF**.
