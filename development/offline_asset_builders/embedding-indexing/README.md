# Embedding Models & Indexing Tools — Offline Asset Builder

This directory contains the Kaggle asset builder script to package multiple sentence embedding models (MiniLM, BGE, GTE-Qwen2, MPNet) and vector indexing dependencies for 100% network-free execution.

## Packaged Models

| Model ID | Short Name | Dims | Size | Description |
|:---|:---|:---:|:---:|:---|
| `sentence-transformers/all-MiniLM-L6-v2` | `all-minilm-l6-v2` | 384 | ~90 MB | Lightweight 22M param fast dense baseline (starter default) |
| `BAAI/bge-small-en-v1.5` | `bge-small-en-v1-5` | 384 | ~133 MB | SOTA compact 33M param MTEB retrieval champion |
| `BAAI/bge-base-en-v1.5` | `bge-base-en-v1-5` | 768 | ~438 MB | Standard 110M param dense retriever |
| `sentence-transformers/all-mpnet-base-v2` | `all-mpnet-base-v2` | 768 | ~438 MB | High-accuracy RoBERTa/MPNet 110M embedder |
| `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | `gte-qwen2-1-5b-instruct` | 1536 | ~3.1 GB | SOTA 1.5B param Qwen2 instruction-aware embedding model |

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
