# Embedding Models & Indexing Tools — Offline Asset Builders

This directory contains modular Kaggle asset builder scripts to package embedding models and indexing dependencies for 100% network-free execution.

## Available Modular Asset Builders

| Script | Dataset Name | Size | Included Models | Best Use Case |
|:---|:---|:---:|:---|:---|
| **[`package-core-embeddings.py`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/development/offline_asset_builders/embedding-indexing/package-core-embeddings.py)** | `core-embeddings-offline-assets` | **~2.2 GB** | `MiniLM-L6-v2`, `bge-small-en-v1.5`, `nomic-embed-text-v1.5`, `Qwen3-Embedding-0.6B`, `ms-marco-MiniLM` | ⚡ **Fastest build (<45s)**, covers all core dense benchmarking. |
| **[`package-qwen3-family.py`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/development/offline_asset_builders/embedding-indexing/package-qwen3-family.py)** | `qwen3-embedding-reranker-assets` | **~5.2 GB** | `Qwen3-Embedding-0.6B`, `Qwen3-Embedding-0.6B-GGUF` (Q4_K_M), `Qwen3-Embedding-4B-GGUF` (Q4_K_M), `Qwen3-Reranker-0.6B` | 🚀 **Dedicated Qwen3 Suite**: Modern instruction-aware embedders & rerankers. |
| **[`package-multimodal-vl.py`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/development/offline_asset_builders/embedding-indexing/package-multimodal-vl.py)** | `multimodal-vl-embedding-assets` | **~6.4 GB** | `Qwen3-VL-Embedding-2B` (text + figures), `bge-m3` (dense + sparse BM25) | 🎨 **Multimodal & Hybrid**: Joint medical text and figure crops indexing. |
| **[`package-embedding-models.py`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/development/offline_asset_builders/embedding-indexing/package-embedding-models.py)** | `embedding-indexing-offline-assets` | **~12.4 GB** | All 10 models combined + wheels | 📦 **Unified Full Suite**: Everything in one single Kaggle dataset. |

---

## Packaged Python Wheels (Included in every builder)
- `sentence-transformers>=3.0.0`
- `transformers>=4.44.0`
- `faiss-cpu>=1.8.0`
- `tiktoken>=0.7.0`
- `safetensors>=0.4.0`
- `accelerate>=0.30.0`
- `einops>=0.7.0`
- `rank-bm25>=0.2.2` (hybrid search)
- `langchain-text-splitters>=0.2.0` (semantic & recursive chunking)

---

## How to Build on Kaggle

1. Create a new Kaggle notebook (**CPU / No accelerator**, **Internet: ON**).
2. Paste the contents of your chosen builder script into a notebook cell and run it:
   - For quick baseline & Stage 4 tests: use [`package-core-embeddings.py`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/development/offline_asset_builders/embedding-indexing/package-core-embeddings.py).
   - For Qwen3 embeddings & rerankers: use [`package-qwen3-family.py`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/development/offline_asset_builders/embedding-indexing/package-qwen3-family.py).
   - For multimodal figures: use [`package-multimodal-vl.py`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/development/offline_asset_builders/embedding-indexing/package-multimodal-vl.py).
3. When finished, create a private Kaggle Dataset from the output folder using the matching dataset name.
4. Attach the dataset to your offline indexing/benchmarking notebook with **Internet: OFF**.
