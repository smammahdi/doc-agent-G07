# Embedding Models & Indexing Tools — Offline Asset Builders

To package all models cleanly on Kaggle without hitting disk limits, you only need to run **2 notebooks** (or just Bundle 1 if focusing on text retrieval):

---

## 🎯 The 2-Bundle Workflow

| Bundle | Script | Output Kaggle Dataset Name | Size | What It Includes | Role in Pipeline |
|:---|:---|:---|:---:|:---|:---|
| **Bundle 1** | [**`package-text-embeddings.py`**](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/development/offline_asset_builders/embedding-indexing/package-text-embeddings.py) | `text-embeddings-offline-assets` | **~5.8 GB** | • `all-MiniLM-L6-v2`<br>• `bge-small-en-v1.5`<br>• `nomic-embed-text-v1.5`<br>• `Qwen3-Embedding-0.6B`<br>• `Qwen3-Embedding-4B-GGUF` (Q4_K_M)<br>• `Qwen3-Reranker-0.6B`<br>• `ms-marco-MiniLM-L6-v2`<br>• All Python wheels | **All Text Embeddings + Rerankers + Indexing Tools** (Stage 4 & Stage 5). |
| **Bundle 2** | [**`package-multimodal-vl.py`**](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/development/offline_asset_builders/embedding-indexing/package-multimodal-vl.py) | `multimodal-vl-embedding-assets` | **~6.4 GB** | • `Qwen3-VL-Embedding-2B`<br>• `BAAI/bge-m3`<br>• All Python wheels | **Multimodal Vision-Language** (text + 350 figure crops) & hybrid dense/sparse search. |

---

## 📦 Alternative: All-in-One Single Bundle
If you prefer running a single script that packages everything in one go:
- [**`package-embedding-models.py`**](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/development/offline_asset_builders/embedding-indexing/package-embedding-models.py) (~12.4 GB total, saves as `embedding-indexing-offline-assets`).

---

## Python Wheels Included in Both Bundles
- `sentence-transformers>=3.0.0`
- `transformers>=4.44.0`
- `faiss-cpu>=1.8.0`
- `tiktoken>=0.7.0`
- `safetensors>=0.4.0`
- `accelerate>=0.30.0`
- `einops>=0.7.0`
- `rank-bm25>=0.2.2` (BM25 sparse ranking)
- `langchain-text-splitters>=0.2.0` (recursive text chunking)

---

## How to Run on Kaggle (CPU, Internet: ON)

1. Open a new Kaggle Notebook.
2. Set **Accelerator: None (CPU)** and **Internet: ON**.
3. Copy-paste [`package-text-embeddings.py`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/development/offline_asset_builders/embedding-indexing/package-text-embeddings.py) and click **Run All** (~1.5 minutes).
4. Save notebook output as Kaggle Dataset: `text-embeddings-offline-assets`.
5. *(Optional for figures)* Repeat with [`package-multimodal-vl.py`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/development/offline_asset_builders/embedding-indexing/package-multimodal-vl.py) (~2 minutes).
