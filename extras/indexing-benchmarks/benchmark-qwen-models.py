# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec: {display_name: Python 3, language: python, name: python3}
# ---
# %% [markdown]
# # Stage 4: Qwen3 IR & Multimodal Embedding Suite Benchmark
#
# Benchmarks modern Qwen3 IR models:
# 1. `Qwen/Qwen3-Embedding-0.6B` (PyTorch, 1024-d)
# 2. `Qwen/Qwen3-Embedding-0.6B-GGUF` (Quantized Q4_K_M, 1024-d)
# 3. `Qwen/Qwen3-Embedding-4B-GGUF` (Quantized Q4_K_M, 2560-d)
# 4. `Qwen/Qwen3-VL-2B` (Vision-Language Multimodal, 1536-d)
#
# Outputs: `results_qwen_family.json`

# %%
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if Path("/kaggle/working").is_dir():
    INPUT_ROOT = Path("/kaggle/input")
    WORK = Path("/kaggle/working")
else:
    INPUT_ROOT = Path("extras/indexing-benchmarks")
    WORK = Path("extras/indexing-benchmarks/output")

OUT_DIR = WORK / "indexing-benchmark-outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
USE_GPU = True
BATCH_SIZE = 16

# %%
def is_llama_cpp_available() -> bool:
    try:
        import llama_cpp
        return True
    except Exception:
        return False

class GGUFEmbeddingWrapper:
    def __init__(self, model_path: str, n_ctx: int = 2048):
        from llama_cpp import Llama
        p = Path(model_path)
        if p.is_dir():
            gguf_files = sorted(list(p.glob("*.gguf")) + list(p.glob("*.GGUF")), key=lambda x: x.stat().st_size, reverse=True)
            if not gguf_files:
                raise FileNotFoundError(f"No .gguf file found in {model_path}")
            gguf_file = str(gguf_files[0])
        else:
            gguf_file = str(p)
        self.llm = Llama(model_path=gguf_file, embedding=True, n_ctx=n_ctx, verbose=False)
        self.model_name = Path(gguf_file).stem

    def encode(self, texts: list[str], normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        vectors = []
        for text in texts:
            emb = self.llm.create_embedding(text)
            vec = np.array(emb["data"][0]["embedding"], dtype=np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

class QwenVLTextEmbeddingWrapper:
    def __init__(self, model_path: str, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.model_name = Path(model_path).name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<|endoftext|>"

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = 16, normalize_embeddings: bool = True, **kwargs) -> np.ndarray:
        import torch
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**encoded)
                hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                sum_emb = torch.sum(hidden * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                pooled = sum_emb / sum_mask
                if normalize_embeddings:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embs.append(pooled.float().cpu().numpy())
        return np.vstack(all_embs) if all_embs else np.empty((0, 1536), dtype=np.float32)

# %%
def load_source_pages() -> list[dict[str, str]]:
    pages = []
    for p in [INPUT_ROOT / "chunks.jsonl", *INPUT_ROOT.rglob("chunks.jsonl"), Path("extras/output/chandra/chunks.jsonl")]:
        if p.is_file():
            try:
                page_blocks: dict[str, list[str]] = {}
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        item = json.loads(line)
                        p_id = item.get("page_id", "p0001")
                        txt = item.get("text", "").strip()
                        if txt:
                            page_blocks.setdefault(p_id, []).append(txt)
                if len(page_blocks) >= 50:
                    for pid in sorted(page_blocks.keys()):
                        pages.append({"doc_id": "pierce-1890", "page_id": pid, "text": "\n\n".join(page_blocks[pid])})
                    print(f"Loaded {len(pages)} FULL BOOK pages reconstructed from {p}")
                    return pages
            except Exception:
                pass
    return []

def load_curated_tasks() -> list[dict[str, Any]]:
    for p in [INPUT_ROOT / "tasks.jsonl", *INPUT_ROOT.rglob("tasks.jsonl"), Path("grading_kit/tasks.jsonl")]:
        if p.is_file():
            try:
                tasks = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
                if tasks:
                    print(f"Loaded {len(tasks)} curated tasks from {p}")
                    return tasks
            except Exception:
                pass
    return []

# %%
def run():
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
    print(f"Device: {device.upper()}")

    pages = load_source_pages()
    tasks = load_curated_tasks()
    grounded_tasks = [t for t in tasks if t.get("target_pages") or t.get("gold_pages")]

    # Baseline 256-token chunks
    chunks = []
    step = max(1, 256 - 32)
    for page in pages:
        p_id = page.get("page_id", "p0001")
        doc_id = page.get("doc_id", "pierce-1890")
        words = page.get("text", "").split()
        if not words:
            continue
        for i in range(0, len(words), step):
            win = words[i : i + 256]
            chunks.append({
                "chunk_id": f"{doc_id}_{p_id}_c{len(chunks):04d}",
                "doc_id": doc_id, "page_id": p_id,
                "text": " ".join(win), "token_count": len(win)
            })
            if i + 256 >= len(words):
                break

    print(f"Baseline chunks: {len(chunks)} | Grounded Tasks: {len(grounded_tasks)}")

    # Discover Qwen Models
    models_to_test = []
    for m_cand in ["qwen3-embedding-0-6b", "qwen3-embedding-0-6b-gguf", "qwen3-embedding-4b-gguf", "qwen3-vl-embedding-2b"]:
        for search_dir in [INPUT_ROOT, Path(".")]:
            matches = list(search_dir.rglob(f"*{m_cand}*"))
            dirs = [m for m in matches if (m.is_dir() or m.suffix.lower() == ".gguf") and "reranker" not in m.name.lower()]
            if dirs:
                models_to_test.append(str(dirs[0]))
                break

    if not models_to_test:
        print("No Qwen models discovered. Attach package-qwen3-family dataset.")
        return

    results = []
    for m_path in models_to_test:
        m_name = Path(m_path).name
        is_gguf = "gguf" in m_path.lower() or Path(m_path).suffix.lower() == ".gguf"
        is_vl = "vl" in m_path.lower()

        print(f"\n--- Benchmarking: {m_name} ---")
        try:
            t0 = time.perf_counter()
            if is_gguf:
                if not is_llama_cpp_available():
                    print(f"[GGUF SKIPPED] llama-cpp-python not available for {m_name}")
                    continue
                model = GGUFEmbeddingWrapper(m_path)
                m_type = "GGUF-Q4_K_M"
            elif is_vl:
                model = QwenVLTextEmbeddingWrapper(m_path, device=device)
                m_type = "Multimodal-VL"
            else:
                model = SentenceTransformer(m_path, device=device, trust_remote_code=True)
                m_type = "PyTorch-Dense"
            load_time = time.perf_counter() - t0

            texts = [c["text"] for c in chunks]
            t_enc = time.perf_counter()
            embs = model.encode(texts, normalize_embeddings=True)
            enc_time = time.perf_counter() - t_enc
            embs_np = np.asarray(embs, dtype=np.float32)
            dim = embs_np.shape[1]
            tput = len(texts) / enc_time if enc_time > 0 else 0

            # Single Query Latency
            lats = []
            for q in ["What is the treatment for catarrh?", "Golden Seal medicinal virtues"]:
                t_q = time.perf_counter()
                _ = model.encode([q], normalize_embeddings=True)
                lats.append((time.perf_counter() - t_q) * 1000)
            p50_lat = float(np.median(lats))

            # Retrieval accuracy
            index = faiss.IndexFlatIP(dim)
            index.add(embs_np)

            recalls = {1: 0, 3: 0, 5: 0, 10: 0}
            rr_total = 0.0
            for t in grounded_tasks:
                gold = set(t.get("target_pages") or t.get("gold_pages") or [])
                q_vec = model.encode([t["question"]], normalize_embeddings=True).astype(np.float32)
                _, indices = index.search(q_vec, 10)
                retrieved = [chunks[idx]["page_id"] for idx in indices[0] if idx < len(chunks)]
                for k in [1, 3, 5, 10]:
                    if any(p in gold for p in retrieved[:k]):
                        recalls[k] += 1
                rank = 0
                for r, p in enumerate(retrieved, start=1):
                    if p in gold:
                        rank = r
                        break
                if rank > 0:
                    rr_total += 1.0 / rank

            n = max(1, len(grounded_tasks))
            res = {
                "model": m_name, "path": m_path, "device": device, "type": m_type,
                "dimension": dim, "total_chunks": len(texts),
                "load_time_seconds": round(load_time, 3),
                "encode_time_seconds": round(enc_time, 3),
                "single_query_p50_ms": round(p50_lat, 2),
                "chunks_per_second": round(tput, 1),
                "recall@1": round(recalls[1]/n, 4), "recall@3": round(recalls[3]/n, 4),
                "recall@5": round(recalls[5]/n, 4), "recall@10": round(recalls[10]/n, 4),
                "mrr": round(rr_total/n, 4),
            }
            results.append(res)
            print(f"MRR: {res['mrr']:.4f} | Recall@5: {res['recall@5']:.3f} | Tput: {res['chunks_per_second']:.1f} ch/s | P50: {res['single_query_p50_ms']:.1f} ms")
        except Exception as e:
            print(f"[ERROR] Failed {m_name}: {e}")

    out_file = OUT_DIR / "results_qwen_family.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_file}")

if __name__ == "__main__":
    run()
