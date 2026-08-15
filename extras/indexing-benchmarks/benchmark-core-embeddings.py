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
# # Stage 4: Core Embedding Models Benchmark
#
# Dedicated, self-contained benchmark for standard dense embedding models:
# 1. `sentence-transformers/all-MiniLM-L6-v2` (384-d)
# 2. `BAAI/bge-small-en-v1.5` (384-d)
# 3. `BAAI/bge-m3` (1024-d)
# 4. `cross-encoder/ms-marco-MiniLM-L-6-v2` (384-d)
#
# Runs in ~20 seconds on GPU / ~45 seconds on CPU.
# Output: `results_core_embeddings.json`

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
BATCH_SIZE = 32

# %%
def load_source_pages() -> list[dict[str, str]]:
    pages = []
    # 1. Reconstruct from chunks.jsonl
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

    # 2. full_book_pages.jsonl
    for p in [INPUT_ROOT / "full_book_pages.jsonl", *INPUT_ROOT.rglob("full_book_pages.jsonl"), Path("extras/indexing-benchmarks/full_book_pages.jsonl")]:
        if p.is_file():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        pages.append(json.loads(line))
                if len(pages) >= 50:
                    print(f"Loaded {len(pages)} pages from {p}")
                    return pages
            except Exception:
                pass

    # 3. labels.jsonl fallback
    for p in [INPUT_ROOT / "labels.jsonl", *INPUT_ROOT.rglob("labels.jsonl"), Path("grading_kit/labels.jsonl")]:
        if p.is_file():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        item = json.loads(line)
                        pages.append({"doc_id": "pierce-1890", "page_id": item["page_id"], "text": item["text"]})
                if pages:
                    print(f"Loaded {len(pages)} pages from {p}")
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
def build_baseline_chunks(pages: list[dict[str, str]], chunk_size: int = 256, overlap: int = 32) -> list[dict[str, Any]]:
    chunks = []
    step = max(1, chunk_size - overlap)
    for page in pages:
        p_id = page.get("page_id", "p0001")
        doc_id = page.get("doc_id", "pierce-1890")
        words = page.get("text", "").split()
        if not words:
            continue
        for i in range(0, len(words), step):
            win = words[i : i + chunk_size]
            chunks.append({
                "chunk_id": f"{doc_id}_{p_id}_c{len(chunks):04d}",
                "doc_id": doc_id,
                "page_id": p_id,
                "text": " ".join(win),
                "token_count": len(win),
            })
            if i + chunk_size >= len(words):
                break
    return chunks

# %%
def evaluate_model(model_path: str, chunks: list[dict[str, Any]], tasks: list[dict[str, Any]], device: str = "cpu"):
    from sentence_transformers import SentenceTransformer
    import faiss

    print(f"\n--- Benchmarking: {Path(model_path).name} ({device.upper()}) ---")
    t0 = time.perf_counter()
    model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
    load_time = time.perf_counter() - t0

    texts = [c["text"] for c in chunks]
    # Warmup
    _ = model.encode(texts[:4], batch_size=4, normalize_embeddings=True)

    # Bulk Throughput
    t_start = time.perf_counter()
    embs = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True)
    enc_time = time.perf_counter() - t_start
    embs_np = np.asarray(embs, dtype=np.float32)
    dim = embs_np.shape[1]
    tput = len(texts) / enc_time if enc_time > 0 else 0

    # Single-Query Latency
    queries = ["What is the treatment for catarrh?", "Describe the medicinal virtues of Golden Seal.", "How is the heart constructed?"]
    lats = []
    for q in queries:
        t_q = time.perf_counter()
        _ = model.encode([q], normalize_embeddings=True)
        lats.append((time.perf_counter() - t_q) * 1000)
    p50_lat = float(np.median(lats))

    # Retrieval Accuracy
    index = faiss.IndexFlatIP(dim)
    index.add(embs_np)

    grounded_tasks = [t for t in tasks if t.get("target_pages") or t.get("gold_pages")]
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
    metrics = {
        "model": Path(model_path).name,
        "path": str(model_path),
        "device": device,
        "type": "PyTorch-Dense",
        "dimension": dim,
        "total_chunks": len(texts),
        "load_time_seconds": round(load_time, 3),
        "encode_time_seconds": round(enc_time, 3),
        "single_query_p50_ms": round(p50_lat, 2),
        "chunks_per_second": round(tput, 1),
        "recall@1": round(recalls[1] / n, 4),
        "recall@3": round(recalls[3] / n, 4),
        "recall@5": round(recalls[5] / n, 4),
        "recall@10": round(recalls[10] / n, 4),
        "mrr": round(rr_total / n, 4),
    }
    return model, embs_np, metrics

# %%
def run():
    import torch
    device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
    print(f"Device: {device.upper()}")

    pages = load_source_pages()
    tasks = load_curated_tasks()
    chunks = build_baseline_chunks(pages, 256, 32)
    print(f"Total Chunks: {len(chunks)} | Tasks: {len(tasks)}")

    # Target core models
    target_names = ["all-minilm-l6-v2", "bge-small-en-v1-5", "bge-m3", "ms-marco-minilm-l6-v2"]
    discovered = []
    for t_name in target_names:
        for search_dir in [INPUT_ROOT, Path(".")]:
            matches = list(search_dir.rglob(f"*{t_name}*"))
            dirs = [m for m in matches if m.is_dir() and "reranker" not in m.name.lower()]
            if dirs:
                discovered.append(str(dirs[0]))
                break

    if not discovered:
        discovered = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]

    results = []
    for m_path in discovered:
        try:
            _, _, met = evaluate_model(m_path, chunks, tasks, device=device)
            results.append(met)
        except Exception as e:
            print(f"[ERROR] Failed {m_path}: {e}")

    out_file = OUT_DIR / "results_core_embeddings.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_file}")

    print("\n" + "=" * 105)
    print(f"{ 'Model Name':<26} {'Dims':<6} {'Tput (ch/s)':<14} {'P50 Lat (ms)':<14} {'Recall@1':<10} {'Recall@5':<10} {'MRR':<8}")
    print("-" * 105)
    for m in results:
        print(f"{m['model']:<26} {m['dimension']:<6} {m['chunks_per_second']:<14} {m['single_query_p50_ms']:<14} {m['recall@1']:<10} {m['recall@5']:<10} {m['mrr']:<8}")
    print("=" * 105)

if __name__ == "__main__":
    run()
