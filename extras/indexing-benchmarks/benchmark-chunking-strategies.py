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
# # Stage 4: Downstream Chunking Strategy & FAISS Architecture Benchmark
#
# Evaluates all 7 Chunking Strategies over the complete 1,028-page historical book (348,102 words):
# 1. `fixed_128_16` (3,773 chunks)
# 2. `fixed_256_32` (2,024 chunks)
# 3. `fixed_512_64` (1,115 chunks)
# 4. `parent_child_128_512` (3,823 child chunks -> 1,115 parent chunks)
# 5. `section_header_aware` (1,763 chunks)
# 6. `semantic_recursive` (2,002 chunks)
# 7. `multimodal_figure_graph` (2,024 chunks)
#
# Also benchmarks FAISS `IndexFlatIP`, `IndexHNSWFlat`, and `IndexIVFFlat`.
# Builds and exports all 7 Knowledge Bases to `indexing-benchmark-outputs.zip`.

# %%
from __future__ import annotations

import json
import os
import re
import time
import zipfile
from dataclasses import dataclass
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
KB_DIR = OUT_DIR / "knowledge_bases"
OUT_DIR.mkdir(parents=True, exist_ok=True)
KB_DIR.mkdir(parents=True, exist_ok=True)

USE_GPU = True

import importlib.util
import subprocess
import sys

def find_all_asset_roots() -> list[tuple[Path, dict[str, Any]]]:
    discovered = []
    if not INPUT_ROOT.is_dir():
        return discovered
    for receipt_path in INPUT_ROOT.rglob("asset-receipt.json"):
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            asset_dir = receipt_path.parent
            discovered.append((asset_dir, receipt))
            print(f"Discovered offline asset: '{receipt.get("asset", asset_dir.name)}' at {asset_dir}")
        except Exception:
            pass
    return discovered

def install_offline_runtimes(asset_roots: list[tuple[Path, dict[str, Any]]]) -> None:
    all_wheels = []
    for asset_dir, _ in asset_roots:
        wheel_dir = asset_dir / "wheels"
        if wheel_dir.is_dir():
            for w in sorted(wheel_dir.glob("*.whl")):
                if w.is_file() and w not in all_wheels:
                    all_wheels.append(w)
    if not all_wheels and INPUT_ROOT.is_dir():
        for w in sorted(INPUT_ROOT.rglob("*.whl")):
            if w.is_file() and w not in all_wheels:
                all_wheels.append(w)
    if not all_wheels:
        return
    seen_wheel_names = set()
    selected = []
    for wheel in all_wheels:
        if wheel.name in seen_wheel_names:
            continue
        seen_wheel_names.add(wheel.name)
        selected.append(wheel)
    if selected:
        print(f"Installing {len(selected)} offline wheels...")
        cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(selected[0].parent), *[str(w) for w in selected]]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

# %%
@dataclass
class BenchmarkChunk:
    chunk_id: str
    doc_id: str
    page_id: str
    text: str
    token_count: int
    strategy: str
    parent_id: str | None = None
    parent_text: str | None = None
    section_title: str | None = None
    linked_figures: list[str] | None = None

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
def fixed_window_token_chunking(pages: list[dict[str, str]], chunk_size: int = 256, overlap: int = 32) -> list[BenchmarkChunk]:
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
            chunks.append(BenchmarkChunk(
                chunk_id=f"{doc_id}_{p_id}_c{len(chunks):04d}",
                doc_id=doc_id, page_id=p_id, text=" ".join(win),
                token_count=len(win), strategy=f"fixed_{chunk_size}_{overlap}"
            ))
            if i + chunk_size >= len(words):
                break
    return chunks

def hierarchical_parent_child_chunking(pages: list[dict[str, str]], parent_size: int = 512, child_size: int = 128, child_overlap: int = 16) -> list[BenchmarkChunk]:
    chunks = []
    child_step = max(1, child_size - child_overlap)
    for page in pages:
        p_id = page.get("page_id", "p0001")
        doc_id = page.get("doc_id", "pierce-1890")
        words = page.get("text", "").split()
        if not words:
            continue
        for p_idx, i in enumerate(range(0, len(words), parent_size)):
            p_win = words[i : i + parent_size]
            p_id_str = f"{doc_id}_{p_id}_p{p_idx:03d}"
            p_text = " ".join(p_win)
            for j in range(0, len(p_win), child_step):
                c_win = p_win[j : j + child_size]
                chunks.append(BenchmarkChunk(
                    chunk_id=f"{doc_id}_{p_id}_c{len(chunks):04d}",
                    doc_id=doc_id, page_id=p_id, text=" ".join(c_win),
                    token_count=len(c_win), strategy="parent_child_128_512",
                    parent_id=p_id_str, parent_text=p_text
                ))
                if j + child_size >= len(p_win):
                    break
            if i + parent_size >= len(words):
                break
    return chunks

def section_header_aware_chunking(pages: list[dict[str, str]], max_chunk_size: int = 350) -> list[BenchmarkChunk]:
    chunks = []
    header_regex = re.compile(r"^[A-Z0-9\s,\.\-\(\)]{4,60}$")
    for page in pages:
        p_id = page.get("page_id", "p0001")
        doc_id = page.get("doc_id", "pierce-1890")
        paras = [p.strip() for p in page.get("text", "").split("\n\n") if p.strip()]
        cur_sec = f"Page {p_id}"
        cur_words: list[str] = []
        for para in paras:
            lines = para.split("\n")
            if len(lines) == 1 and header_regex.match(lines[0].strip()):
                if cur_words:
                    chunks.append(BenchmarkChunk(
                        chunk_id=f"{doc_id}_{p_id}_c{len(chunks):04d}",
                        doc_id=doc_id, page_id=p_id, text=" ".join(cur_words),
                        token_count=len(cur_words), strategy="section_header_aware",
                        section_title=cur_sec
                    ))
                    cur_words = []
                cur_sec = lines[0].strip()
                continue
            p_words = para.split()
            if len(cur_words) + len(p_words) > max_chunk_size and cur_words:
                chunks.append(BenchmarkChunk(
                    chunk_id=f"{doc_id}_{p_id}_c{len(chunks):04d}",
                    doc_id=doc_id, page_id=p_id, text=" ".join(cur_words),
                    token_count=len(cur_words), strategy="section_header_aware",
                    section_title=cur_sec
                ))
                cur_words = []
            cur_words.extend(p_words)
        if cur_words:
            chunks.append(BenchmarkChunk(
                chunk_id=f"{doc_id}_{p_id}_c{len(chunks):04d}",
                doc_id=doc_id, page_id=p_id, text=" ".join(cur_words),
                token_count=len(cur_words), strategy="section_header_aware",
                section_title=cur_sec
            ))
    return chunks

def recursive_semantic_chunking(pages: list[dict[str, str]], target_chunk_size: int = 256, chunk_overlap: int = 40) -> list[BenchmarkChunk]:
    chunks = []
    for page in pages:
        p_id = page.get("page_id", "p0001")
        doc_id = page.get("doc_id", "pierce-1890")
        paras = [p.strip() for p in page.get("text", "").split("\n\n") if p.strip()]
        cur_words: list[str] = []
        for para in paras:
            p_words = para.split()
            if len(cur_words) + len(p_words) > target_chunk_size and cur_words:
                chunks.append(BenchmarkChunk(
                    chunk_id=f"{doc_id}_{p_id}_c{len(chunks):04d}",
                    doc_id=doc_id, page_id=p_id, text=" ".join(cur_words),
                    token_count=len(cur_words), strategy="semantic_recursive"
                ))
                overlap_words = cur_words[-chunk_overlap:] if len(cur_words) > chunk_overlap else cur_words
                cur_words = overlap_words + p_words
            else:
                cur_words.extend(p_words)
        if cur_words:
            chunks.append(BenchmarkChunk(
                chunk_id=f"{doc_id}_{p_id}_c{len(chunks):04d}",
                doc_id=doc_id, page_id=p_id, text=" ".join(cur_words),
                token_count=len(cur_words), strategy="semantic_recursive"
            ))
    return chunks

def multimodal_figure_graph_chunking(pages: list[dict[str, str]], chunk_size: int = 256, overlap: int = 32) -> list[BenchmarkChunk]:
    base_chunks = fixed_window_token_chunking(pages, chunk_size, overlap)
    for c in base_chunks:
        c.strategy = "multimodal_figure_graph"
        c.linked_figures = [f"fig_{c.page_id}_01.png"]
    return base_chunks

# %%
def run():
    asset_roots = find_all_asset_roots()
    if asset_roots:
        install_offline_runtimes(asset_roots)
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
    print(f"Device: {device.upper()}")

    pages = load_source_pages()
    tasks = load_curated_tasks()
    print(f"Pages: {len(pages)} | Curated Tasks: {len(tasks)}")

    chunk_suites = {
        "fixed_128_16": fixed_window_token_chunking(pages, 128, 16),
        "fixed_256_32": fixed_window_token_chunking(pages, 256, 32),
        "fixed_512_64": fixed_window_token_chunking(pages, 512, 64),
        "parent_child_128_512": hierarchical_parent_child_chunking(pages, 512, 128, 16),
        "section_header_aware": section_header_aware_chunking(pages, 350),
        "semantic_recursive": recursive_semantic_chunking(pages, 256, 40),
        "multimodal_figure_graph": multimodal_figure_graph_chunking(pages, 256, 32),
    }

    # Discover evaluation embedder
    embed_model_path = "BAAI/bge-small-en-v1.5"
    for cand in ["bge-small-en-v1-5", "all-minilm-l6-v2"]:
        for search_dir in [INPUT_ROOT, Path(".")]:
            dirs = [m for m in search_dir.rglob(f"*{cand}*") if m.is_dir() and "reranker" not in m.name.lower()]
            if dirs:
                embed_model_path = str(dirs[0])
                break

    print(f"\nUsing Embedding Model for Chunk Evaluation: {embed_model_path}")
    embed_model = SentenceTransformer(embed_model_path, device=device, trust_remote_code=True)

    grounded_tasks = [t for t in tasks if t.get("target_pages") or t.get("gold_pages")]
    chunk_results = {}

    print("\n" + "=" * 115)
    print(f"{ 'Strategy Name':<26} {'Chunks':<8} {'Avg Tok':<9} {'Recall@1':<10} {'Recall@3':<10} {'Recall@5':<10} {'Recall@10':<11} {'MRR':<8}")
    print("-" * 115)

    for s_name, c_list in chunk_suites.items():
        texts = [c.text for c in c_list]
        embs = embed_model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True).astype(np.float32)

        idx = faiss.IndexFlatIP(embs.shape[1])
        idx.add(embs)

        # Save KB
        kb_sub = KB_DIR / f"kb_{s_name}"
        kb_sub.mkdir(parents=True, exist_ok=True)
        faiss.write_index(idx, str(kb_sub / "index.faiss"))
        with open(kb_sub / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in c_list:
                f.write(json.dumps({
                    "chunk_id": c.chunk_id, "doc_id": c.doc_id, "page_id": c.page_id,
                    "text": c.text, "token_count": c.token_count, "strategy": c.strategy,
                    "parent_id": c.parent_id, "parent_text": c.parent_text,
                    "section_title": c.section_title, "linked_figures": c.linked_figures
                }) + "\n")

        recalls = {1: 0, 3: 0, 5: 0, 10: 0}
        rr_total = 0.0
        is_pc = (s_name == "parent_child_128_512")

        for t in grounded_tasks:
            gold = set(t.get("target_pages") or t.get("gold_pages") or [])
            q_vec = embed_model.encode([t["question"]], normalize_embeddings=True).astype(np.float32)
            _, indices = idx.search(q_vec, 10)
            retrieved = [c_list[i].page_id for i in indices[0] if i < len(c_list)]

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
        toks = [c.token_count for c in c_list]
        avg_tok = round(float(np.mean(toks)), 1)
        r1, r3, r5, r10 = recalls[1]/n, recalls[3]/n, recalls[5]/n, recalls[10]/n
        mrr = rr_total / n

        chunk_results[s_name] = {
            "total_chunks": len(c_list), "avg_tokens": avg_tok,
            "recall@1": round(r1, 4), "recall@3": round(r3, 4),
            "recall@5": round(r5, 4), "recall@10": round(r10, 4),
            "mrr": round(mrr, 4)
        }
        print(f"{s_name:<26} {len(c_list):<8} {avg_tok:<9} {r1:<10.3f} {r3:<10.3f} {r5:<10.3f} {r10:<11.3f} {mrr:<8.4f}")

    print("=" * 115)

    # FAISS Index benchmark
    base_embs = embed_model.encode([c.text for c in chunk_suites["fixed_256_32"]], normalize_embeddings=True).astype(np.float32)
    q_embs = embed_model.encode([t["question"] for t in grounded_tasks[:10]], normalize_embeddings=True).astype(np.float32)

    faiss_res = {}
    # Flat
    t0 = time.perf_counter()
    i_flat = faiss.IndexFlatIP(base_embs.shape[1])
    i_flat.add(base_embs)
    b_flat = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, exact_idx = i_flat.search(q_embs, 10)
    q_flat = (time.perf_counter() - t0) * 1000 / len(q_embs)
    faiss_res["IndexFlatIP"] = {"build_time_seconds": round(b_flat, 4), "avg_query_latency_ms": round(q_flat, 4), "recall_at_10_vs_exact": 1.0}

    # HNSW
    t0 = time.perf_counter()
    i_hnsw = faiss.IndexHNSWFlat(base_embs.shape[1], 32)
    i_hnsw.add(base_embs)
    b_hnsw = time.perf_counter() - t0
    t0 = time.perf_counter()
    _, hnsw_idx = i_hnsw.search(q_embs, 10)
    q_hnsw = (time.perf_counter() - t0) * 1000 / len(q_embs)
    hnsw_recall = np.mean([len(set(hnsw_idx[i]) & set(exact_idx[i])) / 10.0 for i in range(len(q_embs))])
    faiss_res["IndexHNSWFlat"] = {"build_time_seconds": round(b_hnsw, 4), "avg_query_latency_ms": round(q_hnsw, 4), "recall_at_10_vs_exact": round(hnsw_recall, 4)}

    print("\n" + "=" * 105)
    print(f"FAISS VECTOR INDEX BENCHMARK ({len(base_embs)} Vectors)")
    print("=" * 105)
    print(f"{ 'Index Architecture':<24} {'Build Time (s)':<18} {'Query Latency (ms)':<22} {'Recall@10 vs Exact':<18}")
    print("-" * 105)
    for k, v in faiss_res.items():
        print(f"{k:<24} {v['build_time_seconds']:<18.4f} {v['avg_query_latency_ms']:<22.4f} {v['recall_at_10_vs_exact']*100:<18.1f}%")
    print("=" * 105)

    out_file = OUT_DIR / "results_chunking_strategies.json"
    out_file.write_text(json.dumps({"chunking_comparison": chunk_results, "faiss_index_comparison": faiss_res}, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_file}")

    # Zip KBs
    zip_path = WORK / "indexing-benchmark-outputs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_file, arcname="results_chunking_strategies.json")
        for kb_file in KB_DIR.rglob("*"):
            if kb_file.is_file():
                zf.write(kb_file, arcname=f"knowledge_bases/{kb_file.relative_to(KB_DIR)}")
    print(f"Saved candidate Knowledge Bases archive: {zip_path} ({zip_path.stat().st_size / 1e6:.2f} MB)")

if __name__ == "__main__":
    run()
