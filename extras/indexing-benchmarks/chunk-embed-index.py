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
# # Stage 4 Comprehensive Indexing Benchmark
#
# Benchmarks all three core dimensions of the Stage 4 Knowledge Base pipeline:
# 1. **Chunking Strategies**: Fixed-size token window (128/16, 256/32, 512/64) vs.
#    Recursive Markdown/Semantic character splitting.
# 2. **Embedding Models**: Lightweight MiniLM-384 vs. BGE-small-384 vs. BGE-base-768 vs.
#    GTE-Qwen2-1.5B (1536-d) vs. MPNet-base-768.
# 3. **Vector Index Architectures**: `IndexFlatIP` (exact) vs. `IndexHNSWFlat` (graph ANN)
#    vs. `IndexIVFFlat` (inverted cluster) vs. `IndexIVFPQ` (product quantized).
#
# Runs 100% offline when attached to `embedding-indexing-offline-assets` or online via Hugging Face.

# %%
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# %%
INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUT_DIR = WORK / "indexing-benchmark-outputs"
ASSET_NAME = "embedding-indexing-offline-assets"

# Historical medical domain test queries for Pierce 1890
BENCHMARK_QUERIES = [
    {
        "query": "What are the symptoms and treatment of catarrh in the respiratory organs?",
        "target_keywords": ["catarrh", "mucous", "membrane", "respiratory", "inhalation"],
    },
    {
        "query": "How is Golden Seal or Hydrastis Canadensis prepared and what are its medical properties?",
        "target_keywords": ["golden seal", "hydrastis", "tonic", "tincture", "powder"],
    },
    {
        "query": "What ingredients and preparation are recommended for cough syrup or pulmonary balsam?",
        "target_keywords": ["cough", "syrup", "balsam", "pulmonary", "expectorant", "dose"],
    },
    {
        "query": "Describe the anatomy of the heart and the circulation of blood through arteries and veins.",
        "target_keywords": ["heart", "circulation", "artery", "vein", "ventricle", "blood"],
    },
    {
        "query": "What are the causes, stages, and remedies for typhoid fever?",
        "target_keywords": ["typhoid", "fever", "temperature", "pulse", "delirium", "diet"],
    },
]


# %%
def install_offline_runtime(asset_root: Path) -> None:
    wheel_dir = asset_root / "wheels"
    if not wheel_dir.is_dir():
        print(f"No wheels directory under {asset_root}, skipping offline pip install.")
        return
    wheels = sorted(
        path for path in wheel_dir.iterdir() if path.is_file() and path.name.endswith(".whl")
    )
    if not wheels:
        return

    # Skip preinstalled binary packages to prevent C-extension ABI conflicts
    has_pil = importlib.util.find_spec("PIL") is not None
    has_numpy = importlib.util.find_spec("numpy") is not None
    has_cv2 = importlib.util.find_spec("cv2") is not None

    selected = []
    for wheel in wheels:
        norm = wheel.name.lower().replace("_", "-")
        if has_pil and norm.startswith("pillow-"):
            continue
        if has_numpy and norm.startswith("numpy-"):
            continue
        if has_cv2 and (norm.startswith("opencv-") or norm.startswith("opencv_python-")):
            continue
        selected.append(wheel)

    if selected:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-index",
                "--no-deps",
                "--upgrade",
                *map(str, selected),
            ],
            check=True,
        )
    importlib.invalidate_caches()


def find_asset_root() -> tuple[Path | None, dict[str, Any] | None]:
    if not INPUT_ROOT.is_dir():
        return None, None
    for candidate in INPUT_ROOT.iterdir():
        if not candidate.is_dir():
            continue
        receipt_path = candidate / "asset-receipt.json"
        if receipt_path.is_file():
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt.get("asset") == ASSET_NAME:
                    return candidate, receipt
            except Exception:
                pass
        # Check subdirectories
        for sub in candidate.iterdir():
            if sub.is_dir() and (sub / "asset-receipt.json").is_file():
                try:
                    receipt = json.loads((sub / "asset-receipt.json").read_text(encoding="utf-8"))
                    if receipt.get("asset") == ASSET_NAME:
                        return sub, receipt
                except Exception:
                    pass
    return None, None


# %% [markdown]
# ### 1. Chunking Strategy Implementations


# %%
@dataclass
class BenchmarkChunk:
    chunk_id: str
    doc_id: str
    page_id: str
    text: str
    token_count: int
    strategy: str


def fixed_window_token_chunking(
    pages: list[dict[str, str]],
    chunk_size: int = 256,
    overlap: int = 32,
) -> list[BenchmarkChunk]:
    chunks: list[BenchmarkChunk] = []
    step = max(1, chunk_size - overlap)

    for page in pages:
        page_id = page.get("page_id", "p0001")
        doc_id = page.get("doc_id", "pierce-1890")
        text = page.get("text", "").strip()
        words = text.split()
        if not words:
            continue

        for i in range(0, len(words), step):
            window = words[i : i + chunk_size]
            chunk_text = " ".join(window)
            chunk_id = f"{doc_id}_{page_id}_c{len(chunks):04d}"
            chunks.append(
                BenchmarkChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page_id=page_id,
                    text=chunk_text,
                    token_count=len(window),
                    strategy=f"fixed_{chunk_size}_{overlap}",
                )
            )
            if i + chunk_size >= len(words):
                break

    return chunks


def recursive_semantic_chunking(
    pages: list[dict[str, str]],
    target_chunk_size: int = 256,
    min_chunk_size: int = 40,
) -> list[BenchmarkChunk]:
    chunks: list[BenchmarkChunk] = []

    for page in pages:
        page_id = page.get("page_id", "p0001")
        doc_id = page.get("doc_id", "pierce-1890")
        text = page.get("text", "").strip()
        if not text:
            continue

        # Split on markdown headers or paragraph breaks
        paragraphs = re.split(r"\n\s*\n|(?=^#{1,3}\s)", text, flags=re.MULTILINE)
        current_words: list[str] = []

        for p in paragraphs:
            p_words = p.strip().split()
            if not p_words:
                continue

            if len(current_words) + len(p_words) <= target_chunk_size:
                current_words.extend(p_words)
            else:
                if current_words:
                    chunk_text = " ".join(current_words)
                    chunks.append(
                        BenchmarkChunk(
                            chunk_id=f"{doc_id}_{page_id}_c{len(chunks):04d}",
                            doc_id=doc_id,
                            page_id=page_id,
                            text=chunk_text,
                            token_count=len(current_words),
                            strategy="semantic_recursive",
                        )
                    )
                    current_words = []

                if len(p_words) > target_chunk_size:
                    # Sub-split long paragraph by sentences
                    sentences = re.split(r"(?<=[.?!])\s+", " ".join(p_words))
                    for s in sentences:
                        s_words = s.strip().split()
                        if not s_words:
                            continue
                        if len(current_words) + len(s_words) <= target_chunk_size:
                            current_words.extend(s_words)
                        else:
                            if current_words:
                                chunks.append(
                                    BenchmarkChunk(
                                        chunk_id=f"{doc_id}_{page_id}_c{len(chunks):04d}",
                                        doc_id=doc_id,
                                        page_id=page_id,
                                        text=" ".join(current_words),
                                        token_count=len(current_words),
                                        strategy="semantic_recursive",
                                    )
                                )
                            current_words = list(s_words)
                else:
                    current_words = list(p_words)

        if current_words and len(current_words) >= min_chunk_size:
            chunks.append(
                BenchmarkChunk(
                    chunk_id=f"{doc_id}_{page_id}_c{len(chunks):04d}",
                    doc_id=doc_id,
                    page_id=page_id,
                    text=" ".join(current_words),
                    token_count=len(current_words),
                    strategy="semantic_recursive",
                )
            )

    return chunks


# %% [markdown]
# ### 2. Embedding Benchmark Function


# %%
def benchmark_embedding_model(
    model_name_or_path: str,
    chunks: list[BenchmarkChunk],
    device: str = "cpu",
    batch_size: int = 32,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_name_or_path} on {device}...", flush=True)
    start_load = time.perf_counter()
    try:
        model = SentenceTransformer(model_name_or_path, device=device, trust_remote_code=True)
    except Exception:
        model = SentenceTransformer(model_name_or_path, device=device)
    load_time = time.perf_counter() - start_load

    texts = [c.text for c in chunks]
    start_encode = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    encode_time = time.perf_counter() - start_encode

    embeddings = embeddings.astype(np.float32)
    dim = int(embeddings.shape[1])
    throughput = len(chunks) / max(1e-5, encode_time)

    metrics = {
        "model": model_name_or_path,
        "dimension": dim,
        "chunk_count": len(chunks),
        "load_time_sec": round(load_time, 3),
        "encode_time_sec": round(encode_time, 3),
        "throughput_chunks_per_sec": round(throughput, 2),
        "embedding_matrix_bytes": int(embeddings.nbytes),
    }
    return embeddings, metrics


# %% [markdown]
# ### 3. FAISS Vector Index Architecture Benchmark


# %%
def benchmark_faiss_indexes(
    embeddings: np.ndarray,
    queries_embeddings: np.ndarray,
    dim: int,
    k: int = 10,
) -> dict[str, dict[str, Any]]:
    import faiss

    results: dict[str, dict[str, Any]] = {}
    n_vectors = embeddings.shape[0]

    # --- A. FlatIP (Exact Baseline) ---
    start_b = time.perf_counter()
    index_flat = faiss.IndexFlatIP(dim)
    index_flat.add(embeddings)
    build_time_flat = time.perf_counter() - start_b

    start_q = time.perf_counter()
    scores_flat, ids_flat = index_flat.search(queries_embeddings, k)
    query_time_flat = (time.perf_counter() - start_q) * 1000 / len(queries_embeddings)

    results["IndexFlatIP"] = {
        "build_time_ms": round(build_time_flat * 1000, 2),
        "avg_query_latency_ms": round(query_time_flat, 4),
        "approx_recall_vs_flat": 1.0,
        "is_exact": True,
        "index_type": "faiss:flat_ip",
    }

    # --- B. HNSW (Hierarchical Navigable Small World) ---
    for m_val in (16, 32):
        start_b = time.perf_counter()
        index_hnsw = faiss.IndexHNSWFlat(dim, m_val, faiss.METRIC_INNER_PRODUCT)
        index_hnsw.hnsw.efSearch = 64
        index_hnsw.add(embeddings)
        build_time_hnsw = time.perf_counter() - start_b

        start_q = time.perf_counter()
        scores_hnsw, ids_hnsw = index_hnsw.search(queries_embeddings, k)
        query_time_hnsw = (time.perf_counter() - start_q) * 1000 / len(queries_embeddings)

        # Measure 1-Recall@k against exact FlatIP
        overlap_counts = [
            len(set(ids_flat[i]).intersection(set(ids_hnsw[i]))) for i in range(len(ids_flat))
        ]
        recall_at_k = float(np.mean(overlap_counts) / k)

        results[f"IndexHNSWFlat_M{m_val}"] = {
            "build_time_ms": round(build_time_hnsw * 1000, 2),
            "avg_query_latency_ms": round(query_time_hnsw, 4),
            "approx_recall_vs_flat": round(recall_at_k, 4),
            "is_exact": False,
            "index_type": f"faiss:hnsw_M{m_val}",
        }

    # --- C. IVFFlat (Inverted File) ---
    nlist = max(4, min(32, int(np.sqrt(n_vectors))))
    quantizer = faiss.IndexFlatIP(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    start_b = time.perf_counter()
    index_ivf.train(embeddings)
    index_ivf.add(embeddings)
    index_ivf.nprobe = 8
    build_time_ivf = time.perf_counter() - start_b

    start_q = time.perf_counter()
    scores_ivf, ids_ivf = index_ivf.search(queries_embeddings, k)
    query_time_ivf = (time.perf_counter() - start_q) * 1000 / len(queries_embeddings)
    overlap_counts_ivf = [
        len(set(ids_flat[i]).intersection(set(ids_ivf[i]))) for i in range(len(ids_flat))
    ]
    recall_ivf = float(np.mean(overlap_counts_ivf) / k)

    results[f"IndexIVFFlat_nlist{nlist}"] = {
        "build_time_ms": round(build_time_ivf * 1000, 2),
        "avg_query_latency_ms": round(query_time_ivf, 4),
        "approx_recall_vs_flat": round(recall_ivf, 4),
        "is_exact": False,
        "index_type": f"faiss:ivf_flat_nlist{nlist}",
    }

    # --- D. IVFPQ (Product Quantization Byte Compression) ---
    if dim % 8 == 0 and n_vectors >= 256:
        m_pq = dim // 8
        index_pq = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, 8)
        start_b = time.perf_counter()
        index_pq.train(embeddings)
        index_pq.add(embeddings)
        index_pq.nprobe = 8
        build_time_pq = time.perf_counter() - start_b

        start_q = time.perf_counter()
        scores_pq, ids_pq = index_pq.search(queries_embeddings, k)
        query_time_pq = (time.perf_counter() - start_q) * 1000 / len(queries_embeddings)
        overlap_pq = [
            len(set(ids_flat[i]).intersection(set(ids_pq[i]))) for i in range(len(ids_flat))
        ]
        recall_pq = float(np.mean(overlap_pq) / k)

        results[f"IndexIVFPQ_M{m_pq}"] = {
            "build_time_ms": round(build_time_pq * 1000, 2),
            "avg_query_latency_ms": round(query_time_pq, 4),
            "approx_recall_vs_flat": round(recall_pq, 4),
            "is_exact": False,
            "index_type": f"faiss:ivfpq_M{m_pq}",
        }

    return results


# %% [markdown]
# ### 4. End-to-End Benchmark Execution


# %%
def run_benchmark() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    asset_root, receipt = find_asset_root()

    if asset_root:
        print(f"Discovered offline asset: {asset_root}", flush=True)
        install_offline_runtime(asset_root)
    else:
        print("Running in online mode or using local environment packages.", flush=True)

    # 1. Load mock or real pages
    # Generate representative historical medical pages for testing if raw OCR not present
    pages: list[dict[str, str]] = []
    chandra_md = Path("data/raw/chandra_pages.md")
    if chandra_md.is_file():
        content = chandra_md.read_text(encoding="utf-8")
        raw_pages = content.split("<!-- PAGE ")
        for p in raw_pages[1:]:
            p_id, text = p.split(" -->", 1)
            pages.append({"doc_id": "pierce-1890", "page_id": f"p{int(p_id):04d}", "text": text})
    else:
        # 100 synthetic medical pages mimicking Pierce 1890 structure
        sample_topics = [
            "Catarrh and affections of the air-passages require soothing expectorants and proper hygienic care.",
            "Golden Seal (Hydrastis Canadensis) is a most valuable native remedy, possessing bitter tonic and alterative virtues.",
            "Typhoid or enteric fever is marked by sustained high temperature, great prostration, and abdominal tenderness.",
            "The heart consists of four distinct chambers: right and left auricles, and right and left ventricles.",
            "To prepare the Golden Medical Discovery: combine active extractives with pure vegetable glycerine.",
        ]
        for i in range(1, 101):
            text = f"Chapter {(i % 12) + 1}. Medical Advice for Family Use. " + " ".join(
                sample_topics * 4
            )
            pages.append({"doc_id": "pierce-1890", "page_id": f"p{i:04d}", "text": text})

    print(f"Loaded {len(pages)} source pages for chunking benchmarks.", flush=True)

    # 2. Benchmark Chunking Strategies
    chunk_suites = {
        "fixed_128_16": fixed_window_token_chunking(pages, 128, 16),
        "fixed_256_32": fixed_window_token_chunking(pages, 256, 32),
        "fixed_512_64": fixed_window_token_chunking(pages, 512, 64),
        "semantic_recursive": recursive_semantic_chunking(pages, 256, 40),
    }

    chunk_summary = {}
    for name, c_list in chunk_suites.items():
        lengths = [c.token_count for c in c_list]
        chunk_summary[name] = {
            "total_chunks": len(c_list),
            "avg_tokens": round(float(np.mean(lengths)), 1) if lengths else 0,
            "min_tokens": int(np.min(lengths)) if lengths else 0,
            "max_tokens": int(np.max(lengths)) if lengths else 0,
        }
    print("\n=== Chunking Strategies Comparison ===")
    print(json.dumps(chunk_summary, indent=2))

    # 3. Benchmark Embedding Models on 256/32 Baseline Chunks
    eval_chunks = chunk_suites["fixed_256_32"]
    candidate_models = ["sentence-transformers/all-MiniLM-L6-v2"]

    if asset_root:
        models_dir = asset_root / "models"
        if models_dir.is_dir():
            candidate_models = [str(p) for p in models_dir.iterdir() if p.is_dir()]

    embed_summary = []
    embeddings_store = {}

    for model_path in candidate_models:
        try:
            emb, met = benchmark_embedding_model(
                model_path, eval_chunks, device="cpu", batch_size=32
            )
            embed_summary.append(met)
            embeddings_store[met["model"]] = emb
        except Exception as e:
            print(f"Could not benchmark {model_path}: {e}")

    print("\n=== Embedding Models Comparison ===")
    print(json.dumps(embed_summary, indent=2))

    # 4. Benchmark Vector Indexes
    first_model = list(embeddings_store.keys())[0]
    base_embeddings = embeddings_store[first_model]
    from sentence_transformers import SentenceTransformer

    try:
        st_model = SentenceTransformer(first_model, device="cpu", trust_remote_code=True)
    except Exception:
        st_model = SentenceTransformer(first_model, device="cpu")
    query_texts = [q["query"] for q in BENCHMARK_QUERIES]
    query_vectors = st_model.encode(query_texts, normalize_embeddings=True).astype(np.float32)

    faiss_summary = benchmark_faiss_indexes(
        base_embeddings,
        query_vectors,
        dim=base_embeddings.shape[1],
        k=10,
    )
    print("\n=== Vector Index Architectures Comparison ===")
    print(json.dumps(faiss_summary, indent=2))

    # 5. Save Artifacts
    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_pages": len(pages),
        "chunking_comparison": chunk_summary,
        "embedding_models_comparison": embed_summary,
        "faiss_index_comparison": faiss_summary,
    }

    report_path = OUT_DIR / "indexing_comparison_results.json"
    report_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\nReport written to: {report_path}")

    # Create zip archive
    zip_path = WORK / "indexing-benchmark-outputs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(report_path, arcname="indexing_comparison_results.json")
    print(f"Created archive: {zip_path}")


if __name__ == "__main__":
    run_benchmark()
