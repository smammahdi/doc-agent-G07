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
# # Stage 4 Offline Indexing Benchmark & Interactive Query Playground
#
# Benchmarks all core dimensions of the Stage 4 Knowledge Base pipeline:
# 1. **Chunking Strategies**: Fixed-size token window (128/16, 256/32, 512/64) vs.
#    Recursive Markdown/Semantic character splitting.
# 2. **Embedding Models**: Lightweight MiniLM-384 vs. BGE-small-384 vs. Nomic-768 vs.
#    Qwen3-0.6B (1024-d) vs. BGE-M3 (1024-d).
# 3. **Retrieval Quality Evaluation**: Evaluates Recall@1, Recall@3, Recall@5, Recall@10, and MRR
#    against the 25 verified gold tasks in `grading_kit/tasks.jsonl`.
# 4. **Vector Index Architectures**: `IndexFlatIP` (exact) vs. `IndexHNSWFlat` (graph ANN)
#    vs. `IndexIVFFlat` (inverted cluster) vs. `IndexIVFPQ` (product quantized).
# 5. **Interactive Retrieval Playground**: Query custom questions and inspect retrieved passages with scores and page citations.
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

# Runtime hardware selection:
# Set USE_GPU = False to benchmark in realistic CPU-only mode.
# Set USE_GPU = True to use CUDA acceleration if available.
USE_GPU = True
BATCH_SIZE = 32


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
        print(f"Installing {len(selected)} offline wheels...", flush=True)
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
                if receipt.get("asset") in [
                    ASSET_NAME,
                    "text-embeddings-offline-assets",
                    "core-embeddings-offline-assets",
                    "qwen3-embedding-reranker-assets",
                ]:
                    return candidate, receipt
            except Exception:
                pass
        # Check subdirectories
        for sub in candidate.iterdir():
            if sub.is_dir() and (sub / "asset-receipt.json").is_file():
                try:
                    receipt = json.loads((sub / "asset-receipt.json").read_text(encoding="utf-8"))
                    if receipt.get("asset") in [
                        ASSET_NAME,
                        "text-embeddings-offline-assets",
                        "core-embeddings-offline-assets",
                        "qwen3-embedding-reranker-assets",
                    ]:
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
    parent_id: str | None = None
    parent_text: str | None = None
    section_title: str | None = None
    linked_figures: list[str] | None = None


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


def hierarchical_parent_child_chunking(
    pages: list[dict[str, str]],
    parent_size: int = 512,
    child_size: int = 128,
    child_overlap: int = 16,
) -> list[BenchmarkChunk]:
    """Hierarchical Small-to-Big chunking: Embeds 128-token child for precision, retrieves 512-token parent for context."""
    chunks: list[BenchmarkChunk] = []
    child_step = max(1, child_size - child_overlap)

    for page in pages:
        page_id = page.get("page_id", "p0001")
        doc_id = page.get("doc_id", "pierce-1890")
        text = page.get("text", "").strip()
        words = text.split()
        if not words:
            continue

        for p_idx, p_start in enumerate(range(0, len(words), parent_size)):
            parent_words = words[p_start : p_start + parent_size]
            parent_text = " ".join(parent_words)
            parent_id = f"{doc_id}_{page_id}_p{p_idx:03d}"

            # Create child chunks within parent window
            for c_start in range(0, len(parent_words), child_step):
                child_words = parent_words[c_start : c_start + child_size]
                child_text = " ".join(child_words)
                child_id = f"{parent_id}_c{len(chunks):04d}"
                chunks.append(
                    BenchmarkChunk(
                        chunk_id=child_id,
                        doc_id=doc_id,
                        page_id=page_id,
                        text=child_text,
                        token_count=len(child_words),
                        strategy="parent_child_128_512",
                        parent_id=parent_id,
                        parent_text=parent_text,
                    )
                )
                if c_start + child_size >= len(parent_words):
                    break

    return chunks


def section_header_aware_chunking(
    pages: list[dict[str, str]],
    max_section_tokens: int = 350,
) -> list[BenchmarkChunk]:
    """Structural chunking: Splits text cleanly along Markdown headers and anatomical titles."""
    chunks: list[BenchmarkChunk] = []

    # Regex detecting chapters, uppercase headings, and markdown headers
    header_pattern = re.compile(
        r"(?=^#{1,4}\s+|^CHAPTER\s+[IVXLCDM\d]+|^[A-Z\s]{4,}\.)", re.MULTILINE
    )

    for page in pages:
        page_id = page.get("page_id", "p0001")
        doc_id = page.get("doc_id", "pierce-1890")
        text = page.get("text", "").strip()
        if not text:
            continue

        sections = header_pattern.split(text)
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue

            words = sec.split()
            first_line = sec.split("\n")[0].strip()[:60]

            if len(words) <= max_section_tokens:
                chunks.append(
                    BenchmarkChunk(
                        chunk_id=f"{doc_id}_{page_id}_c{len(chunks):04d}",
                        doc_id=doc_id,
                        page_id=page_id,
                        text=sec,
                        token_count=len(words),
                        strategy="section_header_aware",
                        section_title=first_line,
                    )
                )
            else:
                # Sub-split long section by sentences
                sub_chunks = fixed_window_token_chunking(
                    [{"page_id": page_id, "doc_id": doc_id, "text": sec}],
                    chunk_size=max_section_tokens,
                    overlap=32,
                )
                for sc in sub_chunks:
                    sc.strategy = "section_header_aware"
                    sc.section_title = first_line
                    chunks.append(sc)

    return chunks


def multimodal_figure_graph_chunking(
    pages: list[dict[str, str]],
    chunk_size: int = 256,
    overlap: int = 32,
) -> list[BenchmarkChunk]:
    """Multimodal Graph Association: Links chunks with detected figure diagrams (Fig. X) and crop references."""
    chunks = fixed_window_token_chunking(pages, chunk_size=chunk_size, overlap=overlap)
    fig_pattern = re.compile(r"Fig\.\s*(\d+)", re.IGNORECASE)

    for c in chunks:
        c.strategy = "multimodal_figure_graph"
        matches = fig_pattern.findall(c.text)
        if matches:
            c.linked_figures = [f"Fig_{m}" for m in set(matches)]
            # Enrich chunk representation with explicit figure metadata tag
            fig_tag = f"[FIGURE_LINKS: {', '.join(c.linked_figures)}] "
            c.text = fig_tag + c.text

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

    return chunks


# %% [markdown]
# ### 2. Embedding Model Benchmarking & Accuracy Evaluation


# %%
class GGUFEmbeddingWrapper:
    """Wrapper for running GGUF quantized embedding models via llama_cpp."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 4):
        from llama_cpp import Llama

        p = Path(model_path)
        if p.is_dir():
            gguf_files = list(p.glob("*.gguf")) + list(p.glob("*.GGUF"))
            if not gguf_files:
                raise FileNotFoundError(f"No .gguf file found in {model_path}")
            gguf_file = str(gguf_files[0])
        else:
            gguf_file = str(p)
        self.llm = Llama(
            model_path=gguf_file,
            embedding=True,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        self.model_name = p.name

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
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


def benchmark_embedding_model(
    model_name_or_path: str,
    chunks: list[BenchmarkChunk],
    device: str = "cpu",
    batch_size: int = 32,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    is_gguf = (
        "gguf" in model_name_or_path.lower() or Path(model_name_or_path).suffix.lower() == ".gguf"
    )

    if is_gguf:
        print(f"\n--- Benchmarking GGUF Model: {model_name_or_path} ({device.upper()}) ---")
        start_load = time.perf_counter()
        model = GGUFEmbeddingWrapper(model_name_or_path)
        load_time_s = time.perf_counter() - start_load
    else:
        print(f"\n--- Benchmarking PyTorch Model: {model_name_or_path} ({device.upper()}) ---")
        start_load = time.perf_counter()
        try:
            model = SentenceTransformer(model_name_or_path, device=device, trust_remote_code=True)
        except Exception:
            model = SentenceTransformer(model_name_or_path, device=device)
        load_time_s = time.perf_counter() - start_load

    texts = [c.text for c in chunks]

    # Warmup
    if len(texts) > 0:
        _ = model.encode(texts[: min(4, len(texts))], batch_size=4, normalize_embeddings=True)

    start_enc = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    enc_time_s = time.perf_counter() - start_enc

    embeddings_np = np.asarray(embeddings, dtype=np.float32)
    dim = embeddings_np.shape[1] if len(embeddings_np.shape) > 1 else 0
    throughput = len(texts) / enc_time_s if enc_time_s > 0 else 0

    metrics = {
        "model": Path(model_name_or_path).name,
        "device": device,
        "type": "GGUF-Q4_K_M" if is_gguf else "PyTorch-Dense",
        "dimension": dim,
        "total_chunks": len(texts),
        "load_time_seconds": round(load_time_s, 3),
        "encode_time_seconds": round(enc_time_s, 3),
        "chunks_per_second": round(throughput, 1),
        "memory_estimate_mb": round((embeddings_np.nbytes) / (1024 * 1024), 2),
    }
    return model, embeddings_np, metrics


def evaluate_retrieval_accuracy(
    model: Any,
    chunks: list[BenchmarkChunk],
    tasks: list[dict[str, Any]],
    k_values: list[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Computes Recall@k and MRR against gold evaluation tasks."""
    import faiss

    if not tasks or not chunks:
        return {}

    eval_tasks = [t for t in tasks if t.get("gold_pages")]
    if not eval_tasks:
        return {}

    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(
        np.float32
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    queries = [t["question"] for t in eval_tasks]
    q_vecs = model.encode(queries, normalize_embeddings=True, show_progress_bar=False).astype(
        np.float32
    )

    max_k = max(k_values)
    scores, indices = index.search(q_vecs, max_k)

    recalls = {f"recall@{k}": 0.0 for k in k_values}
    mrr_sum = 0.0

    for i, t in enumerate(eval_tasks):
        gold_pages = set(t["gold_pages"])
        retrieved_indices = indices[i]
        retrieved_pages = [
            chunks[idx].page_id for idx in retrieved_indices if 0 <= idx < len(chunks)
        ]

        for k in k_values:
            top_k_pages = set(retrieved_pages[:k])
            if gold_pages.intersection(top_k_pages):
                recalls[f"recall@{k}"] += 1.0

        rr = 0.0
        for rank, p_id in enumerate(retrieved_pages, start=1):
            if p_id in gold_pages:
                rr = 1.0 / rank
                break
        mrr_sum += rr

    n = len(eval_tasks)
    accuracy_results = {
        "evaluated_tasks": n,
        "mrr": round(mrr_sum / n, 4) if n > 0 else 0.0,
    }
    for k in k_values:
        accuracy_results[f"recall@{k}"] = round(recalls[f"recall@{k}"] / n, 4) if n > 0 else 0.0

    return accuracy_results


# %% [markdown]
# ### 3. FAISS Vector Index Architecture Benchmarking


# %%
def benchmark_faiss_indexes(
    embeddings: np.ndarray,
    query_vectors: np.ndarray,
    dim: int,
    k: int = 10,
) -> dict[str, Any]:
    import faiss

    results: dict[str, Any] = {}

    # A. IndexFlatIP (Exact Cosine / Inner Product baseline)
    start = time.perf_counter()
    index_flat = faiss.IndexFlatIP(dim)
    index_flat.add(embeddings)
    build_time = time.perf_counter() - start

    start_q = time.perf_counter()
    scores_flat, ids_flat = index_flat.search(query_vectors, k)
    query_time = (time.perf_counter() - start_q) / len(query_vectors)

    results["IndexFlatIP"] = {
        "type": "exact",
        "build_time_seconds": round(build_time, 4),
        "avg_query_latency_ms": round(query_time * 1000, 3),
        "recall_at_10_vs_exact": 1.0,
        "description": "Exact inner-product search (100% recall, baseline)",
    }

    # B. IndexHNSWFlat (Hierarchical Navigable Small World Graph)
    start = time.perf_counter()
    m_links = 32
    index_hnsw = faiss.IndexHNSWFlat(dim, m_links, faiss.METRIC_INNER_PRODUCT)
    index_hnsw.hnsw.efSearch = 64
    index_hnsw.add(embeddings)
    build_time_hnsw = time.perf_counter() - start

    start_q = time.perf_counter()
    scores_hnsw, ids_hnsw = index_hnsw.search(query_vectors, k)
    query_time_hnsw = (time.perf_counter() - start_q) / len(query_vectors)

    recalls_hnsw = []
    for i in range(len(query_vectors)):
        gt_set = set(ids_flat[i])
        hnsw_set = set(ids_hnsw[i])
        recalls_hnsw.append(len(gt_set.intersection(hnsw_set)) / k)

    results["IndexHNSWFlat"] = {
        "type": "graph_ann",
        "build_time_seconds": round(build_time_hnsw, 4),
        "avg_query_latency_ms": round(query_time_hnsw * 1000, 3),
        "recall_at_10_vs_exact": round(float(np.mean(recalls_hnsw)), 3),
        "description": "Graph-based ANN with efSearch=64, M=32",
    }

    # C. IndexIVFFlat (Inverted File Clustering)
    if len(embeddings) >= 64:
        try:
            start = time.perf_counter()
            nlist = min(32, max(4, len(embeddings) // 16))
            quantizer = faiss.IndexFlatIP(dim)
            index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index_ivf.train(embeddings)
            index_ivf.add(embeddings)
            index_ivf.nprobe = min(8, nlist)
            build_time_ivf = time.perf_counter() - start

            start_q = time.perf_counter()
            scores_ivf, ids_ivf = index_ivf.search(query_vectors, k)
            query_time_ivf = (time.perf_counter() - start_q) / len(query_vectors)

            recalls_ivf = []
            for i in range(len(query_vectors)):
                gt_set = set(ids_flat[i])
                ivf_set = set(ids_ivf[i])
                recalls_ivf.append(len(gt_set.intersection(ivf_set)) / k)

            results["IndexIVFFlat"] = {
                "type": "inverted_file",
                "build_time_seconds": round(build_time_ivf, 4),
                "avg_query_latency_ms": round(query_time_ivf * 1000, 3),
                "recall_at_10_vs_exact": round(float(np.mean(recalls_ivf)), 3),
                "description": f"Inverted list clustering (nlist={nlist}, nprobe=8)",
            }
        except Exception as e:
            results["IndexIVFFlat"] = {"error": str(e)}

    return results


# %%
def load_source_pages() -> list[dict[str, str]]:
    """Loads corpus pages from attached Kaggle datasets or local labels/fallback."""
    pages: list[dict[str, str]] = []

    if INPUT_ROOT.is_dir():
        for path in INPUT_ROOT.rglob("labels.jsonl"):
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        item = json.loads(line)
                        pages.append(
                            {
                                "doc_id": "pierce-1890",
                                "page_id": item["page_id"],
                                "text": item["text"],
                            }
                        )
                if pages:
                    print(f"Loaded {len(pages)} pages from {path}")
                    return pages
            except Exception:
                pass

        for path in INPUT_ROOT.rglob("*.md"):
            if "chandra" in path.name.lower() or "page" in path.name.lower():
                try:
                    content = path.read_text(encoding="utf-8")
                    if "<!-- PAGE " in content:
                        raw_pages = content.split("<!-- PAGE ")
                        for p in raw_pages[1:]:
                            p_id, text = p.split(" -->", 1)
                            pages.append(
                                {
                                    "doc_id": "pierce-1890",
                                    "page_id": f"p{int(p_id):04d}",
                                    "text": text.strip(),
                                }
                            )
                        if pages:
                            print(f"Loaded {len(pages)} pages from {path}")
                            return pages
                except Exception:
                    pass

    local_labels = Path("grading_kit/labels.jsonl")
    if local_labels.is_file():
        try:
            for line in local_labels.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    pages.append(
                        {
                            "doc_id": "pierce-1890",
                            "page_id": item["page_id"],
                            "text": item["text"],
                        }
                    )
            if pages:
                print(f"Loaded {len(pages)} pages from local grading_kit/labels.jsonl")
                return pages
        except Exception:
            pass

    print("Using 100 synthetic historical medical test pages for benchmarking.")
    sample_topics = [
        "Catarrh and affections of the air-passages require soothing expectorants and proper hygienic care.",
        "Golden Seal (Hydrastis Canadensis) is a most valuable native remedy, possessing bitter tonic and alterative virtues.",
        "Typhoid or enteric fever is marked by sustained high temperature, great prostration, and abdominal tenderness.",
        "The heart consists of four distinct chambers: right and left auricles, and right and left ventricles.",
        "To prepare the Golden Medical Discovery: combine active extractives with pure vegetable glycerine.",
        "The bones contain more earthy matter than any other part of the human body, being firm and lime-colored.",
        "The stomach is a musculo-membranous sac communicating with the esophagus by the cardiac orifice.",
    ]
    for i in range(1, 101):
        text = f"Chapter {(i % 12) + 1}. Medical Advice for Family Use. " + " ".join(
            sample_topics * 3
        )
        pages.append({"doc_id": "pierce-1890", "page_id": f"p{i:04d}", "text": text})

    return pages


def load_gold_tasks() -> list[dict[str, Any]]:
    """Loads gold evaluation tasks from attached Kaggle datasets or local grading_kit/tasks.jsonl."""
    tasks: list[dict[str, Any]] = []

    if INPUT_ROOT.is_dir():
        for path in INPUT_ROOT.rglob("tasks.jsonl"):
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        tasks.append(json.loads(line))
                if tasks:
                    print(f"Loaded {len(tasks)} gold tasks from {path}")
                    return tasks
            except Exception:
                pass

    local_tasks = Path("grading_kit/tasks.jsonl")
    if local_tasks.is_file():
        try:
            for line in local_tasks.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    tasks.append(json.loads(line))
            if tasks:
                print(f"Loaded {len(tasks)} gold tasks from local grading_kit/tasks.jsonl")
                return tasks
        except Exception:
            pass

    return []


# %%
_INTERACTIVE_CACHE = {
    "chunks": [],
    "model_name": None,
    "model": None,
    "index": None,
    "embeddings": None,
}


def build_interactive_index(
    model_name: str = "qwen3-embedding-0-6b",
    chunk_strategy: str = "fixed_256_32",
    device: str = "cpu",
) -> None:
    """Builds and caches a FAISS index for instant interactive querying."""
    import faiss
    from sentence_transformers import SentenceTransformer

    pages = load_source_pages()
    if chunk_strategy == "semantic_recursive":
        chunks = recursive_semantic_chunking(pages, 256, 40)
    else:
        chunks = fixed_window_token_chunking(pages, 256, 32)

    asset_root, _ = find_asset_root()
    model_path = model_name
    if asset_root:
        candidate_dir = asset_root / "models" / model_name
        if candidate_dir.is_dir():
            model_path = str(candidate_dir)

    print(f"\n[Playground] Loading embedding model: {model_path} on {device.upper()}...")
    try:
        model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
    except Exception:
        model = SentenceTransformer(model_path, device=device)

    print(f"[Playground] Encoding {len(chunks)} chunks...")
    texts = [c.text for c in chunks]
    embeddings = model.encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )
    embeddings_np = np.asarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)

    _INTERACTIVE_CACHE["chunks"] = chunks
    _INTERACTIVE_CACHE["model_name"] = model_name
    _INTERACTIVE_CACHE["model"] = model
    _INTERACTIVE_CACHE["index"] = index
    _INTERACTIVE_CACHE["embeddings"] = embeddings_np
    print(f"[Playground] Index ready with {len(chunks)} passages!\n")


def interactive_search(
    query: str,
    top_k: int = 5,
    model_name: str = "qwen3-embedding-0-6b",
) -> list[dict[str, Any]]:
    """Runs instant semantic retrieval and displays highlighted result cards."""
    if _INTERACTIVE_CACHE["index"] is None or _INTERACTIVE_CACHE["model_name"] != model_name:
        import torch

        dev = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
        build_interactive_index(model_name=model_name, device=dev)

    model = _INTERACTIVE_CACHE["model"]
    index = _INTERACTIVE_CACHE["index"]
    chunks = _INTERACTIVE_CACHE["chunks"]

    q_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_vec, top_k)

    results = []
    print("=" * 80)
    print(f'QUERY: "{query}"')
    print(f"MODEL: {model_name} | TOP-{top_k} PASSAGES")
    print("=" * 80)

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        snippet = chunk.text[:280] + "..." if len(chunk.text) > 280 else chunk.text
        res_item = {
            "rank": rank,
            "score": round(float(score), 4),
            "page_id": chunk.page_id,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
        }
        results.append(res_item)

        print(
            f"\n[Rank #{rank}]  Score: {score:.4f}  |  Page: {chunk.page_id}  |  ID: {chunk.chunk_id}"
        )
        print(f'"{snippet}"')
        print("-" * 80)

    return results


# %%
def run_benchmark() -> None:
    import torch

    device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
    print(f"Execution environment: PyTorch device = {device.upper()} (USE_GPU={USE_GPU})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    asset_root, receipt = find_asset_root()

    if asset_root:
        print(f"Discovered offline asset: {asset_root}", flush=True)
        install_offline_runtime(asset_root)
    else:
        print("Running in online mode or using local environment packages.", flush=True)

    # 1. Load source pages and gold tasks
    pages = load_source_pages()
    tasks = load_gold_tasks()
    print(
        f"Loaded {len(pages)} source pages and {len(tasks)} gold evaluation tasks.",
        flush=True,
    )

    # 2. Benchmark Chunking Strategies
    chunk_suites = {
        "fixed_128_16": fixed_window_token_chunking(pages, 128, 16),
        "fixed_256_32": fixed_window_token_chunking(pages, 256, 32),
        "fixed_512_64": fixed_window_token_chunking(pages, 512, 64),
        "parent_child_128_512": hierarchical_parent_child_chunking(pages, 512, 128, 16),
        "section_header_aware": section_header_aware_chunking(pages, 350),
        "multimodal_figure_graph": multimodal_figure_graph_chunking(pages, 256, 32),
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
            "has_parent_links": any(c.parent_id is not None for c in c_list),
            "has_figure_links": any(c.linked_figures is not None for c in c_list),
        }
    print("\n" + "=" * 90)
    print("CHUNKING STRATEGIES COMPARISON (Structural, Hierarchical & Graph Links)")
    print("=" * 90)
    header_chunk = f"{'Strategy Name':<26} {'Total Chunks':<14} {'Avg Tokens':<12} {'Min/Max Tokens':<16} {'Graph/Parent Links'}"
    print(header_chunk)
    print("-" * 90)
    for s_name, s_met in chunk_summary.items():
        min_max = f"{s_met['min_tokens']}/{s_met['max_tokens']}"
        link_str = (
            "Parent (128->512)"
            if s_met["has_parent_links"]
            else ("Figure Graph" if s_met["has_figure_links"] else "None")
        )
        print(
            f"{s_name:<26} {s_met['total_chunks']:<14} {s_met['avg_tokens']:<12} {min_max:<16} {link_str}"
        )
    print("=" * 90)

    # 3. Benchmark Embedding Models & Accuracy on Baseline Chunks
    eval_chunks = chunk_suites["fixed_256_32"]
    candidate_models = []
    if INPUT_ROOT.is_dir():
        for m_dir in INPUT_ROOT.rglob("models/*"):
            if m_dir.is_dir() and "reranker" not in m_dir.name.lower():
                candidate_models.append(str(m_dir))

    if not candidate_models and asset_root:
        models_dir = asset_root / "models"
        if models_dir.is_dir():
            candidate_models = [
                str(p)
                for p in models_dir.iterdir()
                if p.is_dir() and "reranker" not in p.name.lower()
            ]

    if not candidate_models:
        candidate_models = ["sentence-transformers/all-MiniLM-L6-v2"]

    embed_summary = []
    embeddings_store = {}

    for model_path in candidate_models:
        try:
            model_obj, emb, met = benchmark_embedding_model(
                model_path, eval_chunks, device=device, batch_size=32
            )
            # Evaluate retrieval quality against tasks.jsonl
            if tasks:
                accuracy = evaluate_retrieval_accuracy(model_obj, eval_chunks, tasks)
                met.update(accuracy)

            embed_summary.append(met)
            embeddings_store[met["model"]] = emb
        except Exception as e:
            print(f"Could not benchmark {model_path}: {e}")

    print("\n" + "=" * 110)
    print("FINAL EMBEDDING MODEL LEADERBOARD (Ranked by MRR & Recall@5)")
    print("=" * 110)
    sorted_models = sorted(
        embed_summary,
        key=lambda x: (x.get("mrr", 0), x.get("recall@5", 0), x.get("chunks_per_second", 0)),
        reverse=True,
    )
    header = f"{'Rank':<5} {'Model Name':<28} {'Dims':<6} {'Throughput':<15} {'Encode Time':<12} {'Recall@1':<10} {'Recall@5':<10} {'MRR':<8}"
    print(header)
    print("-" * 110)
    for rank_idx, m in enumerate(sorted_models, start=1):
        m_name = m.get("model", "unknown")[:26]
        m_dim = str(m.get("dimension", "-"))
        m_tput = f"{m.get('chunks_per_second', 0):.1f} ch/s"
        m_time = f"{m.get('encode_time_seconds', 0):.2f}s"
        r1 = f"{m.get('recall@1', 0.0):.3f}" if "recall@1" in m else "N/A"
        r5 = f"{m.get('recall@5', 0.0):.3f}" if "recall@5" in m else "N/A"
        mrr_val = f"{m.get('mrr', 0.0):.4f}" if "mrr" in m else "N/A"
        print(
            f"#{rank_idx:<4} {m_name:<28} {m_dim:<6} {m_tput:<15} {m_time:<12} {r1:<10} {r5:<10} {mrr_val:<8}"
        )
    print("=" * 110)

    # 4. Benchmark Vector Indexes
    if embeddings_store:
        first_model = list(embeddings_store.keys())[0]
        base_embeddings = embeddings_store[first_model]
        from sentence_transformers import SentenceTransformer

        matching_path = next(
            (m for m in candidate_models if Path(m).name == first_model), first_model
        )
        try:
            st_model = SentenceTransformer(matching_path, device=device, trust_remote_code=True)
        except Exception:
            st_model = SentenceTransformer(matching_path, device=device)

        eval_queries = (
            [t["question"] for t in tasks if t.get("gold_pages")]
            if tasks
            else ["What are the symptoms and remedies for catarrh?"]
        )
        query_vectors = st_model.encode(eval_queries[:10], normalize_embeddings=True).astype(
            np.float32
        )

        faiss_summary = benchmark_faiss_indexes(
            base_embeddings,
            query_vectors,
            dim=base_embeddings.shape[1],
            k=10,
        )
        print("\n=== Vector Index Architectures Comparison ===")
        print(json.dumps(faiss_summary, indent=2))
    else:
        faiss_summary = {}

    # 5. Save Artifacts
    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_pages": len(pages),
        "gold_tasks_evaluated": len(tasks),
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


# %%
if __name__ == "__main__":
    run_benchmark()


# %% [markdown]
# # Interactive Vector Retrieval Playground
# Run this cell anytime to ask any custom question and inspect the top retrieved passages, scores, and page citations.

# %%
MY_QUERY = "What are the medicinal preparations and healing properties of Golden Seal?"
CHOSEN_MODEL = (
    "qwen3-embedding-0-6b"  # or "bge-small-en-v1-5", "nomic-embed-text-v1-5", "all-minilm-l6-v2"
)
TOP_K = 5

results = interactive_search(query=MY_QUERY, top_k=TOP_K, model_name=CHOSEN_MODEL)
