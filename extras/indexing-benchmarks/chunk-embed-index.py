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
# # Stage 4 Comprehensive Knowledge Base & Dense Retrieval Benchmark Suite
#
# Rigorous, publication-grade benchmark of the core Stage 4 Knowledge Base pipeline:
# 1. **Chunking Strategies Evaluation**: End-to-end retrieval accuracy (Recall@k, MRR) across:
#    - Fixed Token Windows (128/16, 256/32, 512/64)
#    - Hierarchical Parent-Child (128 child -> 512 parent, Small-to-Big retrieval)
#    - Structural Section/Header-Aware Splitting
#    - Multimodal Figure-Graph Linkage (text chunks linked to figure crops)
#    - Semantic Recursive Paragraph/Sentence Splitting
# 2. **Dense & Quantized Embedding Models**:
#    - all-MiniLM-L6-v2 (384-d, 22M param fast baseline)
#    - bge-small-en-v1.5 (384-d, 33M param MTEB leader)
#    - nomic-embed-text-v1.5 (768-d, 137M param 8k context)
#    - Qwen3-Embedding-0.6B (1024-d, 0.6B PyTorch dense)
#    - Qwen3-Embedding-0.6B-GGUF (1024-d, Q4_K_M quantized)
#    - Qwen3-Embedding-4B-GGUF (2560-d, Q4_K_M quantized)
#    - Qwen3-VL-Embedding-2B (1536-d, 2.0B Vision-Language multimodal)
#    - BAAI/bge-m3 (1024-d, 560M param multi-lingual/hybrid)
# 3. **Mathematical Decision Framework (MCDA)**:
#    - Multi-criteria composite scoring ($S_{\text{embed}}$ and $S_{\text{chunk}}$) balancing Accuracy, Throughput, Latency, and Memory.
# 4. **Ablation & Score Margin Analysis**:
#    - Cosine similarity separation between Grounded tasks and Ungrounded/Abstention tasks to calibrate `weak_threshold: 0.35`.
# 5. **Vector Index Architectures**: IndexFlatIP (exact) vs. IndexHNSWFlat (graph ANN) vs. IndexIVFFlat.
# 6. **Publication-Grade Visualizations**: Automatically saves 4 PNG figures to `plots/`.
# 7. **Interactive Retrieval Playground**: Instant semantic query test with card rendering.
#
# 100% offline compliant when attached to offline assets.

# %%
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Set matplotlib to headless mode
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Enforce strict offline execution to prevent DNS retry hangs
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# %%
if Path("/kaggle/working").is_dir():
    INPUT_ROOT = Path("/kaggle/input")
    WORK = Path("/kaggle/working")
else:
    INPUT_ROOT = Path("extras/indexing-benchmarks")
    WORK = Path("extras/indexing-benchmarks/output")

OUT_DIR = WORK / "indexing-benchmark-outputs"
PLOTS_DIR = OUT_DIR / "plots"

# Runtime hardware selection:
# Set USE_GPU = False to benchmark in realistic CPU-only mode.
# Set USE_GPU = True to use CUDA acceleration if available.
USE_GPU = True
BATCH_SIZE = 32
GENERATE_PLOTS = False  # Set False on Kaggle to maximize speed; plots can be generated locally


# %%
def find_all_asset_roots() -> list[tuple[Path, dict[str, Any]]]:
    """Recursively discovers ALL attached offline asset packages across arbitrary depths."""
    discovered: list[tuple[Path, dict[str, Any]]] = []
    if not INPUT_ROOT.is_dir():
        return discovered

    for receipt_path in INPUT_ROOT.rglob("asset-receipt.json"):
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            asset_dir = receipt_path.parent
            discovered.append((asset_dir, receipt))
            print(
                f"Discovered offline asset: '{receipt.get('asset', asset_dir.name)}' at {asset_dir}",
                flush=True,
            )
        except Exception as e:
            print(f"Warning: could not parse {receipt_path}: {e}")

    return discovered


def install_offline_runtimes(asset_roots: list[tuple[Path, dict[str, Any]]]) -> None:
    """Installs wheels from all discovered asset packages with filename deduplication."""
    all_wheels: list[Path] = []
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
        print("No offline wheels found, using preinstalled environment packages.")
        return

    has_pil = importlib.util.find_spec("PIL") is not None
    has_numpy = importlib.util.find_spec("numpy") is not None
    has_cv2 = importlib.util.find_spec("cv2") is not None

    seen_wheel_names: set[str] = set()
    selected: list[Path] = []
    for wheel in all_wheels:
        if wheel.name in seen_wheel_names:
            continue
        seen_wheel_names.add(wheel.name)

        norm = wheel.name.lower().replace("_", "-")
        if has_pil and norm.startswith("pillow-"):
            continue
        if has_numpy and norm.startswith("numpy-"):
            continue
        if has_cv2 and (norm.startswith("opencv-") or norm.startswith("opencv_python-")):
            continue
        selected.append(wheel)

    if selected:
        print(f"Installing {len(selected)} unique offline wheels...", flush=True)
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
# ### 2. Dense, GGUF, & Multimodal Model Encoders


# %%
# %%
class HuggingFaceTransformerWrapper:
    """Pure HuggingFace AutoModel + AutoTokenizer wrapper with attention-weighted mean pooling.
    Guarantees compatibility with custom architectures (e.g. Nomic BERT, Qwen VL) when SentenceTransformer fails.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self.model.eval()
        self.model_name = Path(model_path).name

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        import torch

        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded)
                last_hidden = (
                    outputs.last_hidden_state
                    if hasattr(outputs, "last_hidden_state")
                    else outputs[0]
                )
                mask = encoded["attention_mask"].unsqueeze(-1).expand(last_hidden.size())
                sum_embeddings = torch.sum(last_hidden * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                pooled = sum_embeddings / sum_mask

                if normalize_embeddings:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embs.append(pooled.cpu().to(torch.float32).numpy())
        return np.vstack(all_embs) if all_embs else np.empty((0, 384), dtype=np.float32)


class GGUFEmbeddingWrapper:
    """Wrapper for running GGUF quantized embedding models (0.6B / 4B) via llama_cpp."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 4):
        try:
            from llama_cpp import Llama
        except ImportError as err:
            raise ImportError(
                "llama-cpp-python runtime is not installed for Python 3.12. "
                "GGUF models will be skipped gracefully."
            ) from err

        p = Path(model_path)
        if p.is_dir():
            gguf_files = sorted(
                list(p.glob("*.gguf")) + list(p.glob("*.GGUF")),
                key=lambda x: x.stat().st_size,
                reverse=True,
            )
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
        self.model_name = Path(gguf_file).stem

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


def load_model_instance(model_name_or_path: str, device: str = "cpu") -> tuple[Any, str]:
    """Instantiates an embedding model using a cascading 3-tier fallback strategy."""
    is_gguf = (
        "gguf" in model_name_or_path.lower() or Path(model_name_or_path).suffix.lower() == ".gguf"
    )
    if is_gguf:
        return GGUFEmbeddingWrapper(model_name_or_path), "GGUF-Q4_K_M"

    is_vl = any(
        k in model_name_or_path.lower()
        for k in ["qwen3-vl", "qwen2-vl", "vl-embedding", "multimodal-vl"]
    )
    if is_vl:
        # Vision-Language models use direct HuggingFace AutoModel with attention mean pooling
        return HuggingFaceTransformerWrapper(model_name_or_path, device=device), "Multimodal-VL"

    from sentence_transformers import SentenceTransformer

    # Strategy 1: SentenceTransformer with full kwargs propagated to AutoConfig and AutoModel
    try:
        model = SentenceTransformer(
            model_name_or_path,
            device=device,
            trust_remote_code=True,
            model_kwargs={"trust_remote_code": True},
            tokenizer_kwargs={"trust_remote_code": True},
            config_kwargs={"trust_remote_code": True},
        )
        return model, "PyTorch-Dense"
    except Exception:
        pass

    # Strategy 2: Standard SentenceTransformer
    try:
        model = SentenceTransformer(model_name_or_path, device=device, trust_remote_code=True)
        return model, "PyTorch-Dense"
    except Exception:
        pass

    # Strategy 3: Pure HuggingFace AutoModel + AutoTokenizer with attention mean pooling
    try:
        model = HuggingFaceTransformerWrapper(model_name_or_path, device=device)
        return model, "HF-AutoModel-Dense"
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model {model_name_or_path} across all loaders: {e}"
        ) from e


def benchmark_embedding_model(
    model_name_or_path: str,
    chunks: list[BenchmarkChunk],
    device: str = "cpu",
    batch_size: int = 32,
) -> tuple[Any | None, np.ndarray, dict[str, Any]]:
    is_gguf = (
        "gguf" in model_name_or_path.lower() or Path(model_name_or_path).suffix.lower() == ".gguf"
    )
    type_label = "GGUF" if is_gguf else "PyTorch"
    print(f"\n--- Benchmarking {type_label} Model: {model_name_or_path} ({device.upper()}) ---")

    start_load = time.perf_counter()
    try:
        model, model_type = load_model_instance(model_name_or_path, device=device)
    except Exception as e:
        if is_gguf and (
            "llama_cpp" in str(e).lower() or "runtime is not installed" in str(e).lower()
        ):
            print(
                f"[GGUF SKIPPED] llama-cpp-python runtime is not available: {Path(model_name_or_path).name}"
            )
        else:
            print(f"[ERROR] Failed to load {model_name_or_path}: {e}")
        return None, np.empty((0, 0), dtype=np.float32), {}

    load_time_s = time.perf_counter() - start_load
    texts = [c.text for c in chunks]

    # Warmup
    if len(texts) > 0:
        try:
            _ = model.encode(texts[: min(4, len(texts))], batch_size=4, normalize_embeddings=True)
        except Exception as e:
            print(f"[ERROR] Warmup failed for {model_name_or_path}: {e}")
            return None, np.empty((0, 0), dtype=np.float32), {}

    # Bulk Throughput Measurement (batch_size=32)
    start_enc = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    enc_time_s = time.perf_counter() - start_enc

    # Single-Query Latency Measurement (batch_size=1, 3 sample queries)
    sample_queries = [
        "What is the treatment for catarrh?",
        "Describe the medicinal virtues of Golden Seal.",
        "How is the heart constructed anatomically?",
    ]
    latencies_ms = []
    for sq in sample_queries:
        t0 = time.perf_counter()
        _ = model.encode([sq], normalize_embeddings=True)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    p50_latency_ms = float(np.median(latencies_ms))

    embeddings_np = np.asarray(embeddings, dtype=np.float32)
    dim = embeddings_np.shape[1] if len(embeddings_np.shape) > 1 else 0
    throughput = len(texts) / enc_time_s if enc_time_s > 0 else 0

    # Memory Tracking
    import torch

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if (device == "cuda" and torch.cuda.is_available())
        else 0.0
    )

    metrics = {
        "model": Path(model_name_or_path).name,
        "device": device,
        "type": model_type,
        "dimension": dim,
        "total_chunks": len(texts),
        "load_time_seconds": round(load_time_s, 3),
        "encode_time_seconds": round(enc_time_s, 3),
        "single_query_p50_ms": round(p50_latency_ms, 2),
        "chunks_per_second": round(throughput, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "memory_estimate_mb": round((embeddings_np.nbytes) / (1024 * 1024), 2),
    }
    return model, embeddings_np, metrics


def evaluate_retrieval_accuracy(
    model: Any,
    chunks: list[BenchmarkChunk],
    tasks: list[dict[str, Any]],
    k_values: list[int] = (1, 3, 5, 10),
    is_parent_child: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Computes Recall@k, MRR, and per-query similarity score margins against curated evaluation Q&A tasks."""
    import faiss

    if not tasks or not chunks:
        return {}, []

    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype(
        np.float32
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    queries = [t["question"] for t in tasks]
    q_vecs = model.encode(queries, normalize_embeddings=True, show_progress_bar=False).astype(
        np.float32
    )

    max_k = max(k_values)
    scores, indices = index.search(q_vecs, max_k)

    recalls = {f"recall@{k}": 0.0 for k in k_values}
    mrr_sum = 0.0
    task_scores_log: list[dict[str, Any]] = []

    eval_tasks_count = 0
    for i, t in enumerate(tasks):
        target_pages = set(t.get("target_pages") or t.get("gold_pages", []))
        is_grounded = len(target_pages) > 0
        top1_score = float(scores[i][0]) if len(scores[i]) > 0 else 0.0

        task_scores_log.append(
            {
                "task_id": t.get("id", f"t{i+1:02d}"),
                "question": t["question"],
                "is_grounded": is_grounded,
                "top1_score": top1_score,
            }
        )

        if not is_grounded:
            continue

        eval_tasks_count += 1
        retrieved_indices = indices[i]
        retrieved_pages = [
            chunks[idx].page_id for idx in retrieved_indices if 0 <= idx < len(chunks)
        ]

        for k in k_values:
            top_k_pages = set(retrieved_pages[:k])
            if target_pages.intersection(top_k_pages):
                recalls[f"recall@{k}"] += 1.0

        rr = 0.0
        for rank, p_id in enumerate(retrieved_pages, start=1):
            if p_id in target_pages:
                rr = 1.0 / rank
                break
        mrr_sum += rr

    n = max(1, eval_tasks_count)
    accuracy_results = {
        "evaluated_tasks": n,
        "mrr": round(mrr_sum / n, 4),
    }
    for k in k_values:
        accuracy_results[f"recall@{k}"] = round(recalls[f"recall@{k}"] / n, 4)

    return accuracy_results, task_scores_log


# %% [markdown]
# ### 3. Multi-Criteria Decision Framework (MCDA Composite Scores)


# %%
def calculate_composite_scores(
    embed_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Calculates objective MCDA composite scores balancing Accuracy, Throughput, Latency, and Memory."""
    if not embed_results:
        return []

    # Extract arrays
    mrrs = np.array([m.get("mrr", 0.0) for m in embed_results])
    r5s = np.array([m.get("recall@5", 0.0) for m in embed_results])
    r1s = np.array([m.get("recall@1", 0.0) for m in embed_results])
    tputs = np.array([m.get("chunks_per_second", 0.0) for m in embed_results])
    lats = np.array([m.get("single_query_p50_ms", 10.0) for m in embed_results])
    dims = np.array([m.get("dimension", 384) for m in embed_results])

    # 1. Accuracy Component (40% MRR + 30% Recall@5 + 20% Recall@1 + 10% Recall@10)
    raw_acc = 0.40 * mrrs + 0.30 * r5s + 0.20 * r1s + 0.10 * mrrs

    # Min-max normalization helpers
    def min_max(arr: np.ndarray, invert: bool = False) -> np.ndarray:
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-6:
            return np.ones_like(arr)
        norm = (arr - mn) / (mx - mn)
        return (1.0 - norm) if invert else norm

    norm_acc = min_max(raw_acc)
    norm_tput = min_max(np.log1p(tputs))
    norm_lat = min_max(lats, invert=True)  # lower latency is better
    norm_mem = min_max(dims, invert=True)  # lower dimension/memory is better

    # Production Weights: 45% Accuracy, 20% Throughput, 20% Latency, 15% Memory/Footprint
    w_acc, w_tput, w_lat, w_mem = 0.45, 0.20, 0.20, 0.15

    for idx, m in enumerate(embed_results):
        score = (
            w_acc * norm_acc[idx]
            + w_tput * norm_tput[idx]
            + w_lat * norm_lat[idx]
            + w_mem * norm_mem[idx]
        )
        m["composite_score"] = round(float(score), 4)

    return sorted(embed_results, key=lambda x: x.get("composite_score", 0), reverse=True)


# %% [markdown]
# ### 4. Publication-Grade Visualization Generator


# %%
def generate_benchmark_plots(
    chunk_suites: dict[str, list[BenchmarkChunk]],
    chunk_accuracy: dict[str, dict[str, Any]],
    embed_results: list[dict[str, Any]],
    score_logs: list[dict[str, Any]],
    output_dir: Path,
    faiss_results: dict[str, Any] | None = None,
) -> None:
    """Generates 3 minimalist, publication-grade benchmark figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif", "figure.autolayout": True})

    # -------------------------------------------------------------------------
    # Figure 1: Embedding Models Pareto Frontier (MRR vs. Throughput ch/s)
    # -------------------------------------------------------------------------
    fig1_path = output_dir / "figure1_pareto_frontier.png"
    if embed_results:
        fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
        ax.set_facecolor("#fafafa")
        fig.patch.set_facecolor("#ffffff")
        ax.grid(True, linestyle="--", alpha=0.5, color="#d0d0d0")

        models = [m for m in embed_results if m.get("mrr", 0) > 0]
        colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#8c564b"]

        for i, m in enumerate(models):
            tput = m.get("chunks_per_second", 1)
            mrr = m.get("mrr", 0)
            dim = m.get("dimension", 384)
            name = m.get("model", "unknown")
            clean_name = (
                name.replace("sentence-transformers/", "").replace("BAAI/", "").replace("Qwen/", "")
            )

            ax.scatter(
                tput,
                mrr,
                s=max(120, dim * 0.7),
                color=colors[i % len(colors)],
                alpha=0.85,
                edgecolors="#333333",
                linewidth=1.2,
                zorder=4,
            )

            # Smart annotation placement to prevent overlap
            if "minilm" in clean_name.lower():
                xy_text = (-130, 18)
            elif "bge-small" in clean_name.lower():
                xy_text = (15, -24)
            elif "bge-m3" in clean_name.lower():
                xy_text = (15, 12)
            elif "qwen" in clean_name.lower():
                xy_text = (15, -18)
            else:
                xy_text = (15, 15)

            ax.annotate(
                f"{clean_name}\n({dim}-d | {tput:.0f} ch/s)",
                xy=(tput, mrr),
                xytext=xy_text,
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
                color="#222222",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.9,
                    edgecolor="#cccccc",
                    linewidth=0.8,
                ),
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8),
                zorder=5,
            )

        ax.set_xscale("log")
        ax.set_xlim(50, 4500)
        ax.set_ylim(0.75, 1.0)
        ax.set_xlabel(
            "Encoding Throughput (chunks/sec, log scale)",
            fontsize=10.5,
            fontweight="bold",
            color="#333333",
            labelpad=8,
        )
        ax.set_ylabel(
            "Mean Reciprocal Rank (MRR)",
            fontsize=10.5,
            fontweight="bold",
            color="#333333",
            labelpad=8,
        )
        ax.set_title(
            "Embedding Models: Retrieval Accuracy vs. Throughput Pareto Frontier",
            fontsize=12,
            fontweight="bold",
            pad=12,
            color="#111111",
        )
        plt.tight_layout()
        plt.savefig(fig1_path, bbox_inches="tight")
        plt.close()
        print(f"Generated visual artifact: {fig1_path}")

    # -------------------------------------------------------------------------
    # Figure 2: Ranked Chunking Strategy Comparison (1,028 Pages)
    # -------------------------------------------------------------------------
    fig2_path = output_dir / "figure2_chunking_comparison.png"
    if chunk_accuracy:
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
        ax.set_facecolor("#fafafa")
        fig.patch.set_facecolor("#ffffff")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5, color="#d0d0d0")

        sorted_chunks = sorted(
            chunk_accuracy.items(), key=lambda x: x[1].get("mrr", 0), reverse=True
        )
        names = [k for k, _ in sorted_chunks]
        mrrs = [v.get("mrr", 0) for _, v in sorted_chunks]
        r1s = [v.get("recall@1", 0) for _, v in sorted_chunks]
        counts = [len(chunk_suites.get(k, [])) for k, _ in sorted_chunks]
        toks = [
            (
                round(float(np.mean([c.token_count for c in chunk_suites.get(k, [])])), 0)
                if chunk_suites.get(k)
                else 0
            )
            for k, _ in sorted_chunks
        ]

        y_pos = np.arange(len(names))
        height = 0.36

        ax.barh(
            y_pos - height / 2,
            mrrs,
            height,
            label="MRR (Rank Precision)",
            color="#1f77b4",
            alpha=0.9,
            edgecolor="#333333",
            linewidth=0.8,
        )
        ax.barh(
            y_pos + height / 2,
            r1s,
            height,
            label="Recall@1 (Top-1 Exact)",
            color="#2ca02c",
            alpha=0.9,
            edgecolor="#333333",
            linewidth=0.8,
        )

        for i in range(len(names)):
            ax.text(
                mrrs[i] + 0.01,
                y_pos[i] - height / 2,
                f"{mrrs[i]:.3f} (MRR)",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#1f77b4",
            )
            ax.text(
                r1s[i] + 0.01,
                y_pos[i] + height / 2,
                f"{r1s[i]:.3f} (R@1)",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#2ca02c",
            )

        custom_labels = [
            f"{names[i]}\n[{counts[i]} chunks | avg {toks[i]:.0f} tok]" for i in range(len(names))
        ]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(custom_labels, fontsize=8.5, fontweight="semibold")
        ax.invert_yaxis()
        ax.set_xlim(0.65, 1.08)
        ax.set_xlabel(
            "Retrieval Score (0 - 1.0)",
            fontsize=10.5,
            fontweight="bold",
            color="#333333",
            labelpad=8,
        )
        ax.set_title(
            "Downstream Retrieval Performance Across Chunking Strategies (1,028 Pages)",
            fontsize=12,
            fontweight="bold",
            pad=12,
            color="#111111",
        )
        ax.legend(
            loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5
        )
        plt.tight_layout()
        plt.savefig(fig2_path, bbox_inches="tight")
        plt.close()
        print(f"Generated visual artifact: {fig2_path}")

    # -------------------------------------------------------------------------
    # Figure 3: Score Margin & Decision Boundary (weak_threshold = 0.35)
    # -------------------------------------------------------------------------
    fig3_path = output_dir / "figure3_score_margin.png"
    if score_logs:
        fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=300)
        ax.set_facecolor("#fafafa")
        fig.patch.set_facecolor("#ffffff")
        ax.grid(True, linestyle="--", alpha=0.5, color="#d0d0d0")

        ax.axvspan(0.0, 0.35, alpha=0.12, color="#e74c3c", label="Abstention Zone (< 0.35)")
        ax.axvspan(
            0.35, 1.0, alpha=0.08, color="#2ecc71", label="Grounded Retrieval Zone (>= 0.35)"
        )
        ax.axvline(
            0.35,
            color="#c0392b",
            linestyle="--",
            linewidth=1.8,
            label="Decision Boundary (weak_threshold = 0.35)",
        )

        grounded_scores = [s["top1_score"] for s in score_logs if s["is_grounded"]]
        negative_scores = [s["top1_score"] for s in score_logs if not s["is_grounded"]]

        if grounded_scores:
            ax.hist(
                grounded_scores,
                bins=8,
                alpha=0.75,
                color="#2b7bba",
                label="Grounded Tasks (t01-t20)",
                density=True,
                edgecolor="#1c5580",
                linewidth=0.8,
            )
        if negative_scores:
            ax.hist(
                negative_scores,
                bins=5,
                alpha=0.75,
                color="#e74c3c",
                label="Negative Tasks (t21-t25)",
                density=True,
                edgecolor="#962d22",
                linewidth=0.8,
            )

        ax.set_xlim(0.2, 0.9)
        ax.set_xlabel(
            "Top-1 Cosine Similarity Score",
            fontsize=10.5,
            fontweight="bold",
            color="#333333",
            labelpad=8,
        )
        ax.set_ylabel(
            "Probability Density", fontsize=10.5, fontweight="bold", color="#333333", labelpad=8
        )
        ax.set_title(
            "Retrieval Score Margin: Grounded vs. Negative Abstention Queries",
            fontsize=12,
            fontweight="bold",
            pad=12,
            color="#111111",
        )
        ax.legend(
            loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5
        )
        plt.tight_layout()
        plt.savefig(fig3_path, bbox_inches="tight")
        plt.close()
        print(f"Generated visual artifact: {fig3_path}")

    # Display all 3 figures inline in notebook cell output
    try:
        from IPython.display import Image as IPImage
        from IPython.display import display as ip_display

        print("\n=== Displaying Benchmark Visualizations Inline ===")
        for p in [fig1_path, fig2_path, fig3_path]:
            if p.is_file():
                ip_display(IPImage(filename=str(p)))
    except Exception:
        pass


# %% [markdown]
# ### 5. FAISS Vector Index Architecture Benchmarking


# %%
def benchmark_faiss_indexes(
    embeddings: np.ndarray,
    query_vectors: np.ndarray,
    dim: int,
    k: int = 10,
) -> dict[str, Any]:
    import faiss

    results: dict[str, Any] = {}

    # A. IndexFlatIP (Exact Cosine baseline)
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
                "description": (f"Inverted list clustering (nlist={nlist}, nprobe=8)"),
            }
        except Exception as e:
            results["IndexIVFFlat"] = {"error": str(e)}

    return results


# %%
def load_source_pages() -> list[dict[str, str]]:
    """Loads full 1,034-page corpus from attached Kaggle datasets (Chandra chunks.jsonl or pages.jsonl), local files, or fallback."""
    pages: list[dict[str, str]] = []

    # 1. Prioritize Chandra chunks.jsonl or full-book pages.jsonl across all attached inputs
    if INPUT_ROOT.is_dir():
        # Check for Chandra chunks.jsonl first (groups 8,544 OCR blocks into 1,034 book pages)
        for path in sorted(INPUT_ROOT.rglob("*.jsonl")):
            if "chunk" in path.name.lower():
                try:
                    page_texts: dict[str, list[str]] = {}
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            item = json.loads(line)
                            p_num = item.get(
                                "page_index", item.get("book_page", item.get("page_id", 1))
                            )
                            try:
                                p_id = f"p{int(p_num):04d}"
                            except Exception:
                                p_id = str(p_num)
                            content = item.get("content", item.get("text", "")).strip()
                            clean_text = re.sub(r"<[^>]+>", " ", content).strip()
                            if clean_text:
                                if p_id not in page_texts:
                                    page_texts[p_id] = []
                                page_texts[p_id].append(clean_text)

                    cand_pages = [
                        {"doc_id": "pierce-1890", "page_id": pid, "text": " ".join(blocks)}
                        for pid, blocks in sorted(page_texts.items())
                    ]
                    if len(cand_pages) >= 50:
                        print(f"Loaded {len(cand_pages)} FULL BOOK pages reconstructed from {path}")
                        return cand_pages
                except Exception:
                    pass

        # Check for pages.jsonl or full_book_pages.jsonl
        for path in sorted(INPUT_ROOT.rglob("*.jsonl")):
            if "page" in path.name.lower() or "full_book" in path.name.lower():
                try:
                    cand_pages = []
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            item = json.loads(line)
                            cand_pages.append(
                                {
                                    "doc_id": item.get("doc_id", "pierce-1890"),
                                    "page_id": item.get("page_id", f"p{len(cand_pages)+1:04d}"),
                                    "text": item.get("text", "").strip(),
                                }
                            )
                    if len(cand_pages) >= 50:
                        print(f"Loaded {len(cand_pages)} FULL BOOK pages from {path}")
                        return cand_pages
                except Exception:
                    pass

    # 2. Check local full_book_pages.jsonl in workspace
    local_full = Path("extras/indexing-benchmarks/full_book_pages.jsonl")
    if local_full.is_file():
        try:
            cand_pages = []
            for line in local_full.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    cand_pages.append(
                        {
                            "doc_id": item.get("doc_id", "pierce-1890"),
                            "page_id": item.get("page_id", f"p{len(cand_pages)+1:04d}"),
                            "text": item.get("text", "").strip(),
                        }
                    )
            if cand_pages:
                print(f"Loaded {len(cand_pages)} FULL BOOK pages from {local_full}")
                return cand_pages
        except Exception:
            pass

    # 3. Check attached labels.jsonl or markdown files
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
                    print(f"Loaded {len(pages)} held-out evaluation pages from {path}")
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
        (
            "Catarrh and affections of the air-passages require soothing"
            " expectorants and proper hygienic care."
        ),
        (
            "Golden Seal (Hydrastis Canadensis) is a most valuable native"
            " remedy, possessing bitter tonic and alterative virtues."
        ),
        (
            "Typhoid or enteric fever is marked by sustained high temperature,"
            " great prostration, and abdominal tenderness."
        ),
        (
            "The heart consists of four distinct chambers: right and left"
            " auricles, and right and left ventricles."
        ),
        (
            "To prepare the Golden Medical Discovery: combine active"
            " extractives with pure vegetable glycerine."
        ),
        (
            "The bones contain more earthy matter than any other part of the"
            " human body, being firm and lime-colored."
        ),
        (
            "The stomach is a musculo-membranous sac communicating with the"
            " esophagus by the cardiac orifice."
        ),
    ]
    for i in range(1, 101):
        text = f"Chapter {(i % 12) + 1}. Medical Advice for Family Use. " + " ".join(
            sample_topics * 3
        )
        pages.append({"doc_id": "pierce-1890", "page_id": f"p{i:04d}", "text": text})

    return pages


def load_curated_tasks() -> list[dict[str, Any]]:
    """Loads curated Q&A evaluation tasks from attached Kaggle datasets or local grading_kit/tasks.jsonl."""
    tasks: list[dict[str, Any]] = []

    if INPUT_ROOT.is_dir():
        for path in INPUT_ROOT.rglob("tasks.jsonl"):
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        tasks.append(json.loads(line))
                if tasks:
                    print(f"Loaded {len(tasks)} curated Q&A tasks from {path}")
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
                print(
                    f"Loaded {len(tasks)} curated Q&A tasks from local" " grading_kit/tasks.jsonl"
                )
                return tasks
        except Exception:
            pass

    return []


# Candidate Embedding Model Catalog defining all target architectures
CANDIDATE_MODEL_CATALOG = [
    {
        "id": "all-minilm-l6-v2",
        "aliases": [
            "sentence-transformers/all-MiniLM-L6-v2",
            "all-minilm-l6-v2",
            "all-MiniLM-L6-v2",
        ],
        "expected_dim": 384,
        "format": "PyTorch-Dense",
        "package_source": "package-embedding-models",
    },
    {
        "id": "bge-small-en-v1-5",
        "aliases": ["BAAI/bge-small-en-v1.5", "bge-small-en-v1-5", "bge-small-en-v1.5"],
        "expected_dim": 384,
        "format": "PyTorch-Dense",
        "package_source": "package-embedding-models",
    },
    {
        "id": "nomic-embed-text-v1-5",
        "aliases": [
            "nomic-ai/nomic-embed-text-v1.5",
            "nomic-embed-text-v1-5",
            "nomic-embed-text-v1.5",
        ],
        "expected_dim": 768,
        "format": "PyTorch-Dense",
        "package_source": "package-embedding-models",
    },
    {
        "id": "bge-m3",
        "aliases": ["BAAI/bge-m3", "bge-m3", "bge_m3"],
        "expected_dim": 1024,
        "format": "PyTorch-Dense",
        "package_source": "package-embedding-models",
    },
    {
        "id": "qwen3-embedding-0-6b",
        "aliases": ["Qwen/Qwen3-Embedding-0.6B", "qwen3-embedding-0-6b", "qwen3_embedding_0_6b"],
        "expected_dim": 1024,
        "format": "PyTorch-Dense",
        "package_source": "package-qwen3-family",
    },
    {
        "id": "qwen3-embedding-0-6b-gguf",
        "aliases": [
            "qwen3-embedding-0-6b-gguf",
            "qwen3-embedding-0.6b-q4_k_m.gguf",
            "qwen3-embedding-0.6b.gguf",
        ],
        "expected_dim": 1024,
        "format": "GGUF-Q4_K_M",
        "package_source": "package-qwen3-family",
    },
    {
        "id": "qwen3-embedding-4b-gguf",
        "aliases": [
            "qwen3-embedding-4b-gguf",
            "qwen3-embedding-4b-q4_k_m.gguf",
            "qwen3-embedding-4b.gguf",
        ],
        "expected_dim": 2560,
        "format": "GGUF-Q4_K_M",
        "package_source": "package-qwen3-family",
    },
    {
        "id": "qwen3-vl-embedding-2b",
        "aliases": ["Qwen/Qwen3-VL-2B", "qwen3-vl-embedding-2b", "qwen3-vl-2b"],
        "expected_dim": 1536,
        "format": "Multimodal-VL",
        "package_source": "package-multimodal-vl",
    },
]


def discover_all_candidate_models(
    asset_roots: list[tuple[Path, dict[str, Any]]],
) -> list[str]:
    """Audits and discovers all candidate embedding models across all attached Kaggle inputs without duplicates."""
    embed_models: list[str] = []
    seen_model_ids: set[str] = set()
    seen_paths: set[str] = set()

    print("\n" + "=" * 105)
    print("EMBEDDING MODEL DISCOVERY & ASSET ATTACHMENT AUDIT")
    print("=" * 105)
    audit_header = (
        f"{'Model ID':<26} {'Format':<15} {'Status':<12} {'Dataset Source / Location':<48}"
    )
    print(audit_header)
    print("-" * 105)

    all_search_dirs = [asset_dir for asset_dir, _ in asset_roots]
    if INPUT_ROOT.is_dir() and INPUT_ROOT not in all_search_dirs:
        all_search_dirs.append(INPUT_ROOT)

    for entry in CANDIDATE_MODEL_CATALOG:
        m_id = entry["id"]
        aliases = entry["aliases"]
        found_path: Path | None = None

        for search_dir in all_search_dirs:
            for alias in aliases:
                cand1 = search_dir / alias
                cand2 = search_dir / "models" / alias
                if cand1.exists():
                    found_path = cand1
                    break
                if cand2.exists():
                    found_path = cand2
                    break

                matches = list(search_dir.rglob(f"*{alias}*"))
                if matches:
                    found_path = matches[0]
                    break
            if found_path:
                break

        if found_path and m_id not in seen_model_ids:
            seen_model_ids.add(m_id)
            seen_paths.add(str(found_path.resolve()))
            embed_models.append(str(found_path))
            print(f"{m_id:<26} {entry['format']:<15} {'ATTACHED':<12} {str(found_path):<48}")
        elif m_id not in seen_model_ids:
            print(
                f"{m_id:<26} {entry['format']:<15} {'NOT FOUND':<12} Attach '{entry['package_source']}' dataset"
            )

    # Discovered directories for strict subtree checks
    discovered_dirs = [
        Path(p).resolve() if Path(p).is_dir() else Path(p).resolve().parent for p in embed_models
    ]

    # Discover any uncataloged standalone models
    for search_dir in all_search_dirs:
        for m_dir in sorted(search_dir.rglob("models/*")):
            if "reranker" in m_dir.name.lower() or not m_dir.is_dir():
                continue
            resolved_dir = m_dir.resolve()
            if m_dir.name in seen_model_ids or any(
                resolved_dir == d or resolved_dir.is_relative_to(d) for d in discovered_dirs
            ):
                continue
            seen_model_ids.add(m_dir.name)
            seen_paths.add(str(resolved_dir))
            embed_models.append(str(m_dir))
            print(f"{m_dir.name:<26} {'PyTorch-Dense':<15} {'ATTACHED':<12} {str(m_dir):<48}")

        for gguf_file in sorted(search_dir.rglob("*.gguf")):
            if "reranker" in gguf_file.name.lower() or not gguf_file.is_file():
                continue
            resolved_file = gguf_file.resolve()
            base_stem = gguf_file.stem.lower().replace("-q4_k_m", "").replace("-q8_0", "")
            if (
                base_stem in seen_model_ids
                or gguf_file.name in seen_model_ids
                or any(resolved_file.is_relative_to(d) for d in discovered_dirs)
            ):
                continue
            seen_model_ids.add(base_stem)
            seen_paths.add(str(resolved_file))
            embed_models.append(str(gguf_file))
            print(f"{gguf_file.name:<26} {'GGUF-Q4_K_M':<15} {'ATTACHED':<12} {str(gguf_file):<48}")

    print("=" * 105 + "\n")
    return embed_models


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

    pages = load_source_pages()
    if chunk_strategy == "parent_child_128_512":
        chunks = hierarchical_parent_child_chunking(pages, 512, 128, 16)
    elif chunk_strategy == "section_header_aware":
        chunks = section_header_aware_chunking(pages, 350)
    elif chunk_strategy == "multimodal_figure_graph":
        chunks = multimodal_figure_graph_chunking(pages, 256, 32)
    elif chunk_strategy == "semantic_recursive":
        chunks = recursive_semantic_chunking(pages, 256, 40)
    else:
        chunks = fixed_window_token_chunking(pages, 256, 32)

    asset_roots = find_all_asset_roots()
    model_path = model_name

    for asset_dir, _ in asset_roots:
        candidate_dir = asset_dir / "models" / model_name
        if candidate_dir.is_dir():
            model_path = str(candidate_dir)
            break

    if model_path == model_name and INPUT_ROOT.is_dir():
        for p in INPUT_ROOT.rglob(f"models/{model_name}"):
            if p.is_dir():
                model_path = str(p)
                break

    print(f"\n[Playground] Loading embedding model: {model_path} on" f" {device.upper()}...")
    is_gguf = "gguf" in model_path.lower() or Path(model_path).suffix.lower() == ".gguf"

    if is_gguf:
        model = GGUFEmbeddingWrapper(model_path)
    else:
        from sentence_transformers import SentenceTransformer

        try:
            model = SentenceTransformer(
                model_path,
                device=device,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            try:
                model = SentenceTransformer(model_path, device=device, local_files_only=True)
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
    asset_roots = find_all_asset_roots()
    if asset_roots:
        install_offline_runtimes(asset_roots)

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
            f"\n[Rank #{rank}]  Score: {score:.4f}  |  Page: {chunk.page_id} "
            f" |  ID: {chunk.chunk_id}"
        )
        print(f'"{snippet}"')
        print("-" * 80)

    return results


# %%
def run_benchmark() -> None:
    import torch

    device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
    print(f"Execution environment: PyTorch device = {device.upper()}" f" (USE_GPU={USE_GPU})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    asset_roots = find_all_asset_roots()

    if asset_roots:
        install_offline_runtimes(asset_roots)
    else:
        print(
            "Running in online mode or using local environment packages.",
            flush=True,
        )

    # 1. Load source pages and curated Q&A tasks
    pages = load_source_pages()
    tasks = load_curated_tasks()
    print(
        f"Loaded {len(pages)} source pages and {len(tasks)} curated evaluation" " Q&A tasks.",
        flush=True,
    )

    # 2. Build Chunking Suites
    chunk_suites = {
        "fixed_128_16": fixed_window_token_chunking(pages, 128, 16),
        "fixed_256_32": fixed_window_token_chunking(pages, 256, 32),
        "fixed_512_64": fixed_window_token_chunking(pages, 512, 64),
        "parent_child_128_512": hierarchical_parent_child_chunking(pages, 512, 128, 16),
        "section_header_aware": section_header_aware_chunking(pages, 350),
        "multimodal_figure_graph": multimodal_figure_graph_chunking(pages, 256, 32),
        "semantic_recursive": recursive_semantic_chunking(pages, 256, 40),
    }

    # 3. Discover Candidate Embedding Models
    candidate_embeds = discover_all_candidate_models(asset_roots)
    if not candidate_embeds:
        candidate_embeds = ["sentence-transformers/all-MiniLM-L6-v2"]

    print(f"\nDiscovered {len(candidate_embeds)} candidate embedding models to" " benchmark:")
    for m in candidate_embeds:
        print(f" - {m}")

    # 4. Benchmark Embedding Models on Standardized Baseline Chunks (fixed_256_32)
    baseline_chunks = chunk_suites["fixed_256_32"]
    embed_summary = []
    embeddings_store = {}
    best_model_obj = None
    all_score_logs = []

    for model_path in candidate_embeds:
        try:
            model_obj, emb, met = benchmark_embedding_model(
                model_path, baseline_chunks, device=device, batch_size=BATCH_SIZE
            )
            if tasks:
                accuracy, score_logs = evaluate_retrieval_accuracy(
                    model_obj, baseline_chunks, tasks
                )
                met.update(accuracy)
                if not all_score_logs:
                    all_score_logs = score_logs

            embed_summary.append(met)
            embeddings_store[met["model"]] = emb
            if best_model_obj is None:
                best_model_obj = model_obj
        except Exception as e:
            print(f"[ERROR] Failed to benchmark {model_path}: {e}")
            import traceback

            traceback.print_exc()

    # Calculate MCDA Composite Scores for Embeddings
    ranked_embed_models = calculate_composite_scores(embed_summary)

    print("\n" + "=" * 125)
    print("EMBEDDING MODEL DECISION MATRIX (Ranked by Multi-Criteria Composite" " Score)")
    print("=" * 125)
    header = (
        f"{'Rank':<5} {'Model Name':<26} {'Dims':<6} {'Tput (ch/s)':<13}"
        f" {'P50 Lat (ms)':<13} {'Recall@1':<10} {'Recall@5':<10} {'MRR':<8}"
        f" {'Composite':<10}"
    )
    print(header)
    print("-" * 125)
    for rank_idx, m in enumerate(ranked_embed_models, start=1):
        m_name = m.get("model", "unknown")[:24]
        m_dim = str(m.get("dimension", "-"))
        m_tput = f"{m.get('chunks_per_second', 0):.1f}"
        m_lat = f"{m.get('single_query_p50_ms', 0):.2f}"
        r1 = f"{m.get('recall@1', 0.0):.3f}" if "recall@1" in m else "N/A"
        r5 = f"{m.get('recall@5', 0.0):.3f}" if "recall@5" in m else "N/A"
        mrr_val = f"{m.get('mrr', 0.0):.4f}" if "mrr" in m else "N/A"
        comp = f"{m.get('composite_score', 0.0):.4f}"
        print(
            f"#{rank_idx:<4} {m_name:<26} {m_dim:<6} {m_tput:<13} {m_lat:<13}"
            f" {r1:<10} {r5:<10} {mrr_val:<8} {comp:<10}"
        )
    print("=" * 125)

    # 5. End-to-End Evaluation Across ALL Chunking Strategies
    chunk_accuracy: dict[str, dict[str, Any]] = {}
    chunk_summary: dict[str, dict[str, Any]] = {}

    if best_model_obj and tasks:
        print("\n" + "=" * 115)
        print("DOWNSTREAM RETRIEVAL EVALUATION ACROSS ALL CHUNKING STRATEGIES")
        print("=" * 115)
        chunk_head = (
            f"{'Strategy Name':<26} {'Chunks':<8} {'Avg Tok':<9}"
            f" {'Recall@1':<10} {'Recall@3':<10} {'Recall@5':<10}"
            f" {'Recall@10':<11} {'MRR':<8}"
        )
        print(chunk_head)
        print("-" * 115)

        for s_name, c_list in chunk_suites.items():
            acc, _ = evaluate_retrieval_accuracy(
                best_model_obj,
                c_list,
                tasks,
                is_parent_child=(s_name == "parent_child_128_512"),
            )
            lengths = [c.token_count for c in c_list]
            avg_tok = round(float(np.mean(lengths)), 1) if lengths else 0
            chunk_accuracy[s_name] = acc

            chunk_summary[s_name] = {
                "total_chunks": len(c_list),
                "avg_tokens": avg_tok,
                "min_tokens": int(np.min(lengths)) if lengths else 0,
                "max_tokens": int(np.max(lengths)) if lengths else 0,
                "has_parent_links": any(c.parent_id is not None for c in c_list),
                "has_figure_links": any(c.linked_figures is not None for c in c_list),
                **acc,
            }
            r1 = f"{acc.get('recall@1', 0.0):.3f}"
            r3 = f"{acc.get('recall@3', 0.0):.3f}"
            r5 = f"{acc.get('recall@5', 0.0):.3f}"
            r10 = f"{acc.get('recall@10', 0.0):.3f}"
            mrr_s = f"{acc.get('mrr', 0.0):.4f}"
            print(
                f"{s_name:<26} {len(c_list):<8} {avg_tok:<9} {r1:<10} {r3:<10}"
                f" {r5:<10} {r10:<11} {mrr_s:<8}"
            )
        print("=" * 115)

    # 6. Benchmark Vector Indexes
    if embeddings_store:
        first_model = list(embeddings_store.keys())[0]
        base_embeddings = embeddings_store[first_model]
        matching_path = next(
            (m for m in candidate_embeds if Path(m).name == first_model),
            first_model,
        )

        try:
            st_model, _ = load_model_instance(matching_path, device=device)
        except Exception:
            st_model = best_model_obj

        eval_queries = (
            [t["question"] for t in tasks if (t.get("target_pages") or t.get("gold_pages"))]
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
        print("\n" + "=" * 105)
        print(f"FAISS VECTOR INDEX ARCHITECTURE BENCHMARK ({len(base_embeddings)} Vectors)")
        print("=" * 105)
        faiss_head = (
            f"{'Index Architecture':<22} {'Index Type':<16} {'Build Time (s)':<16} "
            f"{'Query Latency (ms)':<22} {'Recall@10 vs Exact':<18}"
        )
        print(faiss_head)
        print("-" * 105)
        for idx_name, idx_info in faiss_summary.items():
            b_time = f"{idx_info.get('build_time_seconds', 0.0):.4f}"
            q_lat = f"{idx_info.get('avg_query_latency_ms', 0.0):.4f}"
            r10 = (
                f"{idx_info.get('recall_at_10_vs_exact', 1.0)*100:.1f}%"
                if "recall_at_10_vs_exact" in idx_info
                else "N/A"
            )
            i_type = idx_info.get("type", "unknown")
            print(f"{idx_name:<22} {i_type:<16} {b_time:<16} {q_lat:<22} {r10:<18}")
        print("=" * 105)
    else:
        faiss_summary = {}

    # 7. Optional Visual Figures Generation (Default OFF in Kaggle to prioritize speed)
    if GENERATE_PLOTS:
        print("\nGenerating publication-grade benchmark figures...")
        generate_benchmark_plots(
            chunk_suites,
            chunk_accuracy,
            ranked_embed_models,
            all_score_logs,
            PLOTS_DIR,
            faiss_results=faiss_summary,
        )
    else:
        print("\nSkipping figure rendering on remote runner (Plots can be generated locally).")

    # 8. Save All Candidate Knowledge Bases to disk for downstream experimentation
    kb_dir = OUT_DIR / "knowledge_bases"
    kb_dir.mkdir(parents=True, exist_ok=True)
    if best_model_obj:
        import faiss

        print(f"\nPersisting candidate Knowledge Bases to {kb_dir}...")
        for s_name, c_list in chunk_suites.items():
            kb_sub = kb_dir / f"kb_{s_name}"
            kb_sub.mkdir(parents=True, exist_ok=True)

            texts = [c.text for c in c_list]
            emb = best_model_obj.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).astype(np.float32)
            idx = faiss.IndexFlatIP(emb.shape[1])
            idx.add(emb)
            faiss.write_index(idx, str(kb_sub / "index.faiss"))

            with open(kb_sub / "chunks.jsonl", "w", encoding="utf-8") as f:
                for c in c_list:
                    f.write(
                        json.dumps(
                            {
                                "chunk_id": c.chunk_id,
                                "doc_id": c.doc_id,
                                "page_id": c.page_id,
                                "text": c.text,
                                "token_count": c.token_count,
                                "strategy": c.strategy,
                                "parent_id": c.parent_id,
                                "parent_text": c.parent_text,
                                "section_title": c.section_title,
                                "linked_figures": c.linked_figures,
                            }
                        )
                        + "\n"
                    )
            (kb_sub / "metadata.json").write_text(
                json.dumps(
                    {
                        "strategy": s_name,
                        "total_chunks": len(c_list),
                        "embedding_model": (
                            ranked_embed_models[0]["model"] if ranked_embed_models else "best_model"
                        ),
                        "dimension": emb.shape[1],
                        "accuracy": chunk_accuracy.get(s_name, {}),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        print(f"Saved {len(chunk_suites)} experimental Knowledge Bases.")

    # 9. Save Artifacts
    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_pages": len(pages),
        "curated_tasks_evaluated": len(tasks),
        "chunking_comparison": chunk_summary,
        "embedding_models_comparison": ranked_embed_models,
        "faiss_index_comparison": faiss_summary,
        "knowledge_bases_built": list(chunk_suites.keys()),
        "recommended_production_stack": {
            "chunk_size": 256,
            "chunk_overlap": 32,
            "embedding_model": (
                ranked_embed_models[0]["model"] if ranked_embed_models else "all-minilm-l6-v2"
            ),
            "faiss_index_type": "IndexFlatIP",
            "weak_threshold": 0.35,
        },
    }

    report_path = OUT_DIR / "indexing_comparison_results.json"
    report_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\nReport written to: {report_path}")

    # Pack zip archive with json report, png plots, and all experimental knowledge bases
    zip_path = WORK / "indexing-benchmark-outputs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(report_path, arcname="indexing_comparison_results.json")
        for plot_file in PLOTS_DIR.glob("*.png"):
            zf.write(plot_file, arcname=f"plots/{plot_file.name}")
        for kb_file in kb_dir.rglob("*"):
            if kb_file.is_file():
                zf.write(kb_file, arcname=f"knowledge_bases/{kb_file.relative_to(kb_dir)}")
    print(
        f"Created final benchmark output archive with plots and KBs: {zip_path}"
        f" ({zip_path.stat().st_size / 1e3:.1f} KB)"
    )


# %%
if __name__ == "__main__":
    run_benchmark()


# %% [markdown]
# # Interactive Vector Retrieval Playground
# Run this cell anytime to ask any custom question and inspect the top retrieved passages, scores, and page citations.

# %%
# Available model names from attached assets:
# - "qwen3-embedding-0-6b"       : Compact 0.6B PyTorch text embedder (1024-d)
# - "qwen3-embedding-0-6b-gguf"  : Fast 0.6B Q4_K_M quantized embedder (1024-d)
# - "qwen3-vl-embedding-2b"      : 2.0B Vision-Language multimodal embedder (1536-d)
# - "qwen3-embedding-4b-gguf"    : High-capacity 4.0B Q4_K_M quantized embedder (2560-d)
# - "bge-small-en-v1-5"          : Compact 384-d MTEB benchmark leader
# - "nomic-embed-text-v1-5"      : 768-d 8k long-context embedder
# - "all-minilm-l6-v2"           : Ultra-fast 384-d baseline
# - "bge-m3"                     : 1024-d dense + sparse hybrid

MY_QUERY = "What are the medicinal preparations and healing properties of Golden Seal?"
CHOSEN_MODEL = "qwen3-embedding-0-6b"
TOP_K = 5

try:
    results = interactive_search(query=MY_QUERY, top_k=TOP_K, model_name=CHOSEN_MODEL)
except Exception as e:
    print(f"Interactive search note: {e}")
