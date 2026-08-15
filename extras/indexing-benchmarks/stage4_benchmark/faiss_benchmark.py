from __future__ import annotations

import time
from typing import Any

import numpy as np


def benchmark_faiss_architectures(
    embeddings: np.ndarray,
    queries: np.ndarray,
    ground_truth_top10: np.ndarray,
    num_iterations: int = 10,
) -> dict[str, Any]:
    """Benchmarks FlatIP, HNSW, and IVFFlat under identical normalized inner product conditions."""
    import faiss

    results: dict[str, Any] = {}
    dim = embeddings.shape[1]

    # 1. IndexFlatIP (Exact baseline)
    t0 = time.perf_counter()
    flat_idx = faiss.IndexFlatIP(dim)
    flat_idx.add(embeddings)
    b_time = time.perf_counter() - t0

    _ = flat_idx.search(queries[:5], 10)  # warm-up
    lats = []
    for _ in range(num_iterations):
        t_s = time.perf_counter()
        flat_idx.search(queries, 10)
        lats.append((time.perf_counter() - t_s) * 1000 / len(queries))

    results["IndexFlatIP"] = {
        "build_time_s": round(b_time, 4),
        "p50_query_latency_ms": round(float(np.median(lats)), 3),
        "p95_query_latency_ms": round(float(np.percentile(lats, 95)), 3),
        "top10_agreement_with_flat": 1.0,
        "parameters": "metric=INNER_PRODUCT, exact=true",
    }

    # 2. IndexHNSWFlat
    t0 = time.perf_counter()
    hnsw_idx = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw_idx.hnsw.efConstruction = 64
    hnsw_idx.hnsw.efSearch = 32
    hnsw_idx.add(embeddings)
    b_hnsw = time.perf_counter() - t0

    _ = hnsw_idx.search(queries[:5], 10)
    lats = []
    for _ in range(num_iterations):
        t_s = time.perf_counter()
        _, h_indices = hnsw_idx.search(queries, 10)
        lats.append((time.perf_counter() - t_s) * 1000 / len(queries))

    agreements = []
    for q_i in range(len(queries)):
        match = len(set(h_indices[q_i]) & set(ground_truth_top10[q_i])) / 10.0
        agreements.append(match)

    results["IndexHNSWFlat"] = {
        "build_time_s": round(b_hnsw, 4),
        "p50_query_latency_ms": round(float(np.median(lats)), 3),
        "p95_query_latency_ms": round(float(np.percentile(lats, 95)), 3),
        "top10_agreement_with_flat": round(float(np.mean(agreements)), 4),
        "parameters": "M=32, efConstruction=64, efSearch=32, metric=INNER_PRODUCT",
    }

    # 3. IndexIVFFlat
    nlist = min(64, max(4, len(embeddings) // 30))
    quantizer = faiss.IndexFlatIP(dim)
    ivf_idx = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    t0 = time.perf_counter()
    ivf_idx.train(embeddings)
    ivf_idx.add(embeddings)
    b_ivf = time.perf_counter() - t0
    ivf_idx.nprobe = 8

    _ = ivf_idx.search(queries[:5], 10)
    lats = []
    for _ in range(num_iterations):
        t_s = time.perf_counter()
        _, ivf_indices = ivf_idx.search(queries, 10)
        lats.append((time.perf_counter() - t_s) * 1000 / len(queries))

    agreements_ivf = []
    for q_i in range(len(queries)):
        match = len(set(ivf_indices[q_i]) & set(ground_truth_top10[q_i])) / 10.0
        agreements_ivf.append(match)

    results["IndexIVFFlat"] = {
        "build_time_s": round(b_ivf, 4),
        "p50_query_latency_ms": round(float(np.median(lats)), 3),
        "p95_query_latency_ms": round(float(np.percentile(lats, 95)), 3),
        "top10_agreement_with_flat": round(float(np.mean(agreements_ivf)), 4),
        "parameters": f"nlist={nlist}, nprobe=8, metric=INNER_PRODUCT",
    }

    return results
