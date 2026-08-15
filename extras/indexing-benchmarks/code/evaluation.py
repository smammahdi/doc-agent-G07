from __future__ import annotations

import time
from typing import Any

import numpy as np

from .chunking import BenchmarkChunk
from .models import EmbeddingModelAdapter
from .queries import RetrievalQuery


def calculate_bootstrap_ci(
    scores: list[float], num_resamples: int = 1000, ci: float = 0.95
) -> tuple[float, float]:
    """Calculates non-parametric 95% bootstrap confidence interval."""
    if not scores:
        return 0.0, 0.0
    arr = np.array(scores)
    boot_means = []
    n = len(arr)
    np.random.seed(42)
    for _ in range(num_resamples):
        sample = np.random.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return lower, upper


def evaluate_retrieval_suite(
    model: EmbeddingModelAdapter,
    chunks: list[BenchmarkChunk],
    queries: list[RetrievalQuery],
    top_k: int = 10,
    is_parent_child: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluates full retrieval metrics on a set of queries and chunks."""
    import faiss

    # 1. Encode all chunks
    doc_texts = [c.text for c in chunks]
    t0 = time.perf_counter()
    doc_embs = model.encode_documents(doc_texts, batch_size=32)
    enc_time = time.perf_counter() - t0

    dim = doc_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embs)

    # 2. Encode queries
    grounded_queries = [q for q in queries if q.page_ids]
    q_texts = [q.question for q in grounded_queries]
    t_q0 = time.perf_counter()
    q_embs = model.encode_queries(q_texts)
    q_enc_time = (time.perf_counter() - t_q0) * 1000 / max(1, len(q_texts))

    # 3. Search Index
    scores, indices = index.search(q_embs, top_k)

    per_query_logs = []
    single_page_hits = {1: [], 5: [], 10: []}
    single_page_r1 = []
    single_page_r5 = []
    single_page_r10 = []
    single_page_mrrs = []
    single_page_span_containment = []

    multi_page_coverages = []
    multi_page_all_found = []
    multi_page_first_mrrs = []

    for i, q in enumerate(grounded_queries):
        target_pids = set(q.page_ids)
        retrieved_chunk_indices = [idx for idx in indices[i] if 0 <= idx < len(chunks)]
        retrieved_chunks = [chunks[idx] for idx in retrieved_chunk_indices]
        retrieved_pids = [c.page_id for c in retrieved_chunks]
        retrieved_scores = [float(s) for s in scores[i][: len(retrieved_chunks)]]

        # Deduplicated retrieved pages preserving order
        unique_retrieved_pids = []
        for pid in retrieved_pids:
            if pid not in unique_retrieved_pids:
                unique_retrieved_pids.append(pid)

        # Single-page evaluation
        if q.type == "single_page":
            target_pid = q.page_ids[0]
            # Unique page recall
            r1 = (
                1.0
                if (len(unique_retrieved_pids) >= 1 and unique_retrieved_pids[0] == target_pid)
                else 0.0
            )
            r5 = 1.0 if target_pid in unique_retrieved_pids[:5] else 0.0
            r10 = 1.0 if target_pid in unique_retrieved_pids[:10] else 0.0
            single_page_r1.append(r1)
            single_page_r5.append(r5)
            single_page_r10.append(r10)

            # Chunk-level hit
            c_h1 = (
                1.0
                if (len(retrieved_chunks) >= 1 and retrieved_chunks[0].page_id == target_pid)
                else 0.0
            )
            c_h5 = 1.0 if any(c.page_id == target_pid for c in retrieved_chunks[:5]) else 0.0
            c_h10 = 1.0 if any(c.page_id == target_pid for c in retrieved_chunks[:10]) else 0.0
            single_page_hits[1].append(c_h1)
            single_page_hits[5].append(c_h5)
            single_page_hits[10].append(c_h10)

            # MRR@10
            rank = 0
            for r_idx, pid in enumerate(unique_retrieved_pids[:10], start=1):
                if pid == target_pid:
                    rank = r_idx
                    break
            mrr_val = 1.0 / rank if rank > 0 else 0.0
            single_page_mrrs.append(mrr_val)

            # Answer span containment in returned text
            span_found = False
            for c in retrieved_chunks[:5]:
                context = c.parent_text if (is_parent_child and c.parent_text) else c.text
                if q.exact_answer_span and q.exact_answer_span.lower() in context.lower():
                    span_found = True
                    break
            single_page_span_containment.append(1.0 if span_found else 0.0)

            per_query_logs.append({
                "query_id": q.query_id,
                "split": q.split,
                "type": q.type,
                "target_pages": q.page_ids,
                "retrieved_chunk_ids": [c.chunk_id for c in retrieved_chunks],
                "retrieved_page_ids": unique_retrieved_pids,
                "top_scores": retrieved_scores[:5],
                "rank": rank,
                "recall@5": r5,
                "span_contained@5": span_found,
            })

        # Multi-page evaluation
        elif q.type == "multi_page":
            found_targets = target_pids.intersection(set(unique_retrieved_pids[:10]))
            fraction_recovered = len(found_targets) / max(1, len(target_pids))
            all_found = 1.0 if len(found_targets) == len(target_pids) else 0.0
            multi_page_coverages.append(fraction_recovered)
            multi_page_all_found.append(all_found)

            first_rank = 0
            for r_idx, pid in enumerate(unique_retrieved_pids[:10], start=1):
                if pid in target_pids:
                    first_rank = r_idx
                    break
            first_mrr = 1.0 / first_rank if first_rank > 0 else 0.0
            multi_page_first_mrrs.append(first_mrr)

            per_query_logs.append({
                "query_id": q.query_id,
                "split": q.split,
                "type": q.type,
                "target_pages": q.page_ids,
                "retrieved_page_ids": unique_retrieved_pids,
                "fraction_recovered@10": fraction_recovered,
                "all_pages_found@10": bool(all_found),
                "first_page_rank": first_rank,
            })

    # 4. Bootstrap 95% Confidence Interval for Single-Page Recall@5
    ci_lower, ci_upper = calculate_bootstrap_ci(single_page_r5, num_resamples=1000)

    # 5. Intrinsic Chunk Statistics
    chunk_lengths = [c.word_count for c in chunks]
    avg_words = float(np.mean(chunk_lengths)) if chunk_lengths else 0.0

    metrics = {
        "canonical_model_id": model.model_id,
        "resolved_model_path": model.resolved_path,
        "dimension": dim,
        "strategy": chunks[0].strategy if chunks else "unknown",
        "total_chunks": len(chunks),
        "avg_words": round(avg_words, 1),
        "encoding_time_s": round(enc_time, 2),
        "throughput_chunks_per_s": round(len(doc_texts) / max(0.001, enc_time), 1),
        "single_query_latency_ms": round(q_enc_time, 2),
        # Single-page metrics
        "single_page_recall@1": round(float(np.mean(single_page_r1)), 4) if single_page_r1 else 0.0,
        "single_page_recall@5": round(float(np.mean(single_page_r5)), 4) if single_page_r5 else 0.0,
        "single_page_recall@10": round(float(np.mean(single_page_r10)), 4) if single_page_r10 else 0.0,
        "single_page_mrr@10": round(float(np.mean(single_page_mrrs)), 4) if single_page_mrrs else 0.0,
        "single_page_span_containment@5": round(float(np.mean(single_page_span_containment)), 4) if single_page_span_containment else 0.0,
        "recall@5_ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
        # Multi-page metrics
        "multi_page_coverage@10": round(float(np.mean(multi_page_coverages)), 4) if multi_page_coverages else 0.0,
        "multi_page_all_found@10": round(float(np.mean(multi_page_all_found)), 4) if multi_page_all_found else 0.0,
        "multi_page_first_mrr@10": round(float(np.mean(multi_page_first_mrrs)), 4) if multi_page_first_mrrs else 0.0,
    }
    return metrics, per_query_logs
