from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .chunking import build_chunk_suites
from .corpus import load_canonical_corpus
from .evaluation import evaluate_retrieval_suite
from .faiss_benchmark import benchmark_faiss_architectures
from .models import EmbeddingModelAdapter, discover_candidate_models
from .queries import load_retrieval_queries


def run_stage4_benchmark(
    output_dir: Path,
    device: str = "cpu",
    search_root: Path | None = None,
) -> dict[str, Any]:
    """Executes the complete Stage-4 5x5 Factorial Benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    kb_dir = output_dir / "knowledge_bases"
    kb_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    pages = load_canonical_corpus(search_root)
    all_queries = load_retrieval_queries(search_root)

    dev_queries = [q for q in all_queries if q.split == "dev"]
    test_queries = [q for q in all_queries if q.split == "test"]
    out_of_corpus_queries = [q for q in all_queries if q.split == "out_of_corpus"]

    # 2. Build Chunk Suites
    chunk_suites = build_chunk_suites(pages)

    # 3. Discover Models
    discovered = discover_candidate_models([search_root] if search_root else None)

    grid_results = []
    all_dev_logs = []

    # 4. Run 5x5 Grid on Dev
    for c_id, (_, m_path) in discovered.items():
        adapter = EmbeddingModelAdapter(m_path, canonical_id=c_id, device=device)
        for s_name, c_list in chunk_suites.items():
            is_pc = (s_name == "parent_child_128_512")
            metrics, logs = evaluate_retrieval_suite(
                adapter, c_list, dev_queries, top_k=10, is_parent_child=is_pc
            )
            grid_results.append(metrics)
            all_dev_logs.extend(logs)

    # 5. Lock Winner on Dev Set
    grid_results.sort(
        key=lambda x: (x["single_page_recall@5"], x["single_page_mrr@10"], x["multi_page_coverage@10"]),
        reverse=True,
    )
    winner = grid_results[0]

    # 6. Evaluate Winner Once on Final Test
    winner_c_id = winner["canonical_model_id"]
    winner_path = discovered[winner_c_id][1]
    winning_adapter = EmbeddingModelAdapter(winner_path, canonical_id=winner_c_id, device=device)
    winning_chunks = chunk_suites[winner["strategy"]]

    test_metrics, test_logs = evaluate_retrieval_suite(
        winning_adapter,
        winning_chunks,
        test_queries,
        top_k=10,
        is_parent_child=(winner["strategy"] == "parent_child_128_512"),
    )

    # 7. FAISS Benchmark
    import faiss

    final_doc_embs = winning_adapter.encode_documents([c.text for c in winning_chunks])
    final_q_embs = winning_adapter.encode_queries([q.question for q in dev_queries[:20]])
    exact_idx = faiss.IndexFlatIP(final_doc_embs.shape[1])
    exact_idx.add(final_doc_embs)
    _, gt_top10 = exact_idx.search(final_q_embs, 10)
    faiss_summary = benchmark_faiss_architectures(final_doc_embs, final_q_embs, gt_top10, num_iterations=10)

    # 8. Persist Winning KBs and JSON Report
    for s_name, c_list in chunk_suites.items():
        sub_kb = kb_dir / f"kb_{s_name}"
        sub_kb.mkdir(parents=True, exist_ok=True)
        c_embs = winning_adapter.encode_documents([c.text for c in c_list])
        idx = faiss.IndexFlatIP(c_embs.shape[1])
        idx.add(c_embs)
        faiss.write_index(idx, str(sub_kb / "index.faiss"))
        with open(sub_kb / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in c_list:
                f.write(
                    json.dumps({
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "page_id": c.page_id,
                        "text": c.text,
                        "word_count": c.word_count,
                        "strategy": c.strategy,
                        "parent_id": c.parent_id,
                        "parent_text": c.parent_text,
                        "section_title": c.section_title,
                    })
                    + "\n"
                )

    with open(output_dir / "dev_retrieval_log.jsonl", "w", encoding="utf-8") as f:
        for log in all_dev_logs:
            f.write(json.dumps(log) + "\n")

    with open(output_dir / "test_retrieval_log.jsonl", "w", encoding="utf-8") as f:
        for log in test_logs:
            f.write(json.dumps(log) + "\n")

    full_report = {
        "benchmark_version": "2.0-reproduced",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_statistics": {
            "pdf_total_pages": len(pages),
            "nonempty_indexed_pages": sum(1 for p in pages if p.word_count > 0),
            "total_corpus_words": sum(p.word_count for p in pages),
        },
        "query_suite_statistics": {
            "total_queries": len(all_queries),
            "dev_single_page": len([q for q in dev_queries if q.type == "single_page"]),
            "dev_multi_page": len([q for q in dev_queries if q.type == "multi_page"]),
            "final_test_single_page": len([q for q in test_queries if q.type == "single_page"]),
            "final_test_multi_page": len([q for q in test_queries if q.type == "multi_page"]),
            "out_of_corpus": len(out_of_corpus_queries),
        },
        "factorial_grid_dev_results": grid_results,
        "selection_decision": {
            "winning_embedding_model": winner["canonical_model_id"],
            "winning_chunking_strategy": winner["strategy"],
            "winning_faiss_index": "IndexFlatIP",
        },
        "final_test_evaluation": test_metrics,
        "faiss_index_comparison": faiss_summary,
    }

    (output_dir / "indexing_comparison_results.json").write_text(
        json.dumps(full_report, indent=2), encoding="utf-8"
    )
    return full_report
