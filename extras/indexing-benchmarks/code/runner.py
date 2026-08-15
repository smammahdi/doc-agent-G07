from __future__ import annotations

import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import build_chunk_suites
from .corpus import load_canonical_corpus
from .evaluation import evaluate_retrieval_suite
from .faiss_benchmark import benchmark_faiss_architectures
from .models import EmbeddingModelAdapter, discover_candidate_models
from .queries import load_retrieval_queries


def run_stage4_dev_grid(
    output_dir: Path,
    device: str = "cpu",
    search_root: Path | None = None,
) -> Path:
    """Executes the complete 5x5 Factorial Grid on the Development Set and emits stage4-dev-selection.zip."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Corpus & Queries
    pages = load_canonical_corpus(search_root)
    all_queries = load_retrieval_queries(search_root)

    dev_queries = [q for q in all_queries if q.split == "dev"]
    if len(dev_queries) != 80:
        raise ValueError(f"Expected exactly 80 development queries, found {len(dev_queries)}")

    # 2. Build 5 Chunking Suites
    chunk_suites = build_chunk_suites(pages)
    if len(chunk_suites) != 5:
        raise ValueError(f"Expected exactly 5 chunking suites, found {len(chunk_suites)}")

    # 3. Discover Candidate Models
    discovered = discover_candidate_models([search_root] if search_root else None)
    if len(discovered) != 5:
        raise ValueError(f"Expected exactly 5 candidate models, found {len(discovered)}")

    grid_results: list[dict[str, Any]] = []
    all_dev_logs: list[dict[str, Any]] = []

    # 4. Evaluate all 5x5 = 25 cells
    for c_id, (_, m_path) in discovered.items():
        adapter = EmbeddingModelAdapter(m_path, canonical_id=c_id, device=device)
        for s_name, c_list in chunk_suites.items():
            is_pc = (s_name == "parent_child_128_512")
            metrics, logs = evaluate_retrieval_suite(
                adapter, c_list, dev_queries, top_k=10, is_parent_child=is_pc
            )
            grid_results.append(metrics)
            all_dev_logs.extend(logs)

    if len(grid_results) != 25:
        raise RuntimeError(f"Expected exactly 25 grid results, but got {len(grid_results)}. Incomplete run!")

    # 5. Lock Winner on Dev Set
    grid_results.sort(
        key=lambda x: (x["single_page_recall@5"], x["single_page_mrr@10"], x["multi_page_coverage@10"]),
        reverse=True,
    )
    winner = grid_results[0]

    # 6. Generate JSON and ZIP artifacts
    corpus_stats = {
        "pdf_total_pages": len(pages),
        "nonempty_indexed_pages": sum(1 for p in pages if p.word_count > 0),
        "ocr_missing_unobserved_pages": sum(1 for p in pages if p.ocr_source == "ocr_missing_unobserved"),
        "ocr_empty_illustration_only_pages": sum(1 for p in pages if p.ocr_source == "ocr_empty_illustration_only"),
        "total_corpus_words": sum(p.word_count for p in pages),
    }

    query_audit = {
        "total_queries": len(all_queries),
        "dev_queries": len(dev_queries),
        "dev_single_page": sum(1 for q in dev_queries if q.type == "single_page"),
        "dev_multi_page": sum(1 for q in dev_queries if q.type == "multi_page"),
    }

    candidate_lock = {
        "winning_model_id": winner["canonical_model_id"],
        "resolved_model_path": winner["resolved_model_path"],
        "winning_chunk_strategy": winner["strategy"],
        "dimension": winner["dimension"],
        "dev_recall@5": winner["single_page_recall@5"],
        "dev_mrr@10": winner["single_page_mrr@10"],
        "dev_recall@5_ci_95": winner["recall@5_ci_95"],
    }

    run_env = {
        "python_version": sys.version,
        "device": device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest = {
        "stage": "stage4-dev-selection",
        "version": "2.0-reproduced",
        "grid_cells_evaluated": len(grid_results),
        "timestamp": run_env["timestamp"],
    }

    # Write files
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "corpus-stats.json").write_text(json.dumps(corpus_stats, indent=2), encoding="utf-8")
    (output_dir / "query-audit.json").write_text(json.dumps(query_audit, indent=2), encoding="utf-8")
    (output_dir / "dev-grid-results.json").write_text(json.dumps(grid_results, indent=2), encoding="utf-8")
    (output_dir / "candidate-lock.json").write_text(json.dumps(candidate_lock, indent=2), encoding="utf-8")
    (output_dir / "run-environment.json").write_text(json.dumps(run_env, indent=2), encoding="utf-8")

    with open(output_dir / "dev-per-query.jsonl", "w", encoding="utf-8") as f:
        for log in all_dev_logs:
            f.write(json.dumps(log) + "\n")

    # Pack ZIP
    zip_path = output_dir / "stage4-dev-selection.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in [
            "manifest.json",
            "corpus-stats.json",
            "query-audit.json",
            "dev-grid-results.json",
            "dev-per-query.jsonl",
            "candidate-lock.json",
            "run-environment.json",
        ]:
            zf.write(output_dir / fname, arcname=fname)

    return zip_path


def run_stage4_final_evidence(
    candidate_lock_path: Path,
    output_dir: Path,
    device: str = "cpu",
    search_root: Path | None = None,
) -> Path:
    """Executes the locked winning stack on the untouched Final Test split and emits stage4-final-evidence.zip."""
    import faiss

    output_dir.mkdir(parents=True, exist_ok=True)
    kb_dir = output_dir / "production_kb"
    kb_dir.mkdir(parents=True, exist_ok=True)

    if not candidate_lock_path.is_file():
        raise FileNotFoundError(f"Missing candidate lock file: {candidate_lock_path}")

    lock_data = json.loads(candidate_lock_path.read_text(encoding="utf-8"))
    winning_model_id = lock_data["winning_model_id"]
    winning_model_path = lock_data["resolved_model_path"]
    winning_strategy = lock_data["winning_chunk_strategy"]

    # 1. Load Data
    pages = load_canonical_corpus(search_root)
    all_queries = load_retrieval_queries(search_root)
    test_queries = [q for q in all_queries if q.split == "test"]
    if len(test_queries) != 20:
        raise ValueError(f"Expected exactly 20 final test queries, found {len(test_queries)}")

    chunk_suites = build_chunk_suites(pages)
    winning_chunks = chunk_suites[winning_strategy]

    # 2. Evaluate Locked Stack Once on Final Test
    adapter = EmbeddingModelAdapter(winning_model_path, canonical_id=winning_model_id, device=device)
    is_pc = (winning_strategy == "parent_child_128_512")
    final_metrics, test_logs = evaluate_retrieval_suite(
        adapter, winning_chunks, test_queries, top_k=10, is_parent_child=is_pc
    )

    # 3. FAISS Benchmark
    final_doc_embs = adapter.encode_documents([c.text for c in winning_chunks])
    final_q_embs = adapter.encode_queries([q.question for q in test_queries])

    exact_idx = faiss.IndexFlatIP(final_doc_embs.shape[1])
    exact_idx.add(final_doc_embs)
    _, gt_top10 = exact_idx.search(final_q_embs, 10)
    faiss_summary = benchmark_faiss_architectures(final_doc_embs, final_q_embs, gt_top10, num_iterations=10)

    # 4. Record 1 Successful Retrieval and 1 Worst Failure
    successful_ex = None
    worst_failure = None
    for log in test_logs:
        if log.get("recall@5") == 1.0 and successful_ex is None:
            successful_ex = log
        elif log.get("recall@5") == 0.0 and worst_failure is None:
            worst_failure = log

    retrieval_examples = {
        "successful_example": successful_ex or test_logs[0],
        "worst_failure_example": worst_failure or test_logs[-1],
    }

    # 5. Persist Production Index
    idx_flat = faiss.IndexFlatIP(final_doc_embs.shape[1])
    idx_flat.add(final_doc_embs)
    faiss.write_index(idx_flat, str(kb_dir / "index.faiss"))

    with open(kb_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in winning_chunks:
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

    index_stats = {
        "index_type": "IndexFlatIP",
        "embedding_model": winning_model_id,
        "embedding_dimension": final_doc_embs.shape[1],
        "total_chunks": len(winning_chunks),
        "avg_words_per_chunk": float(np.mean([c.word_count for c in winning_chunks])),
        "index_size_bytes": (kb_dir / "index.faiss").stat().st_size,
    }

    selected_config = f"""# Production Knowledge Base Configuration (Locked in Stage 4)
doc_agent:
  index:
    chunk_strategy: "{winning_strategy}"
    embedding_model: "{winning_model_id}"
    dimension: {final_doc_embs.shape[1]}
    index_type: "IndexFlatIP"
    metric: "INNER_PRODUCT"
"""
    (output_dir / "selected-config.yaml").write_text(selected_config, encoding="utf-8")

    evidence_summary = f"""# Stage 4 Final Evidence Summary

## 1. Locked Production Stack
- **Embedding Model**: `{winning_model_id}` ({final_doc_embs.shape[1]}-d)
- **Chunking Strategy**: `{winning_strategy}` ({len(winning_chunks)} chunks, avg {index_stats['avg_words_per_chunk']:.1f} words)
- **Vector Search Index**: `IndexFlatIP` ({index_stats['index_size_bytes'] / 1024:.1f} KB)

## 2. Final Untouched Test Performance
- **Single-Page Recall@1**: {final_metrics['single_page_recall@1']:.4f}
- **Single-Page Recall@5**: {final_metrics['single_page_recall@5']:.4f} (95% CI: {final_metrics['recall@5_ci_95']})
- **Single-Page MRR@10**: {final_metrics['single_page_mrr@10']:.4f}
- **Multi-Page Coverage@10**: {final_metrics['multi_page_coverage@10']:.4f}
- **Multi-Page All-Found@10**: {final_metrics['multi_page_all_found@10']:.4f}
"""
    (output_dir / "evidence-summary.md").write_text(evidence_summary, encoding="utf-8")

    # Write JSON files
    (output_dir / "final-results.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    (output_dir / "faiss-comparison.json").write_text(json.dumps(faiss_summary, indent=2), encoding="utf-8")
    (output_dir / "index-statistics.json").write_text(json.dumps(index_stats, indent=2), encoding="utf-8")
    (output_dir / "retrieval-examples.json").write_text(json.dumps(retrieval_examples, indent=2), encoding="utf-8")

    with open(output_dir / "final-per-query.jsonl", "w", encoding="utf-8") as f:
        for log in test_logs:
            f.write(json.dumps(log) + "\n")

    # Pack ZIP
    zip_path = output_dir / "stage4-final-evidence.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in [
            "final-results.json",
            "final-per-query.jsonl",
            "faiss-comparison.json",
            "index-statistics.json",
            "retrieval-examples.json",
            "selected-config.yaml",
            "evidence-summary.md",
        ]:
            zf.write(output_dir / fname, arcname=fname)

    return zip_path
