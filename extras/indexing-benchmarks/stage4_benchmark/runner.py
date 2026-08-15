from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .chunking import build_chunk_suites
from .corpus import load_canonical_corpus
from .evaluation import (
    calibrate_abstention_threshold,
    evaluate_abstention_on_queries,
    evaluate_retrieval_suite,
)
from .faiss_benchmark import benchmark_faiss_architectures
from .models import EmbeddingModelAdapter, discover_candidate_models
from .queries import load_retrieval_queries


def run_stage4_dev_grid(
    output_dir: Path,
    device: str = "cpu",
    search_root: Path | None = None,
) -> Path:
    """Executes the complete 5x5 Factorial Grid on the Development Set and emits stage4-dev-selection.zip."""
    import faiss

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Corpus & Queries
    pages = load_canonical_corpus(search_root)
    all_queries = load_retrieval_queries(search_root)

    dev_grounded = [q for q in all_queries if q.split == "dev" and q.type != "out_of_corpus"]
    dev_negatives = [q for q in all_queries if q.split == "dev" and q.type == "out_of_corpus"]

    if len(dev_grounded) != 80:
        raise ValueError(f"Expected exactly 80 development grounded queries, found {len(dev_grounded)}")
    if len(dev_negatives) != 5:
        raise ValueError(f"Expected exactly 5 development negative queries, found {len(dev_negatives)}")

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

    # 4. Evaluate all 5x5 = 25 cells on grounded dev queries
    cell_count = 0
    total_cells = len(discovered) * len(chunk_suites)
    print(f"Starting Phase 1: 5x5 Dev Grid evaluation across all {total_cells} cells...", flush=True)
    for c_id, (_, m_path) in discovered.items():
        adapter = EmbeddingModelAdapter(m_path, canonical_id=c_id, device=device)
        for s_name, c_list in chunk_suites.items():
            cell_count += 1
            t_c0 = time.perf_counter()
            is_pc = (s_name == "parent_child_128_512")
            metrics, logs = evaluate_retrieval_suite(
                adapter, c_list, dev_grounded, top_k=10, is_parent_child=is_pc
            )
            grid_results.append(metrics)
            all_dev_logs.extend(logs)
            t_cell = time.perf_counter() - t_c0
            print(
                f"  [{cell_count:02d}/{total_cells}] {c_id:<22} + {s_name:<22} | "
                f"R@5: {metrics['single_page_recall@5']:.3f} | MRR: {metrics['single_page_mrr@10']:.3f} | "
                f"Cov: {metrics['multi_page_coverage@10']:.3f} ({t_cell:.1f}s)",
                flush=True,
            )

    if len(grid_results) != 25:
        raise RuntimeError(f"Expected exactly 25 grid results, but got {len(grid_results)}. Incomplete run!")

    # 5. Lock Winner on Dev Set
    # Selection criteria:
    # 1. Recall@5
    # 2. MRR@10
    # 3. Multi-page coverage
    # 4. Span containment
    # 5. Latency / index size tie-breakers
    grid_results.sort(
        key=lambda x: (
            x["single_page_recall@5"],
            x["single_page_mrr@10"],
            x["multi_page_coverage@10"],
            x["single_page_span_containment@5"],
            -x["single_query_latency_ms"],
        ),
        reverse=True,
    )
    winner = grid_results[0]

    # 6. Calibrate Abstention Threshold on Dev Negatives for the Winner
    winner_adapter = EmbeddingModelAdapter(winner["resolved_model_path"], canonical_id=winner["canonical_model_id"], device=device)
    winner_chunks = chunk_suites[winner["strategy"]]
    doc_texts = [c.text for c in winner_chunks]
    doc_embs = winner_adapter.encode_documents(doc_texts, batch_size=32)
    win_index = faiss.IndexFlatIP(doc_embs.shape[1])
    win_index.add(doc_embs)

    abstention_threshold, abstention_stats = calibrate_abstention_threshold(
        winner_adapter, win_index, dev_grounded, dev_negatives
    )

    # 7. Generate JSON and ZIP artifacts
    corpus_stats = {
        "pdf_total_pages": len(pages),
        "nonempty_indexed_pages": sum(1 for p in pages if p.word_count > 0),
        "ocr_missing_unobserved_pages": sum(1 for p in pages if p.ocr_source == "ocr_missing_unobserved"),
        "ocr_empty_illustration_only_pages": sum(1 for p in pages if p.ocr_source == "ocr_empty_illustration_only"),
        "total_corpus_words": sum(p.word_count for p in pages),
    }

    query_audit = {
        "total_queries": len(all_queries),
        "dev_grounded": len(dev_grounded),
        "dev_negatives": len(dev_negatives),
        "dev_single_page": sum(1 for q in dev_grounded if q.type == "single_page"),
        "dev_multi_page": sum(1 for q in dev_grounded if q.type == "multi_page"),
    }

    candidate_lock = {
        "winning_model_id": winner["canonical_model_id"],
        "resolved_model_path": winner["resolved_model_path"],
        "winning_chunk_strategy": winner["strategy"],
        "dimension": winner["dimension"],
        "dev_recall@5": winner["single_page_recall@5"],
        "dev_mrr@10": winner["single_page_mrr@10"],
        "dev_recall@5_ci_95": winner["recall@5_ci_95"],
        "abstention_threshold": abstention_threshold,
        "abstention_dev_stats": abstention_stats,
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
    """Executes the locked configuration against untouched final test queries and reserved negatives."""
    import faiss

    if not candidate_lock_path.exists():
        raise FileNotFoundError(f"Missing required candidate lock file: {candidate_lock_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    kb_dir = output_dir / "production_kb"
    kb_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Candidate Lock
    lock_data = json.loads(candidate_lock_path.read_text(encoding="utf-8"))
    winning_model_id = lock_data["winning_model_id"]
    resolved_model_path = lock_data.get("resolved_model_path")
    winning_strategy = lock_data["winning_chunk_strategy"]
    locked_threshold = float(lock_data.get("abstention_threshold", 0.5))

    # Ingest data
    pages = load_canonical_corpus(search_root)
    all_queries = load_retrieval_queries(search_root)

    test_grounded = [q for q in all_queries if q.split == "test" and q.type != "out_of_corpus"]
    test_negatives = [q for q in all_queries if q.split == "test" and q.type == "out_of_corpus"]

    if len(test_grounded) != 20:
        raise ValueError(f"Expected exactly 20 final test grounded queries, found {len(test_grounded)}")
    if len(test_negatives) != 5:
        raise ValueError(f"Expected exactly 5 final test negative queries, found {len(test_negatives)}")

    chunk_suites = build_chunk_suites(pages)
    winning_chunks = chunk_suites[winning_strategy]

    # Resolve model
    discovered = discover_candidate_models([search_root] if search_root else None)
    if winning_model_id in discovered:
        resolved_model_path = discovered[winning_model_id][1]
    elif not resolved_model_path or not Path(resolved_model_path).exists():
        raise FileNotFoundError(f"Cannot resolve locked model {winning_model_id}")

    adapter = EmbeddingModelAdapter(resolved_model_path, canonical_id=winning_model_id, device=device)

    # 2. Evaluate Locked Model on Untouched Test Set
    is_pc = (winning_strategy == "parent_child_128_512")
    final_metrics, test_logs = evaluate_retrieval_suite(
        adapter, winning_chunks, test_grounded, top_k=10, is_parent_child=is_pc
    )

    # 3. Evaluate Abstention on Reserved Test Negatives
    doc_texts = [c.text for c in winning_chunks]
    t0_enc = time.perf_counter()
    final_doc_embs = adapter.encode_documents(doc_texts, batch_size=32)
    doc_enc_time = time.perf_counter() - t0_enc

    exact_idx = faiss.IndexFlatIP(final_doc_embs.shape[1])
    exact_idx.add(final_doc_embs)

    abstention_results, negative_logs = evaluate_abstention_on_queries(
        adapter, exact_idx, test_grounded, test_negatives, threshold=locked_threshold
    )

    final_metrics["abstention_evaluation"] = abstention_results

    # 4. Benchmark FAISS Vector Architectures (FlatIP vs HNSW vs IVFFlat)
    q_texts = [q.question for q in test_grounded]
    final_q_embs = adapter.encode_queries(q_texts)
    _, gt_top10 = exact_idx.search(final_q_embs, 10)
    faiss_summary = benchmark_faiss_architectures(final_doc_embs, final_q_embs, gt_top10, num_iterations=10)

    # 5. Record 1 Genuine Successful Retrieval and 1 Genuine Worst Failure
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
        "figure_linked_example": {
            "page_id": "p0024",
            "figure_reference": "Plate I - Woodcut anatomy",
            "note": "Visual figure illustration metadata linked to page p0024 without claiming separate figure retrieval accuracy.",
        }
    }

    # 6. Persist Production Index & KB
    faiss.write_index(exact_idx, str(kb_dir / "index.faiss"))

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
        "total_corpus_words": sum(c.word_count for c in winning_chunks),
        "avg_words_per_chunk": float(np.mean([c.word_count for c in winning_chunks])),
        "index_size_bytes": (kb_dir / "index.faiss").stat().st_size,
        "build_time_seconds": round(doc_enc_time, 2),
        "query_latency_ms": final_metrics["single_query_latency_ms"],
    }

    selected_config = f"""# Production Knowledge Base Configuration (Locked in Stage 4)
doc_agent:
  index:
    chunk_strategy: "{winning_strategy}"
    embedding_model: "{winning_model_id}"
    dimension: {final_doc_embs.shape[1]}
    index_type: "IndexFlatIP"
    metric: "INNER_PRODUCT"
    abstention_threshold: {locked_threshold}
"""
    (output_dir / "selected-config.yaml").write_text(selected_config, encoding="utf-8")

    evidence_summary = f"""# Stage 4 Final Evidence Summary

## 1. Locked Production Stack
- **Embedding Model**: `{winning_model_id}` ({final_doc_embs.shape[1]}-d)
- **Chunking Strategy**: `{winning_strategy}` ({len(winning_chunks)} chunks, avg {index_stats['avg_words_per_chunk']:.1f} words)
- **Vector Search Index**: `IndexFlatIP` ({index_stats['index_size_bytes'] / 1024:.1f} KB, build time {index_stats['build_time_seconds']:.2f}s)
- **Abstention Threshold**: `{locked_threshold:.4f}`

## 2. Final Untouched Test Performance
- **Single-Page Recall@1**: {final_metrics['single_page_recall@1']:.4f}
- **Single-Page Recall@5**: {final_metrics['single_page_recall@5']:.4f} (95% CI: {final_metrics['recall@5_ci_95']})
- **Single-Page MRR@10**: {final_metrics['single_page_mrr@10']:.4f}
- **Multi-Page Coverage@10**: {final_metrics['multi_page_coverage@10']:.4f}
- **Multi-Page All-Found@10**: {final_metrics['multi_page_all_found@10']:.4f}

## 3. Negative Query Abstention Evaluation
- **Abstention Precision**: {abstention_results.get('abstention_precision', 1.0):.4f}
- **Abstention Recall**: {abstention_results.get('abstention_recall', 1.0):.4f}
- **Abstention F1**: {abstention_results.get('abstention_f1', 1.0):.4f}
- **Abstention Accuracy**: {abstention_results.get('abstention_accuracy', 1.0):.4f}
"""
    (output_dir / "evidence-summary.md").write_text(evidence_summary, encoding="utf-8")

    # Write JSON files
    (output_dir / "final-results.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    (output_dir / "abstention-evaluation.json").write_text(json.dumps(abstention_results, indent=2), encoding="utf-8")
    (output_dir / "faiss-comparison.json").write_text(json.dumps(faiss_summary, indent=2), encoding="utf-8")
    (output_dir / "index-statistics.json").write_text(json.dumps(index_stats, indent=2), encoding="utf-8")
    (output_dir / "retrieval-examples.json").write_text(json.dumps(retrieval_examples, indent=2), encoding="utf-8")

    with open(output_dir / "final-per-query.jsonl", "w", encoding="utf-8") as f:
        for log in test_logs + negative_logs:
            f.write(json.dumps(log) + "\n")

    # Pack ZIP
    zip_path = output_dir / "stage4-final-evidence.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in [
            "final-results.json",
            "abstention-evaluation.json",
            "final-per-query.jsonl",
            "faiss-comparison.json",
            "index-statistics.json",
            "retrieval-examples.json",
            "selected-config.yaml",
            "evidence-summary.md",
        ]:
            zf.write(output_dir / fname, arcname=fname)

    return zip_path


def run_stage4_unified_benchmark(
    output_dir: Path,
    device: str = "cpu",
    search_root: Path | None = None,
    require_local_models: bool = False,
) -> Path:
    """Executes the complete unified Stage-4 Benchmark (Phase 1 Dev Grid + Phase 2 Final Evidence).

    Emits one comprehensive downloadable archive: stage4-benchmark-results.zip.
    """
    import faiss

    output_dir.mkdir(parents=True, exist_ok=True)
    kb_dir = output_dir / "production_kb"
    kb_dir.mkdir(parents=True, exist_ok=True)

    t0_start = time.perf_counter()

    # 1. Ingest Corpus & Queries
    pages = load_canonical_corpus(search_root)
    all_queries = load_retrieval_queries(search_root)

    dev_grounded = [q for q in all_queries if q.split == "dev" and q.type != "out_of_corpus"]
    dev_negatives = [q for q in all_queries if q.split == "dev" and q.type == "out_of_corpus"]
    test_grounded = [q for q in all_queries if q.split == "test" and q.type != "out_of_corpus"]
    test_negatives = [q for q in all_queries if q.split == "test" and q.type == "out_of_corpus"]

    if len(dev_grounded) != 80:
        raise ValueError(f"Expected exactly 80 dev grounded queries, found {len(dev_grounded)}")
    if len(dev_negatives) != 5:
        raise ValueError(f"Expected exactly 5 dev negative queries, found {len(dev_negatives)}")
    if len(test_grounded) != 20:
        raise ValueError(f"Expected exactly 20 test grounded queries, found {len(test_grounded)}")
    if len(test_negatives) != 5:
        raise ValueError(f"Expected exactly 5 test negative queries, found {len(test_negatives)}")

    # 2. Build 5 Chunking Suites
    chunk_suites = build_chunk_suites(pages)
    if len(chunk_suites) != 5:
        raise ValueError(f"Expected exactly 5 chunking suites, found {len(chunk_suites)}")

    # 3. Discover Candidate Models
    # Benchmark data and model snapshots are separate Kaggle inputs. The explicit
    # search_root selects corpus/query data; model discovery keeps its standard
    # offline roots (notably /kaggle/input) so attached model assets remain visible.
    discovered = discover_candidate_models(require_local=require_local_models)
    if len(discovered) != 5:
        raise ValueError(f"Expected exactly 5 candidate models, found {len(discovered)}")

    # =========================================================================
    # PHASE 1: 5x5 DEV GRID EVALUATION (Resumable)
    # =========================================================================
    lock_file = output_dir / "candidate-lock.json"
    grid_file = output_dir / "dev-grid-results.json"
    dev_logs_file = output_dir / "dev-per-query.jsonl"

    if lock_file.exists() and grid_file.exists() and dev_logs_file.exists():
        print(f"[RESUME] Found existing Phase 1 artifacts in {output_dir}. Loading dev lock...")
        lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
        grid_results = json.loads(grid_file.read_text(encoding="utf-8"))
        all_dev_logs = [json.loads(line) for line in dev_logs_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        winner = [r for r in grid_results if r["canonical_model_id"] == lock_data["winning_model_id"] and r["strategy"] == lock_data["winning_chunk_strategy"]][0]
    else:
        grid_results = []
        all_dev_logs = []

        print("Starting Phase 1: 5x5 Dev Grid evaluation across all 25 cells...")
        for c_id, (_, m_path) in discovered.items():
            adapter = EmbeddingModelAdapter(m_path, canonical_id=c_id, device=device)
            for s_name, c_list in chunk_suites.items():
                is_pc = (s_name == "parent_child_128_512")
                metrics, logs = evaluate_retrieval_suite(
                    adapter, c_list, dev_grounded, top_k=10, is_parent_child=is_pc
                )
                grid_results.append(metrics)
                all_dev_logs.extend(logs)

        if len(grid_results) != 25:
            raise RuntimeError(f"Expected exactly 25 grid results, but got {len(grid_results)}. Incomplete run!")

        # Multi-criterion ranking on dev grounded set
        grid_results.sort(
            key=lambda x: (
                x["single_page_recall@5"],
                x["single_page_mrr@10"],
                x["multi_page_coverage@10"],
                x["single_page_span_containment@5"],
                -x["single_query_latency_ms"],
            ),
            reverse=True,
        )
        winner = grid_results[0]

        # Calibrate abstention threshold on dev negatives
        winner_adapter = EmbeddingModelAdapter(winner["resolved_model_path"], canonical_id=winner["canonical_model_id"], device=device)
        winner_chunks = chunk_suites[winner["strategy"]]
        doc_texts = [c.text for c in winner_chunks]
        doc_embs = winner_adapter.encode_documents(doc_texts, batch_size=32)
        win_index = faiss.IndexFlatIP(doc_embs.shape[1])
        win_index.add(doc_embs)

        abstention_threshold, abstention_stats = calibrate_abstention_threshold(
            winner_adapter, win_index, dev_grounded, dev_negatives
        )

        lock_data = {
            "winning_model_id": winner["canonical_model_id"],
            "resolved_model_path": winner["resolved_model_path"],
            "winning_chunk_strategy": winner["strategy"],
            "dimension": winner["dimension"],
            "dev_recall@5": winner["single_page_recall@5"],
            "dev_mrr@10": winner["single_page_mrr@10"],
            "dev_recall@5_ci_95": winner["recall@5_ci_95"],
            "abstention_threshold": abstention_threshold,
            "abstention_dev_stats": abstention_stats,
        }

        lock_file.write_text(json.dumps(lock_data, indent=2), encoding="utf-8")
        grid_file.write_text(json.dumps(grid_results, indent=2), encoding="utf-8")
        with open(dev_logs_file, "w", encoding="utf-8") as f:
            for log in all_dev_logs:
                f.write(json.dumps(log) + "\n")

    # =========================================================================
    # PHASE 2: FINAL EVALUATION OF LOCKED WINNER (Untouched Test & Reserved Negatives)
    # =========================================================================
    winning_model_id = lock_data["winning_model_id"]
    resolved_model_path = lock_data["resolved_model_path"]
    winning_strategy = lock_data["winning_chunk_strategy"]
    locked_threshold = float(lock_data["abstention_threshold"])

    print(f"\nStarting Phase 2: Evaluating locked winner ({winning_model_id} + {winning_strategy}) on untouched test set...")
    winning_chunks = chunk_suites[winning_strategy]
    adapter = EmbeddingModelAdapter(resolved_model_path, canonical_id=winning_model_id, device=device)

    is_pc = (winning_strategy == "parent_child_128_512")
    final_metrics, test_logs = evaluate_retrieval_suite(
        adapter, winning_chunks, test_grounded, top_k=10, is_parent_child=is_pc
    )

    doc_texts = [c.text for c in winning_chunks]
    t0_enc = time.perf_counter()
    final_doc_embs = adapter.encode_documents(doc_texts, batch_size=32)
    doc_enc_time = time.perf_counter() - t0_enc

    exact_idx = faiss.IndexFlatIP(final_doc_embs.shape[1])
    exact_idx.add(final_doc_embs)

    # Evaluate abstention on reserved test negatives
    abstention_results, negative_logs = evaluate_abstention_on_queries(
        adapter, exact_idx, test_grounded, test_negatives, threshold=locked_threshold
    )
    final_metrics["abstention_evaluation"] = abstention_results

    # FAISS comparison
    q_texts = [q.question for q in test_grounded]
    final_q_embs = adapter.encode_queries(q_texts)
    _, gt_top10 = exact_idx.search(final_q_embs, 10)
    faiss_summary = benchmark_faiss_architectures(final_doc_embs, final_q_embs, gt_top10, num_iterations=10)

    # Retrieval examples
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
        "figure_linked_example": {
            "page_id": "p0024",
            "figure_reference": "Plate I - Woodcut anatomy",
            "note": "Visual figure illustration metadata linked to page p0024 without claiming separate figure retrieval accuracy.",
        },
    }

    # Persist Production KB files
    faiss.write_index(exact_idx, str(kb_dir / "index.faiss"))

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
        "total_corpus_words": sum(c.word_count for c in winning_chunks),
        "avg_words_per_chunk": float(np.mean([c.word_count for c in winning_chunks])),
        "index_size_bytes": (kb_dir / "index.faiss").stat().st_size,
        "build_time_seconds": round(doc_enc_time, 2),
        "query_latency_ms": final_metrics["single_query_latency_ms"],
    }

    selected_config = f"""# Production Knowledge Base Configuration (Locked in Stage 4)
doc_agent:
  index:
    chunk_strategy: "{winning_strategy}"
    embedding_model: "{winning_model_id}"
    dimension: {final_doc_embs.shape[1]}
    index_type: "IndexFlatIP"
    metric: "INNER_PRODUCT"
    abstention_threshold: {locked_threshold}
"""
    (output_dir / "selected-config.yaml").write_text(selected_config, encoding="utf-8")

    evidence_summary = f"""# Stage 4 Final Evidence Summary

## 1. Locked Production Stack
- **Embedding Model**: `{winning_model_id}` ({final_doc_embs.shape[1]}-d)
- **Chunking Strategy**: `{winning_strategy}` ({len(winning_chunks)} chunks, avg {index_stats['avg_words_per_chunk']:.1f} words)
- **Vector Search Index**: `IndexFlatIP` ({index_stats['index_size_bytes'] / 1024:.1f} KB, build time {index_stats['build_time_seconds']:.2f}s)
- **Abstention Threshold**: `{locked_threshold:.4f}`

## 2. Final Untouched Test Performance
- **Single-Page Recall@1**: {final_metrics['single_page_recall@1']:.4f}
- **Single-Page Recall@5**: {final_metrics['single_page_recall@5']:.4f} (95% CI: {final_metrics['recall@5_ci_95']})
- **Single-Page MRR@10**: {final_metrics['single_page_mrr@10']:.4f}
- **Multi-Page Coverage@10**: {final_metrics['multi_page_coverage@10']:.4f}
- **Multi-Page All-Found@10**: {final_metrics['multi_page_all_found@10']:.4f}

## 3. Negative Query Abstention Evaluation
- **Abstention Precision**: {abstention_results.get('abstention_precision', 1.0):.4f}
- **Abstention Recall**: {abstention_results.get('abstention_recall', 1.0):.4f}
- **Abstention F1**: {abstention_results.get('abstention_f1', 1.0):.4f}
- **Abstention Accuracy**: {abstention_results.get('abstention_accuracy', 1.0):.4f}
"""
    (output_dir / "evidence-summary.md").write_text(evidence_summary, encoding="utf-8")

    run_env = {
        "python_version": sys.version,
        "device": device,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_runtime_seconds": round(time.perf_counter() - t0_start, 2),
    }

    manifest = {
        "stage": "stage4-unified-benchmark",
        "version": "2.0-reproduced",
        "grid_cells_evaluated": len(grid_results),
        "timestamp": run_env["timestamp"],
    }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "run-environment.json").write_text(json.dumps(run_env, indent=2), encoding="utf-8")
    (output_dir / "final-results.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    (output_dir / "abstention-evaluation.json").write_text(json.dumps(abstention_results, indent=2), encoding="utf-8")
    (output_dir / "faiss-comparison.json").write_text(json.dumps(faiss_summary, indent=2), encoding="utf-8")
    (output_dir / "index-statistics.json").write_text(json.dumps(index_stats, indent=2), encoding="utf-8")
    (output_dir / "retrieval-examples.json").write_text(json.dumps(retrieval_examples, indent=2), encoding="utf-8")

    with open(output_dir / "final-per-query.jsonl", "w", encoding="utf-8") as f:
        for log in test_logs + negative_logs:
            f.write(json.dumps(log) + "\n")

    # =========================================================================
    # PHASE 3: EMIT SINGLE DOWNLOADABLE ZIP ARCHIVE
    # =========================================================================
    zip_path = output_dir / "stage4-benchmark-results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in [
            "candidate-lock.json",
            "dev-grid-results.json",
            "dev-per-query.jsonl",
            "final-results.json",
            "final-per-query.jsonl",
            "abstention-evaluation.json",
            "faiss-comparison.json",
            "index-statistics.json",
            "retrieval-examples.json",
            "selected-config.yaml",
            "evidence-summary.md",
            "manifest.json",
            "run-environment.json",
        ]:
            zf.write(output_dir / fname, arcname=fname)

        # Include production KB directory inside ZIP
        zf.write(kb_dir / "index.faiss", arcname="production_kb/index.faiss")
        zf.write(kb_dir / "chunks.jsonl", arcname="production_kb/chunks.jsonl")

    return zip_path
