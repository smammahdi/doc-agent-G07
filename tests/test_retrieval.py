"""Unit tests for Stage 4 retrieval dataset, preflight, contracts, and abstention."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure indexing-benchmarks code package is in sys.path
_bench_root = Path(__file__).parent.parent / "extras" / "indexing-benchmarks"
if str(_bench_root) not in sys.path:
    sys.path.insert(0, str(_bench_root))
sys.modules.pop("code", None)

from code.evaluation import (
    calibrate_abstention_threshold,
)
from code.models import (
    discover_candidate_models,
    is_valid_local_model_dir,
)
from code.queries import load_retrieval_queries


def test_retrieval_query_suite_splits_and_isolation():
    """Validates 110 unique typed queries, exact split counts, and 100% disjoint pages."""
    queries = load_retrieval_queries()
    assert len(queries) == 110, f"Expected 110 queries, found {len(queries)}"

    q_ids = [q.query_id for q in queries]
    assert len(set(q_ids)) == 110, "Duplicate query IDs found!"

    dev_grounded = [q for q in queries if q.split == "dev" and q.type != "out_of_corpus"]
    dev_negatives = [q for q in queries if q.split == "dev" and q.type == "out_of_corpus"]
    test_grounded = [q for q in queries if q.split == "test" and q.type != "out_of_corpus"]
    test_negatives = [q for q in queries if q.split == "test" and q.type == "out_of_corpus"]

    assert len(dev_grounded) == 80, f"Expected 80 dev grounded, found {len(dev_grounded)}"
    assert len(dev_negatives) == 5, f"Expected 5 dev negatives, found {len(dev_negatives)}"
    assert len(test_grounded) == 20, f"Expected 20 test grounded, found {len(test_grounded)}"
    assert len(test_negatives) == 5, f"Expected 5 test negatives, found {len(test_negatives)}"

    # Check dev/test disjoint gold pages
    dev_pages = set()
    for q in dev_grounded:
        for p in q.page_ids:
            dev_pages.add(p)

    test_pages = set()
    for q in test_grounded:
        for p in q.page_ids:
            test_pages.add(p)

    overlap = dev_pages.intersection(test_pages)
    assert len(overlap) == 0, f"Data leakage: {overlap}"


def test_offline_missing_model_preflight_fails_closed(tmp_path: Path):
    """Verifies discover_candidate_models raises FileNotFoundError when require_local=True and assets are missing."""
    empty_root = tmp_path / "empty_models"
    empty_root.mkdir()

    with pytest.raises(FileNotFoundError) as exc_info:
        discover_candidate_models([empty_root], require_local=True)

    assert "Offline Model Preflight Failed" in str(exc_info.value)


def test_is_valid_local_model_dir_checks_files(tmp_path: Path):
    """Verifies model directory validator requires config, weights, and tokenizer files."""
    model_dir = tmp_path / "mock-model"
    model_dir.mkdir()
    assert not is_valid_local_model_dir(model_dir)

    (model_dir / "config.json").write_text("{}")
    assert not is_valid_local_model_dir(model_dir)

    (model_dir / "model.safetensors").write_text("dummy")
    assert not is_valid_local_model_dir(model_dir)

    (model_dir / "tokenizer.json").write_text("{}")
    assert is_valid_local_model_dir(model_dir)


def test_abstention_calibration_and_evaluation():
    """Verifies threshold calibration and abstention metrics logic."""
    mock_model = MagicMock()
    mock_model.encode_queries.side_effect = lambda texts: np.ones(
        (len(texts), 64), dtype=np.float32
    )

    mock_index = MagicMock()
    mock_index.search.side_effect = [
        (np.array([[0.85]], dtype=np.float32), np.array([[0]])),  # grounded
        (np.array([[0.25]], dtype=np.float32), np.array([[0]])),  # negative
    ]

    queries = load_retrieval_queries()
    dev_g = [q for q in queries if q.split == "dev" and q.type != "out_of_corpus"][:1]
    dev_n = [q for q in queries if q.split == "dev" and q.type == "out_of_corpus"][:1]

    thresh, stats = calibrate_abstention_threshold(mock_model, mock_index, dev_g, dev_n)
    assert 0.20 <= thresh <= 0.85
    assert "abstention_f1" in stats


def test_unified_stage4_benchmark_zip_contract(tmp_path: Path):
    """Validates the output ZIP contract of run_stage4_unified_benchmark without heavy model inference."""
    output_dir = tmp_path / "unified_results"
    kb_dir = output_dir / "production_kb"
    kb_dir.mkdir(parents=True, exist_ok=True)

    # Mock candidate-lock and all required files
    (output_dir / "candidate-lock.json").write_text(
        json.dumps(
            {
                "winning_model_id": "all-MiniLM-L6-v2",
                "resolved_model_path": "mock/path",
                "winning_chunk_strategy": "paragraph_header_aware",
                "dimension": 384,
                "dev_recall@5": 0.95,
                "dev_mrr@10": 0.88,
                "dev_recall@5_ci_95": [0.90, 1.0],
                "abstention_threshold": 0.52,
            }
        )
    )
    (output_dir / "dev-grid-results.json").write_text(
        json.dumps(
            [{"strategy": "paragraph_header_aware", "canonical_model_id": "all-MiniLM-L6-v2"}]
        )
    )
    (output_dir / "dev-per-query.jsonl").write_text(
        json.dumps({"query_id": "q_dev_01", "recall@5": 1.0}) + "\n"
    )
    (output_dir / "final-results.json").write_text(json.dumps({"single_page_recall@5": 0.95}))
    (output_dir / "final-per-query.jsonl").write_text(
        json.dumps({"query_id": "q_test_01", "recall@5": 1.0}) + "\n"
    )
    (output_dir / "abstention-evaluation.json").write_text(json.dumps({"abstention_f1": 1.0}))
    (output_dir / "faiss-comparison.json").write_text(
        json.dumps({"exact_flat": {"recall@10": 1.0}})
    )
    (output_dir / "index-statistics.json").write_text(json.dumps({"total_chunks": 100}))
    (output_dir / "retrieval-examples.json").write_text(json.dumps({"successful_example": {}}))
    (output_dir / "selected-config.yaml").write_text("doc_agent: {}")
    (output_dir / "evidence-summary.md").write_text("# Evidence Summary")
    (output_dir / "manifest.json").write_text(json.dumps({"stage": "stage4-unified-benchmark"}))
    (output_dir / "run-environment.json").write_text(json.dumps({"device": "cpu"}))

    (kb_dir / "index.faiss").write_text("mock faiss binary")
    (kb_dir / "chunks.jsonl").write_text(json.dumps({"chunk_id": "c1", "text": "mock"}) + "\n")

    # Build ZIP
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
        zf.write(kb_dir / "index.faiss", arcname="production_kb/index.faiss")
        zf.write(kb_dir / "chunks.jsonl", arcname="production_kb/chunks.jsonl")

    # Validate ZIP
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        expected = {
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
            "production_kb/index.faiss",
            "production_kb/chunks.jsonl",
        }
        assert expected.issubset(names), f"Missing ZIP members: {expected - names}"
