"""Stage-4 Indexing, Chunking, and Embedding Benchmark Package."""

from .chunking import (
    BenchmarkChunk,
    build_chunk_suites,
    fixed_window_word_chunking,
    hierarchical_parent_child_chunking,
    paragraph_header_aware_chunking,
)
from .corpus import CanonicalPage, load_canonical_corpus
from .evaluation import (
    calculate_bootstrap_ci,
    calibrate_abstention_threshold,
    evaluate_abstention_on_queries,
    evaluate_retrieval_suite,
)
from .faiss_benchmark import benchmark_faiss_architectures
from .models import EmbeddingModelAdapter, discover_candidate_models
from .queries import RetrievalQuery, load_retrieval_queries
from .runner import run_stage4_dev_grid, run_stage4_final_evidence

__all__ = [
    "CanonicalPage",
    "load_canonical_corpus",
    "RetrievalQuery",
    "load_retrieval_queries",
    "BenchmarkChunk",
    "fixed_window_word_chunking",
    "paragraph_header_aware_chunking",
    "hierarchical_parent_child_chunking",
    "build_chunk_suites",
    "EmbeddingModelAdapter",
    "discover_candidate_models",
    "evaluate_retrieval_suite",
    "calculate_bootstrap_ci",
    "calibrate_abstention_threshold",
    "evaluate_abstention_on_queries",
    "benchmark_faiss_architectures",
    "run_stage4_dev_grid",
    "run_stage4_final_evidence",
]
