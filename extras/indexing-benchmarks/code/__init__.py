"""Stage-4 Indexing, Chunking, and Embedding Benchmark Package."""

from .corpus import CanonicalPage, load_canonical_corpus
from .queries import RetrievalQuery, load_retrieval_queries
from .chunking import (
    BenchmarkChunk,
    fixed_window_word_chunking,
    paragraph_header_aware_chunking,
    hierarchical_parent_child_chunking,
    build_chunk_suites,
)
from .models import EmbeddingModelAdapter, discover_candidate_models
from .evaluation import evaluate_retrieval_suite, calculate_bootstrap_ci
from .faiss_benchmark import benchmark_faiss_architectures

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
    "benchmark_faiss_architectures",
]
