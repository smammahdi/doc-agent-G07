"""Unit tests for Stage 4 index chunking, embedding, and FAISS FlatIP persistence."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from doc_agent.contracts import Chunk
from doc_agent.index import chunk, embed, store


def test_word_chunking_boundaries():
    """Validates fixed-size whitespace word chunking and step overlap logic."""
    raw_text = " ".join([f"word_{i}" for i in range(300)])
    source_chunk = Chunk(id="doc-p0001", doc_id="doc", text=raw_text, page_ids=["p0001"])

    cfg = {"index": {"chunk_words": 128, "overlap": 16}}
    chunks = chunk.split([source_chunk], cfg)

    assert len(chunks) == 3
    assert chunks[0].id == "doc-p0001-c0000"
    assert chunks[1].id == "doc-p0001-c0001"
    assert chunks[2].id == "doc-p0001-c0002"
    assert len(chunks[0].text.split()) == 128
    assert len(chunks[1].text.split()) == 128
    assert len(chunks[2].text.split()) == 76
    assert chunks[0].page_ids == ["p0001"]


def test_chunking_invalid_settings():
    """Validates that invalid chunking parameters raise appropriate errors."""
    source_chunk = Chunk(id="c1", doc_id="d1", text="some text", page_ids=["p1"])

    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk.split([source_chunk], {"index": {"chunk_words": 50, "overlap": 50}})

    with pytest.raises(ValueError, match="positive integer"):
        chunk.split([source_chunk], {"index": {"chunk_words": -10, "overlap": 5}})


def test_qwen_query_and_document_prefix():
    """Verifies that Qwen instruction prefixes are correctly assigned."""
    qwen_prefix = embed.get_query_prefix("Qwen/Qwen3-Embedding-0.6B")
    assert "Instruct:" in qwen_prefix
    assert "Query:" in qwen_prefix

    doc_prefix = embed.get_doc_prefix("Qwen/Qwen3-Embedding-0.6B")
    assert doc_prefix == ""

    minilm_prefix = embed.get_query_prefix("sentence-transformers/all-MiniLM-L6-v2")
    assert minilm_prefix == ""


def test_embedding_normalization_and_shape():
    """Validates vector normalization and shape matching."""
    chunks = [
        Chunk(id="c1", doc_id="d1", text="First passage about anatomy.", page_ids=["p1"]),
        Chunk(id="c2", doc_id="d1", text="Second passage about physiology.", page_ids=["p1"]),
    ]
    cfg = {
        "embed": {
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "revision": "test-revision",
            "dim": 1024,
            "batch_size": 32,
        }
    }

    mock_vectors = np.random.randn(2, 1024).astype(np.float32)
    with patch("doc_agent.index.embed._encode_texts", return_value=mock_vectors) as mock_encode:
        vectors = embed.encode(chunks, cfg)
        assert vectors.shape == (2, 1024)
        assert mock_encode.called


def test_store_build_and_load_flat_ip(tmp_path: Path):
    """Verifies FAISS IndexFlatIP building, atomic persistence, and exact loading."""
    chunks = [
        Chunk(id="c1", doc_id="d1", text="Text 1", page_ids=["p0001"]),
        Chunk(id="c2", doc_id="d1", text="Text 2", page_ids=["p0002"]),
    ]
    dim = 1024
    raw_vectors = np.random.randn(2, dim).astype(np.float32)

    cfg = {
        "index": {
            "type": "faiss:flat_ip",
            "path": str(tmp_path / "index_out"),
            "chunk_words": 128,
            "overlap": 16,
        }
    }

    store.build(chunks, raw_vectors, cfg)

    out_dir = tmp_path / "index_out"
    assert (out_dir / "index.faiss").is_file()
    assert (out_dir / "chunks.jsonl").is_file()
    assert (out_dir / "metadata.json").is_file()

    meta = json.loads((out_dir / "metadata.json").read_text())
    assert meta["index_type"] == "faiss:flat_ip"
    assert meta["dimension"] == 1024
    assert meta["count"] == 2

    idx, loaded_chunks, loaded_meta = store.load(cfg)
    assert idx.ntotal == 2
    assert idx.d == 1024
    assert len(loaded_chunks) == 2
    assert loaded_chunks[0].id == "c1"
    assert loaded_chunks[1].page_ids == ["p0002"]


def test_store_load_count_mismatch_fails_closed(tmp_path: Path):
    """Verifies store.load raises ValueError when metadata and index counts disagree."""
    chunks = [Chunk(id="c1", doc_id="d1", text="Text 1", page_ids=["p0001"])]
    cfg = {"index": {"type": "faiss:flat_ip", "path": str(tmp_path / "mismatch_idx")}}

    store.build(chunks, np.random.randn(1, 1024).astype(np.float32), cfg)

    meta_path = tmp_path / "mismatch_idx" / "metadata.json"
    meta_path.write_text(
        json.dumps({"index_type": "faiss:flat_ip", "dimension": 1024, "count": 999})
    )

    with pytest.raises(ValueError, match="disagree"):
        store.load(cfg)
