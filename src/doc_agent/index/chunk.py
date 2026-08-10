"""Stage 4: split OCR text into deterministic, provenance-preserving chunks."""

from __future__ import annotations

from typing import Any

from ..contracts import Chunk


def _settings(cfg: dict[str, Any]) -> tuple[int, int]:
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a mapping")
    index_cfg = cfg.get("index", {})
    if not isinstance(index_cfg, dict):
        raise ValueError("cfg['index'] must be a mapping")
    size = index_cfg.get("chunk_tokens", 256)
    overlap = index_cfg.get("overlap", 32)
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("chunk_tokens must be a positive integer")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk_tokens")
    return size, overlap


def split(chunks: list[Chunk], cfg: dict[str, Any]) -> list[Chunk]:
    """Split OCR chunks by whitespace tokens while retaining page provenance.

    The OCR stage already produces page-scoped ``Chunk`` objects. This function only
    changes the text span and deterministic ID; it never merges pages or discards
    the source document ID. Whitespace tokenisation is intentional for the first
    reproducible baseline and can be replaced behind this same function later.
    """
    size, overlap = _settings(cfg)
    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list of Chunk contracts")

    output: list[Chunk] = []
    step = size - overlap
    for source in chunks:
        if not isinstance(source, Chunk):
            raise TypeError("chunks must contain only Chunk contracts")
        words = source.text.split()
        if not words:
            continue
        n = 0
        for start in range(0, len(words), step):
            part = words[start : start + size]
            if not part:
                break
            output.append(
                Chunk(
                    id=f"{source.id}-c{n:04d}",
                    doc_id=source.doc_id,
                    text=" ".join(part),
                    page_ids=list(source.page_ids),
                )
            )
            n += 1
            if start + size >= len(words):
                break
    return output
