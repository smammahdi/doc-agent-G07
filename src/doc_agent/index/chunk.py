"""Stage 4: split OCR text into deterministic, provenance-preserving chunks."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

from ..contracts import Chunk

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PAGE_DELIM = re.compile(r"<!--\s*book\s+p(\d+)\s*-->")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)]+\.webp)\)")
_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


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


def _clean_markdown(raw: str) -> str:
    """Return plain text suitable for dense-vector embedding.

    Images → keep the alt-text description (Chandra's rich AI-generated captions)
    so the embedding learns the visual content of each figure.
    All remaining HTML tags (b, i, hr, sup, …) are stripped.
    """
    # Replace image refs with their alt text only (keeps figure semantics in embedding)
    text = _MD_IMAGE.sub(lambda m: m.group(1), raw)
    # Decode HTML entities and strip remaining tags
    text = html.unescape(_HTML_TAG.sub(" ", text))
    # Collapse whitespace
    return _WHITESPACE.sub(" ", text).strip()


def _extract_images(raw: str, page_id: str) -> list[dict[str, str]]:
    """Return [{page_id, caption, webp}] for every image on this page."""
    return [
        {"page_id": page_id, "caption": m.group(1).strip(), "webp": m.group(2).strip()}
        for m in _MD_IMAGE.finditer(raw)
    ]


# ---------------------------------------------------------------------------
# Chandra pages.md ingestion
# ---------------------------------------------------------------------------


def load_from_pages_markdown(
    pages_md_path: Path,
    doc_id: str,
) -> tuple[list[Chunk], dict[str, list[dict[str, str]]]]:
    """Parse ``chandra/pages.md`` into one ``Chunk`` per book page.

    Returns
    -------
    page_chunks : list[Chunk]
        One Chunk per non-empty page. ``text`` is clean embedding-ready text
        (image alt-texts included so figure captions are searchable).
        ``page_ids`` is ``["p{N:04d}"]`` for provenance and citation.
    image_index : dict[str, list[dict]]
        ``{page_id: [{caption, webp}, ...]}`` for every page that has images.
        Used by the demo notebook and ``read_page`` tool to render figures.
    """
    content = pages_md_path.read_text(encoding="utf-8")
    parts = _PAGE_DELIM.split(content)
    # parts layout: [pre-text, page_num, page_text, page_num, page_text, ...]

    page_chunks: list[Chunk] = []
    image_index: dict[str, list[dict[str, str]]] = {}

    for i in range(1, len(parts), 2):
        page_num = int(parts[i])
        raw_text = parts[i + 1] if i + 1 < len(parts) else ""
        page_id = f"p{page_num:04d}"

        # Build image index (for display, not embedding)
        imgs = _extract_images(raw_text, page_id)
        if imgs:
            image_index[page_id] = imgs

        # Clean text for embedding
        clean = _clean_markdown(raw_text)
        if not clean:
            continue

        page_chunks.append(
            Chunk(
                id=f"{doc_id}-{page_id}",
                doc_id=doc_id,
                text=clean,
                page_ids=[page_id],
            )
        )

    return page_chunks, image_index


def build_image_index(pages_md_path: Path) -> dict[str, list[dict[str, str]]]:
    """Return ``{page_id: [{caption, webp}, ...]}`` for all illustrated pages.

    Convenience wrapper used by the demo notebook independently of the chunker.
    """
    _, img_idx = load_from_pages_markdown(pages_md_path, doc_id="")
    return img_idx


# ---------------------------------------------------------------------------
# Token-level sliding-window chunking
# ---------------------------------------------------------------------------


def split(chunks: list[Chunk], cfg: dict[str, Any]) -> list[Chunk]:
    """Split OCR chunks by whitespace tokens while retaining page provenance."""
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
