"""Stage 1: render the configured corpus PDF into page images."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import fitz

from ..contracts import Page


def _ingest_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return loader settings from the fixed top-level configuration."""
    ingest = cfg.get("ingest")
    if not isinstance(ingest, dict):
        raise ValueError("config must contain an 'ingest' mapping")
    return ingest


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _page_limit(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("page_limit must be a positive integer or null")
    return value


def _jpeg_quality(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ValueError("quality must be an integer from 1 to 100")
    return value


def _atomic_render(page: fitz.Page, target: Path, dpi: int, quality: int) -> None:
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.stem}.", suffix=".jpg")
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        pixmap = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
        pixmap.save(str(temporary_path), output="jpeg", jpg_quality=quality)
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_pages(cfg: dict[str, Any]) -> list[Page]:
    """Render the configured PDF one page at a time into deterministic ``Page`` contracts.

    RGB JPEGs are reused only when their modification time is at least as new as the source PDF.
    The effective output directory encodes render settings, so setting changes cannot reuse stale
    files and interrupted runs can resume without a manifest. Page identifiers are one-indexed.
    """
    ingest = _ingest_config(cfg)
    source_value = ingest.get("source_pdf")
    doc_id = ingest.get("doc_id")
    output_value = ingest.get("output_dir")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("ingest.source_pdf must be a non-empty path")
    if not isinstance(doc_id, str) or not doc_id:
        raise ValueError("ingest.doc_id must be a non-empty string")
    if not isinstance(output_value, str) or not output_value:
        raise ValueError("ingest.output_dir must be a non-empty path")

    dpi = _positive_integer(ingest.get("dpi"), "dpi")
    page_limit = _page_limit(ingest.get("page_limit"))
    image_format = ingest.get("format", "jpeg")
    if image_format != "jpeg":
        raise ValueError("format must be 'jpeg'")
    quality = _jpeg_quality(ingest.get("quality", 90))
    source = Path(source_value)
    if not source.is_file():
        raise FileNotFoundError(
            f"Corpus PDF not found: {source}. Run scripts/get_data.sh to download it."
        )

    output_dir = Path(output_value) / doc_id / f"{dpi}dpi-{image_format}-q{quality}"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_mtime_ns = source.stat().st_mtime_ns
    pages: list[Page] = []

    with fitz.open(str(source)) as document:
        count = len(document) if page_limit is None else min(page_limit, len(document))
        for index in range(count):
            page_id = f"p{index + 1:04d}"
            image_path = output_dir / f"{page_id}.jpg"
            if not image_path.is_file() or image_path.stat().st_mtime_ns < source_mtime_ns:
                page = document.load_page(index)
                _atomic_render(page, image_path, dpi, quality)
            pages.append(Page(id=page_id, image_path=str(image_path), doc_id=doc_id))

    return pages
