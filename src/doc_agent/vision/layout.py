"""Stage 2: a deterministic, dependency-light layout baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import Page, Region


def _config(cfg: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        raise TypeError("layout config must be a mapping")
    options = cfg.get("layout", {})
    if not isinstance(options, dict):
        raise ValueError("cfg['layout'] must be a mapping")
    mode = options.get("mode", "projection")
    if mode not in {"projection", "full_page"}:
        raise ValueError("layout mode must be 'projection' or 'full_page'")
    min_ink_ratio = options.get("min_ink_ratio", 0.005)
    if isinstance(min_ink_ratio, bool) or not isinstance(min_ink_ratio, (int, float)):
        raise ValueError("layout min_ink_ratio must be a number in (0, 1)")
    if not 0 < min_ink_ratio < 1:
        raise ValueError("layout min_ink_ratio must be a number in (0, 1)")
    max_row_gap = options.get("max_row_gap", 8)
    if isinstance(max_row_gap, bool) or not isinstance(max_row_gap, int) or max_row_gap < 0:
        raise ValueError("layout max_row_gap must be a non-negative integer")
    padding = options.get("padding", 4)
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ValueError("layout padding must be a non-negative integer")
    return {
        "mode": mode,
        "min_ink_ratio": float(min_ink_ratio),
        "max_row_gap": max_row_gap,
        "padding": padding,
    }


def _image_size(path: Path) -> tuple[int, int]:
    try:
        import fitz

        pixmap = fitz.Pixmap(str(path))
        return pixmap.width, pixmap.height
    except Exception as error:
        raise RuntimeError(f"cannot read page image {path}: {error}") from error


def _fallback(page: Page) -> Region:
    width, height = _image_size(Path(page.image_path))
    return Region(page_id=page.id, bbox=(0, 0, width, height), kind="text")


def _projection(page: Page, options: dict[str, Any]) -> list[Region]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return [_fallback(page)]

    path = Path(page.image_path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"cannot read page image {path}")
    height, width = image.shape[:2]
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_ink = np.count_nonzero(binary, axis=1)
    min_ink = max(2, int(round(width * options["min_ink_ratio"])))
    active = row_ink >= min_ink

    runs: list[tuple[int, int]] = []
    start: int | None = None
    gap = 0
    for index, is_active in enumerate(active):
        if is_active:
            if start is None:
                start = index
            gap = 0
        elif start is not None:
            gap += 1
            if gap > options["max_row_gap"]:
                runs.append((start, index - gap + 1))
                start = None
                gap = 0
    if start is not None:
        runs.append((start, height))

    regions: list[Region] = []
    pad = options["padding"]
    for y0, y1 in runs:
        columns = np.count_nonzero(binary[y0:y1], axis=0)
        active_columns = np.flatnonzero(columns)
        if active_columns.size == 0:
            continue
        x0 = max(0, int(active_columns[0]) - pad)
        x1 = min(width, int(active_columns[-1]) + pad + 1)
        y_start = max(0, y0 - pad)
        y_end = min(height, y1 + pad)
        if x1 > x0 and y_end > y_start:
            regions.append(Region(page_id=page.id, bbox=(x0, y_start, x1, y_end), kind="text"))
    return regions or [_fallback(page)]


def detect(pages: list[Page], cfg: dict[str, Any]) -> list[Region]:
    """Return text proposals using projection mode or a truthful full-page fallback."""
    options = _config(cfg)
    if not isinstance(pages, list):
        raise TypeError("pages must be a list of Page contracts")
    regions: list[Region] = []
    for page in pages:
        if not isinstance(page, Page):
            raise TypeError("pages must contain only Page contracts")
        image = Path(page.image_path)
        if not image.is_file():
            raise FileNotFoundError(f"page {page.id} image does not exist: {image}")
        page_regions = (
            [_fallback(page)] if options["mode"] == "full_page" else _projection(page, options)
        )
        regions.extend(page_regions)
    return regions
