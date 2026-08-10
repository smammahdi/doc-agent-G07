"""Stage 2: deterministic projection and optional offline Chandra layout."""

from __future__ import annotations

import json
from math import ceil
from pathlib import Path
from typing import Any

from ..contracts import Page, Region

_CHANDRA_KINDS = {
    "Text": "text",
    "Caption": "text",
    "Footnote": "text",
    "List-Group": "text",
    "Page-Footer": "text",
    "Page-Header": "heading",
    "Section-Header": "heading",
    "Table": "table",
    "Image": "figure",
    "Figure": "figure",
    "Diagram": "figure",
}


def _config(cfg: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        raise TypeError("layout config must be a mapping")
    options = cfg.get("layout", {})
    if not isinstance(options, dict):
        raise ValueError("cfg['layout'] must be a mapping")
    mode = options.get("mode", "projection")
    if mode not in {"projection", "full_page", "chandra"}:
        raise ValueError("layout mode must be 'projection', 'full_page', or 'chandra'")
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
    blocks_path = options.get("blocks_path")
    if blocks_path is not None and (not isinstance(blocks_path, str) or not blocks_path):
        raise ValueError("layout blocks_path must be a non-empty path or null")
    missing = options.get("missing_pages", "error")
    if missing not in {"projection", "full_page", "error"}:
        raise ValueError("layout missing_pages must be 'projection', 'full_page', or 'error'")
    if mode == "chandra" and blocks_path is None:
        raise ValueError("layout chandra mode requires layout.blocks_path")
    return {
        "mode": mode,
        "min_ink_ratio": float(min_ink_ratio),
        "max_row_gap": max_row_gap,
        "padding": padding,
        "blocks_path": blocks_path,
        "missing_pages": missing,
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


def _number(value: Any, field: str, line_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Chandra layout output line {line_number} has non-numeric {field}")
    return float(value)


def _page_id(row: dict[str, Any], line_number: int) -> str:
    book_page = row.get("book_page")
    if isinstance(book_page, bool):
        book_page = None
    if isinstance(book_page, int) and book_page > 0:
        return f"p{book_page:04d}"
    if isinstance(book_page, str) and book_page.isdigit() and int(book_page) > 0:
        return f"p{int(book_page):04d}"
    page_id = row.get("page_id")
    if isinstance(page_id, str) and page_id:
        return page_id
    raise ValueError(f"Chandra layout output line {line_number} has no positive book_page/page_id")


def _chandra_kind(label: Any) -> str:
    if not isinstance(label, str) or not label:
        return "text"
    return _CHANDRA_KINDS.get(label, "text")


def _load_chandra(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    try:
        handle = path.open(encoding="utf-8")
    except OSError as error:
        raise FileNotFoundError(f"Chandra layout output not found: {path}") from error
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Chandra layout output line {line_number} is not valid JSON"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"Chandra layout output line {line_number} must be a JSON object")
            page_id = _page_id(row, line_number)
            bbox = row.get("bbox")
            page_box = row.get("page_box")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Chandra layout output line {line_number} has an invalid bbox")
            if not isinstance(page_box, list) or len(page_box) != 4:
                raise ValueError(
                    f"Chandra layout output line {line_number} has an invalid page_box"
                )
            values = [_number(value, "bbox/page_box", line_number) for value in bbox + page_box]
            x0, y0, x1, y1, bx0, by0, bx1, by1 = values
            if x1 <= x0 or y1 <= y0 or bx1 <= bx0 or by1 <= by0:
                raise ValueError(f"Chandra layout output line {line_number} has a non-positive box")
            rows.setdefault(page_id, []).append(
                {
                    "bbox": (x0, y0, x1, y1),
                    "page_box": (bx0, by0, bx1, by1),
                    "kind": _chandra_kind(row.get("label")),
                }
            )
    return rows


def _chandra_regions(page: Page, rows: list[dict[str, Any]]) -> list[Region]:
    width, height = _image_size(Path(page.image_path))
    regions: list[Region] = []
    for row in rows:
        x0, y0, x1, y1 = row["bbox"]
        bx0, by0, bx1, by1 = row["page_box"]
        nx0 = max(0.0, min(1.0, (x0 - bx0) / (bx1 - bx0)))
        ny0 = max(0.0, min(1.0, (y0 - by0) / (by1 - by0)))
        nx1 = max(0.0, min(1.0, (x1 - bx0) / (bx1 - bx0)))
        ny1 = max(0.0, min(1.0, (y1 - by0) / (by1 - by0)))
        left = max(0, min(width, int(nx0 * width)))
        top = max(0, min(height, int(ny0 * height)))
        right = max(0, min(width, int(ceil(nx1 * width))))
        bottom = max(0, min(height, int(ceil(ny1 * height))))
        if right > left and bottom > top:
            regions.append(
                Region(page_id=page.id, bbox=(left, top, right, bottom), kind=row["kind"])
            )
    return regions


def _missing_chandra(page: Page, options: dict[str, Any]) -> list[Region]:
    behavior = options["missing_pages"]
    if behavior == "projection":
        return _projection(page, options)
    if behavior == "full_page":
        return [_fallback(page)]
    raise ValueError(
        f"Chandra layout output has no rows for page {page.id}; "
        "set layout.missing_pages to 'projection' or 'full_page'"
    )


def detect(pages: list[Page], cfg: dict[str, Any]) -> list[Region]:
    """Return fixed Regions from projection, full-page, or offline Chandra layout output."""
    options = _config(cfg)
    if not isinstance(pages, list):
        raise TypeError("pages must be a list of Page contracts")
    chandra_rows: dict[str, list[dict[str, Any]]] = {}
    if options["mode"] == "chandra":
        chandra_rows = _load_chandra(Path(options["blocks_path"]))
    regions: list[Region] = []
    for page in pages:
        if not isinstance(page, Page):
            raise TypeError("pages must contain only Page contracts")
        image = Path(page.image_path)
        if not image.is_file():
            raise FileNotFoundError(f"page {page.id} image does not exist: {image}")
        if options["mode"] == "full_page":
            page_regions = [_fallback(page)]
        elif options["mode"] == "projection":
            page_regions = _projection(page, options)
        elif page.id in chandra_rows:
            page_regions = _chandra_regions(page, chandra_rows[page.id])
            if not page_regions:
                page_regions = _missing_chandra(page, options)
        else:
            page_regions = _missing_chandra(page, options)
        regions.extend(page_regions)
    return regions
