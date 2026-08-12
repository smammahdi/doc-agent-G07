"""Run TrOCR over fixed Chandra or DocLayout-YOLO regions.

This is an offline research runner. It reads existing layout sidecars, renders
the Pierce PDF at the same 300-DPI settings as the starter loader, and writes
page/region JSONL checkpoints that can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

from doc_agent.contracts import Region
from doc_agent.vision.ocr import Reader

_CHANDRA_FIGURES = {"Image", "Figure", "Diagram"}
_CHANDRA_KINDS = {
    "Page-Header": "heading",
    "Section-Header": "heading",
    "Table": "table",
}
_DLY_KINDS = {
    "title": "heading",
    "figure_caption": "text",
    "table_caption": "text",
    "table_footnote": "text",
    "isolate_formula": "text",
    "formula_caption": "text",
    "table": "table",
}


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object on {path}:{line_number}")
            yield row


def _clip_norm(box: Any) -> tuple[float, float, float, float]:
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"invalid normalized box: {box!r}")
    values = tuple(float(value) for value in box)
    x0, y0, x1, y1 = (max(0.0, min(1.0, value)) for value in values)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"non-positive normalized box: {box!r}")
    return x0, y0, x1, y1


def _chandra_regions(path: Path) -> dict[int, list[dict[str, Any]]]:
    pages: dict[int, list[dict[str, Any]]] = {}
    for row in _jsonl(path):
        page_number = int(row["book_page"])
        label = str(row.get("label", "Text"))
        if label in _CHANDRA_FIGURES:
            continue
        bbox = row["bbox"]
        page_box = row["page_box"]
        bx0, by0, bx1, by1 = (float(value) for value in page_box)
        x0, y0, x1, y1 = (float(value) for value in bbox)
        normalized = [
            (x0 - bx0) / (bx1 - bx0),
            (y0 - by0) / (by1 - by0),
            (x1 - bx0) / (bx1 - bx0),
            (y1 - by0) / (by1 - by0),
        ]
        pages.setdefault(page_number, []).append(
            {
                "source_class": label,
                "kind": _CHANDRA_KINDS.get(label, "text"),
                "bbox_norm": list(_clip_norm(normalized)),
            }
        )
    return pages


def _doclayout_regions(path: Path) -> dict[int, list[dict[str, Any]]]:
    pages: dict[int, list[dict[str, Any]]] = {}
    for row in _jsonl(path):
        if bool(row.get("is_figure")):
            continue
        page_number = int(row["page_number"])
        label = str(row.get("class_name", "text"))
        pages.setdefault(page_number, []).append(
            {
                "source_class": label,
                "kind": _DLY_KINDS.get(label, "text"),
                "bbox_norm": list(_clip_norm(row["bbox_norm"])),
            }
        )
    for records in pages.values():
        records.sort(key=lambda record: (record["bbox_norm"][1], record["bbox_norm"][0]))
    return pages


def _page_dimensions(page: fitz.Page, dpi: int) -> tuple[int, int]:
    scale = dpi / 72.0
    return round(page.rect.width * scale), round(page.rect.height * scale)


def _render_page(page: fitz.Page, target: Path, dpi: int, quality: int = 80) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.stem}.", suffix=".jpg")
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        pixmap = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
        pixmap.save(str(temporary_path), output="jpeg", jpg_quality=quality)
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


def _bbox_pixels(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    left = max(0, min(width, int(x0 * width)))
    top = max(0, min(height, int(y0 * height)))
    right = max(0, min(width, int(x1 * width + 0.999999)))
    bottom = max(0, min(height, int(y1 * height + 0.999999)))
    if right <= left or bottom <= top:
        raise ValueError(f"normalized box became empty at {width}x{height}: {box}")
    return left, top, right, bottom


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _read_pages(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return list(_jsonl(path))


def _recover(output: Path, expected: list[int]) -> tuple[list[dict[str, Any]], set[int]]:
    page_path = output / "pages.jsonl"
    region_path = output / "regions.jsonl"
    pages: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in _read_pages(page_path):
        page_number = int(row["page_number"])
        if page_number in expected and page_number not in seen:
            pages.append(row)
            seen.add(page_number)
    pages.sort(key=lambda row: int(row["page_number"]))
    committed = {int(row["page_number"]) for row in pages}
    if region_path.is_file():
        rows = [row for row in _jsonl(region_path) if int(row["page_number"]) in committed]
        temporary = region_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, region_path)
    with page_path.open("w", encoding="utf-8") as handle:
        for row in pages:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return pages, committed


def _append(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(args: argparse.Namespace) -> None:
    source = Path(args.source_pdf)
    if not source.is_file():
        raise FileNotFoundError(source)
    layout_path = Path(args.layout_path)
    layout_pages = (
        _chandra_regions(layout_path)
        if args.layout == "chandra"
        else _doclayout_regions(layout_path)
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    with fitz.open(str(source)) as document:
        page_count = len(document)
        if page_count != 1034:
            raise ValueError(f"expected the Pierce source to have 1,034 pages; found {page_count}")
        expected = args.pages or list(range(1, page_count + 1))
        if any(page_number < 1 or page_number > page_count for page_number in expected):
            raise ValueError(f"--pages must be between 1 and {page_count}")
        pages, committed = _recover(output, expected)
        image_paths = {f"p{number:04d}": cache / f"p{number:04d}.jpg" for number in expected}
        reader = Reader(
            {
                "ocr": {
                    "mode": "trocr",
                    "trocr_model": args.model,
                    "trocr_device": args.device,
                    "trocr_batch_size": args.batch_size,
                    "trocr_max_length": args.max_length,
                },
                "page_images": {page_id: str(path) for page_id, path in image_paths.items()},
            }
        )
        started = time.monotonic()
        for page_number in expected:
            if page_number in committed:
                continue
            page = document.load_page(page_number - 1)
            page_id = f"p{page_number:04d}"
            width, height = _page_dimensions(page, args.dpi)
            records = layout_pages.get(page_number, [])
            if args.layout == "chandra" and page_number not in layout_pages:
                page_row = {
                    "page_number": page_number,
                    "page_id": page_id,
                    "status": "layout_missing",
                    "width": width,
                    "height": height,
                    "region_count": 0,
                    "page_text": "",
                }
                _append(output / "pages.jsonl", [page_row])
                pages.append(page_row)
                committed.add(page_number)
                continue

            image_path = image_paths[page_id]
            if records and not image_path.is_file():
                _render_page(page, image_path, args.dpi)
            if records:
                with Image.open(image_path) as image:
                    width, height = image.size
            regions: list[Region] = []
            output_rows: list[dict[str, Any]] = []
            for index, record in enumerate(records):
                normalized = tuple(record["bbox_norm"])
                bbox = _bbox_pixels(normalized, width, height)
                regions.append(Region(page_id=page_id, bbox=bbox, kind=record["kind"]))
                output_rows.append(
                    {
                        "region_id": f"{page_id}:r{index:04d}",
                        "page_number": page_number,
                        "page_id": page_id,
                        "source_class": record["source_class"],
                        "kind": record["kind"],
                        "bbox_norm": list(normalized),
                        "bbox_px": list(bbox),
                    }
                )
            page_started = time.monotonic()
            texts = [reader.transcribe_region(region) for region in regions]
            for row, text in zip(output_rows, texts, strict=True):
                row["text"] = text
                row["status"] = "complete"
            page_text = "\n\n".join(text for text in texts if text)
            for row in output_rows:
                row["elapsed_ms"] = round((time.monotonic() - page_started) * 1000, 3)
            _append(output / "regions.jsonl", output_rows)
            page_row = {
                "page_number": page_number,
                "page_id": page_id,
                "status": "complete",
                "width": width,
                "height": height,
                "region_count": len(output_rows),
                "page_text": page_text,
            }
            _append(output / "pages.jsonl", [page_row])
            reader.clear_page_cache()
            pages.append(page_row)
            committed.add(page_number)
            _atomic_json(
                output / "summary.json",
                {
                    "status": "running",
                    "layout": args.layout,
                    "model": args.model,
                    "device": reader._trocr_device_name,
                    "dpi": args.dpi,
                    "pages_requested": len(expected),
                    "pages_completed": sum(row["status"] == "complete" for row in pages),
                    "pages_layout_missing": sum(row["status"] == "layout_missing" for row in pages),
                    "regions_completed": sum(int(row["region_count"]) for row in pages),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
            )

        _atomic_json(
            output / "summary.json",
            {
                "status": "complete",
                "layout": args.layout,
                "model": args.model,
                "device": reader._trocr_device_name,
                "dpi": args.dpi,
                "pages_requested": len(expected),
                "pages_completed": sum(row["status"] == "complete" for row in pages),
                "pages_layout_missing": sum(row["status"] == "layout_missing" for row in pages),
                "regions_completed": sum(int(row["region_count"]) for row in pages),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            },
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("chandra", "doclayout_yolo"), required=True)
    parser.add_argument("--layout-path", required=True, type=Path)
    parser.add_argument("--source-pdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--model", default="microsoft/trocr-base-printed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--pages",
        type=lambda value: [int(item) for item in value.split(",") if item.strip()],
        default=None,
        help="comma-separated 1-based pages for a bounded smoke run; omit for all pages",
    )
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
