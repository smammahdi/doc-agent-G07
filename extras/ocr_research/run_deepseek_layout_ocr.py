"""Save DeepSeek-OCR text from regions supplied by an existing layout.

This is a research exporter, not the production OCR path.  It uses the
official ``deepseek-ai/DeepSeek-OCR`` custom Transformers model on one cropped
layout region at a time.  It never runs a layout detector and never consumes
text from Chandra or another OCR engine.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import fitz
from PIL import Image
from run_layout_trocr import (
    _bbox_pixels,
    _chandra_observed_pages,
    _chandra_regions,
    _doclayout_regions,
    _jsonl,
    _page_dimensions,
)


def _append(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


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


def _atomic_json(path: Path, value: Any) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _render_page(page: fitz.Page, target: Path, dpi: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.stem}.", suffix=".jpg")
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        pixmap = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
        pixmap.save(str(temporary_path), output="jpeg", jpg_quality=80)
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


def _crop(image: Image.Image, bbox: tuple[int, int, int, int], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.stem}.", suffix=".jpg")
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        image.crop(bbox).convert("RGB").save(temporary_path, format="JPEG", quality=90)
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_model(model_name: str, attention: str, device: str) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    selected = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected == "auto":
        selected = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_safetensors=True,
        _attn_implementation=attention,
    ).eval()
    if selected == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DeepSeek-OCR requested CUDA, but CUDA is unavailable")
        model = model.cuda().to(torch.bfloat16)
    elif selected == "cpu":
        model = model.to(torch.float32)
    else:
        raise ValueError("--device must be auto, cuda, or cpu")
    return tokenizer, model, selected


def _infer(
    model: Any, tokenizer: Any, image_path: Path, output_path: Path, args: argparse.Namespace
) -> str:
    prompt = "<image>\nFree OCR."
    kwargs = {
        "tokenizer": tokenizer,
        "prompt": prompt,
        "image_file": str(image_path),
        "output_path": str(output_path),
        "base_size": args.base_size,
        "image_size": args.image_size,
        "crop_mode": args.crop_mode,
        "save_results": False,
        "test_compress": False,
        "eval_mode": True,
    }
    try:
        result = model.infer(**kwargs)
    except TypeError:
        kwargs.pop("eval_mode")
        result = model.infer(**kwargs)
    if not isinstance(result, str):
        raise RuntimeError(
            "DeepSeek-OCR did not return text; use the official custom-code runtime "
            "and eval_mode=True rather than treating saved side effects as OCR output"
        )
    return result.strip()


def run(args: argparse.Namespace) -> None:
    source = Path(args.source_pdf)
    layout_path = Path(args.layout_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not layout_path.is_file():
        raise FileNotFoundError(layout_path)
    layout_pages = (
        _chandra_regions(layout_path)
        if args.layout == "chandra"
        else _doclayout_regions(layout_path)
    )
    observed = (
        _chandra_observed_pages(layout_path) if args.layout == "chandra" else set(layout_pages)
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    with fitz.open(str(source)) as document:
        if len(document) != 1034:
            raise ValueError(
                f"expected the Pierce source to have 1,034 pages; found {len(document)}"
            )
        expected = args.pages or list(range(1, len(document) + 1))
        pages, committed = _recover(output, expected)
        tokenizer, model, device = _load_model(args.model, args.attention, args.device)
        started = time.monotonic()
        for page_number in expected:
            if page_number in committed:
                continue
            page = document.load_page(page_number - 1)
            page_id = f"p{page_number:04d}"
            width, height = _page_dimensions(page, args.dpi)
            records = layout_pages.get(page_number, [])
            if args.layout == "chandra" and page_number not in observed:
                row = {
                    "page_number": page_number,
                    "page_id": page_id,
                    "status": "layout_missing",
                    "width": width,
                    "height": height,
                    "region_count": 0,
                    "page_text": "",
                }
                _append(output / "pages.jsonl", [row])
                pages.append(row)
                committed.add(page_number)
                continue
            page_image = cache / "pages" / f"{page_id}.jpg"
            if records and not page_image.is_file():
                _render_page(page, page_image, args.dpi)
            if records:
                with Image.open(page_image) as image:
                    width, height = image.size
                    image = image.convert("RGB")
                    region_rows: list[dict[str, Any]] = []
                    texts: list[str] = []
                    for index, record in enumerate(records):
                        normalized = tuple(record["bbox_norm"])
                        bbox = _bbox_pixels(normalized, width, height)
                        crop_path = cache / "regions" / page_id / f"r{index:04d}.jpg"
                        if not crop_path.is_file():
                            _crop(image, bbox, crop_path)
                        text = _infer(model, tokenizer, crop_path, cache / "deepseek_output", args)
                        texts.append(text)
                        region_rows.append(
                            {
                                "region_id": f"{page_id}:r{index:04d}",
                                "page_number": page_number,
                                "page_id": page_id,
                                "source_class": record["source_class"],
                                "kind": record["kind"],
                                "bbox_norm": list(normalized),
                                "bbox_px": list(bbox),
                                "text": text,
                                "status": "complete",
                            }
                        )
            else:
                region_rows = []
                texts = []
            _append(output / "regions.jsonl", region_rows)
            row = {
                "page_number": page_number,
                "page_id": page_id,
                "status": "complete",
                "width": width,
                "height": height,
                "region_count": len(region_rows),
                "page_text": "\n\n".join(text for text in texts if text),
            }
            _append(output / "pages.jsonl", [row])
            pages.append(row)
            committed.add(page_number)
            _atomic_json(
                output / "summary.json",
                {
                    "status": "running",
                    "layout": args.layout,
                    "model": args.model,
                    "device": device,
                    "dpi": args.dpi,
                    "pages_requested": len(expected),
                    "pages_completed": sum(item["status"] == "complete" for item in pages),
                    "pages_layout_missing": sum(
                        item["status"] == "layout_missing" for item in pages
                    ),
                    "regions_completed": sum(int(item["region_count"]) for item in pages),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
            )
        _atomic_json(
            output / "summary.json",
            {
                "status": "complete",
                "layout": args.layout,
                "model": args.model,
                "device": device,
                "dpi": args.dpi,
                "pages_requested": len(expected),
                "pages_completed": sum(item["status"] == "complete" for item in pages),
                "pages_layout_missing": sum(item["status"] == "layout_missing" for item in pages),
                "regions_completed": sum(int(item["region_count"]) for item in pages),
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
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--attention", default="eager", choices=("eager", "flash_attention_2"))
    parser.add_argument("--base-size", type=int, default=640)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--crop-mode", action="store_true")
    parser.add_argument(
        "--pages",
        type=lambda value: [int(item) for item in value.split(",") if item.strip()],
        default=None,
        help="comma-separated 1-based pages for a bounded smoke run; omit for all pages",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
