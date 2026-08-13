"""Save PaddleOCR and DeepSeek-OCR text from existing layout regions.

This is a Kaggle research runner. It never runs a layout detector and never
uses text produced by Chandra. It reads a real Pierce PDF plus an existing
layout sidecar, crops each non-figure region, and writes independent output
trees for PaddleOCR and DeepSeek-OCR.

The default is a one-page smoke run (page 34). Use ``--pages all`` for the
full 1,034-page book after the smoke run succeeds.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

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


@dataclass(frozen=True)
class LayoutRegion:
    page_number: int
    source_class: str
    kind: str
    bbox_norm: tuple[float, float, float, float]
    order: int


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


def _number(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {label}: {value!r}") from error


def _norm_box(values: Any) -> tuple[float, float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"expected four box coordinates, got {values!r}")
    clipped = [max(0.0, min(1.0, _number(value, "box coordinate"))) for value in values]
    x0, y0, x1, y1 = clipped
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"non-positive normalized box: {values!r}")
    return x0, y0, x1, y1


def _chandra_box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    bbox = row.get("bbox")
    page_box = row.get("page_box")
    if (
        not isinstance(bbox, list)
        or not isinstance(page_box, list)
        or len(bbox) != 4
        or len(page_box) != 4
    ):
        raise ValueError("Chandra rows need bbox and page_box arrays")
    px0, py0, px1, py1 = (_number(value, "page_box") for value in page_box)
    if px1 <= px0 or py1 <= py0:
        raise ValueError(f"invalid Chandra page_box: {page_box!r}")
    x0, y0, x1, y1 = (_number(value, "bbox") for value in bbox)
    return _norm_box(
        [
            (x0 - px0) / (px1 - px0),
            (y0 - py0) / (py1 - py0),
            (x1 - px0) / (px1 - px0),
            (y1 - py0) / (py1 - py0),
        ]
    )


def load_layout(path: Path, name: str) -> tuple[dict[int, list[LayoutRegion]], set[int]]:
    """Load existing layout geometry; return regions and pages represented by the sidecar."""

    pages: dict[int, list[LayoutRegion]] = {}
    observed: set[int] = set()
    for order, row in enumerate(_jsonl(path)):
        if name == "chandra":
            page_number = int(row["book_page"])
            observed.add(page_number)
            source_class = str(row.get("label", "Text"))
            if source_class in _CHANDRA_FIGURES:
                continue
            bbox_norm = _chandra_box(row)
            kind = _CHANDRA_KINDS.get(source_class, "text")
        else:
            page_number = int(row["page_number"])
            if bool(row.get("is_figure")):
                continue
            source_class = str(row.get("class_name", row.get("label", "text")))
            bbox_norm = _norm_box(row["bbox_norm"])
            kind = _DLY_KINDS.get(source_class, "text")
            observed.add(page_number)
        pages.setdefault(page_number, []).append(
            LayoutRegion(page_number, source_class, kind, bbox_norm, order)
        )
    if name != "chandra":
        for records in pages.values():
            records.sort(
                key=lambda record: (record.bbox_norm[1], record.bbox_norm[0], record.order)
            )
    return pages, observed


def _ranked_choice(candidates: list[Path], label: str, tokens: tuple[str, ...]) -> Path:
    if not candidates:
        raise FileNotFoundError(f"could not find {label}")
    ranked = sorted(
        candidates,
        key=lambda path: (
            sum(token in str(path).lower() for token in tokens),
            -len(str(path)),
            str(path),
        ),
        reverse=True,
    )
    score = sum(token in str(ranked[0]).lower() for token in tokens)
    tied = [path for path in ranked if sum(token in str(path).lower() for token in tokens) == score]
    if len(tied) != 1:
        raise RuntimeError(f"ambiguous {label}; pass --source-pdf or --layout-path")
    return ranked[0]


def discover(input_root: Path, layout_name: str) -> tuple[Path, Path]:
    files = [path for path in input_root.rglob("*") if path.is_file()]
    source = _ranked_choice(
        [path for path in files if path.suffix.lower() == ".pdf"],
        "Pierce PDF",
        ("pierce", "1890", "medical"),
    )
    if layout_name == "chandra":
        return source, _ranked_choice(
            [path for path in files if path.name == "chunks.jsonl"],
            "Chandra chunks.jsonl",
            ("chandra",),
        )
    return source, _ranked_choice(
        [path for path in files if path.name == "detections.jsonl"],
        f"{layout_name} detections.jsonl",
        tuple(layout_name.split("_")),
    )


def _pages(value: str | None, page_count: int) -> list[int]:
    if value is None or value == "all":
        return list(range(1, page_count + 1))
    selected = sorted({int(item) for item in value.split(",") if item.strip()})
    if not selected or any(number < 1 or number > page_count for number in selected):
        raise ValueError(f"--pages must contain numbers from 1 to {page_count}, or all")
    return selected


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


def _atomic_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _append(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _recover(output: Path, requested: list[int]) -> tuple[list[dict[str, Any]], set[int]]:
    page_path = output / "pages.jsonl"
    region_path = output / "regions.jsonl"
    pages: list[dict[str, Any]] = []
    seen: set[int] = set()
    if page_path.is_file():
        for row in _jsonl(page_path):
            number = int(row["page_number"])
            if number in requested and number not in seen:
                pages.append(row)
                seen.add(number)
    pages.sort(key=lambda row: int(row["page_number"]))
    committed = {int(row["page_number"]) for row in pages}
    if region_path.is_file():
        rows = [row for row in _jsonl(region_path) if int(row["page_number"]) in committed]
        _atomic_write(region_path, rows)
    _atomic_write(page_path, pages)
    return pages, committed


def _render(page: fitz.Page, target: Path, dpi: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.stem}.", suffix=".jpg")
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False).save(
            str(temporary_path), output="jpeg", jpg_quality=80
        )
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


def _crop(image: Image.Image, bbox: tuple[int, int, int, int], target: Path, padding: int) -> None:
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width, right + padding)
    bottom = min(image.height, bottom + padding)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.stem}.", suffix=".jpg")
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        image.crop((left, top, right, bottom)).convert("RGB").save(
            temporary_path, format="JPEG", quality=90
        )
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)


class PaddleRunner:
    def __init__(self, requested_device: str) -> None:
        import paddle
        from paddleocr import PaddleOCR

        has_cuda = bool(getattr(paddle, "is_compiled_with_cuda", lambda: False)())
        self.device = "gpu:0" if requested_device == "auto" and has_cuda else "cpu"
        if requested_device in {"cpu", "gpu:0"}:
            self.device = requested_device
        if self.device.startswith("gpu") and not has_cuda:
            raise RuntimeError("PaddleGPU requested but the installed PaddlePaddle has no CUDA")
        try:
            self.engine = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=self.device,
            )
        except TypeError:
            try:
                self.engine = PaddleOCR(
                    lang="en",
                    use_angle_cls=False,
                    show_log=False,
                    use_gpu=self.device.startswith("gpu"),
                )
            except TypeError:
                self.engine = PaddleOCR(lang="en")

    @staticmethod
    def _dict_result(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            return value
        payload = getattr(value, "json", None)
        if callable(payload):
            payload = payload()
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return None
        return payload if isinstance(payload, dict) else None

    @classmethod
    def _lines(cls, result: Any) -> list[dict[str, Any]]:
        data = cls._dict_result(result)
        if data is None and isinstance(result, list):
            for item in result:
                item_data = cls._dict_result(item)
                if item_data is not None:
                    nested_lines = cls._lines(item_data)
                    if nested_lines:
                        return nested_lines
        if data is not None:
            data = data.get("res", data)
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            if isinstance(texts, list):
                return [
                    {
                        "text": str(text),
                        "score": float(scores[index]) if index < len(scores) else None,
                    }
                    for index, text in enumerate(texts)
                    if str(text).strip()
                ]
        lines: list[dict[str, Any]] = []
        if isinstance(result, list):
            for item in result:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                    and isinstance(item[1], (list, tuple))
                    and item[1]
                    and isinstance(item[1][0], str)
                ):
                    lines.append(
                        {
                            "text": item[1][0],
                            "score": float(item[1][1]) if len(item[1]) > 1 else None,
                        }
                    )
                elif isinstance(item, list):
                    lines.extend(cls._lines(item))
        return lines

    def recognize(self, image_path: Path) -> tuple[str, list[dict[str, Any]], float | None]:
        if hasattr(self.engine, "predict"):
            result = list(self.engine.predict(str(image_path)))
        else:
            result = self.engine.ocr(str(image_path), cls=False)
        lines = self._lines(result)
        scores = [line["score"] for line in lines if line["score"] is not None]
        mean_score = sum(scores) / len(scores) if scores else None
        return " ".join(line["text"] for line in lines), lines, mean_score


class DeepSeekRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        requested = args.deepseek_device
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
        if self.device == "auto":
            self.device = "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DeepSeek CUDA requested but CUDA is unavailable")
        self.dtype_name = "float32"
        dtype = torch.float32
        if self.device == "cuda":
            major, _ = torch.cuda.get_device_capability()
            use_bf16 = args.deepseek_dtype == "bf16" or (
                args.deepseek_dtype == "auto" and major >= 8
            )
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            self.dtype_name = "bfloat16" if dtype == torch.bfloat16 else "float16"
        self.tokenizer = AutoTokenizer.from_pretrained(args.deepseek_model, trust_remote_code=True)
        model_kwargs = {
            "trust_remote_code": True,
            "use_safetensors": True,
            "_attn_implementation": args.deepseek_attention,
        }
        try:
            self.model = AutoModel.from_pretrained(args.deepseek_model, **model_kwargs).eval()
        except (TypeError, ValueError):
            model_kwargs.pop("_attn_implementation")
            self.model = AutoModel.from_pretrained(args.deepseek_model, **model_kwargs).eval()
        self.model = self.model.to(self.device)
        if self.device == "cuda":
            self.model = self.model.to(dtype)
        self.base_size = args.deepseek_base_size
        self.image_size = args.deepseek_image_size
        self.crop_mode = args.deepseek_crop_mode

    def recognize(self, image_path: Path, output_dir: Path) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        result = self.model.infer(
            tokenizer=self.tokenizer,
            prompt="<image>\nFree OCR.",
            image_file=str(image_path),
            output_path=str(output_dir),
            base_size=self.base_size,
            image_size=self.image_size,
            crop_mode=self.crop_mode,
            save_results=False,
            test_compress=False,
            eval_mode=True,
        )
        if not isinstance(result, str):
            raise RuntimeError("DeepSeek-OCR returned no text string")
        return result.strip()


def _summary(
    args: argparse.Namespace,
    output: Path,
    pages: list[dict[str, Any]],
    started: float,
    devices: dict[str, str],
) -> None:
    _atomic_json(
        output / "summary.json",
        {
            "status": "running",
            "layout": args.layout_name,
            "engines": args.engines,
            "devices": devices,
            "dpi": args.dpi,
            "pages_requested": len(args.page_numbers),
            "pages_completed": sum(row["status"] == "complete" for row in pages),
            "pages_layout_missing": sum(row["status"] == "layout_missing" for row in pages),
            "regions_completed": sum(int(row["region_count"]) for row in pages),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
    )


def run(args: argparse.Namespace) -> None:
    input_root = args.input_root
    if args.source_pdf and args.layout_path:
        source, layout_path = args.source_pdf, args.layout_path
    else:
        discovered_source, discovered_layout = discover(input_root, args.layout_name)
        source = args.source_pdf or discovered_source
        layout_path = args.layout_path or discovered_layout
    with fitz.open(str(source)) as document:
        args.page_numbers = _pages(args.pages, len(document))
        layout_pages, observed = load_layout(layout_path, args.layout_name)
        print(f"PDF: {source}\nLayout: {layout_path}\nPages: {len(args.page_numbers)}", flush=True)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "regions": sum(
                            len(layout_pages.get(page, [])) for page in args.page_numbers
                        ),
                        "layout_missing": (
                            sorted(set(args.page_numbers) - observed)
                            if args.layout_name == "chandra"
                            else []
                        ),
                    },
                    indent=2,
                )
            )
            return
        output_root = args.output_root
        cache_root = args.cache_root
        output_root.mkdir(parents=True, exist_ok=True)
        cache_root.mkdir(parents=True, exist_ok=True)
        for engine in args.engines:
            if engine == "paddleocr":
                runner: Any = PaddleRunner(args.paddle_device)
                device_name = runner.device
            else:
                runner = DeepSeekRunner(args)
                device_name = f"{runner.device}:{runner.dtype_name}"
            output = output_root / ("deepseek-ocr" if engine == "deepseek" else engine)
            output.mkdir(parents=True, exist_ok=True)
            pages, committed = _recover(output, args.page_numbers)
            started = time.monotonic()
            for page_number in args.page_numbers:
                if page_number in committed:
                    continue
                page_id = f"p{page_number:04d}"
                page = document.load_page(page_number - 1)
                width = round(page.rect.width * args.dpi / 72.0)
                height = round(page.rect.height * args.dpi / 72.0)
                records = layout_pages.get(page_number, [])
                if args.layout_name == "chandra" and page_number not in observed:
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
                    _summary(args, output, pages, started, {engine: device_name})
                    continue
                page_image = cache_root / "pages" / f"{page_id}.jpg"
                if records and not page_image.is_file():
                    _render(page, page_image, args.dpi)
                region_rows: list[dict[str, Any]] = []
                page_text: list[str] = []
                if records:
                    with Image.open(page_image) as image:
                        image = image.convert("RGB")
                        width, height = image.size
                        for index, record in enumerate(records):
                            bbox = _bbox_pixels(record.bbox_norm, width, height)
                            crop = (
                                cache_root
                                / "crops"
                                / engine
                                / args.layout_name
                                / page_id
                                / f"r{index:04d}.jpg"
                            )
                            if not crop.is_file():
                                _crop(image, bbox, crop, args.crop_padding)
                            try:
                                if engine == "paddleocr":
                                    text, lines, score = runner.recognize(crop)
                                    row_extra = {"lines": lines, "score": score}
                                else:
                                    text = runner.recognize(
                                        crop, cache_root / "deepseek-side-effects"
                                    )
                                    row_extra = {}
                                status = "complete"
                                error = None
                            except Exception as exc:  # keep a page-complete audit trail
                                text, row_extra, status, error = (
                                    "",
                                    {},
                                    "error",
                                    f"{type(exc).__name__}: {exc}",
                                )
                            if text:
                                page_text.append(text)
                            region_rows.append(
                                {
                                    "region_id": f"{page_id}:r{index:04d}",
                                    "page_number": page_number,
                                    "page_id": page_id,
                                    "source_class": record.source_class,
                                    "kind": record.kind,
                                    "bbox_norm": list(record.bbox_norm),
                                    "bbox_px": list(bbox),
                                    "text": text,
                                    "status": status,
                                    **row_extra,
                                    **({"error": error} if error else {}),
                                }
                            )
                _append(output / "regions.jsonl", region_rows)
                page_row = {
                    "page_number": page_number,
                    "page_id": page_id,
                    "status": "complete",
                    "width": width,
                    "height": height,
                    "region_count": len(region_rows),
                    "page_text": "\n\n".join(page_text),
                }
                _append(output / "pages.jsonl", [page_row])
                pages.append(page_row)
                committed.add(page_number)
                _summary(args, output, pages, started, {engine: device_name})
                print(f"{engine}: {page_id} regions={len(region_rows)}", flush=True)
            completed = sum(row["status"] == "complete" for row in pages)
            layout_missing = sum(row["status"] == "layout_missing" for row in pages)
            errors = (
                sum(1 for row in _jsonl(output / "regions.jsonl") if row.get("status") == "error")
                if (output / "regions.jsonl").is_file()
                else 0
            )
            _atomic_json(
                output / "summary.json",
                {
                    "status": "complete",
                    "layout": args.layout_name,
                    "engine": engine,
                    "device": device_name,
                    "dpi": args.dpi,
                    "pages_requested": len(args.page_numbers),
                    "pages_completed": completed,
                    "pages_layout_missing": layout_missing,
                    "regions_completed": sum(int(row["region_count"]) for row in pages),
                    "regions_error": errors,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
            )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--input-root",
        type=Path,
        default=Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input")),
    )
    result.add_argument("--source-pdf", type=Path)
    result.add_argument("--layout-path", type=Path)
    result.add_argument(
        "--layout-name",
        choices=(
            "chandra",
            "doclayout_yolo",
            "ppdoclayout_v3",
            "ppdoclayout_plus_l",
            "picodet_s",
        ),
        default="doclayout_yolo",
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working/paddle-deepseek-ocr"),
    )
    result.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/kaggle/working/paddle-deepseek-ocr-cache"),
    )
    result.add_argument("--pages", default="34", help="comma-separated pages or all (default: 34)")
    result.add_argument(
        "--engines", default="paddleocr,deepseek", help="comma-separated: paddleocr,deepseek"
    )
    result.add_argument("--dpi", type=int, default=300)
    result.add_argument("--crop-padding", type=int, default=2)
    result.add_argument("--paddle-device", choices=("auto", "cpu", "gpu:0"), default="auto")
    result.add_argument("--deepseek-model", default="deepseek-ai/DeepSeek-OCR")
    result.add_argument("--deepseek-device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--deepseek-dtype", choices=("auto", "fp16", "bf16"), default="auto")
    result.add_argument(
        "--deepseek-attention",
        choices=("eager", "flash_attention_2"),
        default="eager",
    )
    result.add_argument("--deepseek-base-size", type=int, default=640)
    result.add_argument("--deepseek-image-size", type=int, default=640)
    result.add_argument("--deepseek-crop-mode", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.engines = [item.strip() for item in arguments.engines.split(",") if item.strip()]
    if not set(arguments.engines) <= {"paddleocr", "deepseek"} or not arguments.engines:
        raise SystemExit("--engines must contain paddleocr and/or deepseek")
    run(arguments)
