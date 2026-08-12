"""Stage 3: Tesseract baseline and optional reference/model OCR modes."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..contracts import Chunk, Page, Region

_CLOSING_PUNCTUATION = frozenset({".", ",", ";", ":", "!", "?", ")", "]", "}", "%", "…"})


def _join_reference_tokens(tokens: list[str]) -> str:
    parts: list[str] = []
    for raw_token in tokens:
        token = raw_token.strip()
        if not token:
            continue
        if token in _CLOSING_PUNCTUATION and parts:
            parts[-1] += token
        else:
            parts.append(token)
    return " ".join(parts)


def _ocr_config(cfg: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        raise TypeError("ocr config must be a mapping")
    options = cfg.get("ocr")
    if not isinstance(options, dict):
        raise ValueError("cfg['ocr'] must be a mapping")
    mode = options.get("mode", "tesseract")
    if mode not in {"tesseract", "empty", "fallback", "document_ai_reference", "trocr"}:
        raise ValueError(
            "ocr mode must be 'tesseract', 'empty', 'fallback', 'document_ai_reference', or 'trocr'"
        )
    language = options.get("lang", "eng")
    if not isinstance(language, str) or not language:
        raise ValueError("ocr lang must be a non-empty string")
    tesseract_config = options.get("tesseract_config", "--psm 6")
    if not isinstance(tesseract_config, str):
        raise ValueError("ocr tesseract_config must be a string")
    words_path = options.get("words_path")
    if words_path is not None and (not isinstance(words_path, str) or not words_path):
        raise ValueError("ocr words_path must be a non-empty path or null")
    missing = options.get("reference_missing", "empty")
    if missing not in {"tesseract", "empty", "error"}:
        raise ValueError("ocr reference_missing must be 'tesseract', 'empty', or 'error'")
    if mode == "document_ai_reference" and words_path is None:
        raise ValueError("document_ai_reference mode requires ocr.words_path")
    trocr_model = options.get("trocr_model", "microsoft/trocr-base-printed")
    if not isinstance(trocr_model, str) or not trocr_model:
        raise ValueError("ocr trocr_model must be a non-empty model name or path")
    trocr_device = options.get("trocr_device", "auto")
    if trocr_device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("ocr trocr_device must be 'auto', 'cpu', 'cuda', or 'mps'")
    trocr_batch_size = options.get("trocr_batch_size", 8)
    if (
        isinstance(trocr_batch_size, bool)
        or not isinstance(trocr_batch_size, int)
        or trocr_batch_size <= 0
    ):
        raise ValueError("ocr trocr_batch_size must be a positive integer")
    trocr_max_length = options.get("trocr_max_length", 64)
    if (
        isinstance(trocr_max_length, bool)
        or not isinstance(trocr_max_length, int)
        or trocr_max_length <= 0
    ):
        raise ValueError("ocr trocr_max_length must be a positive integer")
    return {
        "mode": mode,
        "lang": language,
        "tesseract_config": tesseract_config,
        "words_path": words_path,
        "reference_missing": missing,
        "trocr_model": trocr_model,
        "trocr_device": trocr_device,
        "trocr_batch_size": trocr_batch_size,
        "trocr_max_length": trocr_max_length,
    }


def _page_images(cfg: dict[str, Any]) -> dict[str, Path]:
    images: dict[str, Path] = {}
    explicit = cfg.get("page_images")
    if isinstance(explicit, dict):
        for page_id, value in explicit.items():
            if isinstance(page_id, str) and isinstance(value, str):
                images[page_id] = Path(value)
    pages = cfg.get("pages")
    if isinstance(pages, list):
        for page in pages:
            if isinstance(page, Page):
                images.setdefault(page.id, Path(page.image_path))
    ingest = cfg.get("ingest")
    if isinstance(ingest, dict):
        output = ingest.get("output_dir")
        doc_id = ingest.get("doc_id")
        if isinstance(output, str) and isinstance(doc_id, str) and doc_id:
            root = Path(output) / doc_id
            if root.is_dir():
                for image in sorted(root.rglob("p*.jpg")):
                    images.setdefault(image.stem, image)
    return images


def _load_reference_words(path: Path) -> dict[str, list[dict[str, Any]]]:
    words: dict[str, list[dict[str, Any]]] = {}
    try:
        handle = path.open(encoding="utf-8")
    except OSError as error:
        raise FileNotFoundError(f"Document AI reference output not found: {path}") from error
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Document AI reference line {line_number} is not valid JSON"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"Document AI reference line {line_number} must be an object")
            page_id = row.get("page_id")
            text = row.get("text")
            bbox = row.get("bbox_norm")
            if not isinstance(page_id, str) or not page_id:
                raise ValueError(f"Document AI reference line {line_number} has no page_id")
            if not isinstance(text, str):
                raise ValueError(f"Document AI reference line {line_number} has non-text text")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Document AI reference line {line_number} has invalid bbox_norm")
            values = []
            for value in bbox:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not isfinite(value)
                ):
                    raise ValueError(
                        f"Document AI reference line {line_number} has non-numeric bbox_norm"
                    )
                values.append(float(value))
            x0, y0, x1, y1 = values
            if not (0 <= x0 <= x1 <= 1 and 0 <= y0 <= y1 <= 1) or x0 == x1 or y0 == y1:
                raise ValueError(
                    f"Document AI reference line {line_number} has an out-of-range bbox_norm"
                )
            words.setdefault(page_id, []).append({"text": text, "bbox_norm": (x0, y0, x1, y1)})
    return words


def _image_size(path: Path) -> tuple[int, int]:
    try:
        import fitz

        pixmap = fitz.Pixmap(str(path))
        return pixmap.width, pixmap.height
    except Exception as error:
        raise RuntimeError(f"cannot read page image {path}: {error}") from error


def _trocr_device(requested: str) -> str:
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ocr trocr_device='cuda' requested but CUDA is unavailable")
    if requested == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("ocr trocr_device='mps' requested but MPS is unavailable")
    return requested


def _line_spans(
    gray: np.ndarray, min_ink_ratio: float = 0.01, max_gap: int = 3
) -> list[tuple[int, int]]:
    """Return vertical spans for printed text lines inside a layout crop."""
    if gray.ndim != 2 or gray.size == 0:
        return []
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_ink = np.count_nonzero(binary, axis=1)
    min_ink = max(1, int(round(gray.shape[1] * min_ink_ratio)))
    active = row_ink >= min_ink
    spans: list[tuple[int, int]] = []
    start: int | None = None
    gap = 0
    for y, is_active in enumerate(active):
        if is_active:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                end = y - gap + 1
                if end > start:
                    spans.append((start, end))
                start = None
                gap = 0
    if start is not None and len(active) > start:
        spans.append((start, len(active)))
    return spans


class Reader:
    """Read Tesseract text or precomputed Document AI words over configured pages."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = _ocr_config(cfg)
        self._images = _page_images(cfg)
        self._image_sizes: dict[str, tuple[int, int]] = {}
        self._reference_words: dict[str, list[dict[str, Any]]] = {}
        self._trocr_processor: Any | None = None
        self._trocr_model: Any | None = None
        self._trocr_torch: Any | None = None
        self._trocr_device_name: str | None = None
        if self.cfg["mode"] == "document_ai_reference":
            self._reference_words = _load_reference_words(Path(self.cfg["words_path"]))
        if self.cfg["mode"] == "trocr":
            self._load_trocr()

    def _load_trocr(self) -> None:
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as error:
            raise RuntimeError(
                "TrOCR OCR requires torch and transformers; install them in the "
                "research environment"
            ) from error
        device_name = _trocr_device(self.cfg["trocr_device"])
        self._trocr_torch = torch
        self._trocr_device_name = device_name
        self._trocr_processor = TrOCRProcessor.from_pretrained(self.cfg["trocr_model"])
        self._trocr_model = VisionEncoderDecoderModel.from_pretrained(self.cfg["trocr_model"])
        self._trocr_model.to(device_name).eval()

    def _transcribe_tesseract(self, region: Region) -> str:
        try:
            import pytesseract
        except ImportError as error:
            raise RuntimeError("Tesseract OCR requires cv2 and pytesseract") from error

        image_path = self._images.get(region.page_id)
        if image_path is None:
            raise FileNotFoundError(
                f"no image path for page {region.page_id}; pass cfg['page_images'] or loader output"
            )
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read page image {image_path}")
        height, width = image.shape[:2]
        x0, y0, x1, y1 = region.bbox
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"region {region.page_id} has an empty bbox: {region.bbox}")
        crop = image[y0:y1, x0:x1]
        text = pytesseract.image_to_string(
            crop,
            lang=self.cfg["lang"],
            config=self.cfg["tesseract_config"],
        )
        return text.strip()

    def _region_image(self, region: Region) -> np.ndarray:
        image_path = self._images.get(region.page_id)
        if image_path is None:
            raise FileNotFoundError(
                f"no image path for page {region.page_id}; pass cfg['page_images'] or loader output"
            )
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read page image {image_path}")
        height, width = image.shape[:2]
        x0, y0, x1, y1 = region.bbox
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"region {region.page_id} has an empty bbox: {region.bbox}")
        return image[y0:y1, x0:x1]

    def _region_lines(self, region: Region) -> list[Any]:
        image = self._region_image(region)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        spans = _line_spans(gray)
        if not spans:
            spans = [(0, image.shape[0])]
        from PIL import Image

        lines = []
        for y0, y1 in spans:
            if y1 - y0 < 6:
                continue
            line = image[y0:y1, :, :]
            lines.append(Image.fromarray(cv2.cvtColor(line, cv2.COLOR_BGR2RGB)))
        return lines

    def _trocr_read_lines(self, lines: list[Any]) -> list[str]:
        if not lines:
            return []
        if self._trocr_processor is None or self._trocr_model is None or self._trocr_torch is None:
            raise RuntimeError("TrOCR model is not loaded")
        torch = self._trocr_torch
        texts: list[str] = []
        for start in range(0, len(lines), self.cfg["trocr_batch_size"]):
            batch = lines[start : start + self.cfg["trocr_batch_size"]]
            inputs = self._trocr_processor(images=batch, return_tensors="pt").pixel_values
            inputs = inputs.to(self._trocr_device_name)
            with torch.no_grad():
                ids = self._trocr_model.generate(
                    inputs,
                    max_length=self.cfg["trocr_max_length"],
                )
            texts.extend(
                text.strip()
                for text in self._trocr_processor.batch_decode(ids, skip_special_tokens=True)
            )
        return texts

    def _transcribe_trocr(self, region: Region) -> str:
        return " ".join(text for text in self._trocr_read_lines(self._region_lines(region)) if text)

    def _reference_text(self, region: Region) -> str | None:
        rx0, ry0, rx1, ry1, _ = self._region_bounds(region)
        tokens = []
        for row in self._reference_words.get(region.page_id, []):
            wx0, wy0, wx1, wy1 = row["bbox_norm"]
            center_x, center_y = (wx0 + wx1) / 2, (wy0 + wy1) / 2
            if rx0 <= center_x <= rx1 and ry0 <= center_y <= ry1 and row["text"].strip():
                tokens.append(row["text"].strip())
        text = _join_reference_tokens(tokens)
        return text or None

    def _region_bounds(self, region: Region) -> tuple[float, float, float, float, float]:
        image_path = self._images.get(region.page_id)
        if image_path is None:
            raise FileNotFoundError(
                f"no image path for page {region.page_id}; pass cfg['page_images'] or loader output"
            )
        if region.page_id not in self._image_sizes:
            self._image_sizes[region.page_id] = _image_size(image_path)
        width, height = self._image_sizes[region.page_id]
        x0, y0, x1, y1 = region.bbox
        rx0, ry0, rx1, ry1 = x0 / width, y0 / height, x1 / width, y1 / height
        return rx0, ry0, rx1, ry1, (rx1 - rx0) * (ry1 - ry0)

    def _missing_reference(self, region: Region) -> str:
        missing = self.cfg["reference_missing"]
        if missing == "empty":
            return ""
        if missing == "tesseract":
            return self._transcribe_tesseract(region)
        raise ValueError(
            f"Document AI reference has no words inside page {region.page_id} "
            f"region {region.bbox}"
        )

    def _reference_assignments(self, regions: list[Region]) -> list[list[dict[str, Any]]]:
        assignments: list[list[dict[str, Any]]] = [[] for _ in regions]
        by_page: dict[str, list[int]] = {}
        for index, region in enumerate(regions):
            by_page.setdefault(region.page_id, []).append(index)
        for page_id, indices in by_page.items():
            bounds = {index: self._region_bounds(regions[index]) for index in indices}
            for row in self._reference_words.get(page_id, []):
                wx0, wy0, wx1, wy1 = row["bbox_norm"]
                center_x, center_y = (wx0 + wx1) / 2, (wy0 + wy1) / 2
                eligible = [
                    (bounds[index][4], index)
                    for index in indices
                    if bounds[index][0] <= center_x <= bounds[index][2]
                    and bounds[index][1] <= center_y <= bounds[index][3]
                ]
                if eligible:
                    _, selected = min(eligible, key=lambda item: (item[0], item[1]))
                    assignments[selected].append(row)
        return assignments

    def transcribe_regions(self, regions: list[Region]) -> list[str]:
        """Batch reference OCR with one deterministic owner per reference token."""
        assignments = self._reference_assignments(regions)
        texts: list[str] = []
        for region, rows in zip(regions, assignments, strict=True):
            text = _join_reference_tokens([row["text"] for row in rows])
            texts.append(text if text else self._missing_reference(region))
        return texts

    def transcribe_region(self, region: Region) -> str:
        if self.cfg["mode"] in {"empty", "fallback"}:
            return ""
        if self.cfg["mode"] == "document_ai_reference":
            text = self._reference_text(region)
            if text is not None:
                return text
            return self._missing_reference(region)
        if self.cfg["mode"] == "trocr":
            return self._transcribe_trocr(region)
        return self._transcribe_tesseract(region)


def _doc_id(cfg: dict[str, Any]) -> str:
    ingest = cfg.get("ingest")
    if isinstance(ingest, dict) and isinstance(ingest.get("doc_id"), str):
        if ingest["doc_id"]:
            return ingest["doc_id"]
    return "document"


def transcribe(regions: list[Region], cfg: dict[str, Any]) -> list[Chunk]:
    """OCR each region into a deterministic chunk while retaining page provenance."""
    if not isinstance(regions, list):
        raise TypeError("regions must be a list of Region contracts")
    reader = Reader(cfg)
    doc_id = _doc_id(cfg)
    for region in regions:
        if not isinstance(region, Region):
            raise TypeError("regions must contain only Region contracts")
    texts = (
        reader.transcribe_regions(regions)
        if reader.cfg["mode"] == "document_ai_reference"
        else [reader.transcribe_region(region) for region in regions]
    )
    chunks: list[Chunk] = []
    for index, region in enumerate(regions):
        chunks.append(
            Chunk(
                id=f"{region.page_id}:r{index:04d}",
                doc_id=doc_id,
                text=texts[index],
                page_ids=[region.page_id],
            )
        )
    return chunks
