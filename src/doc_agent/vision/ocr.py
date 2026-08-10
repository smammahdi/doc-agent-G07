"""Stage 3: a reproducible Tesseract OCR baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import Chunk, Page, Region


def _ocr_config(cfg: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        raise TypeError("ocr config must be a mapping")
    options = cfg.get("ocr")
    if not isinstance(options, dict):
        raise ValueError("cfg['ocr'] must be a mapping")
    mode = options.get("mode", "tesseract")
    if mode not in {"tesseract", "empty", "fallback"}:
        raise ValueError("ocr mode must be 'tesseract', 'empty', or 'fallback'")
    language = options.get("lang", "eng")
    if not isinstance(language, str) or not language:
        raise ValueError("ocr lang must be a non-empty string")
    tesseract_config = options.get("tesseract_config", "--psm 6")
    if not isinstance(tesseract_config, str):
        raise ValueError("ocr tesseract_config must be a string")
    return {"mode": mode, "lang": language, "tesseract_config": tesseract_config}


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


class Reader:
    """Tesseract reader over page-image paths supplied in cfg or loader output."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = _ocr_config(cfg)
        self._images = _page_images(cfg)

    def transcribe_region(self, region: Region) -> str:
        if self.cfg["mode"] in {"empty", "fallback"}:
            return ""
        try:
            import cv2
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
    chunks: list[Chunk] = []
    for index, region in enumerate(regions):
        if not isinstance(region, Region):
            raise TypeError("regions must contain only Region contracts")
        chunks.append(
            Chunk(
                id=f"{region.page_id}:r{index:04d}",
                doc_id=doc_id,
                text=reader.transcribe_region(region),
                page_ids=[region.page_id],
            )
        )
    return chunks
