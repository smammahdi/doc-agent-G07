"""Stage 1 classical preprocessing boundary.

The current A2 baseline is deliberately identity-preserving: the loader's measured JPEG render
is the source image, and no deskew/denoise/binarize transform is claimed until it is evaluated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import Page


def _config(cfg: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        raise TypeError("preprocess config must be a mapping")
    options = cfg.get("preprocess", {})
    if not isinstance(options, dict):
        raise ValueError("cfg['preprocess'] must be a mapping")
    unknown = set(options) - {"enabled", "mode"}
    if unknown:
        names = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(f"unsupported preprocess option(s): {names}")
    enabled = options.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("cfg['preprocess']['enabled'] must be a boolean")
    mode = options.get("mode", "identity")
    if mode != "identity":
        raise ValueError("the measured baseline supports only preprocess mode 'identity'")
    return {"enabled": enabled, "mode": mode}


def _validate_pages(pages: list[Page]) -> None:
    if not isinstance(pages, list):
        raise TypeError("pages must be a list of Page contracts")
    for page in pages:
        if not isinstance(page, Page):
            raise TypeError("pages must contain only Page contracts")
        image = Path(page.image_path)
        if not image.is_file():
            raise FileNotFoundError(f"page {page.id} image does not exist: {image}")


def run(pages: list[Page], cfg: dict[str, Any]) -> list[Page]:
    """Validate pages and return an idempotent, source-preserving baseline.

    No files are written and no Page fields are changed. ``enabled`` is accepted for pipeline
    configuration symmetry, but the only measured mode is ``identity`` until a transform has
    real-image quality evidence.
    """
    _config(cfg)
    _validate_pages(pages)
    return list(pages)
