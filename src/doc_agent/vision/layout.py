"""Stage 2: projection, orphan-ink, Chandra, and optional ONNX layout modes."""

from __future__ import annotations

import ast
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

# DocLayout-YOLO's DocStructBench label set.  The model has finer classes than
# the fixed Region contract; captions and formula/footnote classes therefore
# intentionally collapse to ``text``.
_DOC_LAYOUT_KINDS = {
    "title": "heading",
    "plain text": "text",
    "text": "text",
    "abandon": "text",
    "figure": "figure",
    "figure_caption": "text",
    "caption": "text",
    "table": "table",
    "table_caption": "text",
    "table_footnote": "text",
    "isolate_formula": "text",
    "formula_caption": "text",
}
_DOC_LAYOUT_CLASSES = {
    0: "title",
    1: "plain text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}
_FIGURE_MIN_WIDTH = 0.07
_FIGURE_MIN_HEIGHT = 0.045
_FIGURE_MIN_AREA = 0.006
_FIGURE_CLOSE_RATIO = 0.022
_FIGURE_WORD_PADDING = 3


def _config(cfg: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        raise TypeError("layout config must be a mapping")
    options = cfg.get("layout", {})
    if not isinstance(options, dict):
        raise ValueError("cfg['layout'] must be a mapping")
    mode = options.get("mode", "projection")
    if mode not in {"projection", "full_page", "chandra", "orphan_ink", "doclayout_yolo"}:
        raise ValueError(
            "layout mode must be 'projection', 'full_page', 'chandra', 'orphan_ink', "
            "or 'doclayout_yolo'"
        )
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
    score_thr = options.get("score_thr", options.get("confidence", 0.5))
    if mode == "doclayout_yolo":
        if isinstance(score_thr, bool) or not isinstance(score_thr, (int, float)):
            raise ValueError("layout score_thr must be a number in [0, 1]")
        if not 0 <= score_thr <= 1:
            raise ValueError("layout score_thr must be a number in [0, 1]")
    else:
        score_thr = 0.5

    # ``weights_path`` is the public name.  ``onnx_path`` is accepted as a
    # readable alias because this mode is specifically tied to an ONNX file.
    weights_path = options.get("weights_path", options.get("onnx_path"))
    if mode == "doclayout_yolo" and (
        weights_path is not None and (not isinstance(weights_path, str) or not weights_path)
    ):
        raise ValueError("layout weights_path must be a non-empty path or null")

    # The heuristic uses the PDF's actual text layer.  Prefer an explicit
    # layout path (useful for callers with a sidecar PDF), otherwise inherit
    # the configured ingest source.
    source_pdf = options.get("source_pdf", options.get("pdf_path"))
    if source_pdf is None:
        ingest = cfg.get("ingest", {})
        if isinstance(ingest, dict):
            source_pdf = ingest.get("source_pdf")
    if (
        mode == "orphan_ink"
        and source_pdf is not None
        and (not isinstance(source_pdf, str) or not source_pdf)
    ):
        raise ValueError("layout source_pdf must be a non-empty path or null")
    if mode == "orphan_ink" and source_pdf is None:
        raise ValueError(
            "layout orphan_ink mode requires layout.source_pdf or ingest.source_pdf "
            "to read the PDF text layer"
        )
    if mode == "doclayout_yolo" and weights_path is None:
        raise ValueError("layout doclayout_yolo mode requires layout.weights_path")
    return {
        "mode": mode,
        "min_ink_ratio": float(min_ink_ratio),
        "max_row_gap": max_row_gap,
        "padding": padding,
        "blocks_path": blocks_path,
        "missing_pages": missing,
        "score_thr": float(score_thr),
        "weights_path": weights_path,
        "source_pdf": source_pdf,
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


def _merge_boxes(
    boxes: list[tuple[int, int, int, int]], gap: int
) -> list[tuple[int, int, int, int]]:
    """Merge overlapping or nearby connected components deterministically."""
    pending = [list(box) for box in boxes]
    changed = True
    while changed:
        changed = False
        for index, first in enumerate(pending):
            for other_index in range(index + 1, len(pending)):
                second = pending[other_index]
                if (
                    first[0] - gap < second[2]
                    and second[0] - gap < first[2]
                    and first[1] - gap < second[3]
                    and second[1] - gap < first[3]
                ):
                    pending[index] = [
                        min(first[0], second[0]),
                        min(first[1], second[1]),
                        max(first[2], second[2]),
                        max(first[3], second[3]),
                    ]
                    pending.pop(other_index)
                    changed = True
                    break
            if changed:
                break
    return [
        (box[0], box[1], box[2], box[3])
        for box in sorted(pending, key=lambda box: (box[1], box[0]))
    ]


def _pdf_word_boxes(
    pdf_page: Any, image_width: int, image_height: int
) -> list[tuple[int, int, int, int]]:
    """Scale embedded PDF word boxes into rendered image coordinates."""
    rect = pdf_page.rect
    scale_x = image_width / float(rect.width)
    scale_y = image_height / float(rect.height)
    boxes: list[tuple[int, int, int, int]] = []
    for row in pdf_page.get_text("words"):
        if len(row) < 4:
            continue
        x0, y0, x1, y1 = (float(value) for value in row[:4])
        left = max(0, min(image_width, int(round(x0 * scale_x))))
        top = max(0, min(image_height, int(round(y0 * scale_y))))
        right = max(0, min(image_width, int(round(x1 * scale_x))))
        bottom = max(0, min(image_height, int(round(y1 * scale_y))))
        if right > left and bottom > top:
            boxes.append((left, top, right, bottom))
    return boxes


def _orphan_figures(page: Page, pdf_page: Any) -> list[Region]:
    """Find figure ink not covered by the PDF's real embedded text layer."""
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "layout orphan_ink mode requires opencv-python-headless and numpy"
        ) from error

    image_path = Path(page.image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"cannot read page image {image_path}")
    height, width = image.shape[:2]
    _, inverted = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    covered = np.zeros((height, width), np.uint8)
    for x0, y0, x1, y1 in _pdf_word_boxes(pdf_page, width, height):
        cv2.rectangle(
            covered,
            (max(0, x0 - _FIGURE_WORD_PADDING), max(0, y0 - _FIGURE_WORD_PADDING)),
            (min(width - 1, x1 + _FIGURE_WORD_PADDING), min(height - 1, y1 + _FIGURE_WORD_PADDING)),
            255,
            -1,
        )
    mask = ((inverted > 0) & (covered == 0)).astype(np.uint8) * 255
    kernel_size = max(3, int(_FIGURE_CLOSE_RATIO * min(height, width)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)  # type: ignore[call-overload]
    boxes: list[tuple[int, int, int, int]] = []
    for index in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[index])
        if (
            box_width < _FIGURE_MIN_WIDTH * width
            or box_height < _FIGURE_MIN_HEIGHT * height
            or area < _FIGURE_MIN_AREA * width * height
        ):
            continue
        if box_width > 0.97 * width and box_height > 0.97 * height:
            continue
        boxes.append((x, y, x + box_width, y + box_height))

    merged = _merge_boxes(boxes, gap=int(0.02 * min(height, width)))
    return [Region(page_id=page.id, bbox=box, kind="figure") for box in merged]


def _subtract_box(
    box: tuple[int, int, int, int], blocker: tuple[int, int, int, int]
) -> list[tuple[int, int, int, int]]:
    """Return the parts of ``box`` not covered by ``blocker``."""
    x0, y0, x1, y1 = box
    bx0, by0, bx1, by1 = blocker
    ix0, iy0, ix1, iy1 = max(x0, bx0), max(y0, by0), min(x1, bx1), min(y1, by1)
    if ix0 >= ix1 or iy0 >= iy1:
        return [box]
    pieces: list[tuple[int, int, int, int]] = []
    if y0 < iy0:
        pieces.append((x0, y0, x1, iy0))
    if iy1 < y1:
        pieces.append((x0, iy1, x1, y1))
    if x0 < ix0:
        pieces.append((x0, iy0, ix0, iy1))
    if ix1 < x1:
        pieces.append((ix1, iy0, x1, iy1))
    return [piece for piece in pieces if piece[2] > piece[0] and piece[3] > piece[1]]


def _without_figures(text_regions: list[Region], figures: list[Region]) -> list[Region]:
    """Subtract figure boxes so text and figure Regions never overlap."""
    output: list[Region] = []
    for region in text_regions:
        pieces = [region.bbox]
        for figure in figures:
            pieces = [piece for box in pieces for piece in _subtract_box(box, figure.bbox)]
        output.extend(
            Region(page_id=region.page_id, bbox=piece, kind=region.kind) for piece in pieces
        )
    return output


def _orphan_regions(page: Page, pdf_page: Any, options: dict[str, Any]) -> list[Region]:
    figures = _orphan_figures(page, pdf_page)
    text = _without_figures(_projection(page, options), figures)
    return text + figures


def _load_doclayout(path: Path) -> tuple[Any, dict[int, str], str]:
    if not path.is_file():
        raise FileNotFoundError(f"DocLayout-YOLO ONNX weights not found: {path}")
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "layout doclayout_yolo mode requires onnxruntime; install the project dependencies"
        ) from error
    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in ort.get_available_providers()
    ] or ["CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(str(path), providers=providers)
    except Exception as error:
        raise RuntimeError(f"cannot load DocLayout-YOLO ONNX weights {path}: {error}") from error
    classes = dict(_DOC_LAYOUT_CLASSES)
    names = session.get_modelmeta().custom_metadata_map.get("names")
    if names:
        try:
            parsed = ast.literal_eval(names)
            if isinstance(parsed, dict):
                classes = {int(key): str(value) for key, value in parsed.items()}
        except (SyntaxError, ValueError, TypeError):
            pass
    inputs = session.get_inputs()
    if not inputs:
        raise RuntimeError(f"DocLayout-YOLO ONNX weights have no inputs: {path}")
    return session, classes, inputs[0].name


def _doclayout_regions(
    page: Page,
    options: dict[str, Any],
    session: Any,
    classes: dict[int, str],
    input_name: str,
) -> list[Region]:
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "layout doclayout_yolo mode requires opencv-python-headless and numpy"
        ) from error
    image_path = Path(page.image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read page image {image_path}")
    height, width = image.shape[:2]
    image_size = 1024
    scale = min(image_size / height, image_size / width)
    resized_height = int(round(height * scale))
    resized_width = int(round(width * scale))
    canvas: Any = np.full((image_size, image_size, 3), 114, np.uint8)
    canvas[:resized_height, :resized_width] = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
    tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255.0
    try:
        output_values = session.run(None, {input_name: tensor})
    except Exception as error:
        raise RuntimeError(
            f"DocLayout-YOLO inference failed for page {page.id}: {error}"
        ) from error
    if not output_values:
        raise RuntimeError("DocLayout-YOLO returned no outputs")
    detections = np.asarray(output_values[0])
    if detections.ndim == 3 and detections.shape[0] == 1:
        detections = detections[0]
    if detections.ndim != 2 or detections.shape[1] < 6:
        raise RuntimeError(
            "DocLayout-YOLO output must have shape (N, 6), " f"received {tuple(detections.shape)}"
        )

    regions: list[Region] = []
    for raw in detections:
        x0, y0, x1, y1, score, class_id = (float(value) for value in raw[:6])
        if score < options["score_thr"]:
            continue
        left = max(0, min(width, int(x0 / scale)))
        top = max(0, min(height, int(y0 / scale)))
        right = max(0, min(width, int(x1 / scale)))
        bottom = max(0, min(height, int(y1 / scale)))
        if right <= left or bottom <= top:
            continue
        label = classes.get(int(class_id), str(int(class_id))).strip().lower()
        kind = _DOC_LAYOUT_KINDS.get(label, "text")
        regions.append(Region(page_id=page.id, bbox=(left, top, right, bottom), kind=kind))
    return sorted(regions, key=lambda region: (region.bbox[1], region.bbox[0], region.kind))


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
    """Return fixed Regions from the configured layout detector."""
    options = _config(cfg)
    if not isinstance(pages, list):
        raise TypeError("pages must be a list of Page contracts")
    chandra_rows: dict[str, list[dict[str, Any]]] = {}
    if options["mode"] == "chandra":
        chandra_rows = _load_chandra(Path(options["blocks_path"]))
    pdf_document: Any = None
    doclayout_bundle: tuple[Any, dict[int, str], str] | None = None
    if options["mode"] == "orphan_ink":
        source_pdf = Path(options["source_pdf"])
        if not source_pdf.is_file():
            raise FileNotFoundError(f"layout orphan_ink source PDF not found: {source_pdf}")
        try:
            import fitz

            pdf_document = fitz.open(str(source_pdf))
        except Exception as error:
            raise RuntimeError(
                f"cannot open layout orphan_ink source PDF {source_pdf}: {error}"
            ) from error
    elif options["mode"] == "doclayout_yolo":
        doclayout_bundle = _load_doclayout(Path(options["weights_path"]))
    regions: list[Region] = []
    try:
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
            elif options["mode"] == "orphan_ink":
                if pdf_document is None:
                    raise RuntimeError("orphan_ink mode requires an open pdf_document")
                try:
                    page_number = int(page.id.removeprefix("p")) - 1
                except ValueError as error:
                    raise ValueError(
                        f"page id must be pXXXX for orphan_ink mode: {page.id}"
                    ) from error
                if page_number < 0 or page_number >= pdf_document.page_count:
                    raise ValueError(
                        f"page {page.id} is outside source PDF page range "
                        f"1..{pdf_document.page_count}"
                    )
                page_regions = _orphan_regions(page, pdf_document.load_page(page_number), options)
            elif options["mode"] == "doclayout_yolo":
                if doclayout_bundle is None:
                    raise RuntimeError("doclayout_yolo mode requires loaded doclayout_bundle")
                page_regions = _doclayout_regions(page, options, *doclayout_bundle)
            elif page.id in chandra_rows:
                page_regions = _chandra_regions(page, chandra_rows[page.id])
                if not page_regions:
                    page_regions = _missing_chandra(page, options)
            else:
                page_regions = _missing_chandra(page, options)
            regions.extend(page_regions)
    finally:
        if pdf_document is not None:
            pdf_document.close()
    return regions
