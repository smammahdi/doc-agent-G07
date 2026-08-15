"""Resumable real-Pierce layout detector preflight/runner.

Modes are intentionally loaded from their official local detector paths:
``orphan_ink`` uses the PDF text layer, ``doclayout_yolo`` uses the local ONNX
export through onnxruntime, and ``ppdoclayout_v3`` uses the local Transformers
checkpoint. Each page gets exactly one JSON row, including empty detections.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDF = ROOT / "data/raw/pierce-peoples-common-sense-medical-adviser-1890.pdf"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"
GT_LABELS = {"Image", "Figure", "Diagram"}
PREDICTED_FIGURE_LABELS = {"image", "chart", "figure"}
DOC_CLASSES = {
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


def versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        "cv2",
        "numpy",
        "psutil",
        "pymupdf",
        "onnxruntime",
        "torch",
        "transformers",
    ):
        try:
            module = importlib.import_module(name)
            result[name] = str(getattr(module, "__version__", "present"))
        except Exception as error:
            result[name] = f"unavailable:{type(error).__name__}"
    return result


def load_reference(
    path: Path,
) -> tuple[dict[int, list[list[float]]], set[int]]:
    """Load Chandra boxes and retain pages with non-figure blocks as negatives."""
    rows: dict[int, list[list[float]]] = {}
    observed_pages: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            page = row.get("book_page")
            if not isinstance(page, int):
                raise ValueError(f"invalid Chandra page at line {line_number}")
            observed_pages.add(page)
            if row.get("label") not in GT_LABELS:
                continue
            page_box = row.get("page_box")
            bbox = row.get("bbox")
            if not isinstance(page_box, list) or not isinstance(bbox, list):
                raise ValueError(f"invalid Chandra row {line_number}")
            if len(page_box) != 4 or len(bbox) != 4:
                raise ValueError(f"invalid Chandra boxes at line {line_number}")
            px0, py0, px1, py1 = (float(value) for value in page_box)
            x0, y0, x1, y1 = (float(value) for value in bbox)
            if px1 <= px0 or py1 <= py0 or x1 <= x0 or y1 <= y0:
                raise ValueError(f"non-positive Chandra box at line {line_number}")
            rows.setdefault(page, []).append(
                [
                    (x0 - px0) / (px1 - px0),
                    (y0 - py0) / (py1 - py0),
                    (x1 - px0) / (px1 - px0),
                    (y1 - py0) / (py1 - py0),
                ]
            )
    return rows, observed_pages


def iou(left: list[float], right: list[float]) -> float:
    ix0, iy0 = max(left[0], right[0]), max(left[1], right[1])
    ix1, iy1 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / (la + ra - inter) if la + ra - inter else 0.0


def merge_boxes(boxes: list[list[int]], gap: int) -> list[tuple[int, int, int, int]]:
    """Iteratively union overlapping or near-touching boxes, matching orphan_ink."""
    pending = [list(box) for box in boxes]
    changed = True
    while changed:
        changed = False
        for index in range(len(pending)):
            for other in range(index + 1, len(pending)):
                left, right = pending[index], pending[other]
                if (
                    left[0] - gap < right[2]
                    and right[0] - gap < left[2]
                    and left[1] - gap < right[3]
                    and right[1] - gap < left[3]
                ):
                    pending[index] = [
                        min(left[0], right[0]),
                        min(left[1], right[1]),
                        max(left[2], right[2]),
                        max(left[3], right[3]),
                    ]
                    pending.pop(other)
                    changed = True
                    break
            if changed:
                break
    return [(box[0], box[1], box[2], box[3]) for box in pending]


def greedy(predicted: list[list[float]], truth: list[list[float]]) -> list[float]:
    candidates = sorted(
        ((iou(a, b), ai, bi) for ai, a in enumerate(predicted) for bi, b in enumerate(truth)),
        reverse=True,
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[float] = []
    for score, ai, bi in candidates:
        if score < 0.5 or ai in used_a or bi in used_b:
            continue
        used_a.add(ai)
        used_b.add(bi)
        matches.append(score)
    return matches


def normalize(box: list[float], width: int, height: int) -> list[float]:
    return [
        max(0.0, min(1.0, box[0] / width)),
        max(0.0, min(1.0, box[1] / height)),
        max(0.0, min(1.0, box[2] / width)),
        max(0.0, min(1.0, box[3] / height)),
    ]


def clamp_bbox(box: list[float], width: int, height: int) -> list[float]:
    """Clamp detector pixels to the rendered page and reject degenerate boxes."""
    clamped = [
        max(0.0, min(float(width), box[0])),
        max(0.0, min(float(height), box[1])),
        max(0.0, min(float(width), box[2])),
        max(0.0, min(float(height), box[3])),
    ]
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        raise ValueError(f"non-positive detection box after clamping: {box}")
    return clamped


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    os.replace(temporary, path)


def recover_outputs(
    pages_path: Path, detections_path: Path, requested_pages: set[int]
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Keep only page commits whose detection rows are complete after a crash."""
    page_rows = read_jsonl(pages_path)
    detection_rows = read_jsonl(detections_path)
    seen_pages: set[int] = set()
    for row in page_rows:
        page = row.get("page_number")
        if not isinstance(page, int) or page in seen_pages:
            raise ValueError(f"duplicate/invalid page row in {pages_path}")
        if page not in requested_pages:
            raise ValueError("existing pages are outside --pages; use a new --output-root")
        seen_pages.add(page)
    by_page: dict[int, list[dict[str, Any]]] = {}
    for row in detection_rows:
        page = row.get("page_number")
        if not isinstance(page, int) or page not in requested_pages:
            continue
        by_page.setdefault(page, []).append(row)
    committed: dict[int, dict[str, Any]] = {}
    for row in page_rows:
        page = row["page_number"]
        detections = by_page.get(page, [])
        figure_count = sum(bool(item.get("is_figure")) for item in detections)
        if (
            row.get("status") == "complete"
            and row.get("detection_count") == len(detections)
            and row.get("figure_count") == figure_count
        ):
            committed[page] = row
    kept_detections = [row for page in committed for row in by_page.get(page, [])]
    kept_detections.sort(key=lambda row: (row["page_number"], row["detection_id"]))
    kept_pages = [committed[page] for page in sorted(committed)]
    write_jsonl_atomic(pages_path, kept_pages)
    write_jsonl_atomic(detections_path, kept_detections)
    return committed, kept_detections


class Detector:
    def __init__(self, mode: str, confidence: float, weights: Path | None):
        self.mode = mode
        self.confidence = confidence
        self.session: Any = None
        self.classes = dict(DOC_CLASSES)
        self.processor: Any = None
        self.model: Any = None
        if mode == "orphan_ink":
            return
        if weights is None:
            raise ValueError(f"--{mode.replace('_', '-')}-weights is required")
        if not weights.exists():
            raise FileNotFoundError(f"{mode} weights not found: {weights}")
        if mode == "doclayout_yolo":
            import onnxruntime as ort

            providers = [
                name
                for name in ("CUDAExecutionProvider", "CPUExecutionProvider")
                if name in ort.get_available_providers()
            ] or ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(weights), providers=providers)
            names = self.session.get_modelmeta().custom_metadata_map.get("names")
            if names:
                self.classes = {
                    int(key): str(value) for key, value in ast.literal_eval(names).items()
                }
            self.input_name = self.session.get_inputs()[0].name
            self.provider = providers[0]
        else:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection

            self.processor = AutoImageProcessor.from_pretrained(str(weights))
            self.model = AutoModelForObjectDetection.from_pretrained(str(weights)).eval()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

    def orphan(self, image: np.ndarray, pdf_page: Any) -> list[dict[str, Any]]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, inverted = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        height, width = gray.shape
        covered = np.zeros((height, width), np.uint8)
        sx, sy = width / pdf_page.rect.width, height / pdf_page.rect.height
        for row in pdf_page.get_text("words"):
            x0, y0, x1, y1 = (float(value) for value in row[:4])
            cv2.rectangle(
                covered,
                (int(x0 * sx) - 3, int(y0 * sy) - 3),
                (int(x1 * sx) + 3, int(y1 * sy) + 3),
                255,
                -1,
            )
        mask = ((inverted > 0) & (covered == 0)).astype(np.uint8) * 255
        kernel = max(3, int(0.022 * min(height, width)) | 1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        boxes: list[list[int]] = []
        for index in range(1, count):
            x, y, bw, bh, area = (int(value) for value in stats[index])
            if bw < 0.07 * width or bh < 0.045 * height or area < 0.006 * width * height:
                continue
            if bw > 0.97 * width and bh > 0.97 * height:
                continue
            boxes.append([x, y, x + bw, y + bh])
        merged = merge_boxes(boxes, gap=int(0.02 * min(height, width)))
        return [
            {"cls": "figure", "score": 0.0, "bbox": list(box)}
            for box in sorted(merged, key=lambda box: (box[1], box[0]))
        ]

    def detect(self, image: np.ndarray, pdf_page: Any) -> list[dict[str, Any]]:
        if self.mode == "orphan_ink":
            return self.orphan(image, pdf_page)
        height, width = image.shape[:2]
        if self.mode == "doclayout_yolo":
            size = 1024
            scale = min(size / height, size / width)
            nh, nw = int(round(height * scale)), int(round(width * scale))
            canvas: np.ndarray = np.full((size, size, 3), 114, np.uint8)
            canvas[:nh, :nw] = cv2.resize(image, (nw, nh))
            # PyMuPDF's csRGB pixmap is already RGB; feed the ONNX model RGB
            # channels after letterboxing (the official reader converts BGR to RGB).
            tensor = canvas.transpose(2, 0, 1)[None]
            tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255.0
            outputs = self.session.run(None, {self.input_name: tensor})[0][0]
            result = []
            for x0, y0, x1, y1, score, class_id in outputs:
                if float(score) < self.confidence:
                    continue
                result.append(
                    {
                        "cls": self.classes.get(int(class_id), str(int(class_id))),
                        "score": float(score),
                        "bbox": [
                            max(0.0, x0 / scale),
                            max(0.0, y0 / scale),
                            min(width, x1 / scale),
                            min(height, y1 / scale),
                        ],
                    }
                )
            return result
        import torch
        from PIL import Image

        pil = Image.fromarray(image)
        with torch.no_grad():
            output = self.model(**self.processor(images=[pil], return_tensors="pt").to(self.device))
        for key in ("logits", "pred_boxes"):
            if getattr(output, key, None) is not None:
                setattr(output, key, getattr(output, key).cpu())
        result = self.processor.post_process_object_detection(
            output, target_sizes=torch.tensor([pil.size[::-1]])
        )[0]
        return [
            {
                "cls": self.model.config.id2label[label.item()],
                "score": float(score.item()),
                "bbox": [float(value) for value in box.tolist()],
            }
            for score, label, box in zip(
                result["scores"], result["labels"], result["boxes"], strict=True
            )
            if float(score.item()) >= self.confidence
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("orphan_ink", "doclayout_yolo", "ppdoclayout_v3"),
        required=True,
    )
    parser.add_argument(
        "--pages",
        default="34,74,987",
        help="1-based comma-separated pages, or 'all'; default is preflight set",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument(
        "--chandra",
        type=Path,
        required=True,
        help="Chandra chunks.jsonl used as a provisional layout reference",
    )
    parser.add_argument("--doclayout-yolo-weights", type=Path)
    parser.add_argument("--ppdoclayout-v3-weights", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.pdf.is_file():
        raise FileNotFoundError(f"Pierce PDF not found: {args.pdf}")
    if not args.chandra.is_file():
        raise FileNotFoundError(f"Chandra reference not found: {args.chandra}")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    with fitz.open(str(args.pdf)) as source:
        page_count = source.page_count
    if args.pages.strip().lower() == "all":
        pages = list(range(1, page_count + 1))
    else:
        pages = sorted({int(value) for value in args.pages.split(",")})
    if any(page < 1 or page > page_count for page in pages):
        raise ValueError(f"pages must be within 1..{page_count}")
    mode_dir = args.output_root / args.mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    pages_path = mode_dir / "pages.jsonl"
    detections_path = mode_dir / "detections.jsonl"
    summary_path = mode_dir / "summary.json"
    log_path = mode_dir / "run.log"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("mode") != args.mode:
            raise ValueError("existing summary mode differs; use a new --output-root")
        if previous.get("dpi") != args.dpi or previous.get("confidence") != args.confidence:
            raise ValueError("existing summary settings differ; use a new --output-root")
    requested = set(pages)
    committed, detection_rows = recover_outputs(pages_path, detections_path, requested)
    done = set(committed)
    model_path = {
        "orphan_ink": None,
        "doclayout_yolo": args.doclayout_yolo_weights,
        "ppdoclayout_v3": args.ppdoclayout_v3_weights,
    }[args.mode]
    detector = Detector(args.mode, args.confidence, model_path)
    gt, observed_reference_pages = load_reference(args.chandra)
    started = time.perf_counter()
    process = psutil.Process()
    max_rss = max((int(row.get("rss_bytes", 0)) for row in committed.values()), default=0)

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    log(
        f"start mode={args.mode} pages={len(pages)} resumed={len(done)} "
        f"dpi={args.dpi} confidence={args.confidence}"
    )
    with fitz.open(str(args.pdf)) as document:
        for page_number in pages:
            if page_number in done:
                continue
            page_started = time.perf_counter()
            pdf_page = document.load_page(page_number - 1)
            pixmap = pdf_page.get_pixmap(dpi=args.dpi, colorspace=fitz.csRGB, alpha=False)
            image = np.frombuffer(pixmap.samples, np.uint8).reshape(pixmap.height, pixmap.width, 3)
            detections = detector.detect(image, pdf_page)
            height, width = image.shape[:2]
            elapsed_seconds = time.perf_counter() - page_started
            rss_bytes = process.memory_info().rss
            max_rss = max(max_rss, rss_bytes)
            page_detection_rows: list[dict[str, Any]] = []
            for index, detection in enumerate(detections):
                bbox = clamp_bbox([float(value) for value in detection["bbox"]], width, height)
                class_name = str(detection["cls"])
                page_detection_rows.append(
                    {
                        "detection_id": f"p{page_number:04d}-d{index:03d}",
                        "page_number": page_number,
                        "page_id": f"p{page_number:04d}",
                        "class_name": class_name,
                        "score": float(detection["score"]),
                        "bbox_px": bbox,
                        "bbox_norm": normalize(bbox, width, height),
                        "is_figure": class_name.lower() in PREDICTED_FIGURE_LABELS,
                    }
                )
            with detections_path.open("a", encoding="utf-8") as handle:
                for row in page_detection_rows:
                    handle.write(json.dumps(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            page_row = {
                "page_number": page_number,
                "page_id": f"p{page_number:04d}",
                "status": "complete",
                "width": width,
                "height": height,
                "landscape": width > height,
                "detection_count": len(page_detection_rows),
                "figure_count": sum(1 for row in page_detection_rows if row["is_figure"]),
                "runtime_ms": int(round(elapsed_seconds * 1000)),
                "rss_bytes": rss_bytes,
            }
            with pages_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(page_row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            committed[page_number] = page_row
            detection_rows.extend(page_detection_rows)
            done.add(page_number)
            log(
                f"page={page_number} detections={len(page_detection_rows)} "
                f"figures={page_row['figure_count']} runtime_ms={page_row['runtime_ms']} "
                f"rss_mb={rss_bytes / 1048576:.1f}"
            )
            if len(done) == 3 or len(done) % 200 == 0:
                log(
                    f"checkpoint completed={len(done)}/{len(pages)} "
                    f"elapsed_s={time.perf_counter() - started:.2f}"
                )
    if len(done) == len(pages):
        page_rows = [committed[page] for page in sorted(committed)]
        metrics = score(
            page_rows,
            detection_rows,
            pages,
            gt,
            observed_reference_pages,
        )
        summary = {
            "status": "complete",
            "mode": args.mode,
            "model_filename": model_path.name if model_path else None,
            "pdf_filename": args.pdf.name,
            "chandra_filename": args.chandra.name,
            "page_count_pdf": page_count,
            "pages_requested": len(pages),
            "pages_completed": len(done),
            "dpi": args.dpi,
            "confidence": args.confidence,
            "environment": {
                "python": sys.executable,
                "platform": platform.platform(),
                "dependencies": versions(),
            },
            "counts": {
                "detections": len(detection_rows),
                "figure_detections": sum(bool(row["is_figure"]) for row in detection_rows),
            },
            "runtime_seconds": metrics["runtime_seconds"],
            "rss_max_bytes": max_rss,
            "chandra": metrics,
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        log(
            f"complete pages={len(done)} detections={len(detection_rows)} "
            f"runtime_s={metrics['runtime_seconds']:.3f} rss_max_mb={max_rss / 1048576:.1f}"
        )
        print(json.dumps(summary, indent=2))


def score(
    page_rows: list[dict[str, Any]],
    detection_rows: list[dict[str, Any]],
    pages: list[int],
    gt: dict[int, list[list[float]]],
    observed_reference_pages: set[int],
) -> dict[str, Any]:
    assert len(page_rows) == len({row["page_number"] for row in page_rows})
    assert {row["page_number"] for row in page_rows} == set(pages)
    scored_pages = set(pages) & observed_reference_pages
    truth_pages = set(gt) & scored_pages
    pred_by_page: dict[int, list[list[float]]] = {}
    for row in detection_rows:
        page = row["page_number"]
        if page not in scored_pages or not row.get("is_figure"):
            continue
        pred_by_page.setdefault(page, []).append(row["bbox_norm"])
    tp = fp = fn = 0
    matched: list[float] = []
    failures = []
    for page in sorted(truth_pages | set(pred_by_page)):
        pred = pred_by_page.get(page, [])
        expected = gt.get(page, [])
        scores = greedy(pred, expected)
        tp += len(scores)
        fp += len(pred) - len(scores)
        fn += len(expected) - len(scores)
        matched.extend(scores)
        if len(pred) != len(scores) or len(expected) != len(scores):
            failures.append(
                {
                    "page": page,
                    "tp": len(scores),
                    "fp": len(pred) - len(scores),
                    "fn": len(expected) - len(scores),
                }
            )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    predicted_presence = {page for page, boxes in pred_by_page.items() if boxes}
    presence_tp = len(predicted_presence & truth_pages)
    presence_fp = len(predicted_presence - truth_pages)
    presence_fn = len(truth_pages - predicted_presence)
    presence_precision = (
        presence_tp / (presence_tp + presence_fp) if presence_tp + presence_fp else 0.0
    )
    presence_recall = (
        presence_tp / (presence_tp + presence_fn) if presence_tp + presence_fn else 0.0
    )
    presence_f1 = (
        2 * presence_precision * presence_recall / (presence_precision + presence_recall)
        if presence_precision + presence_recall
        else 0.0
    )
    runtime_seconds = sum(float(row.get("runtime_ms", 0.0)) for row in page_rows) / 1000
    rss_values = [int(row.get("rss_bytes", 0)) for row in page_rows]
    return {
        "pages_requested": len(pages),
        "pages_scored": len(scored_pages),
        "chandra_observed_gt_pages": len(truth_pages),
        "chandra_gt_boxes": sum(len(gt.get(page, [])) for page in truth_pages),
        "chandra_total_observed_gt_pages": len(gt),
        "chandra_total_gt_boxes": sum(len(boxes) for boxes in gt.values()),
        "prediction_boxes": sum(len(pred_by_page.get(page, [])) for page in scored_pages),
        "box_iou_at_0_5": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_matched_iou": sum(matched) / len(matched) if matched else 0.0,
        },
        "page_presence": {
            "tp": presence_tp,
            "fp": presence_fp,
            "fn": presence_fn,
            "precision": presence_precision,
            "recall": presence_recall,
            "f1": presence_f1,
        },
        "failure_pages": failures,
        "excluded_missing_chandra_pages": sorted(set(pages) - scored_pages),
        "runtime_seconds": runtime_seconds,
        "rss_max_bytes": max(rss_values, default=0),
    }


if __name__ == "__main__":
    main()
