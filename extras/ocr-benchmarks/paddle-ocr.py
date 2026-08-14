# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PaddleOCR benchmark
#
# Runs the same 24 Pierce pages in two modes:
#
# 1. full-page PP-OCRv6 detection and recognition;
# 2. PP-DocLayoutV3 non-figure crops followed by PP-OCRv6 line detection and
#    recognition inside each crop.
#
# This runner is network-free. Attach the Paddle OCR family asset and the
# Pierce layout/output bundle. It writes one downloadable
# `paddle-ocr-benchmark.zip` archive under `/kaggle/working`.

# %%
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

os.environ["PADDLE_PDX_CACHE_HOME"] = "/kaggle/working/paddlex-cache"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["DISABLE_PADDLE_UPDATE_CHECK"] = "1"
os.environ["PADDLEOCR_SHOW_LOG"] = "False"

INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUT = WORK / "paddle-ocr-benchmark"
EXTRACTED_INPUT = WORK / "pierce-layout-input"
PAGES = [f"p{i:04d}" for i in range(24, 48)]
ASSET_NAME = "paddle-ocr-family-offline-assets"
DETECTION_MODEL_ID = "PaddlePaddle/PP-OCRv6_medium_det"
RECOGNITION_MODEL_ID = "PaddlePaddle/PP-OCRv6_medium_rec"
MODES = ("full-page", "ppdoclayout-v3")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            target = (destination / item.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"unsafe ZIP member: {item.filename}")
        archive.extractall(destination)


def prepare_support_roots() -> list[Path]:
    roots = [INPUT_ROOT]
    direct_labels = [
        path for path in INPUT_ROOT.rglob("labels.jsonl") if not path.is_relative_to(WORK)
    ]
    direct_layout = [
        path
        for path in INPUT_ROOT.rglob("detections.jsonl")
        if path.parent.name.lower() == "ppdoclayout-v3" and not path.is_relative_to(WORK)
    ]
    direct_pages = [
        path
        for path in INPUT_ROOT.rglob("heldout_pages")
        if path.is_dir()
        and all((path / f"{pid}.jpg").is_file() for pid in PAGES)
        and not path.is_relative_to(WORK)
    ]
    if direct_labels and direct_layout and direct_pages:
        return roots

    archives: list[Path] = []
    for path in sorted(INPUT_ROOT.rglob("*.zip")):
        if path.is_relative_to(WORK):
            continue
        try:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
                if any("labels.jsonl" in name for name in names) and any(
                    "detections.jsonl" in name for name in names
                ):
                    archives.append(path)
        except (zipfile.BadZipFile, OSError):
            continue

    if not archives:
        archives = sorted(
            path
            for path in INPUT_ROOT.rglob("*.zip")
            if not path.is_relative_to(WORK)
            and any(term in path.name.lower() for term in ("pierce", "layout", "bundle"))
            and not path.name.lower().startswith("paddle")
        )

    if len(archives) != 1:
        raise FileNotFoundError(
            "Attach the Pierce layout/output bundle; expected one layout-output ZIP "
            f"when direct files are absent, found {archives}"
        )
    if EXTRACTED_INPUT.exists():
        shutil.rmtree(EXTRACTED_INPUT)
    safe_extract(archives[0], EXTRACTED_INPUT)
    roots.append(EXTRACTED_INPUT)
    return roots


def find_receipt() -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in INPUT_ROOT.rglob("asset-receipt.json"):
        if path.is_relative_to(WORK):
            continue
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if receipt.get("asset") == ASSET_NAME:
            matches.append((path.parent, receipt))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one attached {ASSET_NAME} receipt, found {len(matches)}")
    return matches[0]


def model_paths(asset_root: Path, receipt: dict[str, Any]) -> tuple[Path, Path]:
    entries = {
        str(model.get("model_id")): model
        for model in receipt.get("models", [])
        if isinstance(model, dict)
    }
    paths: list[Path] = []
    for model_id in (DETECTION_MODEL_ID, RECOGNITION_MODEL_ID):
        entry = entries.get(model_id)
        if entry is None:
            raise ValueError(f"asset receipt does not contain {model_id}")
        model_dir = (asset_root / str(entry.get("directory", ""))).resolve()
        required = entry.get("required_files", [])
        missing = [name for name in required if not (model_dir / str(name)).is_file()]
        if not model_dir.is_dir():
            raise FileNotFoundError(f"model directory does not exist for {model_id}: {model_dir}")
        if missing:
            raise FileNotFoundError(
                f"incomplete local model {model_id}: missing {missing} in {model_dir}"
            )
        if not required and not any(model_dir.iterdir()):
            raise FileNotFoundError(f"model directory is empty for {model_id}: {model_dir}")
        paths.append(model_dir)
    return paths[0], paths[1]


def install_runtime(asset_root: Path) -> None:
    wheel_dir = asset_root / "wheels"
    if not wheel_dir.is_dir():
        raise FileNotFoundError(f"wheels directory not found under {asset_root}")
    wheels = sorted(
        path for path in wheel_dir.iterdir() if path.is_file() and path.name.endswith(".whl")
    )
    if not wheels:
        raise FileNotFoundError(f"no offline wheels found under {wheel_dir}")

    has_cuda_paddle = False
    try:
        res = subprocess.run(
            [
                sys.executable,
                "-c",
                "import paddle; print(bool(paddle.is_compiled_with_cuda()))",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip().lower() == "true":
            has_cuda_paddle = True
    except Exception:
        has_cuda_paddle = False

    selected = []
    for wheel in wheels:
        normalized = wheel.name.lower().replace("_", "-")
        if has_cuda_paddle and (
            normalized.startswith("paddlepaddle-")
            or normalized.startswith("paddlepaddle_gpu-")
            or normalized.startswith("paddlepaddle_cu")
        ):
            continue
        selected.append(wheel)

    if selected:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-index",
                "--no-deps",
                "--upgrade",
                *map(str, selected),
            ],
            check=True,
        )
    for name in list(sys.modules):
        if name in ("PIL", "paddle", "paddleocr", "paddlex") or name.startswith(
            ("PIL.", "paddle.", "paddleocr.", "paddlex.")
        ):
            del sys.modules[name]
    importlib.invalidate_caches()


def find_support_files(roots: list[Path]) -> tuple[Path, Path, Path]:
    labels_matches: list[Path] = []
    layout_matches: list[Path] = []
    heldout_matches: list[Path] = []
    for root in roots:
        for path in root.rglob("labels.jsonl"):
            try:
                page_ids = [row.get("page_id") for row in read_jsonl(path)]
            except (json.JSONDecodeError, OSError):
                continue
            if page_ids == PAGES:
                labels_matches.append(path)
        layout_matches.extend(
            path
            for path in root.rglob("detections.jsonl")
            if path.parent.name.lower() == "ppdoclayout-v3"
        )
        heldout_matches.extend(
            path
            for path in root.rglob("heldout_pages")
            if path.is_dir() and all((path / f"{pid}.jpg").is_file() for pid in PAGES)
        )
        if not heldout_matches:
            heldout_matches.extend(
                path
                for path in root.rglob("*")
                if path.is_dir() and all((path / f"{pid}.jpg").is_file() for pid in PAGES)
            )

    labels_matches = sorted(set(labels_matches))
    layout_matches = sorted(set(layout_matches))
    heldout_matches = sorted(set(heldout_matches))

    if len(roots) > 1 and EXTRACTED_INPUT in roots:
        extracted_labels = [p for p in labels_matches if p.is_relative_to(EXTRACTED_INPUT)]
        if len(extracted_labels) == 1:
            labels_matches = extracted_labels
        extracted_layout = [p for p in layout_matches if p.is_relative_to(EXTRACTED_INPUT)]
        if len(extracted_layout) == 1:
            layout_matches = extracted_layout
        extracted_heldout = [p for p in heldout_matches if p.is_relative_to(EXTRACTED_INPUT)]
        if len(extracted_heldout) == 1:
            heldout_matches = extracted_heldout

    if len(labels_matches) != 1 or len(layout_matches) != 1 or len(heldout_matches) != 1:
        raise FileNotFoundError(
            "support bundle must provide exactly one held-out labels file, image folder, "
            "and PP-DocLayoutV3 detections file; found "
            f"labels={labels_matches}, pages={heldout_matches}, layout={layout_matches}"
        )
    return heldout_matches[0], labels_matches[0], layout_matches[0]


def labels_by_page(path: Path) -> dict[str, str]:
    labels = {str(row["page_id"]): str(row["text"]) for row in read_jsonl(path)}
    if list(labels) != PAGES:
        raise ValueError("labels must contain exactly p0024 through p0047 in order")
    return labels


def load_regions(path: Path) -> dict[str, list[dict[str, Any]]]:
    regions: dict[str, list[dict[str, Any]]] = {pid: [] for pid in PAGES}
    for line_number, row in enumerate(read_jsonl(path), 1):
        pid = row.get("page_id")
        if pid not in regions or bool(row.get("is_figure", False)):
            continue
        box = row.get("bbox_norm")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"invalid PP-DocLayoutV3 box at line {line_number}")
        values = [max(0.0, min(1.0, float(v))) for v in box]
        if values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError(f"non-positive PP-DocLayoutV3 box at line {line_number}: {box}")
        regions[pid].append(
            {
                "source_id": str(row.get("detection_id", f"line-{line_number}")),
                "class_name": str(row.get("class_name", "region")),
                "score": float(row.get("score", 0.0)),
                "bbox_norm": values,
            }
        )
    for pid in PAGES:
        regions[pid].sort(
            key=lambda row: (
                row["bbox_norm"][1],
                row["bbox_norm"][0],
                row["bbox_norm"][3],
                row["bbox_norm"][2],
                row["source_id"],
            )
        )
        if not regions[pid]:
            raise ValueError(f"PP-DocLayoutV3 has no non-figure regions for {pid}")
    return regions


def bbox_pixels(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, min(width - 1, int(box[0] * width)))
    top = max(0, min(height - 1, int(box[1] * height)))
    right = max(left + 1, min(width, int(box[2] * width + 0.999999)))
    bottom = max(top + 1, min(height, int(box[3] * height + 0.999999)))
    return left, top, right, bottom


def latin_ratio(text: str) -> float:
    letters = [character for character in text if character.isalpha()]
    return sum(character.isascii() for character in letters) / max(1, len(letters))


def lines_from(result: Any) -> list[dict[str, Any]]:
    data: dict[str, Any] = {}
    if hasattr(result, "json"):
        payload = result.json() if callable(result.json) else result.json
        if isinstance(payload, str):
            try:
                data = json.loads(payload)
            except Exception:
                data = {}
        elif isinstance(payload, dict):
            data = payload
    elif hasattr(result, "res") and isinstance(result.res, dict):
        data = result.res
    elif isinstance(result, dict):
        data = result

    if isinstance(data.get("res"), dict):
        data = data["res"]

    lines: list[dict[str, Any]] = []
    if "rec_texts" in data and isinstance(data["rec_texts"], list):
        texts = data["rec_texts"]
        scores = data.get("rec_scores", [])
        for index, value in enumerate(texts):
            text = str(value).strip()
            if not text:
                continue
            ratio = latin_ratio(text)
            score_val = (
                float(scores[index]) if index < len(scores) and scores[index] is not None else None
            )
            lines.append(
                {
                    "text": text,
                    "score": score_val,
                    "latin_ratio": ratio,
                    "suspicious_non_latin": any(char.isalpha() for char in text) and ratio < 0.8,
                }
            )
        return lines

    if isinstance(result, (list, tuple)):
        items = result[0] if (len(result) == 1 and isinstance(result[0], (list, tuple))) else result
        for item in items:
            if (
                isinstance(item, (list, tuple))
                and len(item) >= 2
                and isinstance(item[1], (list, tuple))
            ):
                text = str(item[1][0]).strip()
                score_val = (
                    float(item[1][1]) if len(item[1]) > 1 and item[1][1] is not None else None
                )
                if not text:
                    continue
                ratio = latin_ratio(text)
                lines.append(
                    {
                        "text": text,
                        "score": score_val,
                        "latin_ratio": ratio,
                        "suspicious_non_latin": any(char.isalpha() for char in text)
                        and ratio < 0.8,
                    }
                )
    return lines


def recognize(ocr: Any, source: Any) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    if hasattr(ocr, "predict"):
        prediction = ocr.predict(source)
        if isinstance(prediction, (list, tuple)):
            for result in prediction:
                lines.extend(lines_from(result))
        else:
            try:
                for result in iter(prediction):
                    lines.extend(lines_from(result))
            except TypeError:
                lines.extend(lines_from(prediction))
    elif hasattr(ocr, "ocr"):
        prediction = ocr.ocr(source, cls=False)
        lines.extend(lines_from(prediction))
    return lines


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def recover_mode(
    mode_dir: Path, run_signature: str
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    pages_path = mode_dir / "pages.jsonl"
    regions_path = mode_dir / "regions.jsonl"
    page_rows = read_jsonl(pages_path) if pages_path.is_file() else []
    region_rows = read_jsonl(regions_path) if regions_path.is_file() else []
    regions_by_page: dict[str, list[dict[str, Any]]] = {page_id: [] for page_id in PAGES}
    for row in region_rows:
        page_id = row.get("page_id")
        if isinstance(page_id, str) and page_id in regions_by_page:
            regions_by_page[page_id].append(row)
    complete: dict[str, dict[str, Any]] = {}
    for row in page_rows:
        page_id = row.get("page_id")
        rows = regions_by_page.get(page_id, []) if isinstance(page_id, str) else []
        region_ids = [region.get("region_id") for region in rows]
        if (
            page_id in regions_by_page
            and row.get("status") == "complete"
            and row.get("run_signature") == run_signature
            and bool(rows)
            and all(region.get("run_signature") == run_signature for region in rows)
            and row.get("region_count") == len(rows)
            and len(region_ids) == len(set(region_ids))
            and page_id not in complete
        ):
            complete[page_id] = row
    kept_pages = [complete[page_id] for page_id in PAGES if page_id in complete]
    kept_regions = [
        region for page_id in PAGES if page_id in complete for region in regions_by_page[page_id]
    ]
    mode_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pages_path, kept_pages)
    write_jsonl(regions_path, kept_regions)
    return complete, {page_id: regions_by_page[page_id] for page_id in complete}


def normalize_exact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str) -> str:
    """Normalize typography while preserving letters and numbers for fair OCR scoring."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    characters = [char if unicodedata.category(char)[0] in {"L", "N"} else " " for char in folded]
    return re.sub(r"\s+", " ", "".join(characters)).strip()


def levenshtein(left: list[Any] | str, right: list[Any] | str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index, value in enumerate(left, 1):
        current = [index]
        for other_index, other in enumerate(right, 1):
            current.append(
                min(
                    previous[other_index] + 1,
                    current[-1] + 1,
                    previous[other_index - 1] + (value != other),
                )
            )
        previous = current
    return previous[-1]


def word_f1(hypothesis: str, reference: str) -> float:
    hyp = Counter(hypothesis.split())
    ref = Counter(reference.split())
    true_positive = sum((hyp & ref).values())
    if not hyp and not ref:
        return 1.0
    if not true_positive:
        return 0.0
    precision = true_positive / sum(hyp.values())
    recall = true_positive / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def score(
    mode_dir: Path,
    labels: dict[str, str],
    engine: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    pages = read_jsonl(mode_dir / "pages.jsonl")
    regions = read_jsonl(mode_dir / "regions.jsonl")
    page_ids = [row["page_id"] for row in pages]
    if page_ids != PAGES or any(row.get("status") != "complete" for row in pages):
        raise ValueError(f"{mode_dir} does not contain exactly 24 completed pages")
    if len({row["region_id"] for row in regions}) != len(regions):
        raise ValueError(f"{mode_dir} contains duplicate region IDs")

    rows = []
    total_ce = total_we = total_chars = total_words = 0
    exact_total_ce = exact_total_we = exact_total_chars = exact_total_words = 0
    for page in pages:
        exact_reference = normalize_exact(labels[page["page_id"]])
        exact_hypothesis = normalize_exact(page.get("text", ""))
        reference = normalize(exact_reference)
        hypothesis = normalize(exact_hypothesis)
        reference_words, hypothesis_words = reference.split(), hypothesis.split()
        char_errors = levenshtein(hypothesis, reference)
        word_errors = levenshtein(hypothesis_words, reference_words)
        exact_char_errors = levenshtein(exact_hypothesis, exact_reference)
        exact_word_errors = levenshtein(exact_hypothesis.split(), exact_reference.split())
        rows.append(
            {
                "page_id": page["page_id"],
                "cer": char_errors / max(1, len(reference)),
                "wer": word_errors / max(1, len(reference_words)),
                "word_f1": word_f1(hypothesis, reference),
                "exact_cer": exact_char_errors / max(1, len(exact_reference)),
                "exact_wer": exact_word_errors / max(1, len(exact_reference.split())),
                "exact_word_f1": word_f1(exact_hypothesis, exact_reference),
                "reference_chars": len(reference),
                "hypothesis_chars": len(hypothesis),
                "reference_words": len(reference_words),
                "hypothesis_words": len(hypothesis_words),
                "regions": int(page.get("region_count", 0)),
                "elapsed_seconds": float(page.get("elapsed_seconds", 0.0)),
            }
        )
        total_ce += char_errors
        total_we += word_errors
        total_chars += len(reference)
        total_words += len(reference_words)
        exact_total_ce += exact_char_errors
        exact_total_we += exact_word_errors
        exact_total_chars += len(exact_reference)
        exact_total_words += len(exact_reference.split())

    metrics = {
        "engine": engine,
        "models": provenance["models"],
        "runtime": provenance["runtime"],
        "primary_scoring": "NFKC, casefold, letters/numbers only, collapsed whitespace",
        "mode": mode_dir.name,
        "pages": len(rows),
        "regions": len(regions),
        "recognized_lines": sum(
            int(row.get("line_count", len(row.get("lines", [])))) for row in regions
        ),
        "suspicious_non_latin_lines": sum(
            int(line.get("suspicious_non_latin", False))
            for region in regions
            for line in region.get("lines", [])
        ),
        "micro_cer": total_ce / max(1, total_chars),
        "micro_wer": total_we / max(1, total_words),
        "macro_cer": sum(row["cer"] for row in rows) / len(rows),
        "macro_wer": sum(row["wer"] for row in rows) / len(rows),
        "macro_word_f1": sum(row["word_f1"] for row in rows) / len(rows),
        "exact_micro_cer": exact_total_ce / max(1, exact_total_chars),
        "exact_micro_wer": exact_total_we / max(1, exact_total_words),
        "exact_macro_cer": sum(row["exact_cer"] for row in rows) / len(rows),
        "exact_macro_wer": sum(row["exact_wer"] for row in rows) / len(rows),
        "exact_macro_word_f1": sum(row["exact_word_f1"] for row in rows) / len(rows),
        "per_page": rows,
    }
    (mode_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def run_mode(
    ocr: Any,
    mode: str,
    heldout: Path,
    labels: dict[str, str],
    layout_regions: dict[str, list[dict[str, Any]]],
    run_signature: str,
    provenance: dict[str, Any],
) -> None:
    from PIL import Image

    mode_dir = OUT / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    pages_path, regions_path = mode_dir / "pages.jsonl", mode_dir / "regions.jsonl"
    complete, existing_regions = recover_mode(mode_dir, run_signature)
    page_rows = list(complete.values())
    region_rows = [region for rows in existing_regions.values() for region in rows]
    for pid in PAGES:
        if pid in complete:
            continue
        started = time.perf_counter()
        with Image.open(heldout / f"{pid}.jpg") as opened:
            image = opened.convert("RGB")
        sources = (
            [
                {
                    "source_id": "full-page",
                    "class_name": "full-page",
                    "score": 1.0,
                    "bbox_norm": [0.0, 0.0, 1.0, 1.0],
                }
            ]
            if mode == "full-page"
            else layout_regions[pid]
        )
        new_regions = []
        with tempfile.TemporaryDirectory(prefix=f"paddle-{pid}-") as scratch_name:
            scratch = Path(scratch_name)
            for index, source in enumerate(sources):
                box = source["bbox_norm"]
                pixel_box = bbox_pixels(box, image.width, image.height)
                crop_path = scratch / f"{pid}-r{index:04d}.png"
                image.crop(pixel_box).save(crop_path)
                lines = recognize(ocr, str(crop_path))
                new_regions.append(
                    {
                        "region_id": f"{pid}-r{index:04d}",
                        "page_id": pid,
                        "source_id": source["source_id"],
                        "source_class": source["class_name"],
                        "score": source["score"],
                        "bbox_norm": box,
                        "bbox_px": list(pixel_box),
                        "text": "\n".join(line["text"] for line in lines),
                        "line_count": len(lines),
                        "lines": lines,
                        "status": "complete",
                        "run_signature": run_signature,
                    }
                )
        page_rows.append(
            {
                "page_id": pid,
                "mode": mode,
                "status": "complete",
                "region_count": len(new_regions),
                "region_ids": [row["region_id"] for row in new_regions],
                "text": "\n".join(row["text"] for row in new_regions if row["text"]),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "run_signature": run_signature,
            }
        )
        region_rows.extend(new_regions)
        page_rows.sort(key=lambda row: row["page_id"])
        region_rows.sort(key=lambda row: row["region_id"])
        write_jsonl(regions_path, region_rows)
        write_jsonl(pages_path, page_rows)
        print(mode, pid, len(new_regions), flush=True)
    metrics = score(mode_dir, labels, provenance["engine"], provenance)
    selected = {key: metrics[key] for key in ("micro_cer", "micro_wer", "macro_word_f1")}
    print(mode, json.dumps(selected), flush=True)


def receipt_revision(receipt: dict[str, Any], model_id: str) -> str:
    for model in receipt.get("models", []):
        if isinstance(model, dict) and model.get("model_id") == model_id:
            return str(model.get("revision", "unknown"))
    return "unknown"


def main() -> None:
    if not INPUT_ROOT.is_dir() or not WORK.is_dir():
        raise RuntimeError("run this file inside a Kaggle notebook")
    asset_root, receipt = find_receipt()
    detection_dir, recognition_dir = model_paths(asset_root, receipt)
    roots = prepare_support_roots()
    heldout, labels_path, layout_path = find_support_files(roots)
    labels = labels_by_page(labels_path)
    layout_regions = load_regions(layout_path)

    install_runtime(asset_root)
    import paddle
    import paddleocr
    from paddleocr import PaddleOCR

    device = "gpu:0" if paddle.is_compiled_with_cuda() else "cpu"
    revisions = {
        "detection": receipt_revision(receipt, DETECTION_MODEL_ID),
        "recognition": receipt_revision(receipt, RECOGNITION_MODEL_ID),
    }
    signature_payload = {
        "asset": ASSET_NAME,
        "models": {
            "detection": {
                "id": DETECTION_MODEL_ID,
                "revision": revisions["detection"],
            },
            "recognition": {
                "id": RECOGNITION_MODEL_ID,
                "revision": revisions["recognition"],
            },
        },
        "scoring": "normalized-and-exact-v1",
    }
    run_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    provenance = {
        "engine": "PaddleOCR 3.7.0 PP-OCRv6 medium detector and recognizer",
        "models": signature_payload["models"],
        "runtime": {
            "offline_asset": ASSET_NAME,
            "paddleocr": getattr(paddleocr, "__version__", "unknown"),
            "paddlepaddle": getattr(paddle, "__version__", "unknown"),
            "device": device,
            "cuda_compiled": bool(paddle.is_compiled_with_cuda()),
            "run_signature": run_signature,
        },
    }
    print(json.dumps(provenance, indent=2), flush=True)

    ocr = PaddleOCR(
        lang="en",
        ocr_version="PP-OCRv6",
        text_detection_model_name="PP-OCRv6_medium_det",
        text_detection_model_dir=str(detection_dir),
        text_recognition_model_name="PP-OCRv6_medium_rec",
        text_recognition_model_dir=str(recognition_dir),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_recognition_batch_size=16,
        device=device,
    )
    smoke_lines = recognize(ocr, str(heldout / "p0024.jpg"))
    smoke_text = " ".join(line["text"] for line in smoke_lines)
    suspicious = sum(int(line.get("suspicious_non_latin", False)) for line in smoke_lines)
    ratio = latin_ratio(smoke_text)
    if len(smoke_text) < 500 or len(smoke_lines) < 5 or ratio < 0.9:
        raise RuntimeError(
            "PP-OCRv6 smoke test failed: expected substantial English text on p0024, "
            f"got {len(smoke_lines)} lines, {len(smoke_text)} chars, "
            f"Latin ratio {ratio:.3f}, suspicious lines {suspicious}"
        )
    print(
        json.dumps(
            {
                "smoke_page": "p0024",
                "lines": len(smoke_lines),
                "chars": len(smoke_text),
                "latin_ratio": round(ratio, 4),
                "suspicious_non_latin_lines": suspicious,
                "preview": smoke_text[:160],
            },
            indent=2,
        ),
        flush=True,
    )

    for mode in MODES:
        run_mode(
            ocr,
            mode,
            heldout,
            labels,
            layout_regions,
            run_signature,
            provenance,
        )
    mode_metrics = {
        mode: json.loads((OUT / mode / "metrics.json").read_text(encoding="utf-8"))
        for mode in MODES
    }
    comparison = {
        "engine": provenance["engine"],
        "models": provenance["models"],
        "runtime": provenance["runtime"],
        "pages": len(PAGES),
        "modes": {
            mode: {key: value for key, value in metrics.items() if key != "per_page"}
            for mode, metrics in mode_metrics.items()
        },
        "per_page": [
            {
                "page_id": page_id,
                **{
                    mode: {
                        key: mode_metrics[mode]["per_page"][index][key]
                        for key in (
                            "cer",
                            "wer",
                            "word_f1",
                            "exact_cer",
                            "exact_wer",
                            "exact_word_f1",
                        )
                    }
                    for mode in MODES
                },
            }
            for index, page_id in enumerate(PAGES)
        ],
    }
    (OUT / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    archive = shutil.make_archive(str(WORK / "paddle-ocr-benchmark"), "zip", root_dir=OUT)
    print("download:", archive, flush=True)


if __name__ == "__main__":
    main()
