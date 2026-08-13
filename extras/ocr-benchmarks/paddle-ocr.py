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
# The same 24 held-out Pierce pages are processed in two modes: direct
# full-page OCR and OCR on non-figure regions supplied by PP-DocLayoutV3.
# Results are written to one downloadable `paddle-ocr-benchmark.zip` archive.

# %%
# Kaggle: Internet ON. A Tesla T4 is recommended.
# %pip install -q 'paddleocr==3.7.0'
# %pip install -q 'paddlepaddle-gpu==3.3.1' -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
# %pip install -q --upgrade --force-reinstall --no-deps \
#     'nvidia-nccl-cu12==2.27.5' \
#     'nvidia-nvjitlink-cu12==12.8.93' \
#     'nvidia-nvtx-cu12==12.8.90'
# %pip install -q 'pymupdf>=1.25,<1.26' 'pillow>=10,<12'

import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

os.environ["PADDLE_PDX_CACHE_HOME"] = "/kaggle/working/paddlex-cache"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

REPO = Path("/kaggle/working/doc-agent-G07")
OUT = Path("/kaggle/working/paddle-ocr-benchmark")
LAYOUT_PATH = REPO / "extras/output/ppdoclayout-v3/detections.jsonl"
HELDOUT = REPO / "grading_kit/heldout_pages"
LABELS = REPO / "grading_kit/labels.jsonl"
PAGES = [f"p{i:04d}" for i in range(24, 48)]


def ensure_repository() -> None:
    if not LABELS.is_file():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "main",
                "https://github.com/smammahdi/doc-agent-G07.git",
                str(REPO),
            ],
            check=True,
        )
    if not LABELS.is_file() or not LAYOUT_PATH.is_file():
        raise FileNotFoundError("main repository is missing labels or PP-DocLayoutV3 detections")
    missing = [pid for pid in PAGES if not (HELDOUT / f"{pid}.jpg").is_file()]
    if missing:
        raise FileNotFoundError(f"missing held-out pages: {missing}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def labels_by_page() -> dict[str, str]:
    rows = read_jsonl(LABELS)
    labels = {row["page_id"]: row["text"] for row in rows}
    if list(labels) != PAGES:
        raise ValueError("labels must contain exactly p0024 through p0047 in order")
    return labels


def load_regions() -> dict[str, list[dict[str, Any]]]:
    regions: dict[str, list[dict[str, Any]]] = {pid: [] for pid in PAGES}
    for row in read_jsonl(LAYOUT_PATH):
        pid = row.get("page_id")
        if pid not in regions or row.get("is_figure", False):
            continue
        box = [float(value) for value in row["bbox_norm"]]
        if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise ValueError(f"invalid PP-DocLayoutV3 box for {pid}: {box}")
        regions[pid].append(
            {
                "source_id": str(row.get("detection_id", f"line-{len(regions[pid])}")),
                "class_name": row.get("class_name", "region"),
                "score": float(row.get("score", 0.0)),
                "bbox_norm": box,
            }
        )
    for pid in PAGES:
        regions[pid].sort(key=lambda row: (row["bbox_norm"][1], row["bbox_norm"][0]))
    return regions


def bbox_pixels(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, min(width - 1, int(box[0] * width)))
    top = max(0, min(height - 1, int(box[1] * height)))
    right = max(left + 1, min(width, int(box[2] * width + 0.999999)))
    bottom = max(top + 1, min(height, int(box[3] * height + 0.999999)))
    return left, top, right, bottom


def result_dict(value: Any) -> dict[str, Any]:
    payload = getattr(value, "json", None)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict):
        data = payload.get("res", payload)
        return data if isinstance(data, dict) else {}
    return value if isinstance(value, dict) else {}


def lines_from(result: Any) -> list[dict[str, Any]]:
    data = result_dict(result)
    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])
    return [
        {"text": str(text), "score": float(scores[index]) if index < len(scores) else None}
        for index, text in enumerate(texts)
        if str(text).strip()
    ]


def recognize(ocr: Any, source: Any) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for result in ocr.predict(source):
        lines.extend(lines_from(result))
    return lines


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def recover_mode(
    mode_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    page_rows = read_jsonl(mode_dir / "pages.jsonl") if (mode_dir / "pages.jsonl").is_file() else []
    region_rows = (
        read_jsonl(mode_dir / "regions.jsonl") if (mode_dir / "regions.jsonl").is_file() else []
    )
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
    write_jsonl(mode_dir / "pages.jsonl", kept_pages)
    write_jsonl(mode_dir / "regions.jsonl", kept_regions)
    return complete, {page_id: regions_by_page[page_id] for page_id in complete}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


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
    hyp = Counter(normalize(hypothesis).split())
    ref = Counter(normalize(reference).split())
    true_positive = sum((hyp & ref).values())
    if not hyp and not ref:
        return 1.0
    if not true_positive:
        return 0.0
    precision = true_positive / sum(hyp.values())
    recall = true_positive / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def score(mode_dir: Path, labels: dict[str, str], engine: str) -> dict[str, Any]:
    pages = read_jsonl(mode_dir / "pages.jsonl")
    page_ids = [row["page_id"] for row in pages]
    if page_ids != PAGES or any(row["status"] != "complete" for row in pages):
        raise ValueError(f"{mode_dir} does not contain exactly 24 completed pages")
    rows = []
    total_ce = total_we = total_chars = total_words = 0
    for page in pages:
        reference = normalize(labels[page["page_id"]])
        hypothesis = normalize(page["text"])
        reference_words, hypothesis_words = reference.split(), hypothesis.split()
        char_errors = levenshtein(hypothesis, reference)
        word_errors = levenshtein(hypothesis_words, reference_words)
        row = {
            "page_id": page["page_id"],
            "cer": char_errors / max(1, len(reference)),
            "wer": word_errors / max(1, len(reference_words)),
            "word_f1": word_f1(hypothesis, reference),
            "reference_chars": len(reference),
            "reference_words": len(reference_words),
        }
        rows.append(row)
        total_ce += char_errors
        total_we += word_errors
        total_chars += len(reference)
        total_words += len(reference_words)
    metrics = {
        "engine": engine,
        "mode": mode_dir.name,
        "pages": len(rows),
        "micro_cer": total_ce / max(1, total_chars),
        "micro_wer": total_we / max(1, total_words),
        "macro_cer": sum(row["cer"] for row in rows) / len(rows),
        "macro_wer": sum(row["wer"] for row in rows) / len(rows),
        "macro_word_f1": sum(row["word_f1"] for row in rows) / len(rows),
        "per_page": rows,
    }
    (mode_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def run_mode(
    ocr: Any,
    mode: str,
    labels: dict[str, str],
    layout_regions: dict[str, list[dict[str, Any]]],
) -> None:
    mode_dir = OUT / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    pages_path, regions_path = mode_dir / "pages.jsonl", mode_dir / "regions.jsonl"
    complete, existing_regions = recover_mode(mode_dir)
    page_rows = list(complete.values())
    region_rows = [region for rows in existing_regions.values() for region in rows]
    completed = set(complete)
    for pid in PAGES:
        if pid in completed:
            continue
        started = time.perf_counter()
        image = Image.open(HELDOUT / f"{pid}.jpg").convert("RGB")
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
        for index, source in enumerate(sources):
            box = source["bbox_norm"]
            pixel_box = bbox_pixels(box, image.width, image.height)
            crop = image.crop(pixel_box)
            lines = recognize(ocr, np.asarray(crop))
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
                }
            )
        page_rows.append(
            {
                "page_id": pid,
                "mode": mode,
                "status": "complete",
                "region_count": len(new_regions),
                "region_ids": [row["region_id"] for row in new_regions],
                "text": "\n".join(row["text"] for row in new_regions),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        region_rows.extend(new_regions)
        page_rows.sort(key=lambda row: row["page_id"])
        region_rows.sort(key=lambda row: row["region_id"])
        write_jsonl(regions_path, region_rows)
        write_jsonl(pages_path, page_rows)
        print(mode, pid, len(new_regions), flush=True)
    metrics = score(mode_dir, labels, "PaddleOCR 3.7.0 PP-OCRv6")
    selected = {key: metrics[key] for key in ("micro_cer", "micro_wer", "macro_word_f1")}
    print(mode, json.dumps(selected))


def main() -> None:
    ensure_repository()
    labels = labels_by_page()
    layout_regions = load_regions()
    import paddle
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="gpu:0" if paddle.is_compiled_with_cuda() else "cpu",
    )
    run_mode(ocr, "full-page", labels, layout_regions)
    run_mode(ocr, "ppdoclayout-v3", labels, layout_regions)
    mode_metrics = {
        mode: json.loads((OUT / mode / "metrics.json").read_text())
        for mode in ("full-page", "ppdoclayout-v3")
    }
    comparison = {
        "engine": "PaddleOCR 3.7.0 PP-OCRv6",
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
                        for key in ("cer", "wer", "word_f1")
                    }
                    for mode in ("full-page", "ppdoclayout-v3")
                },
            }
            for index, page_id in enumerate(PAGES)
        ],
    }
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    archive = shutil.make_archive("/kaggle/working/paddle-ocr-benchmark", "zip", root_dir=OUT)
    print("download:", archive)


if __name__ == "__main__":
    main()
