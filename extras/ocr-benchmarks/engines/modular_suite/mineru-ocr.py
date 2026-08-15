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
# # MinerU OCR benchmark
#
# Run MinerU2.5-Pro on the same 24 Pierce pages in two modes: full-page input
# and existing PP-DocLayoutV3 non-figure crops. No external layout model is run.
# The script writes one downloadable `mineru-ocr-benchmark.zip` archive.

# %%
"""Benchmark MinerU2.5-Pro on full pages and PP-DocLayoutV3 regions.

Kaggle settings: Internet ON and Accelerator = Tesla T4. Run this file as a
Jupytext notebook or execute ``python mineru-ocr.py``. Dependencies and the
official 1.2B checkpoint are downloaded normally from PyPI/Hugging Face.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path("/kaggle/working/doc-agent-G07")
OUT = Path("/kaggle/working/mineru-ocr-benchmark")
MODEL_NAME = "opendatalab/MinerU2.5-Pro-2604-1.2B"
LAYOUT_PATH = REPO / "extras/layout-benchmarks/outputs/ppdoclayout-v3/detections.jsonl"
HELDOUT = REPO / "grading_kit/heldout_pages"
LABELS = REPO / "grading_kit/labels.jsonl"
PAGES = [f"p{number:04d}" for number in range(24, 48)]
MODES = ("full-page", "ppdoclayout-v3")


def install_dependencies() -> None:
    """Install the official lightweight Transformers client on Kaggle."""

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "mineru-vl-utils[transformers]==1.0.5",
            "transformers==4.57.6",
        ],
        check=True,
    )


def ensure_repository() -> dict[str, str]:
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
    labels = {
        row["page_id"]: row["text"]
        for line in LABELS.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    if list(labels) != PAGES:
        raise ValueError("labels must contain exactly p0024 through p0047 in order")
    missing = [page_id for page_id in PAGES if not (HELDOUT / f"{page_id}.jpg").is_file()]
    if missing:
        raise FileNotFoundError(f"missing held-out pages: {missing}")
    return labels


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_regions() -> dict[str, list[dict[str, Any]]]:
    regions: dict[str, list[dict[str, Any]]] = {page_id: [] for page_id in PAGES}
    for line_number, row in enumerate(read_jsonl(LAYOUT_PATH), 1):
        page_id = row.get("page_id")
        if page_id not in regions or bool(row.get("is_figure")):
            continue
        box = row.get("bbox_norm")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"invalid PP-DocLayoutV3 box at line {line_number}")
        values = [max(0.0, min(1.0, float(value))) for value in box]
        if values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError(f"non-positive PP-DocLayoutV3 box at line {line_number}")
        regions[page_id].append(
            {
                "source_id": str(row.get("detection_id", f"line-{line_number}")),
                "class_name": str(row.get("class_name", "text")),
                "score": float(row.get("score", 0.0)),
                "bbox_norm": values,
            }
        )
    for page_regions in regions.values():
        page_regions.sort(
            key=lambda row: (
                row["bbox_norm"][1],
                row["bbox_norm"][0],
                row["bbox_norm"][3],
                row["bbox_norm"][2],
                row["source_id"],
            )
        )
    return regions


def bbox_pixels(box: list[float], width: int, height: int) -> list[int]:
    left = max(0, min(width - 1, int(box[0] * width)))
    top = max(0, min(height - 1, int(box[1] * height)))
    right = max(left + 1, min(width, int(box[2] * width + 0.999999)))
    bottom = max(top + 1, min(height, int(box[3] * height + 0.999999)))
    return [left, top, right, bottom]


def load_model() -> tuple[Any, Any, str]:
    import torch
    from mineru_vl_utils import MinerUClient
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("MinerU2.5-Pro requires a CUDA GPU; select a Tesla T4 in Kaggle")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
    ).eval()
    client = MinerUClient(
        backend="transformers",
        model=model,
        processor=processor,
        image_analysis=False,
    )
    return client, model, torch.cuda.get_device_name(0)


def block_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return dict(block)
    model_dump = getattr(block, "model_dump", None)
    if callable(model_dump):
        value = model_dump()
        if isinstance(value, dict):
            return value
    return {
        "type": getattr(block, "type", "unknown"),
        "bbox": getattr(block, "bbox", None),
        "angle": getattr(block, "angle", None),
        "content": getattr(block, "content", None),
    }


def recognize(client: Any, image: Any) -> tuple[str, list[dict[str, Any]]]:
    blocks = [block_dict(block) for block in client.two_step_extract(image.convert("RGB"))]
    text = "\n\n".join(
        str(block["content"]).strip()
        for block in blocks
        if block.get("content") is not None and str(block["content"]).strip()
    )
    return text, blocks


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


def recover_mode(
    mode_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    page_rows = read_jsonl(mode_dir / "pages.jsonl")
    region_rows = read_jsonl(mode_dir / "regions.jsonl")
    regions_by_page: dict[str, list[dict[str, Any]]] = {page_id: [] for page_id in PAGES}
    for row in region_rows:
        page_id = row.get("page_id")
        if page_id in regions_by_page:
            regions_by_page[page_id].append(row)
    complete: dict[str, dict[str, Any]] = {}
    for row in page_rows:
        page_id = row.get("page_id")
        if (
            page_id in regions_by_page
            and row.get("status") == "complete"
            and int(row.get("region_count", -1)) == len(regions_by_page[page_id])
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


def score(
    mode: str,
    labels: dict[str, str],
    page_rows: dict[str, dict[str, Any]],
    region_count: int,
    device: str,
) -> dict[str, Any]:
    per_page: list[dict[str, Any]] = []
    char_errors = word_errors = total_chars = total_words = 0
    for page_id in PAGES:
        reference = normalize(labels[page_id])
        hypothesis = normalize(str(page_rows[page_id].get("text", "")))
        reference_words = reference.split()
        hypothesis_words = hypothesis.split()
        current_char_errors = levenshtein(hypothesis, reference)
        current_word_errors = levenshtein(hypothesis_words, reference_words)
        row = {
            "page_id": page_id,
            "cer": current_char_errors / max(1, len(reference)),
            "wer": current_word_errors / max(1, len(reference_words)),
            "word_f1": word_f1(hypothesis, reference),
            "reference_chars": len(reference),
            "reference_words": len(reference_words),
            "hypothesis_chars": len(hypothesis),
            "hypothesis_words": len(hypothesis_words),
        }
        per_page.append(row)
        char_errors += current_char_errors
        word_errors += current_word_errors
        total_chars += len(reference)
        total_words += len(reference_words)
    metrics = {
        "engine": MODEL_NAME,
        "mode": mode,
        "layout": (
            "full-page input" if mode == "full-page" else "PP-DocLayoutV3 non-figure regions"
        ),
        "device": device,
        "pages": len(per_page),
        "regions": region_count,
        "micro_cer": char_errors / max(1, total_chars),
        "micro_wer": word_errors / max(1, total_words),
        "macro_cer": sum(row["cer"] for row in per_page) / len(per_page),
        "macro_wer": sum(row["wer"] for row in per_page) / len(per_page),
        "macro_word_f1": sum(row["word_f1"] for row in per_page) / len(per_page),
        "per_page": per_page,
    }
    (OUT / mode / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def run_mode(
    mode: str,
    labels: dict[str, str],
    layout_regions: dict[str, list[dict[str, Any]]],
    client: Any,
    device: str,
) -> dict[str, Any]:
    from PIL import Image

    mode_dir = OUT / mode
    complete, saved_regions = recover_mode(mode_dir)
    page_rows = dict(complete)
    regions_by_page = {page_id: list(rows) for page_id, rows in saved_regions.items()}
    with tempfile.TemporaryDirectory(prefix=f"mineru-{mode}-") as scratch_name:
        scratch = Path(scratch_name)
        for index, page_id in enumerate(PAGES, 1):
            if page_id in complete:
                continue
            with Image.open(HELDOUT / f"{page_id}.jpg") as page_image:
                page_image = page_image.convert("RGB")
                width, height = page_image.size
                sources = [
                    {
                        "source_id": "full-page",
                        "class_name": "full-page",
                        "score": 1.0,
                        "bbox_norm": [0.0, 0.0, 1.0, 1.0],
                        "bbox_px": [0, 0, width, height],
                    }
                ]
                if mode == "ppdoclayout-v3":
                    sources = []
                    for source_row in layout_regions[page_id]:
                        source = dict(source_row)
                        source["bbox_px"] = bbox_pixels(source["bbox_norm"], width, height)
                        sources.append(source)
                page_started = time.perf_counter()
                output_regions: list[dict[str, Any]] = []
                for region_index, source in enumerate(sources):
                    region_started = time.perf_counter()
                    bbox = source["bbox_px"]
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        raise TypeError(f"invalid region pixel box: {bbox!r}")
                    crop_box = tuple(int(value) for value in bbox)
                    crop = page_image if mode == "full-page" else page_image.crop(crop_box)
                    # Materialize once because the official processor accepts PIL images,
                    # while the temporary path also makes failed-region inspection simple.
                    crop_path = scratch / f"{page_id}-r{region_index:04d}.jpg"
                    crop.save(crop_path, quality=95)
                    with Image.open(crop_path) as model_image:
                        text, blocks = recognize(client, model_image)
                    output_regions.append(
                        {
                            "region_id": f"{page_id}-r{region_index:04d}",
                            "page_id": page_id,
                            "source_id": source["source_id"],
                            "source_class": source["class_name"],
                            "score": source["score"],
                            "bbox_norm": source["bbox_norm"],
                            "bbox_px": bbox,
                            "text": text,
                            "mineru_blocks": blocks,
                            "status": "complete",
                            "elapsed_seconds": round(time.perf_counter() - region_started, 3),
                        }
                    )
            page_text = "\n\n".join(row["text"] for row in output_regions if row["text"])
            page_row = {
                "page_id": page_id,
                "mode": mode,
                "status": "complete",
                "text": page_text,
                "region_count": len(output_regions),
                "region_ids": [row["region_id"] for row in output_regions],
                "elapsed_seconds": round(time.perf_counter() - page_started, 3),
            }
            page_rows[page_id] = page_row
            regions_by_page[page_id] = output_regions
            complete[page_id] = page_row
            write_jsonl(
                mode_dir / "regions.jsonl",
                [
                    row
                    for expected_page in PAGES
                    if expected_page in complete
                    for row in regions_by_page[expected_page]
                ],
            )
            write_jsonl(
                mode_dir / "pages.jsonl",
                [page_rows[expected_page] for expected_page in PAGES if expected_page in complete],
            )
            print(f"{mode}: {index}/{len(PAGES)} {page_id}", flush=True)
    return score(
        mode,
        labels,
        {page_id: page_rows[page_id] for page_id in PAGES},
        sum(len(regions_by_page[page_id]) for page_id in PAGES),
        device,
    )


def write_comparison(metrics_by_mode: dict[str, dict[str, Any]]) -> None:
    summaries = {
        mode: {key: value for key, value in metrics.items() if key != "per_page"}
        for mode, metrics in metrics_by_mode.items()
    }
    per_mode = {
        mode: {row["page_id"]: row for row in metrics["per_page"]}
        for mode, metrics in metrics_by_mode.items()
    }
    per_page = [
        {
            "page_id": page_id,
            **{
                mode: {
                    metric: per_mode[mode][page_id][metric] for metric in ("cer", "wer", "word_f1")
                }
                for mode in MODES
            },
        }
        for page_id in PAGES
    ]
    (OUT / "comparison.json").write_text(
        json.dumps(
            {
                "engine": MODEL_NAME,
                "pages": len(PAGES),
                "modes": summaries,
                "per_page": per_page,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    install_dependencies()
    labels = ensure_repository()
    layout_regions = load_regions()
    client, model, device = load_model()
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_by_mode = {
        mode: run_mode(mode, labels, layout_regions, client, device) for mode in MODES
    }
    write_comparison(metrics_by_mode)
    archive = shutil.make_archive("/kaggle/working/mineru-ocr-benchmark", "zip", OUT)
    print("download:", archive)
    del client, model


if __name__ == "__main__":
    main()
