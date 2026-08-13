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
# # GLM-OCR benchmark
#
# Save GLM-OCR text from the same 24 pages in two modes: direct full-page OCR
# and OCR of the existing PP-DocLayoutV3 non-figure regions. No layout model is
# run here. The script writes one downloadable `glm-ocr-benchmark.zip` archive.

# %%
# Kaggle: Internet ON, Accelerator = Tesla T4.
# %pip install -q --upgrade "git+https://github.com/huggingface/transformers.git"

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image

REPO = Path("/kaggle/working/doc-agent-G07")
OUT = Path("/kaggle/working/glm-ocr-benchmark")
MODEL_NAME = "zai-org/GLM-OCR"
LAYOUT_PATH = REPO / "extras/output/ppdoclayout-v3/detections.jsonl"
HELDOUT = REPO / "grading_kit/heldout_pages"
LABELS = REPO / "grading_kit/labels.jsonl"
PAGES = [f"p{i:04d}" for i in range(24, 48)]
MODES = ("full-page", "ppdoclayout-v3")


def ensure_repository() -> tuple[dict[str, str], list[str]]:
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
    return labels, PAGES


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_regions() -> dict[str, list[dict[str, Any]]]:
    regions: dict[str, list[dict[str, Any]]] = {page_id: [] for page_id in PAGES}
    for line_number, row in enumerate(read_jsonl(LAYOUT_PATH), 1):
        page_id = row.get("page_id")
        if page_id not in regions or row.get("is_figure", False):
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
            key=lambda row: (row["bbox_norm"][1], row["bbox_norm"][0], row["source_id"])
        )
    return regions


def bbox_pixels(box: list[float], width: int, height: int) -> list[int]:
    left = max(0, min(width - 1, int(box[0] * width)))
    top = max(0, min(height - 1, int(box[1] * height)))
    right = max(left + 1, min(width, int(box[2] * width + 0.999999)))
    bottom = max(top + 1, min(height, int(box[3] * height + 0.999999)))
    return [left, top, right, bottom]


def load_model() -> tuple[Any, Any, str]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("GLM-OCR requires a CUDA GPU; select a Tesla T4 in Kaggle")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    ).eval()
    return processor, model, torch.cuda.get_device_name(0)


@torch.inference_mode()
def recognize(processor: Any, model: Any, image_path: Path) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": "Text Recognition:"},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    output_ids = model.generate(**inputs, max_new_tokens=2048)
    prompt_length = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][prompt_length:], skip_special_tokens=False).strip()


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    temporary.replace(path)


def score(mode_dir: Path, labels: dict[str, str]) -> dict[str, Any]:
    pages = read_jsonl(mode_dir / "pages.jsonl")
    if [row.get("page_id") for row in pages] != PAGES:
        raise ValueError(f"{mode_dir} does not contain exactly 24 pages")
    per_page = []
    char_errors = word_errors = total_chars = total_words = 0
    for page in pages:
        reference = normalize(labels[page["page_id"]])
        hypothesis = normalize(page.get("text", ""))
        ref_words, hyp_words = reference.split(), hypothesis.split()
        ce = levenshtein(hypothesis, reference)
        we = levenshtein(hyp_words, ref_words)
        row = {
            "page_id": page["page_id"],
            "cer": ce / max(1, len(reference)),
            "wer": we / max(1, len(ref_words)),
            "word_f1": word_f1(hypothesis, reference),
            "reference_chars": len(reference),
            "reference_words": len(ref_words),
        }
        per_page.append(row)
        char_errors += ce
        word_errors += we
        total_chars += len(reference)
        total_words += len(ref_words)
    metrics = {
        "engine": MODEL_NAME,
        "mode": mode_dir.name,
        "pages": len(per_page),
        "micro_cer": char_errors / max(1, total_chars),
        "micro_wer": word_errors / max(1, total_words),
        "macro_cer": sum(row["cer"] for row in per_page) / len(per_page),
        "macro_wer": sum(row["wer"] for row in per_page) / len(per_page),
        "macro_word_f1": sum(row["word_f1"] for row in per_page) / len(per_page),
        "per_page": per_page,
    }
    mode_dir.joinpath("metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def run_mode(
    mode: str,
    labels: dict[str, str],
    layout_regions: dict[str, list[dict[str, Any]]],
    processor: Any,
    model: Any,
) -> dict[str, Any]:
    mode_dir = OUT / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    page_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    for page_id in PAGES:
        started = time.perf_counter()
        image_path = HELDOUT / f"{page_id}.jpg"
        with Image.open(image_path) as image:
            width, height = image.size
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
                else layout_regions[page_id]
            )
            page_regions = []
            for index, source in enumerate(sources):
                box_px = bbox_pixels(source["bbox_norm"], width, height)
                if mode == "full-page":
                    text = recognize(processor, model, image_path)
                else:
                    crop_path = OUT / ".crops" / f"{page_id}-r{index:04d}.jpg"
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    image.crop(tuple(box_px)).convert("RGB").save(crop_path)
                    text = recognize(processor, model, crop_path)
                page_regions.append(
                    {
                        "region_id": f"{page_id}-r{index:04d}",
                        "page_id": page_id,
                        "source_id": source["source_id"],
                        "source_class": source["class_name"],
                        "score": source["score"],
                        "bbox_norm": source["bbox_norm"],
                        "bbox_px": box_px,
                        "text": text,
                        "status": "complete",
                    }
                )
        region_rows.extend(page_regions)
        page_rows.append(
            {
                "page_id": page_id,
                "mode": mode,
                "status": "complete",
                "text": "\n\n".join(row["text"] for row in page_regions if row["text"]),
                "region_count": len(page_regions),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        write_jsonl(mode_dir / "regions.jsonl", region_rows)
        write_jsonl(mode_dir / "pages.jsonl", page_rows)
        print(mode, page_id, len(page_regions), flush=True)
    return score(mode_dir, labels)


def main() -> None:
    labels, pages = ensure_repository()
    layout_regions = load_regions()
    processor, model, device = load_model()
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = {mode: run_mode(mode, labels, layout_regions, processor, model) for mode in MODES}
    comparison = {
        "engine": MODEL_NAME,
        "device": device,
        "pages": len(pages),
        "modes": {
            mode: {key: value for key, value in result.items() if key != "per_page"}
            for mode, result in metrics.items()
        },
        "per_page": [
            {
                "page_id": page_id,
                **{
                    mode: {
                        key: next(
                            row[key]
                            for row in metrics[mode]["per_page"]
                            if row["page_id"] == page_id
                        )
                        for key in ("cer", "wer", "word_f1")
                    }
                    for mode in MODES
                },
            }
            for page_id in pages
        ],
    }
    (OUT / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(OUT / ".crops", ignore_errors=True)
    archive = shutil.make_archive("/kaggle/working/glm-ocr-benchmark", "zip", root_dir=OUT)
    print("download:", archive)


if __name__ == "__main__":
    main()
