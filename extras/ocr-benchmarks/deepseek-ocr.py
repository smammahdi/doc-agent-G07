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

# %%
"""Benchmark DeepSeek-OCR-2 on full pages and existing PP-DocLayoutV3 regions.

The two modes use the same 24 committed held-out Pierce pages. ``full-page``
passes each page image directly to DeepSeek-OCR-2; ``ppdoclayout-v3`` crops
the existing non-figure regions from the committed layout sidecar, orders them
top-to-bottom/left-to-right, and passes each crop to the same model. No layout
model is run here. Each mode writes ``pages.jsonl``, ``regions.jsonl``, and
``metrics.json``. A root ``comparison.json`` and one archive are written after
both modes finish.

Kaggle requirements: Internet enabled and a Tesla T4 GPU. Run with:

    python deepseek-ocr.py

The archive is written to ``/kaggle/working/deepseek-ocr-benchmark.zip``.
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
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path("/kaggle/working/doc-agent-G07")
OUT = Path("/kaggle/working/deepseek-ocr-benchmark")
MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"
LAYOUT_PATH = REPO / "extras/output/ppdoclayout-v3/detections.jsonl"
MODES = ("full-page", "ppdoclayout-v3")
FULL_PAGE_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
REGION_PROMPT = "<image>\nFree OCR."
IMAGE_SIZE = 640  # DeepSeek's supported default and the tested single-T4 profile.


def install_dependencies() -> None:
    """Install the tested Transformers-side dependencies without replacing Kaggle Torch."""

    def pip(*args: str) -> None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

    # Kaggle's torch 2.10 CUDA build may see older libraries in the base image.
    # Align the libraries needed by torch before importing it or Transformers.
    pip(
        "--upgrade",
        "--force-reinstall",
        "--no-deps",
        "nvidia-nccl-cu12==2.27.5",
        "nvidia-nvjitlink-cu12==12.8.93",
        "nvidia-nvtx-cu12==12.8.90",
    )
    pip(
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "sentencepiece",
        "einops",
        "addict",
        "easydict",
        "safetensors",
        "pillow>=10,<12",
    )


def ensure_repository() -> tuple[Path, dict[str, str], list[str]]:
    if not (REPO / "grading_kit/labels.jsonl").is_file():
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
    heldout = REPO / "grading_kit/heldout_pages"
    labels_path = REPO / "grading_kit/labels.jsonl"
    labels = {
        row["page_id"]: row["text"]
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    pages = sorted(labels)
    expected_pages = [f"p{number:04d}" for number in range(24, 48)]
    if pages != expected_pages:
        raise ValueError(f"expected held-out pages p0024-p0047; found {pages}")
    missing = [page_id for page_id in pages if not (heldout / f"{page_id}.jpg").is_file()]
    if missing:
        raise FileNotFoundError(f"missing held-out images: {missing}")
    return heldout, labels, pages


def load_layout_regions(pages: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Load existing PP-DocLayoutV3 non-figure regions for the held-out pages."""

    if not LAYOUT_PATH.is_file():
        raise FileNotFoundError(f"missing committed PP-DocLayoutV3 detections: {LAYOUT_PATH}")
    page_set = set(pages)
    regions: dict[str, list[dict[str, Any]]] = {page_id: [] for page_id in pages}
    with LAYOUT_PATH.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            page_id = row.get("page_id")
            if page_id not in page_set or bool(row.get("is_figure")):
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
    """Convert a clipped normalized box to a non-empty pixel crop."""

    left = max(0, min(width, int(box[0] * width)))
    top = max(0, min(height, int(box[1] * height)))
    right = max(left + 1, min(width, int(box[2] * width + 0.999999)))
    bottom = max(top + 1, min(height, int(box[3] * height + 0.999999)))
    if right <= left or bottom <= top:
        raise ValueError(f"normalized box became empty at {width}x{height}: {box}")
    return [left, top, right, bottom]


def load_model() -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("DeepSeek-OCR-2 requires a CUDA GPU; select a Tesla T4 in Kaggle")
    try:
        bf16 = bool(torch.cuda.is_bf16_supported())
    except (AttributeError, RuntimeError):
        bf16 = False
    dtype = torch.bfloat16 if bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model_kwargs = {
        "trust_remote_code": True,
        "use_safetensors": True,
        "torch_dtype": dtype,
        "_attn_implementation": "eager",
    }
    try:
        model = AutoModel.from_pretrained(MODEL_NAME, **model_kwargs)
    except (TypeError, ValueError):
        model_kwargs.pop("_attn_implementation")
        model = AutoModel.from_pretrained(MODEL_NAME, **model_kwargs)
    model = model.eval().to("cuda").to(dtype)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = False
    if hasattr(model, "generate"):
        orig_generate = model.generate

        def bounded_generate(*args: Any, **kwargs: Any) -> Any:
            max_new = kwargs.get("max_new_tokens")
            if max_new is None or max_new > 4096:
                kwargs["max_new_tokens"] = 4096
            if kwargs.get("temperature") == 0.0 and not kwargs.get("do_sample", False):
                kwargs.pop("temperature", None)
            return orig_generate(*args, **kwargs)

        model.generate = bounded_generate
    return tokenizer, model, str(dtype).replace("torch.", "")


def infer(
    model: Any,
    tokenizer: Any,
    image_path: Path,
    result_dir: Path,
    prompt: str,
    *,
    crop_mode: bool = True,
) -> str:
    kwargs = {
        "tokenizer": tokenizer,
        "prompt": prompt,
        "image_file": str(image_path),
        "output_path": str(result_dir),
        "base_size": 1024,
        "image_size": IMAGE_SIZE,
        "crop_mode": crop_mode,
        "save_results": False,
        "test_compress": False,
    }
    try:
        import torch

        with torch.inference_mode():
            result = model.infer(**kwargs, eval_mode=True)
    except TypeError:
        try:
            import torch

            with torch.inference_mode():
                result = model.infer(**kwargs)
        except Exception:
            result = model.infer(**kwargs)
    except Exception:
        result = model.infer(**kwargs, eval_mode=True)
    if isinstance(result, str):
        return result.strip()
    raise RuntimeError(
        "DeepSeek-OCR-2 did not return text; use the official custom-code runtime "
        "with eval_mode=True rather than treating saved side effects as OCR output"
    )


def normalize_exact(text: str) -> str:
    return re.sub(r"\s+", " ", remove_grounding_markup(text)).strip()


def remove_grounding_markup(text: str) -> str:
    """Remove DeepSeek coordinate annotations while retaining their referenced text."""

    text = re.sub(r"<\|det\|>.*?<\|/det\|>", " ", text, flags=re.DOTALL)
    return re.sub(r"<\|/?ref\|>", "", text)


def normalize(text: str) -> str:
    """Normalize typography while preserving letters and numbers for fair OCR scoring."""

    folded = unicodedata.normalize("NFKC", remove_grounding_markup(text)).casefold()
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def recover_mode(
    mode_dir: Path, pages: list[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Retain only page rows with a matching complete region set after a restart."""

    page_rows = read_jsonl(mode_dir / "pages.jsonl")
    region_rows = read_jsonl(mode_dir / "regions.jsonl")
    allowed = set(pages)
    regions_by_page: dict[str, list[dict[str, Any]]] = {page_id: [] for page_id in pages}
    for row in region_rows:
        page_id = row.get("page_id")
        if page_id in allowed:
            regions_by_page[page_id].append(row)
    complete: dict[str, dict[str, Any]] = {}
    for row in page_rows:
        page_id = row.get("page_id")
        if (
            page_id in allowed
            and row.get("status") == "complete"
            and int(row.get("region_count", -1)) == len(regions_by_page[page_id])
            and page_id not in complete
        ):
            complete[page_id] = row
    kept_pages = [complete[page_id] for page_id in pages if page_id in complete]
    kept_regions = [
        region for page_id in pages if page_id in complete for region in regions_by_page[page_id]
    ]
    mode_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(mode_dir / "pages.jsonl", kept_pages)
    write_jsonl(mode_dir / "regions.jsonl", kept_regions)
    return complete, regions_by_page


def score(
    labels: dict[str, str],
    pages: list[str],
    page_rows: dict[str, dict[str, Any]],
    mode: str,
    dtype: str,
    region_count: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_char_errors = total_word_errors = total_chars = total_words = 0
    exact_total_char_errors = exact_total_word_errors = exact_total_chars = exact_total_words = 0
    for page_id in pages:
        exact_reference = normalize_exact(labels[page_id])
        exact_hypothesis = normalize_exact(str(page_rows[page_id].get("text", "")))
        reference = normalize(exact_reference)
        hypothesis = normalize(exact_hypothesis)
        reference_words = reference.split()
        hypothesis_words = hypothesis.split()
        char_errors = levenshtein(hypothesis, reference)
        word_errors = levenshtein(hypothesis_words, reference_words)
        exact_char_errors = levenshtein(exact_hypothesis, exact_reference)
        exact_word_errors = levenshtein(exact_hypothesis.split(), exact_reference.split())
        row = {
            "page_id": page_id,
            "cer": char_errors / max(1, len(reference)),
            "wer": word_errors / max(1, len(reference_words)),
            "word_f1": word_f1(hypothesis, reference),
            "exact_cer": exact_char_errors / max(1, len(exact_reference)),
            "exact_wer": exact_word_errors / max(1, len(exact_reference.split())),
            "exact_word_f1": word_f1(exact_hypothesis, exact_reference),
            "reference_chars": len(reference),
            "reference_words": len(reference_words),
            "hypothesis_chars": len(hypothesis),
            "hypothesis_words": len(hypothesis_words),
        }
        rows.append(row)
        total_char_errors += char_errors
        total_word_errors += word_errors
        total_chars += len(reference)
        total_words += len(reference_words)
        exact_total_char_errors += exact_char_errors
        exact_total_word_errors += exact_word_errors
        exact_total_chars += len(exact_reference)
        exact_total_words += len(exact_reference.split())
        print(page_id, row)
    metrics = {
        "engine": MODEL_NAME,
        "mode": mode,
        "layout": "full-page input" if mode == "full-page" else "PP-DocLayoutV3 non-figure regions",
        "dtype": dtype,
        "prompts": {
            "full-page": FULL_PAGE_PROMPT,
            "ppdoclayout-v3": REGION_PROMPT,
        },
        "image_size": IMAGE_SIZE,
        "primary_scoring": "NFKC, casefold, letters/numbers only, collapsed whitespace",
        "pages": len(rows),
        "regions": region_count,
        "micro_cer": total_char_errors / max(1, total_chars),
        "micro_wer": total_word_errors / max(1, total_words),
        "macro_cer": sum(row["cer"] for row in rows) / len(rows),
        "macro_wer": sum(row["wer"] for row in rows) / len(rows),
        "macro_word_f1": sum(row["word_f1"] for row in rows) / len(rows),
        "exact_micro_cer": exact_total_char_errors / max(1, exact_total_chars),
        "exact_micro_wer": exact_total_word_errors / max(1, exact_total_words),
        "exact_macro_cer": sum(row["exact_cer"] for row in rows) / len(rows),
        "exact_macro_wer": sum(row["exact_wer"] for row in rows) / len(rows),
        "exact_macro_word_f1": sum(row["exact_word_f1"] for row in rows) / len(rows),
        "per_page": rows,
    }
    mode_dir = OUT / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    mode_dir.joinpath("metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_page"}, indent=2))
    return metrics


def run_mode(
    mode: str,
    heldout: Path,
    pages: list[str],
    labels: dict[str, str],
    layout_regions: dict[str, list[dict[str, Any]]],
    model: Any,
    tokenizer: Any,
    dtype: str,
) -> dict[str, Any]:
    """Run one mode while checkpointing canonical page and region JSONL files."""

    from PIL import Image

    mode_dir = OUT / mode
    complete, existing_regions = recover_mode(mode_dir, pages)
    page_rows = dict(complete)
    region_rows_by_page = {
        page_id: list(existing_regions.get(page_id, [])) for page_id in pages if page_id in complete
    }
    with tempfile.TemporaryDirectory(prefix=f"deepseek-{mode}-") as scratch_name:
        scratch = Path(scratch_name)
        side_effects = scratch / "model-side-effects"
        side_effects.mkdir()
        for index, page_id in enumerate(pages, 1):
            if page_id in complete:
                continue
            image_path = heldout / f"{page_id}.jpg"
            with Image.open(image_path) as image:
                width, height = image.size
                if mode == "full-page":
                    sources = [
                        {
                            "source_id": "full-page",
                            "class_name": "full-page",
                            "score": 1.0,
                            "bbox_norm": [0.0, 0.0, 1.0, 1.0],
                            "bbox_px": [0, 0, width, height],
                            "image_path": image_path,
                        }
                    ]
                else:
                    sources = []
                    for source in layout_regions[page_id]:
                        source = dict(source)
                        source["bbox_px"] = bbox_pixels(source["bbox_norm"], width, height)
                        sources.append(source)
                page_started = time.perf_counter()
                output_regions: list[dict[str, Any]] = []
                for region_index, source in enumerate(sources):
                    region_started = time.perf_counter()
                    crop_path = image_path
                    if mode != "full-page":
                        crop_path = scratch / f"{page_id}-r{region_index:04d}.jpg"
                        region_box = source["bbox_px"]
                        if not isinstance(region_box, list):
                            raise TypeError(f"invalid region pixel box: {region_box!r}")
                        image.crop(tuple(int(value) for value in region_box)).convert("RGB").save(
                            crop_path
                        )
                    prompt = FULL_PAGE_PROMPT if mode == "full-page" else REGION_PROMPT
                    crop_mode = mode == "full-page"
                    text = infer(
                        model,
                        tokenizer,
                        crop_path,
                        side_effects,
                        prompt,
                        crop_mode=crop_mode,
                    )
                    output_regions.append(
                        {
                            "region_id": f"{page_id}-r{region_index:04d}",
                            "page_id": page_id,
                            "source_id": source["source_id"],
                            "source_class": source["class_name"],
                            "score": source["score"],
                            "bbox_norm": source["bbox_norm"],
                            "bbox_px": source["bbox_px"],
                            "text": text,
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
                "region_ids": [region["region_id"] for region in output_regions],
                "elapsed_seconds": round(time.perf_counter() - page_started, 3),
            }
            page_rows[page_id] = page_row
            region_rows_by_page[page_id] = output_regions
            complete[page_id] = page_row
            write_jsonl(
                mode_dir / "regions.jsonl",
                [
                    region
                    for expected_page in pages
                    if expected_page in complete
                    for region in region_rows_by_page[expected_page]
                ],
            )
            write_jsonl(
                mode_dir / "pages.jsonl",
                [page_rows[expected_page] for expected_page in pages if expected_page in complete],
            )
            print(f"{mode}: {index}/{len(pages)} {page_id}", flush=True)
    rows = {page_id: page_rows[page_id] for page_id in pages}
    return score(
        labels,
        pages,
        rows,
        mode,
        dtype,
        sum(len(region_rows_by_page[page_id]) for page_id in pages if page_id in complete),
    )


def write_comparison(metrics_by_mode: dict[str, dict[str, Any]], pages: list[str]) -> None:
    summaries = {
        mode: {key: value for key, value in metrics.items() if key != "per_page"}
        for mode, metrics in metrics_by_mode.items()
    }
    per_page: list[dict[str, Any]] = []
    full_rows = {row["page_id"]: row for row in metrics_by_mode["full-page"]["per_page"]}
    layout_rows = {row["page_id"]: row for row in metrics_by_mode["ppdoclayout-v3"]["per_page"]}
    for page_id in pages:
        full = full_rows[page_id]
        layout = layout_rows[page_id]
        per_page.append(
            {
                "page_id": page_id,
                "full-page": {
                    "cer": full["cer"],
                    "wer": full["wer"],
                    "word_f1": full["word_f1"],
                },
                "ppdoclayout-v3": {
                    "cer": layout["cer"],
                    "wer": layout["wer"],
                    "word_f1": layout["word_f1"],
                },
            }
        )
    (OUT / "comparison.json").write_text(
        json.dumps(
            {
                "engine": MODEL_NAME,
                "pages": len(pages),
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
    heldout, labels, pages = ensure_repository()
    layout_regions = load_layout_regions(pages)
    tokenizer, model, dtype = load_model()
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_by_mode = {
        mode: run_mode(
            mode,
            heldout,
            pages,
            labels,
            layout_regions,
            model,
            tokenizer,
            dtype,
        )
        for mode in MODES
    }
    write_comparison(metrics_by_mode, pages)
    archive = shutil.make_archive("/kaggle/working/deepseek-ocr-benchmark", "zip", OUT)
    print("download:", archive)


if __name__ == "__main__":
    main()
