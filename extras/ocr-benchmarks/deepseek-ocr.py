"""Run DeepSeek-OCR-2 on the repository's committed OCR pages.

It clones the public repository, discovers the committed page images and
labels, writes one JSON record per page, and reports CER, WER, and word-F1.
It is intentionally a full-page OCR runner: it does not run a layout model and
does not use Chandra or Document AI text as input. Existing layout-fed OCR is
kept in the separate combined research runner.

Kaggle requirements: Internet enabled and a Tesla T4 GPU. Run with:

    python kaggle-deepseek-ocr.py

The archive is written to /kaggle/working/deepseek-ocr-results.zip.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path("/kaggle/working/doc-agent-G07")
OUT = Path("/kaggle/working/deepseek-ocr-results")
MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"


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
    missing = [page_id for page_id in pages if not (heldout / f"{page_id}.jpg").is_file()]
    if missing:
        raise FileNotFoundError(f"missing held-out images: {missing}")
    return heldout, labels, pages


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
    return tokenizer, model, str(dtype).replace("torch.", "")


def infer(model: Any, tokenizer: Any, image_path: Path, result_dir: Path) -> str:
    kwargs = {
        "tokenizer": tokenizer,
        "prompt": "<image>\nFree OCR.",
        "image_file": str(image_path),
        "output_path": str(result_dir),
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,
        "save_results": False,
        "test_compress": False,
    }
    try:
        result = model.infer(**kwargs, eval_mode=True)
    except TypeError:
        result = model.infer(
            **kwargs,
        )
    if isinstance(result, str):
        return result.strip()
    raise RuntimeError(
        "DeepSeek-OCR-2 did not return text; use the official custom-code runtime "
        "with eval_mode=True rather than treating saved side effects as OCR output"
    )


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


def score(labels: dict[str, str], pages: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_char_errors = total_word_errors = total_chars = total_words = 0
    for page_id in pages:
        reference = normalize(labels[page_id])
        hypothesis = normalize(
            json.loads((OUT / f"{page_id}.json").read_text(encoding="utf-8"))["text"]
        )
        reference_words = reference.split()
        hypothesis_words = hypothesis.split()
        char_errors = levenshtein(hypothesis, reference)
        word_errors = levenshtein(hypothesis_words, reference_words)
        row = {
            "page_id": page_id,
            "cer": char_errors / max(1, len(reference)),
            "wer": word_errors / max(1, len(reference_words)),
            "word_f1": word_f1(hypothesis, reference),
            "reference_chars": len(reference),
            "reference_words": len(reference_words),
        }
        rows.append(row)
        total_char_errors += char_errors
        total_word_errors += word_errors
        total_chars += len(reference)
        total_words += len(reference_words)
        print(page_id, row)
    metrics = {
        "engine": MODEL_NAME,
        "layout": "full-page input",
        "pages": len(rows),
        "micro_cer": total_char_errors / max(1, total_chars),
        "micro_wer": total_word_errors / max(1, total_words),
        "macro_cer": sum(row["cer"] for row in rows) / len(rows),
        "macro_wer": sum(row["wer"] for row in rows) / len(rows),
        "macro_word_f1": sum(row["word_f1"] for row in rows) / len(rows),
        "per_page": rows,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_page"}, indent=2))
    return metrics


def main() -> None:
    install_dependencies()
    heldout, labels, pages = ensure_repository()
    tokenizer, model, dtype = load_model()
    OUT.mkdir(parents=True, exist_ok=True)
    result_dir = OUT / "model-side-effects"
    result_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for page_id in pages:
        target = OUT / f"{page_id}.json"
        if target.is_file():
            continue
        text = infer(model, tokenizer, heldout / f"{page_id}.jpg", result_dir)
        target.write_text(
            json.dumps(
                {
                    "page_id": page_id,
                    "text": text,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(page_id)
    metrics = score(labels, pages)
    metrics["dtype"] = dtype
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    archive = shutil.make_archive("/kaggle/working/deepseek-ocr-results", "zip", OUT)
    print("download:", archive)


if __name__ == "__main__":
    main()
