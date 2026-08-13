"""Kaggle launcher for a bounded DeepSeek-OCR layout-fed smoke run.

The launcher uses the official custom Transformers model and the repository's
research exporter. It consumes an existing layout sidecar, never runs layout
detection, and defaults to real Pierce page 34 for a cheap acceptance check.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/smammahdi/doc-agent-G07.git"
BRANCH = "a2/trocr-layout-comparison"
REPO = Path("/kaggle/working/doc-agent-G07")


def _run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("chandra", "doclayout_yolo"), default="doclayout_yolo")
    parser.add_argument("--pages", default="34", help="comma-separated 1-based smoke pages")
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument(
        "--output", type=Path, default=Path("/kaggle/working/deepseek_layout_smoke")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("/kaggle/working/deepseek_layout_cache")
    )
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-OCR")
    parser.add_argument("--base-size", type=int, default=640)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--crop-mode", action="store_true")
    return parser


def _prepare() -> Path:
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers==4.46.3",
            "tokenizers==0.20.3",
            "sentencepiece>=0.2,<1",
            "einops",
            "easydict",
            "addict",
            "pymupdf>=1.25.5,<1.26",
            "pillow>=10,<12",
        ]
    )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("DeepSeek-OCR smoke requires a CUDA GPU")
    capability = torch.cuda.get_device_capability(0)
    if capability < (7, 0):
        raise RuntimeError(f"unsupported GPU sm_{capability[0]}{capability[1]}; need sm_70+")
    print("GPU:", torch.cuda.get_device_name(0), f"sm_{capability[0]}{capability[1]}", flush=True)
    if not REPO.exists():
        _run(["git", "clone", "--branch", BRANCH, "--depth", "1", REPOSITORY, str(REPO)])
    else:
        _run(["git", "-C", str(REPO), "fetch", "origin", BRANCH, "--depth", "1"])
        _run(["git", "-C", str(REPO), "checkout", "--force", "FETCH_HEAD"])
    _run(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    return REPO


def run(args: argparse.Namespace) -> None:
    repo = _prepare()
    sys.path.insert(0, str(repo / "extras" / "ocr_research"))
    from run_kaggle_layout_trocr import _discover

    source_pdf, layouts = _discover(args.input_root)
    selected = next(layout for layout in layouts if layout.name == args.layout)
    runner = repo / "extras" / "ocr_research" / "run_deepseek_layout_ocr.py"
    source_root = repo / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(source_root),
            str(repo / "extras" / "ocr_research"),
            environment.get("PYTHONPATH"),
        )
        if value
    )
    command = [
        sys.executable,
        str(runner),
        "--layout",
        selected.name,
        "--layout-path",
        str(selected.path),
        "--source-pdf",
        str(source_pdf),
        "--output",
        str(args.output),
        "--cache-dir",
        str(args.cache_dir),
        "--model",
        args.model,
        "--device",
        "cuda",
        "--attention",
        "eager",
        "--base-size",
        str(args.base_size),
        "--image-size",
        str(args.image_size),
        "--pages",
        args.pages,
    ]
    if args.crop_mode:
        command.append("--crop-mode")
    print(
        f"Saving DeepSeek-OCR text from existing {selected.name} regions on pages {args.pages}",
        flush=True,
    )
    subprocess.run(command, check=True, env=environment)
    print("canonical output:", args.output, flush=True)


if __name__ == "__main__":
    run(_parser().parse_args())
