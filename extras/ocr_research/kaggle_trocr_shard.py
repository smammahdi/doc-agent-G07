"""Run one bounded full-book shard of TrOCR from an existing layout.

Kaggle kernels use this entrypoint for one layout and one inclusive page range.
The output is a normal ``run_layout_trocr.py`` directory, so shards can be
validated and merged without changing the canonical page/region schema.
No layout detector is run here, and the layout sidecar's text is never used as
OCR input.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/smammahdi/doc-agent-G07.git"
BRANCH = "a2/trocr-layout-comparison"
REPO = Path("/kaggle/working/doc-agent-G07")


def _pages(value: str) -> str:
    try:
        start, end = (int(item) for item in value.split(":", 1))
    except ValueError as error:
        raise argparse.ArgumentTypeError("pages must be START:END") from error
    if not (1 <= start <= end <= 1034):
        raise argparse.ArgumentTypeError("pages must be within 1:1034")
    return f"{start},{end}" if start == end else ",".join(map(str, range(start, end + 1)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("chandra", "doclayout_yolo"), required=True)
    parser.add_argument("--pages", required=True, type=_pages, help="inclusive START:END")
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/trocr_shard_output"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/kaggle/working/trocr_page_cache"))
    parser.add_argument("--model", default="microsoft/trocr-base-printed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def _run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _prepare_runtime() -> Path:
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "pymupdf>=1.25.5,<1.26",
            "pillow>=10,<12",
            "transformers>=4.40,<5",
            "sentencepiece>=0.2,<1",
            "opencv-python-headless>=4.10,<5",
        ]
    )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required for a TrOCR shard")
    capability = torch.cuda.get_device_capability(0)
    if capability < (7, 0):
        raise RuntimeError(f"unsupported GPU sm_{capability[0]}{capability[1]}; need sm_70+")
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        f"sm_{capability[0]}{capability[1]}",
        flush=True,
    )
    if not REPO.exists():
        _run(["git", "clone", "--branch", BRANCH, "--depth", "1", REPOSITORY, str(REPO)])
    else:
        _run(["git", "-C", str(REPO), "fetch", "origin", BRANCH, "--depth", "1"])
        _run(["git", "-C", str(REPO), "checkout", "--force", "FETCH_HEAD"])
    print("repository:", end=" ", flush=True)
    _run(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    return REPO


def run(args: argparse.Namespace) -> None:
    repo = _prepare_runtime()
    sys.path.insert(0, str(repo / "extras" / "ocr_research"))
    from run_kaggle_layout_trocr import _discover

    source_pdf, layouts = _discover(args.input_root)
    selected = next(layout for layout in layouts if layout.name == args.layout)
    runner = repo / "extras" / "ocr_research" / "run_layout_trocr.py"
    repository_src = runner.resolve().parents[2] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(repository_src), environment.get("PYTHONPATH")) if value
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
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--max-length",
        str(args.max_length),
        "--dpi",
        str(args.dpi),
        "--pages",
        args.pages,
    ]
    print(f"Saving {selected.name} TrOCR pages {args.pages}", flush=True)
    subprocess.run(command, check=True, env=environment)
    page_numbers = args.pages.split(",")
    range_label = f"{page_numbers[0]}-{page_numbers[-1]}"
    archive = shutil.make_archive(
        str(Path("/kaggle/working") / f"trocr_{selected.name}_{range_label}"),
        "zip",
        args.output,
    )
    print("canonical output:", args.output, flush=True)
    print("downloadable archive:", archive, flush=True)


if __name__ == "__main__":
    run(_parser().parse_args())
