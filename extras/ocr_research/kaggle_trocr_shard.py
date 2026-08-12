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
import subprocess
import sys
from pathlib import Path

from run_kaggle_layout_trocr import _discover


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/kaggle/working/trocr_page_cache"))
    parser.add_argument("--model", default="microsoft/trocr-base-printed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def run(args: argparse.Namespace) -> None:
    source_pdf, layouts = _discover(args.input_root)
    selected = next(layout for layout in layouts if layout.name == args.layout)
    runner = Path(__file__).with_name("run_layout_trocr.py")
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


if __name__ == "__main__":
    run(_parser().parse_args())
