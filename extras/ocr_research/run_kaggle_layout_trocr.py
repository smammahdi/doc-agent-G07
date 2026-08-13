"""Run the existing layout-fed TrOCR exporter over a Kaggle input bundle.

The wrapper discovers the Pierce PDF and the supplied Chandra and
DocLayout-YOLO JSONL sidecars under ``/kaggle/input``. It then saves separate
TrOCR outputs for each existing layout; it does not run a layout model or
compare the layouts.

Example (inside a Kaggle notebook after cloning this repository)::

    python extras/ocr_research/run_kaggle_layout_trocr.py

Use ``--pages 34,74`` for a bounded smoke run. Omitting it processes the full
book through ``run_layout_trocr.py``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LayoutInput:
    name: str
    path: Path


def _ranked_choice(candidates: list[Path], label: str, tokens: tuple[str, ...]) -> Path:
    if not candidates:
        raise FileNotFoundError(f"could not find the {label} under the input root")
    ranked = sorted(
        candidates,
        key=lambda path: (
            sum(token in str(path).lower() for token in tokens),
            -len(str(path)),
            str(path),
        ),
        reverse=True,
    )
    best_score = sum(token in str(ranked[0]).lower() for token in tokens)
    tied = [
        path for path in ranked if sum(token in str(path).lower() for token in tokens) == best_score
    ]
    if len(tied) != 1:
        options = ", ".join(str(path) for path in tied[:6])
        raise RuntimeError(
            f"ambiguous {label}; pass a bundle with one preferred sidecar: {options}"
        )
    return ranked[0]


def _discover(input_root: Path) -> tuple[Path, tuple[LayoutInput, ...]]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Kaggle input root does not exist: {input_root}")
    files = [path for path in input_root.rglob("*") if path.is_file()]
    pdf = _ranked_choice(
        [path for path in files if path.suffix.lower() == ".pdf"],
        "PDF",
        ("pierce", "1890", "medical"),
    )
    chunks = [path for path in files if path.name.lower() == "chunks.jsonl"]
    detections = [path for path in files if path.name.lower() == "detections.jsonl"]
    return pdf, (
        LayoutInput("chandra", _ranked_choice(chunks, "Chandra chunks.jsonl", ("chandra",))),
        LayoutInput(
            "doclayout_yolo",
            _ranked_choice(
                detections,
                "DocLayout-YOLO detections.jsonl",
                ("doclayout", "yolo", "dly"),
            ),
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input")),
        help="Kaggle dataset mount to scan (default: /kaggle/input)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working/trocr_layout_outputs"),
        help="directory for separate chandra/ and doclayout_yolo/ outputs",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/kaggle/working/trocr_page_cache"),
        help="directory for rendered page images shared by both runs",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("run_layout_trocr.py"),
        help="existing layout-fed TrOCR runner",
    )
    parser.add_argument("--model", default="microsoft/trocr-base-printed")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pages", help="optional comma-separated 1-based pages for a smoke run")
    return parser


def run(args: argparse.Namespace) -> None:
    source_pdf, layouts = _discover(args.input_root)
    if not args.runner.is_file():
        raise FileNotFoundError(f"existing TrOCR runner not found: {args.runner}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    repository_src = args.runner.resolve().parents[2] / "src"
    environment = os.environ.copy()
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(repository_src), current_pythonpath) if value
    )
    for layout in layouts:
        command = [
            sys.executable,
            str(args.runner),
            "--layout",
            layout.name,
            "--layout-path",
            str(layout.path),
            "--source-pdf",
            str(source_pdf),
            "--output",
            str(args.output_root / layout.name),
            "--cache-dir",
            str(args.cache_root),
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
        ]
        if args.pages:
            command.extend(("--pages", args.pages))
        print(f"Saving {layout.name} TrOCR output from {layout.path}", flush=True)
        subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    run(_parser().parse_args())
