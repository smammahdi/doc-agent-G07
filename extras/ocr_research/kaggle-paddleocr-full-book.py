"""Run PaddleOCR over the full Pierce book using existing layout regions.

This is a Kaggle-only wrapper around ``kaggle-paddle-deepseek-ocr.py``. It
uses the committed DocLayout-YOLO detections, never reruns a layout model, and
never uses Chandra text as OCR input. Run page 34 first, then rerun with
``--pages all --zip`` for the complete book.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/smammahdi/doc-agent-G07.git"
RUNNER_RELATIVE = Path("extras/ocr_research/kaggle-paddle-deepseek-ocr.py")
LAYOUT_RELATIVE = Path("extras/output/doclayout-yolo/detections.jsonl")


def _find_pdf(input_root: Path) -> Path:
    candidates = sorted(input_root.rglob("*.pdf"))
    preferred = [
        path
        for path in candidates
        if all(token in str(path).lower() for token in ("pierce", "medical"))
    ]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        "Expected one Pierce PDF under "
        f"{input_root}; found {len(candidates)} candidates: {candidates}"
    )


def _ensure_repo(repo: Path) -> Path:
    runner = repo / RUNNER_RELATIVE
    if not runner.is_file():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", "main", REPO_URL, str(repo)],
            check=True,
        )
    if not runner.is_file():
        raise FileNotFoundError(f"runner missing after clone: {runner}")
    return runner


def _archive(engine_output: Path, archive_path: Path) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_base = archive_path.with_suffix("")
    created = Path(shutil.make_archive(str(archive_base), "zip", root_dir=engine_output))
    if created != archive_path:
        created.replace(archive_path)
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--repo", type=Path, default=Path("/kaggle/working/doc-agent-G07"))
    parser.add_argument("--pages", default="34", help="page numbers or all")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working/paddleocr-doclayout-yolo"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/kaggle/working/paddleocr-doclayout-yolo-cache"),
    )
    parser.add_argument(
        "--zip",
        dest="archive_path",
        type=Path,
        default=None,
        help="create a ZIP after a successful run",
    )
    args = parser.parse_args()

    runner = _ensure_repo(args.repo)
    source_pdf = _find_pdf(args.input_root)
    layout_path = args.repo / LAYOUT_RELATIVE
    if not layout_path.is_file():
        raise FileNotFoundError(f"tracked layout sidecar missing: {layout_path}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PADDLE_PDX_CACHE_HOME", str(args.cache_root / "paddlex"))
    env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    command = [
        sys.executable,
        str(runner),
        "--source-pdf",
        str(source_pdf),
        "--layout-path",
        str(layout_path),
        "--layout-name",
        "doclayout_yolo",
        "--pages",
        args.pages,
        "--engines",
        "paddleocr",
        "--paddle-device",
        "gpu:0",
        "--output-root",
        str(args.output_root),
        "--cache-root",
        str(args.cache_root),
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)

    engine_output = args.output_root / "paddleocr"
    summary_path = engine_output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("regions_error", 0):
        raise RuntimeError(f"PaddleOCR run did not complete cleanly: {summary}")
    print(json.dumps(summary, indent=2), flush=True)

    if args.archive_path is not None:
        archive = _archive(engine_output, args.archive_path)
        print(f"Download from the Kaggle output panel: {archive}", flush=True)


if __name__ == "__main__":
    main()
