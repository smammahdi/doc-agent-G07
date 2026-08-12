"""Kaggle GPU entrypoint for the full-book TrOCR output run.

This script saves TrOCR text from two existing layout sidecars. It does not
rerun either layout detector, use Chandra's text, or score OCR quality.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/smammahdi/doc-agent-G07.git"
BRANCH = "a2/trocr-layout-comparison"
WORK = Path("/kaggle/working")
REPO = WORK / "doc-agent-G07"
OUTPUT = WORK / "trocr_layout_outputs"
CACHE = WORK / "trocr_page_cache"


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    run(
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
            "pydantic>=2.7,<3",
            "pydantic-settings>=2.2,<3",
            "pyyaml>=6,<7",
        ]
    )
    if not REPO.exists():
        run(["git", "clone", "--branch", BRANCH, "--depth", "1", REPOSITORY, str(REPO)])
    else:
        run(["git", "-C", str(REPO), "fetch", "origin", BRANCH, "--depth", "1"])
        run(["git", "-C", str(REPO), "checkout", "--force", "FETCH_HEAD"])
    print("repository:", end=" ", flush=True)
    run(["git", "-C", str(REPO), "rev-parse", "HEAD"])

    wrapper = REPO / "extras" / "ocr_research" / "run_kaggle_layout_trocr.py"
    run(
        [
            sys.executable,
            str(wrapper),
            "--input-root",
            "/kaggle/input",
            "--output-root",
            str(OUTPUT),
            "--cache-root",
            str(CACHE),
            "--device",
            "cuda",
            "--batch-size",
            "8",
            "--max-length",
            "64",
            "--dpi",
            "300",
        ]
    )
    archive = shutil.make_archive(str(WORK / "trocr_layout_outputs"), "zip", OUTPUT)
    print("canonical output:", OUTPUT)
    print("downloadable archive:", archive)


if __name__ == "__main__":
    main()
