# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec: {display_name: Python 3, language: python, name: python3}
# ---
# fmt: off
# ruff: noqa
# %% [markdown]
# # Build Embedding Models & Indexing Tools Asset Bundle for Offline Kaggle
#
# Downloads official Hugging Face sentence embedding checkpoints across lightweight,
# standard, and LLM-based architectures (MiniLM, BGE, GTE-Qwen2, MPNet) along with
# offline Python wheels for `sentence-transformers`, `faiss-cpu`, `rank-bm25`, and
# `langchain-text-splitters`.
#
# Run this notebook in Kaggle with **Internet ON** and **CPU (no accelerator)**.
# The resulting output folder `/kaggle/working/embedding-indexing-offline-assets`
# can be saved as a private Kaggle dataset for 100% network-free indexing experiments.

# %%
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

ASSET_NAME = "embedding-indexing-offline-assets"
WORK = Path("/kaggle/working")
ASSET_DIR = WORK / ASSET_NAME
MODEL_ROOT = ASSET_DIR / "models"
WHEEL_DIR = ASSET_DIR / "wheels"
PIP_REPORT = WORK / "embedding-indexing-pip-report.json"

MODEL_SPECS = [
    {
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "name": "all-minilm-l6-v2",
        "dim": 384,
        "license": "apache-2.0",
        "description": "Lightweight 22M param fast dense baseline (starter default)",
    },
    {
        "id": "BAAI/bge-small-en-v1.5",
        "name": "bge-small-en-v1-5",
        "dim": 384,
        "license": "mit",
        "description": "High-accuracy compact 33M param MTEB dense retriever",
    },
    {
        "id": "BAAI/bge-base-en-v1.5",
        "name": "bge-base-en-v1-5",
        "dim": 768,
        "license": "mit",
        "description": "Balanced 110M param BERT-base dense retriever",
    },
    {
        "id": "sentence-transformers/all-mpnet-base-v2",
        "name": "all-mpnet-base-v2",
        "dim": 768,
        "license": "apache-2.0",
        "description": "High-performing RoBERTa/MPNet 110M param semantic embedder",
    },
    {
        "id": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "name": "gte-qwen2-1-5b-instruct",
        "dim": 1536,
        "license": "apache-2.0",
        "description": "SOTA 1.5B param Qwen2-based instruction-aware embedding model",
    },
]

REQUIREMENTS = [
    "sentence-transformers>=3.0.0,<4",
    "transformers>=4.44.0,<5",
    "faiss-cpu>=1.8.0,<2",
    "tiktoken>=0.7.0,<1",
    "safetensors>=0.4.0,<1",
    "accelerate>=0.30.0,<2",
    "einops>=0.7.0,<1",
    "rank-bm25>=0.2.2,<1",
    "langchain-text-splitters>=0.2.0,<1",
]


def run_cmd(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if process.stdout is None:
        raise RuntimeError("subprocess stdout was not captured")
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait():
        raise RuntimeError(f"command failed: {command!r}")


def collect_wheels() -> list[dict[str, Any]]:
    print("\n=== Resolving & Downloading Wheels ===", flush=True)
    run_cmd(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--ignore-installed",
            "--report",
            str(PIP_REPORT),
            *REQUIREMENTS,
        ]
    )
    report = json.loads(PIP_REPORT.read_text(encoding="utf-8"))
    excluded = {"torch", "torchvision", "triton"}
    resolved: list[str] = []
    runtime_provided: list[str] = []

    for item in report.get("install", []):
        metadata = item.get("metadata", {})
        name = str(metadata.get("name", "")).strip()
        version = str(metadata.get("version", "")).strip()
        if not name or not version:
            raise RuntimeError("pip report contained an incomplete package record")
        pinned = f"{name}=={version}"
        normalized = name.lower().replace("_", "-")
        if normalized in excluded or normalized.startswith("nvidia-"):
            runtime_provided.append(pinned)
        else:
            resolved.append(pinned)

    if not resolved:
        raise RuntimeError("pip resolved no downloadable dependencies")

    run_cmd(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--only-binary=:all:",
            "--dest",
            str(WHEEL_DIR),
            *resolved,
        ]
    )
    PIP_REPORT.unlink(missing_ok=True)
    wheels = sorted(path for path in WHEEL_DIR.iterdir() if path.is_file())
    if len(wheels) != len(resolved):
        raise RuntimeError(f"expected {len(resolved)} wheels, found {len(wheels)}")

    print(f"Downloaded {len(wheels)} wheels. Runtime-provided packages: {runtime_provided}")
    return [
        {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in wheels
    ]


def file_records(root: Path) -> list[dict[str, Any]]:
    symlinks = [path for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"snapshot contains symlinks: {symlinks}")
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(path for path in root.rglob("*") if path.is_file())
    ]


def download_model(spec: dict[str, Any]) -> dict[str, Any]:
    model_id = spec["id"]
    name = spec["name"]
    model_dir = MODEL_ROOT / name
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Downloading {model_id} -> {model_dir} ===", flush=True)

    revision = HfApi().model_info(model_id).sha
    if not revision:
        raise RuntimeError(f"Hugging Face returned no immutable revision for {model_id}")

    resolved = snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=model_dir,
        ignore_patterns=["pytorch_model.bin", "*.msgpack", "*.h5", "*.onnx", "*.tflite", "*.ot"],
        max_workers=8,
    )
    if Path(resolved).resolve() != model_dir.resolve():
        raise RuntimeError(f"snapshot landed at unexpected path: {resolved}")

    shutil.rmtree(model_dir / ".cache", ignore_errors=True)
    records = file_records(model_dir)

    # Validate essential files
    files = {path.name for path in model_dir.rglob("*") if path.is_file()}
    if "config.json" not in files:
        raise FileNotFoundError(f"{model_id} missing config.json")
    if not any(f.endswith(".safetensors") for f in files):
        raise FileNotFoundError(f"{model_id} missing .safetensors weights")

    return {
        "model_id": model_id,
        "name": name,
        "dim": spec["dim"],
        "license": spec["license"],
        "description": spec["description"],
        "revision": revision,
        "directory": str(model_dir.relative_to(ASSET_DIR)),
        "files": records,
        "bytes": sum(int(str(record["bytes"])) for record in records),
    }


def build() -> None:
    if not WORK.is_dir():
        raise RuntimeError("run this builder inside a Kaggle notebook")

    if ASSET_DIR.exists():
        shutil.rmtree(ASSET_DIR)
    MODEL_ROOT.mkdir(parents=True)
    WHEEL_DIR.mkdir(parents=True)

    print("Python version:", sys.version.split()[0])
    print(f"Free disk space: {shutil.disk_usage(WORK).free / 1e9:.2f} GB")
    started = time.perf_counter()

    downloaded_models = []
    for spec in MODEL_SPECS:
        rec = download_model(spec)
        downloaded_models.append(rec)

    wheels = collect_wheels()

    receipt = {
        "schema": 1,
        "asset": ASSET_NAME,
        "source": "https://huggingface.co",
        "models": downloaded_models,
        "wheels": wheels,
        "wheel_file_count": len(wheels),
        "portable": {
            "ordinary_files_only": True,
            "symlink_count": 0,
            "internet_required_after_build": False,
            "accelerator_required_to_build": False,
            "pickle_checkpoints_excluded": True,
            "runtime_note": "Target runtime uses Kaggle PyTorch stack with sentence-transformers and FAISS.",
        },
        "download_seconds": round(time.perf_counter() - started, 3),
    }

    receipt_path = ASSET_DIR / "asset-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print(
        json.dumps(
            {
                "asset": ASSET_NAME,
                "models": len(downloaded_models),
                "wheels": len(wheels),
                "total_bytes": sum(m["bytes"] for m in downloaded_models) + sum(w["bytes"] for w in wheels),
            },
            indent=2,
        )
    )
    print("=" * 60)
    print(f"SUCCESS: Package created at {ASSET_DIR}")
    print("Next step: Save notebook output as a private Kaggle dataset named 'embedding-indexing-offline-assets'.")


if __name__ == "__main__":
    build()
