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
# # Package Vision-Language Multimodal & Hybrid Models (~6.4 GB)
#
# Dedicated asset builder packaging:
# 1. `Qwen/Qwen3-VL-Embedding-2B` (1536-d multimodal for text + figure crops, ~4.2 GB)
# 2. `BAAI/bge-m3` (1024-d universal dense + sparse BM25 multi-vector, ~2.2 GB)
# + Offline Python wheels (`sentence-transformers`, `faiss-cpu`, `rank-bm25`, etc.)
#
# Run on Kaggle (CPU, Internet ON). Saves as `multimodal-vl-embedding-assets`.

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

ASSET_NAME = "multimodal-vl-embedding-assets"
WORK = Path("/kaggle/working")
ASSET_DIR = WORK / ASSET_NAME
MODEL_ROOT = ASSET_DIR / "models"
WHEEL_DIR = ASSET_DIR / "wheels"
PIP_REPORT = WORK / "multimodal-vl-pip-report.json"

MODEL_SPECS = [
    {
        "id": "Qwen/Qwen3-VL-Embedding-2B",
        "name": "qwen3-vl-embedding-2b",
        "dim": 1536,
        "type": "multimodal-embedding",
        "license": "apache-2.0",
        "description": "2.0B param Qwen3 Vision-Language multimodal embedding model for text + figures",
    },
    {
        "id": "BAAI/bge-m3",
        "name": "bge-m3",
        "dim": 1024,
        "type": "text-embedding",
        "license": "mit",
        "description": "Universal SOTA model (dense + sparse + multi-vector) for complex document retrieval",
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

    ignore_patterns = ["*.msgpack", "*.h5", "*.onnx", "*.tflite", "*.ot", "*.flax*"]
    resolved = snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=model_dir,
        ignore_patterns=ignore_patterns,
        max_workers=8,
    )
    if Path(resolved).resolve() != model_dir.resolve():
        raise RuntimeError(f"snapshot landed at unexpected path: {resolved}")

    shutil.rmtree(model_dir / ".cache", ignore_errors=True)
    records = file_records(model_dir)

    files = {path.name for path in model_dir.rglob("*") if path.is_file()}
    if "config.json" not in files:
        raise FileNotFoundError(f"{model_id} missing config.json")
    has_weights = any(
        f.endswith(".safetensors") or f.endswith(".bin") or f.endswith(".pt") for f in files
    )
    if not has_weights:
        raise FileNotFoundError(f"{model_id} missing model weights (.safetensors or .bin)")

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
    print(f"Next step: Save notebook output as private Kaggle dataset named '{ASSET_NAME}'.")


if __name__ == "__main__":
    build()
