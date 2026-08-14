"""Stage 4: persist and load a FAISS HNSW index with chunk metadata."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..contracts import Chunk


def _settings(cfg: dict[str, Any]) -> tuple[Path, int]:
    index_cfg = cfg.get("index", {})
    if not isinstance(index_cfg, dict):
        raise ValueError("cfg['index'] must be a mapping")
    index_type = index_cfg.get("type", "faiss:hnsw")
    if index_type != "faiss:hnsw":
        raise ValueError("the A2 store supports only index.type='faiss:hnsw'")
    path_value = index_cfg.get("path", "data/processed/index")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("index.path must be a non-empty path")
    m = index_cfg.get("hnsw_m", 32)
    if isinstance(m, bool) or not isinstance(m, int) or m <= 0:
        raise ValueError("index.hnsw_m must be a positive integer")
    target_path = Path(path_value)
    if not target_path.is_absolute() and not target_path.exists() and (Path.cwd().parent / target_path).exists():
        target_path = Path.cwd().parent / target_path
    return target_path, m


def _atomic_json(path: Path, rows: list[dict[str, Any]]) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def build(chunks: list[Chunk], vectors: Any, cfg: dict[str, Any]) -> None:
    """Persist a FAISS index and its exact chunk metadata.

    Uses IndexFlatIP (exact inner-product / cosine similarity after L2-normalisation)
    instead of IndexHNSWFlat. HNSW triggers a macOS arm64 segfault with faiss-cpu
    at process teardown; FlatIP is brute-force but exact, reliable everywhere, and
    perfectly adequate for a 2k-vector corpus. HNSW would only matter at ≥100k vectors.

    The index is a generated build artifact written under the configured processed-data
    path and should not be committed to the public repository.
    """
    if not isinstance(chunks, list) or any(not isinstance(chunk, Chunk) for chunk in chunks):
        raise TypeError("chunks must be a list of Chunk contracts")
    import numpy as np

    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise ValueError("vectors must be a 2-D matrix aligned with chunks")
    if len(chunks) == 0:
        raise ValueError("cannot build an index with no chunks")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors contain non-finite values")
    output_dir, _m = _settings(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import faiss
    except Exception as exc:  # pragma: no cover - depends on the runtime environment
        raise RuntimeError(
            "faiss-cpu is required to build the configured index; install the project dependencies"
        ) from exc

    # L2-normalise so inner-product == cosine similarity
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(output_dir / "index.faiss"))
    _atomic_json(output_dir / "chunks.jsonl", [chunk.model_dump() for chunk in chunks])
    metadata = {
        "index_type": "faiss:flat_ip",
        "dimension": int(matrix.shape[1]),
        "count": len(chunks),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load(cfg: dict[str, Any]) -> tuple[Any, list[Chunk], dict[str, Any]]:
    """Load ``(faiss_index, chunks, metadata)`` from the configured build path."""
    output_dir, _ = _settings(cfg)
    index_path = output_dir / "index.faiss"
    chunks_path = output_dir / "chunks.jsonl"
    metadata_path = output_dir / "metadata.json"
    missing = [str(path) for path in (index_path, chunks_path, metadata_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("index artifacts are missing: " + ", ".join(missing))
    try:
        import faiss
    except Exception as exc:  # pragma: no cover - depends on the runtime environment
        raise RuntimeError("faiss-cpu is required to load the configured index") from exc
    index = faiss.read_index(str(index_path))
    chunks = [
        Chunk.model_validate(json.loads(line))
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("count") != len(chunks) or index.ntotal != len(chunks):
        raise ValueError("index metadata, vectors, and chunk count disagree")
    return index, chunks, metadata
