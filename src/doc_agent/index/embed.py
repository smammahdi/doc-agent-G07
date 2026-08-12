"""Stage 4: encode chunks with the configured sentence embedding model."""

from __future__ import annotations

from typing import Any

from ..contracts import Chunk


def _settings(cfg: dict[str, Any]) -> tuple[str, int, int]:
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a mapping")
    embed_cfg = cfg.get("embed", {})
    if not isinstance(embed_cfg, dict):
        raise ValueError("cfg['embed'] must be a mapping")
    model = embed_cfg.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("embed.model must be a non-empty model name")
    dim = embed_cfg.get("dim")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise ValueError("embed.dim must be a positive integer")
    batch_size = embed_cfg.get("batch_size", 32)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("embed.batch_size must be a positive integer")
    return model, dim, batch_size


def encode(chunks: list[Chunk], cfg: dict[str, Any]) -> Any:
    """Return a float32 matrix aligned one-to-one with ``chunks``.

    Model loading is deliberately lazy: importing the package and running structure
    checks does not download a checkpoint. A real build raises the original model
    or dependency error instead of silently producing placeholder vectors.
    """
    model_name, expected_dim, batch_size = _settings(cfg)
    if not isinstance(chunks, list) or any(not isinstance(chunk, Chunk) for chunk in chunks):
        raise TypeError("chunks must be a list of Chunk contracts")
    if not chunks:
        import numpy as np

        return np.empty((0, expected_dim), dtype="float32")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - depends on the runtime environment
        raise RuntimeError(
            "sentence-transformers is required to build embeddings; "
            "install the project dependencies"
        ) from exc

    import numpy as np

    model = SentenceTransformer(model_name)
    vectors = model.encode(
        [chunk.text for chunk in chunks],
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2 or matrix.shape != (len(chunks), expected_dim):
        raise ValueError(
            f"embedding shape {matrix.shape} does not match {(len(chunks), expected_dim)}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("embedding output contains non-finite values")
    return matrix
