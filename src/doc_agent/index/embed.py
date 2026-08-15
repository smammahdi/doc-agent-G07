"""Stage 4: encode chunks with the configured sentence embedding model."""

from __future__ import annotations

from typing import Any

from ..contracts import Chunk


def _settings(cfg: dict[str, Any]) -> tuple[str, str | None, int, int]:
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a mapping")
    embed_cfg = cfg.get("embed", {})
    if not isinstance(embed_cfg, dict):
        raise ValueError("cfg['embed'] must be a mapping")
    model = embed_cfg.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("embed.model must be a non-empty model name")
    revision = embed_cfg.get("revision")
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise ValueError("embed.revision must be a non-empty revision string when provided")
    dim = embed_cfg.get("dim")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise ValueError("embed.dim must be a positive integer")
    batch_size = embed_cfg.get("batch_size", 32)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("embed.batch_size must be a positive integer")
    return model, revision, dim, batch_size


def get_query_prefix(model_name: str) -> str:
    """Return model-specific query instruction prefix."""
    lower = model_name.lower()
    if "qwen" in lower:
        return "Instruct: Given a medical query, retrieve relevant passages from the historical medical text that answer the query\nQuery: "
    if "bge-small" in lower:
        return "Represent this sentence for searching relevant passages: "
    if "nomic" in lower:
        return "search_query: "
    return ""


def get_doc_prefix(model_name: str) -> str:
    """Return model-specific document prefix."""
    lower = model_name.lower()
    if "nomic" in lower:
        return "search_document: "
    return ""


def _encode_texts(
    texts: list[str],
    model_name: str,
    revision: str | None,
    expected_dim: int,
    batch_size: int = 32,
    device: str | None = None,
) -> Any:
    import numpy as np

    if not texts:
        return np.empty((0, expected_dim), dtype="float32")

    is_qwen = "qwen" in model_name.lower()

    # Step 1: Attempt SentenceTransformer
    try:
        from sentence_transformers import SentenceTransformer

        st_kwargs: dict[str, Any] = {"device": device, "trust_remote_code": True}
        if revision:
            st_kwargs["revision"] = revision
        model = SentenceTransformer(model_name, **st_kwargs)
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        matrix = np.asarray(vectors, dtype="float32")
    except Exception:
        # Step 2: Fallback to Transformers AutoModel + AutoTokenizer with last-token pool for Qwen
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer, Qwen2Config, Qwen2Model

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    revision=revision,
                    use_fast=False,
                    trust_remote_code=True,
                )
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    revision=revision,
                    trust_remote_code=True,
                )

            try:
                hf_model = AutoModel.from_pretrained(
                    model_name,
                    revision=revision,
                    trust_remote_code=True,
                )
            except Exception:
                if is_qwen:
                    qwen_cfg = Qwen2Config.from_pretrained(model_name, revision=revision)
                    hf_model = Qwen2Model.from_pretrained(
                        model_name,
                        revision=revision,
                        config=qwen_cfg,
                    )
                else:
                    raise

            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            hf_model = hf_model.to(dev)
            hf_model.eval()

            all_embs = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                encoded = tokenizer(
                    batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(dev)
                with torch.no_grad():
                    out = hf_model(**encoded)
                    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                    if is_qwen:
                        # Last-token / EOS token pooling for causal decoder Qwen embeddings
                        attention_mask = encoded["attention_mask"]
                        seq_lens = attention_mask.sum(dim=1) - 1
                        pooled = hidden[torch.arange(hidden.size(0)), seq_lens]
                    else:
                        # Mean pooling for standard BERT / RoBERTa models
                        mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                        sum_emb = torch.sum(hidden * mask, dim=1)
                        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                        pooled = sum_emb / sum_mask
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    all_embs.append(pooled.cpu().to(torch.float32).numpy())
            matrix = (
                np.vstack(all_embs) if all_embs else np.empty((0, expected_dim), dtype="float32")
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to encode texts with embedding model '{model_name}'"
            ) from exc

    # Ensure L2 normalization
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    if matrix.ndim != 2 or matrix.shape != (len(texts), expected_dim):
        raise ValueError(
            f"embedding shape {matrix.shape} does not match {(len(texts), expected_dim)}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("embedding output contains non-finite values")
    return matrix.astype("float32")


def encode(chunks: list[Chunk], cfg: dict[str, Any]) -> Any:
    """Return a float32 matrix aligned one-to-one with ``chunks`` (document encoding).

    Model loading is deliberately lazy: importing the package and running structure
    checks does not download a checkpoint. A real build raises the original model
    or dependency error instead of silently producing placeholder vectors.
    """
    model_name, revision, expected_dim, batch_size = _settings(cfg)
    if not isinstance(chunks, list) or any(not isinstance(chunk, Chunk) for chunk in chunks):
        raise TypeError("chunks must be a list of Chunk contracts")
    if not chunks:
        import numpy as np

        return np.empty((0, expected_dim), dtype="float32")

    doc_prefix = get_doc_prefix(model_name)
    texts = [f"{doc_prefix}{chunk.text}" for chunk in chunks]
    return _encode_texts(
        texts,
        model_name,
        revision,
        expected_dim,
        batch_size=batch_size,
    )


def encode_queries(queries: list[str], cfg: dict[str, Any]) -> Any:
    """Encode search queries using model-specific instruction prefix and L2 normalization."""
    model_name, revision, expected_dim, batch_size = _settings(cfg)
    if not isinstance(queries, list):
        raise TypeError("queries must be a list of strings")
    if not queries:
        import numpy as np

        return np.empty((0, expected_dim), dtype="float32")

    q_prefix = get_query_prefix(model_name)
    formatted = [f"{q_prefix}{q}" for q in queries]
    return _encode_texts(
        formatted,
        model_name,
        revision,
        expected_dim,
        batch_size=batch_size,
    )
