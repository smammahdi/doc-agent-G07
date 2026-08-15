from __future__ import annotations

from pathlib import Path

import numpy as np


def sanitize_offline_model_dir(model_path: str | Path) -> str:
    """Sanitize local model directory by stripping upstream HF repo prefixes from auto_map."""
    path_obj = Path(model_path)
    if not path_obj.is_dir():
        return str(model_path)
    cfg_file = path_obj / "config.json"
    if not cfg_file.is_file():
        return str(model_path)
    try:
        import json
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    except Exception:
        return str(model_path)

    auto_map = cfg.get("auto_map", {})
    if not any("--" in str(v) for v in auto_map.values()):
        return str(model_path)

    sanitized_dir = Path("/tmp/offline_sanitized_models") / path_obj.name
    sanitized_dir.mkdir(parents=True, exist_ok=True)

    for item in path_obj.iterdir():
        dest = sanitized_dir / item.name
        if not dest.exists():
            try:
                dest.symlink_to(item, target_is_directory=item.is_dir())
            except Exception:
                pass

    new_auto_map = {k: str(v).split("--")[-1] for k, v in auto_map.items()}
    cfg["auto_map"] = new_auto_map
    (sanitized_dir / "config.json").unlink(missing_ok=True)
    (sanitized_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return str(sanitized_dir)


class EmbeddingModelAdapter:
    """Standardized model adapter enforcing exact prefixes, pooling, and L2 normalization."""

    def __init__(self, model_name_or_path: str, canonical_id: str | None = None, device: str = "cpu"):
        self.device = device
        self.raw_path = str(model_name_or_path)
        self.resolved_path = sanitize_offline_model_dir(model_name_or_path)
        self.raw_name = Path(model_name_or_path).name.lower()

        # Resolve Canonical Model ID
        if canonical_id:
            self.model_id = canonical_id
        elif "bge-small" in self.raw_name:
            self.model_id = "bge-small-en-v1.5"
        elif "bge-m3" in self.raw_name:
            self.model_id = "bge-m3"
        elif "nomic" in self.raw_name:
            self.model_id = "nomic-embed-text-v1.5"
        elif "qwen" in self.raw_name:
            self.model_id = "Qwen3-Embedding-0.6B"
        else:
            self.model_id = "all-MiniLM-L6-v2"

        # Set official prefixes
        if self.model_id == "bge-small-en-v1.5":
            self.query_prefix = "Represent this sentence for searching relevant passages: "
            self.doc_prefix = ""
        elif self.model_id == "bge-m3":
            self.query_prefix = ""
            self.doc_prefix = ""
        elif self.model_id == "nomic-embed-text-v1.5":
            self.query_prefix = "search_query: "
            self.doc_prefix = "search_document: "
        elif self.model_id == "Qwen3-Embedding-0.6B":
            self.query_prefix = (
                "Instruct: Given a medical query, retrieve relevant passages from the historical medical text that answer the query\nQuery: "
            )
            self.doc_prefix = ""
        else:
            self.query_prefix = ""
            self.doc_prefix = ""

        # Model Loading with robust offline cascading fallback
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModel, AutoTokenizer

        self.model = None
        self.hf_model = None
        self.tokenizer = None
        self.is_qwen = "qwen" in self.model_id.lower()

        # Cascading loader: Try SentenceTransformer -> then HF AutoModel
        try:
            self.model = SentenceTransformer(
                self.resolved_path,
                device=device,
                trust_remote_code=True,
                model_kwargs={"trust_remote_code": True},
                tokenizer_kwargs={"trust_remote_code": True},
                config_kwargs={"trust_remote_code": True},
            )
        except Exception:
            try:
                self.model = SentenceTransformer(self.resolved_path, device=device, trust_remote_code=True)
            except Exception:
                try:
                    self.model = SentenceTransformer(self.resolved_path, device=device)
                except Exception:
                    pass

        if self.model is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.resolved_path, trust_remote_code=True)
                self.hf_model = AutoModel.from_pretrained(self.resolved_path, trust_remote_code=True).to(device)
                self.hf_model.eval()
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(self.resolved_path)
                self.hf_model = AutoModel.from_pretrained(self.resolved_path).to(device)
                self.hf_model.eval()

        if self.model is None and self.hf_model is None:
            raise RuntimeError(f"Failed to load embedding model from {self.resolved_path}")

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        formatted = [f"{self.query_prefix}{q}" for q in queries]
        if self.model is not None:
            embs = self.model.encode(
                formatted, batch_size=32, normalize_embeddings=True, show_progress_bar=False
            )
            return np.asarray(embs, dtype=np.float32)
        return self._encode_hf(formatted, batch_size=32)

    def encode_documents(self, documents: list[str], batch_size: int = 32) -> np.ndarray:
        formatted = [f"{self.doc_prefix}{d}" for d in documents]
        if self.model is not None:
            embs = self.model.encode(
                formatted, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
            )
            return np.asarray(embs, dtype=np.float32)
        return self._encode_hf(formatted, batch_size=batch_size)

    def _encode_hf(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        import torch

        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.hf_model(**encoded)
                hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                if self.is_qwen:
                    # Official last-token / EOS token pooling for Qwen embeddings
                    attention_mask = encoded["attention_mask"]
                    seq_lens = attention_mask.sum(dim=1) - 1
                    pooled = hidden[torch.arange(hidden.size(0)), seq_lens]
                else:
                    # Mean pooling for BERT / RoBERTa / Nomic
                    mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                    sum_emb = torch.sum(hidden * mask, dim=1)
                    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                    pooled = sum_emb / sum_mask
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_embs.append(pooled.cpu().to(torch.float32).numpy())
        return np.vstack(all_embs) if all_embs else np.empty((0, 384), dtype=np.float32)


def is_valid_local_model_dir(path: Path) -> bool:
    """Validates that a local directory contains necessary config, weights, and tokenizer files."""
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False

    has_weights = any(
        (path / name).is_file() or list(path.glob(f"*{ext}"))
        for name in ["model.safetensors", "pytorch_model.bin", "model.onnx"]
        for ext in [".safetensors", ".bin", ".onnx"]
    )
    has_tokenizer = any(
        (path / name).is_file()
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "tokenizer.model",
            "spiece.model",
            "vocab.json",
        ]
    )
    return has_weights and has_tokenizer


def discover_candidate_models(
    search_roots: list[Path] | None = None,
    require_local: bool = False,
) -> dict[str, tuple[str, str]]:
    """Discovers the 5 candidate embedding models from local search roots.

    When require_local is True, only valid local model directories are returned;
    no HuggingFace hub name fallbacks are emitted.
    """
    model_specs = [
        ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", ["all-minilm-l6-v2", "minilm", "all_minilm"]),
        ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", ["bge-small-en-v1-5", "bge-small-en-v1.5", "bge_small_en", "bge-small"]),
        ("bge-m3", "BAAI/bge-m3", ["bge-m3", "bge_m3"]),
        ("nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1.5", ["nomic-embed-text-v1-5", "nomic-embed-text-v1.5", "nomic_embed", "nomic"]),
        ("Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B", ["qwen3-embedding-0-6b", "qwen3-embedding-0.6b", "qwen3_embedding", "qwen3-embedding"]),
    ]

    default_roots = [
        Path("/kaggle/input"),
        Path("extras/indexing-benchmarks/models"),
        Path("extras/indexing-benchmarks"),
        Path("models"),
        Path("."),
    ]
    roots = search_roots or default_roots

    discovered: dict[str, tuple[str, str]] = {}
    missing: list[str] = []

    for c_id, default_hub, aliases in model_specs:
        found_path: str | None = None
        for r in roots:
            if not r or not r.exists():
                continue

            # 1. Direct directory match
            for alias in aliases:
                direct = r / alias
                if is_valid_local_model_dir(direct):
                    found_path = str(direct.resolve())
                    break

            if found_path:
                break

            # 2. Recursive search
            for alias in aliases:
                for d in r.rglob(f"*{alias}*"):
                    if is_valid_local_model_dir(d) and "reranker" not in d.name.lower():
                        found_path = str(d.resolve())
                        break
                if found_path:
                    break

            if found_path:
                break

        if found_path:
            discovered[c_id] = (c_id, found_path)
        elif require_local:
            missing.append(c_id)
        else:
            discovered[c_id] = (c_id, default_hub)

    if require_local and missing:
        raise FileNotFoundError(
            f"Offline Model Preflight Failed: Could not find valid local model directories for: {missing}. "
            f"Searched roots: {[str(r) for r in roots if r and r.exists()]}. "
            f"Ensure all 5 model datasets are attached to the Kaggle notebook."
        )

    return discovered
