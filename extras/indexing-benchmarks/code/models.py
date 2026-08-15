from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class EmbeddingModelAdapter:
    """Standardized model adapter enforcing exact prefixes, pooling, and L2 normalization."""

    def __init__(self, model_name_or_path: str, canonical_id: str | None = None, device: str = "cpu"):
        self.device = device
        self.resolved_path = str(model_name_or_path)
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

        # Model Loading with graceful fallback
        from sentence_transformers import SentenceTransformer

        self.model = None
        self.hf_model = None
        self.tokenizer = None
        self.is_qwen = "qwen" in self.model_id.lower()

        try:
            self.model = SentenceTransformer(
                model_name_or_path,
                device=device,
                trust_remote_code=True,
                model_kwargs={"trust_remote_code": True},
                tokenizer_kwargs={"trust_remote_code": True},
                config_kwargs={"trust_remote_code": True},
            )
        except Exception:
            try:
                self.model = SentenceTransformer(model_name_or_path, device=device, trust_remote_code=True)
            except Exception:
                from transformers import AutoModel, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
                self.hf_model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)
                self.hf_model.eval()

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


def discover_candidate_models(search_roots: list[Path] | None = None) -> dict[str, tuple[str, str]]:
    """Returns mapping of canonical_id -> (canonical_id, resolved_path_or_hub_name)."""
    candidates = [
        ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"),
        ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5"),
        ("bge-m3", "BAAI/bge-m3"),
        ("nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1.5"),
        ("Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B"),
    ]

    roots = search_roots or [Path("/kaggle/input"), Path("extras/indexing-benchmarks"), Path(".")]
    discovered = {}

    for c_id, default_hub in candidates:
        c_short = c_id.lower().replace(".", "-")
        found_path = None
        for r in roots:
            if not r or not r.exists():
                continue
            for d in r.rglob(f"*{c_short}*"):
                if d.is_dir() and "reranker" not in d.name.lower() and (d / "config.json").is_file():
                    found_path = str(d)
                    break
            if found_path:
                break
        discovered[c_id] = (c_id, found_path or default_hub)

    return discovered
