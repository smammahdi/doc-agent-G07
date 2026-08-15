from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


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
        from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel

        self.model = None
        self.hf_model = None
        self.tokenizer = None
        self.llama = None
        m_path = Path(self.resolved_path)
        gguf_candidates = list(m_path.glob("*.gguf")) + list(m_path.glob("*/*.gguf")) if m_path.is_dir() else ([m_path] if str(m_path).endswith(".gguf") else [])
        if gguf_candidates:
            try:
                from llama_cpp import Llama
                gguf_file = str(gguf_candidates[0])
                n_gpu = -1 if ("cuda" in device or device == "cuda") else 0
                self.llama = Llama(model_path=gguf_file, embedding=True, n_gpu_layers=n_gpu, verbose=False)
            except Exception as e:
                pass

        # Step 1: SentenceTransformer loader
        if self.llama is None:
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
                    self.model = SentenceTransformer(self.resolved_path, device=device)
                except Exception:
                    pass

        # Step 2: HF AutoModel + AutoTokenizer loader
        if self.model is None and self.llama is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.resolved_path, trust_remote_code=True)
                self.hf_model = AutoModel.from_pretrained(self.resolved_path, trust_remote_code=True).to(device)
                self.hf_model.eval()
            except Exception:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.resolved_path)
                    self.hf_model = AutoModel.from_pretrained(self.resolved_path).to(device)
                    self.hf_model.eval()
                except Exception:
                    pass

        # Step 3: Offline BERT state-dict fallback (e.g. when custom remote code file is missing from snapshot)
        if self.model is None and self.hf_model is None and self.llama is None:
            m_dir = Path(self.resolved_path)
            cfg_path = m_dir / "config.json"
            if not cfg_path.is_file():
                for sub in m_dir.glob("*/config.json"):
                    cfg_path = sub
                    break
            cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
            bert_cfg = BertConfig(
                vocab_size=cfg_dict.get("vocab_size", 30522),
                hidden_size=cfg_dict.get("hidden_size", 768),
                num_hidden_layers=cfg_dict.get("num_hidden_layers", 12),
                num_attention_heads=cfg_dict.get("num_attention_heads", 12),
                intermediate_size=cfg_dict.get("intermediate_size", 3072),
                max_position_embeddings=cfg_dict.get("max_position_embeddings", 2048),
                type_vocab_size=cfg_dict.get("type_vocab_size", 2),
                pad_token_id=cfg_dict.get("pad_token_id", 0),
            )
            self.hf_model = BertModel(bert_cfg).to(device)
            weight_files = list(m_dir.rglob("*.safetensors")) + list(m_dir.rglob("*.bin"))
            if weight_files:
                wf = weight_files[0]
                if wf.suffix == ".safetensors":
                    from safetensors.torch import load_file
                    sd = load_file(str(wf))
                else:
                    sd = torch.load(str(wf), map_location=device)
                clean_sd = {k.replace("nomic_bert.", "").replace("encoder.bert.", "encoder."): v for k, v in sd.items()}
                self.hf_model.load_state_dict(clean_sd, strict=False)
            self.hf_model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(str(m_dir), trust_remote_code=False)

        if self.model is None and self.hf_model is None and self.llama is None:
            raise RuntimeError(f"Failed to load embedding model from {self.resolved_path}")

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        formatted = [f"{self.query_prefix}{q}" for q in queries]
        if self.llama is not None:
            return self._encode_llama(formatted)
        if self.model is not None:
            embs = self.model.encode(
                formatted, batch_size=32, normalize_embeddings=True, show_progress_bar=False
            )
            return np.asarray(embs, dtype=np.float32)
        return self._encode_hf(formatted, batch_size=32)

    def encode_documents(self, documents: list[str], batch_size: int = 32) -> np.ndarray:
        formatted = [f"{self.doc_prefix}{d}" for d in documents]
        if self.llama is not None:
            return self._encode_llama(formatted)
        if self.model is not None:
            embs = self.model.encode(
                formatted, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
            )
            return np.asarray(embs, dtype=np.float32)
        return self._encode_hf(formatted, batch_size=batch_size)

    def _encode_llama(self, texts: list[str]) -> np.ndarray:
        all_embs = []
        for text in texts:
            res = self.llama.create_embedding(text)  # type: ignore[union-attr]
            vec = np.array(res["data"][0]["embedding"], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-9:
                vec = vec / norm
            all_embs.append(vec)
        return np.vstack(all_embs) if all_embs else np.empty((0, 1024), dtype=np.float32)

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
    """Validates that a local directory contains necessary config, weights, and tokenizer files (or GGUF)."""
    if not path.is_dir():
        return False
    if any(path.glob("*.gguf")) or any(path.glob("*/*.gguf")):
        return True
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
    """Discovers candidate embedding models from local search roots.

    When require_local is True, only valid local model directories are returned;
    no HuggingFace hub name fallbacks are emitted.
    """
    model_specs = [
        ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", ["all-minilm-l6-v2", "minilm", "all_minilm"]),
        ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", ["bge-small-en-v1-5", "bge-small-en-v1.5", "bge_small_en", "bge-small"]),
        ("bge-m3", "BAAI/bge-m3", ["bge-m3", "bge_m3"]),
        ("nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1.5", ["nomic-embed-text-v1-5", "nomic-embed-text-v1.5", "nomic_embed", "nomic"]),
        ("Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B", ["qwen3-embedding-0-6b", "qwen3-embedding-0.6b", "qwen3_embedding", "qwen3-embedding", "qwen3-embedding-0-6b-gguf", "qwen3-embedding-4b-gguf"]),
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
