#!/usr/bin/env python3
"""Stage 4 end-to-end index build.

Usage (from repo root):
    uv run python scripts/run_index.py
    # or via the shell wrapper:
    bash scripts/build_index.sh

What it does:
  1. Parse chandra/pages.md into one Chunk per book page (load_from_pages_markdown).
  2. Apply sliding-window token chunking (split).
  3. Embed with sentence-transformers/all-MiniLM-L6-v2 (encode).
  4. Persist FAISS HNSW + chunks.jsonl + metadata.json (build).
  5. Save image_index.json alongside the FAISS artifacts for the demo notebook.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Make src importable when run directly
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import yaml

from doc_agent.index.chunk import (
    build_image_index,
    load_from_canonical_jsonl,
    load_from_mineru_jsonl,
    load_from_pages_markdown,
    split,
)
from doc_agent.index.store import build

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CFG_PATH = REPO / "configs" / "config.yaml"
MINERU_PAGES = REPO / "extras" / "output" / "mineru-ocr-full-book" / "full-page" / "pages.jsonl"
PAGES_MD = REPO / "chandra" / "pages.md"
CHANDRA_DIR = REPO / "chandra"
CANONICAL_CANDIDATES = [
    REPO / "data" / "canonical-pages.jsonl",
    REPO / "extras" / "indexing-benchmarks" / "data" / "canonical-pages.jsonl",
]


def main() -> None:
    t0 = time.perf_counter()

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    doc_id = cfg.get("ingest", {}).get("doc_id", "pierce-1890")
    index_path = Path(cfg["index"]["path"])

    # Prefer Canonical Corpus -> MinerU -> Chandra fallback
    canonical_file = next((p for p in CANONICAL_CANDIDATES if p.is_file()), None)
    use_mineru = MINERU_PAGES.is_file()

    if canonical_file:
        source_desc = f"Canonical Corpus ({canonical_file})"
    elif use_mineru:
        source_desc = f"MinerU ({MINERU_PAGES})"
    else:
        source_desc = f"Chandra ({PAGES_MD})"

    print("=" * 70)
    print("  Doc-Agent G07 — Stage 4 Index Build")
    print(f"  Source : {source_desc}")
    print(f"  doc_id : {doc_id}")
    print(f"  Index  : {index_path}")
    print("=" * 70)

    # ── Step 1: Load and link pages + image index ────────────────────────────
    print("\n[1/4] Loading pages and building multimodal image index ...", flush=True)
    image_index = build_image_index(PAGES_MD) if PAGES_MD.is_file() else {}

    if canonical_file:
        page_chunks, image_index = load_from_canonical_jsonl(
            canonical_file, doc_id, image_index=image_index
        )
        print(f"      → Loaded {len(page_chunks)} pages from Canonical Corpus", flush=True)
    elif use_mineru:
        page_chunks, _ = load_from_mineru_jsonl(MINERU_PAGES, doc_id, image_index=image_index)
        print(f"      → Loaded {len(page_chunks)} pages from MinerU SOTA OCR", flush=True)
    else:
        page_chunks, image_index = load_from_pages_markdown(PAGES_MD, doc_id)
        print(f"      → Loaded {len(page_chunks)} pages from Chandra OCR", flush=True)

    print(
        f"      → {sum(len(v) for v in image_index.values())} figure references "
        f"linked across {len(image_index)} illustrated pages",
        flush=True,
    )

    # ── Step 2: Sliding-window token chunking ─────────────────────────────────
    print(
        "\n[2/4] Chunking (tokens={}, overlap={}) ...".format(
            cfg["index"]["chunk_tokens"], cfg["index"]["overlap"]
        ),
        flush=True,
    )
    chunks = split(page_chunks, cfg)
    total_tokens = sum(len(c.text.split()) for c in chunks)
    print(f"      → {len(chunks)} chunks  ({total_tokens:,} total tokens)", flush=True)

    # ── Step 3: Embed ──────────────────────────────────────────────────────────
    print(
        f"\n[3/4] Embedding with {cfg['embed']['model']} "
        f"(batch={cfg['embed'].get('batch_size', 32)}) ...",
        flush=True,
    )

    import numpy as np
    from sentence_transformers import SentenceTransformer

    # encode() in embed.py validates shape/finiteness; we call SentenceTransformer
    # directly here to enable show_progress_bar and normalize_embeddings.
    model = SentenceTransformer(cfg["embed"]["model"])
    batch = cfg["embed"].get("batch_size", 32)
    vectors = model.encode(
        [c.text for c in chunks],
        batch_size=batch,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-norm ⟹ inner-product == cosine similarity
    )
    vectors = np.asarray(vectors, dtype="float32")
    print(f"      → Embedding matrix: {vectors.shape}", flush=True)

    # ── Step 4: Persist FAISS HNSW index ──────────────────────────────────────
    print("\n[4/4] Persisting FAISS HNSW index ...", flush=True)
    build(chunks, vectors, cfg)

    # ── Bonus: save image_index.json alongside the FAISS store ────────────────
    index_path.mkdir(parents=True, exist_ok=True)
    img_idx_path = index_path / "image_index.json"
    img_idx_path.write_text(
        json.dumps(image_index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"      → image_index.json written ({len(image_index)} pages with figures)", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\n✅  Index built in {elapsed:.1f}s")
    print(f"   {len(chunks)} chunks · dim {cfg['embed']['dim']} · " f"{index_path}/index.faiss")


if __name__ == "__main__":
    main()
