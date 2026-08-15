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

from doc_agent.contracts import Chunk
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
VERIFIED_KB_DIR = (
    REPO
    / "extras"
    / "indexing-benchmarks"
    / "results"
    / "stage4-benchmark-results"
    / "production_kb"
)


def main() -> None:
    t0 = time.perf_counter()

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    doc_id = cfg.get("ingest", {}).get("doc_id", "pierce-1890")
    index_path = Path(cfg["index"]["path"])

    print("=" * 70)
    print("  Doc-Agent G07 — Stage 4 Index Build (Qwen3-Embedding-0.6B / FlatIP)")
    print(f"  Model  : {cfg['embed']['model']} (dim={cfg['embed']['dim']})")
    print(f"  Index  : {index_path} ({cfg['index']['type']})")
    print("=" * 70)

    # ── Step 1: Multimodal image index ───────────────────────────────────────
    image_index = build_image_index(PAGES_MD) if PAGES_MD.is_file() else {}

    # Check if verified benchmark production KB is present and matches config
    if (
        VERIFIED_KB_DIR.is_dir()
        and (VERIFIED_KB_DIR / "index.faiss").is_file()
        and (VERIFIED_KB_DIR / "chunks.jsonl").is_file()
        and cfg["embed"]["dim"] == 1024
    ):
        print("\n[1/4] Importing verified Stage-4 production Knowledge Base ...", flush=True)
        raw_lines = (VERIFIED_KB_DIR / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        chunks = []
        for line in raw_lines:
            if not line.strip():
                continue
            r = json.loads(line)
            chunks.append(
                Chunk(
                    id=r.get("id") or r.get("chunk_id", ""),
                    doc_id=r.get("doc_id", doc_id),
                    text=r.get("text", ""),
                    page_ids=r.get("page_ids") or ([r["page_id"]] if "page_id" in r else []),
                    score=0.0,
                )
            )

        print(f"      → Converted {len(chunks)} chunks to strict Chunk contracts", flush=True)

        import faiss

        index = faiss.read_index(str(VERIFIED_KB_DIR / "index.faiss"))
        if index.ntotal != len(chunks) or index.d != 1024:
            raise ValueError(
                f"Verified KB index shape mismatch: ntotal={index.ntotal}, dim={index.d}"
            )

        index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path / "index.faiss"))

        # Write atomic chunks.jsonl
        chunks_jsonl_path = index_path / "chunks.jsonl"
        with open(chunks_jsonl_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(c.model_dump_json() + "\n")

        # Write metadata.json
        metadata = {
            "index_type": "faiss:flat_ip",
            "dimension": int(index.d),
            "count": len(chunks),
        }
        (index_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        # Prefer Canonical Corpus -> MinerU -> Chandra fallback
        canonical_file = next((p for p in CANONICAL_CANDIDATES if p.is_file()), None)
        use_mineru = MINERU_PAGES.is_file()

        if canonical_file:
            source_desc = f"Canonical Corpus ({canonical_file})"
        elif use_mineru:
            source_desc = f"MinerU ({MINERU_PAGES})"
        else:
            source_desc = f"Chandra ({PAGES_MD})"

        print(f"\n[1/4] Loading pages from {source_desc} ...", flush=True)
        if canonical_file:
            page_chunks, image_index = load_from_canonical_jsonl(
                canonical_file, doc_id, image_index=image_index
            )
        elif use_mineru:
            page_chunks, _ = load_from_mineru_jsonl(MINERU_PAGES, doc_id, image_index=image_index)
        else:
            page_chunks, image_index = load_from_pages_markdown(PAGES_MD, doc_id)

        # ── Step 2: Sliding-window word chunking ──────────────────────────────
        chunk_words = cfg["index"].get("chunk_words", cfg["index"].get("chunk_tokens", 128))
        overlap = cfg["index"].get("overlap", 16)
        print(f"\n[2/4] Chunking (words={chunk_words}, overlap={overlap}) ...", flush=True)
        chunks = split(page_chunks, cfg)
        total_words = sum(len(c.text.split()) for c in chunks)
        print(f"      → {len(chunks)} chunks  ({total_words:,} total words)", flush=True)

        # ── Step 3: Embed ──────────────────────────────────────────────────────
        print(
            f"\n[3/4] Embedding with {cfg['embed']['model']} "
            f"(batch={cfg['embed'].get('batch_size', 32)}) ...",
            flush=True,
        )
        from doc_agent.index.embed import encode

        vectors = encode(chunks, cfg)
        print(f"      → Embedding matrix: {vectors.shape}", flush=True)

        # ── Step 4: Persist FAISS FlatIP index ──────────────────────────────────
        print("\n[4/4] Persisting FAISS FlatIP index ...", flush=True)
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
    print(f"   {len(chunks)} chunks · dim {cfg['embed']['dim']} · {index_path}/index.faiss")


if __name__ == "__main__":
    main()
