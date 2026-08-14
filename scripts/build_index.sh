#!/usr/bin/env bash
# A2 — Build the Stage 4 vector knowledge base from chandra/pages.md.
# Usage: bash scripts/build_index.sh
# Output: data/processed/index/{index.faiss, chunks.jsonl, metadata.json, image_index.json}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo ">>> Building Stage 4 index for Pierce 1890 Medical Adviser (doc_id: pierce-1890)"
echo ">>> Source: chandra/pages.md   Output: data/processed/index/"
echo ""

# uv run resolves the project's pyproject.toml environment automatically
uv run python scripts/run_index.py

echo ""
echo ">>> Done. Artifacts written to data/processed/index/"
