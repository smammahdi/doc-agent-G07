#!/usr/bin/env bash
# A2 — render, read, embed, and persist the configured knowledge base once.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python scripts/run_index.py
