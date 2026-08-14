#!/usr/bin/env python3
"""
PaddleOCR-VL-1.6 Benchmark: Layout-Aware (PP-DocLayoutV3) & Direct Full-Page
Pierce 1890 Medical Adviser · Team G07 · A2 SOTA Vision-Language Document Parser

Uses the OFFICIAL PaddleOCR pipeline (paddleocr[doc-parser] >= 3.6.0) as recommended
by the model card. NOT the Transformers path (element-level only, slower).

  1. Mode 1 (full-page):     Direct full-page document parsing via PaddleOCRVL
  2. Mode 2 (ppdoclayout):   PP-DocLayoutV3 non-figure crop → PaddleOCRVL per-region OCR
  3. Evaluation:             CER, WER, Word F1 against grading_kit/labels.jsonl
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ["PYTHONUNBUFFERED"] = "1"

# ── Paths ────────────────────────────────────────────────────────────────────
REPO = Path("/kaggle/working/doc-agent-G07")
OUT  = Path("/kaggle/working/paddleocr-vl-benchmark")

LAYOUT_PATH = REPO / "extras/output/ppdoclayout-v3/detections.jsonl"
HELDOUT     = REPO / "grading_kit/heldout_pages"
LABELS      = REPO / "grading_kit/labels.jsonl"
PAGES       = [f"p{n:04d}" for n in range(24, 48)]

# ── Helpers ───────────────────────────────────────────────────────────────────
def ensure_repository() -> dict[str, str]:
    if not LABELS.is_file():
        print("Cloning doc-agent-G07 repository...", flush=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", "main",
             "https://github.com/smammahdi/doc-agent-G07.git", str(REPO)],
            check=True,
        )
    if not LABELS.is_file() or not LAYOUT_PATH.is_file():
        raise FileNotFoundError("Repository missing labels or PP-DocLayoutV3 detections")
    labels = {
        row["page_id"]: row["text"]
        for line in LABELS.read_text("utf-8").splitlines() if line.strip()
        for row in [json.loads(line)]
    }
    if list(labels) != PAGES:
        raise ValueError("labels must contain exactly p0024 through p0047 in order")
    missing = [p for p in PAGES if not (HELDOUT / f"{p}.jpg").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing held-out pages: {missing}")
    return labels


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    os.replace(tmp, path)


def load_regions() -> dict[str, list[dict[str, Any]]]:
    regions: dict[str, list[dict[str, Any]]] = {p: [] for p in PAGES}
    for i, row in enumerate(read_jsonl(LAYOUT_PATH), 1):
        pid = row.get("page_id")
        if pid not in regions or bool(row.get("is_figure")):
            continue
        box = row.get("bbox_norm", [])
        if len(box) != 4:
            continue
        vals = [max(0.0, min(1.0, float(v))) for v in box]
        if vals[2] <= vals[0] or vals[3] <= vals[1]:
            continue
        regions[pid].append({
            "source_id": str(row.get("detection_id", f"line-{i}")),
            "class_name": str(row.get("class_name", "text")),
            "score": float(row.get("score", 0.0)),
            "bbox_norm": vals,
        })
    for pid in PAGES:
        regions[pid].sort(key=lambda r: (r["bbox_norm"][1], r["bbox_norm"][0]))
    return regions


def bbox_pixels(box: list[float], w: int, h: int) -> list[int]:
    l = max(0, min(w - 1, int(box[0] * w)))
    t = max(0, min(h - 1, int(box[1] * h)))
    r = max(l + 1, min(w, int(box[2] * w + 0.999999)))
    b = max(t + 1, min(h, int(box[3] * h + 0.999999)))
    return [l, t, r, b]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(a: list | str, b: list | str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, av in enumerate(a, 1):
        cur = [i]
        for j, bv in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (av != bv)))
        prev = cur
    return prev[-1]


def word_f1(hyp: str, ref: str) -> float:
    h = Counter(normalize(hyp).split())
    r = Counter(normalize(ref).split())
    tp = sum((h & r).values())
    if not h and not r:
        return 1.0
    if not tp:
        return 0.0
    prec = tp / sum(h.values())
    rec  = tp / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def extract_text_from_result(result: Any) -> str:
    """Extract plain text from a PaddleOCRVL pipeline result object."""
    # Try .json() method (returns dict with markdown/text fields)
    try:
        data = result.json() if callable(getattr(result, "json", None)) else result
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, dict):
            # Try markdown output first (preserves structure)
            md = data.get("res", {}).get("markdown", "")
            if md:
                return md.strip()
            # Fall back to rec_texts (like classic PaddleOCR)
            texts = data.get("res", {}).get("rec_texts", [])
            if texts:
                return "\n".join(str(t) for t in texts if str(t).strip())
    except Exception:
        pass
    return str(result)


def score(mode: str, labels: dict[str, str], page_rows: dict[str, dict]) -> dict:
    per_page = []
    total_ce = total_we = total_chars = total_words = 0
    for pid in PAGES:
        ref = normalize(labels[pid])
        hyp = normalize(str(page_rows.get(pid, {}).get("text", "")))
        ce = levenshtein(hyp, ref)
        we = levenshtein(hyp.split(), ref.split())
        per_page.append({
            "page_id": pid,
            "cer": ce / max(1, len(ref)),
            "wer": we / max(1, len(ref.split())),
            "word_f1": word_f1(hyp, ref),
        })
        total_ce    += ce;  total_chars += len(ref)
        total_we    += we;  total_words += len(ref.split())
    metrics = {
        "engine":        "PaddleOCR-VL-1.6 (official pipeline)",
        "mode":          mode,
        "pages":         len(per_page),
        "micro_cer":     total_ce / max(1, total_chars),
        "micro_wer":     total_we / max(1, total_words),
        "macro_word_f1": sum(r["word_f1"] for r in per_page) / len(per_page),
        "per_page":      per_page,
    }
    (OUT / mode).mkdir(parents=True, exist_ok=True)
    (OUT / mode / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    return metrics


def run_mode(
    pipeline,
    mode: str,
    labels: dict[str, str],
    layout_regions: dict[str, list[dict]],
) -> dict:
    from PIL import Image

    mode_dir   = OUT / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    pages_path = mode_dir / "pages.jsonl"

    # Resume: skip already-completed pages
    done_rows  = {r["page_id"]: r for r in read_jsonl(pages_path) if r.get("status") == "complete"}
    page_rows  = {}
    all_pages  = []

    for pid in PAGES:
        if pid in done_rows:
            page_rows[pid] = done_rows[pid]
            all_pages.append(done_rows[pid])
            print(f"  [SKIP]  {pid}  (already complete)", flush=True)
            continue

        t0  = time.perf_counter()
        img = Image.open(HELDOUT / f"{pid}.jpg").convert("RGB")

        if mode == "full-page":
            # Feed the whole page directly to PaddleOCRVL
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                img.save(tf.name, "JPEG")
                tmp_path = tf.name
            try:
                texts = []
                for res in pipeline.predict(tmp_path):
                    texts.append(extract_text_from_result(res))
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            page_text = "\n\n".join(t for t in texts if t).strip()
        else:
            # Crop each non-figure PP-DocLayoutV3 region → OCR each crop
            sources = layout_regions.get(pid, [])
            region_texts = []
            with tempfile.TemporaryDirectory(prefix=f"pvl-{pid}-") as scratch:
                scratch = Path(scratch)
                for ri, src in enumerate(sources):
                    px = bbox_pixels(src["bbox_norm"], img.width, img.height)
                    crop_path = scratch / f"{pid}-r{ri:04d}.jpg"
                    img.crop(px).save(crop_path, "JPEG")
                    try:
                        crop_texts = []
                        for res in pipeline.predict(str(crop_path)):
                            crop_texts.append(extract_text_from_result(res))
                        region_text = "\n".join(t for t in crop_texts if t).strip()
                    except Exception as e:
                        region_text = ""
                        print(f"    [WARN] {pid} region {ri} failed: {e}", flush=True)
                    if region_text:
                        region_texts.append(region_text)
                    print(f"    [{pid}] region {ri+1}/{len(sources)} -> {repr(region_text[:40])}", flush=True)
            page_text = "\n\n".join(region_texts).strip()

        elapsed = time.perf_counter() - t0
        ref     = normalize(labels[pid])
        hyp     = normalize(page_text)
        ce      = levenshtein(hyp, ref)
        we      = levenshtein(hyp.split(), ref.split())
        f1      = word_f1(hyp, ref)

        row = {
            "page_id": pid,
            "mode":    mode,
            "status":  "complete",
            "text":    page_text,
            "cer":     ce / max(1, len(ref)),
            "wer":     we / max(1, len(ref.split())),
            "word_f1": f1,
            "elapsed_seconds": round(elapsed, 3),
        }
        page_rows[pid] = row
        all_pages.append(row)
        write_jsonl(pages_path, all_pages)
        print(f"  [{mode}] {pid}: CER={row['cer']:.4f} | WER={row['wer']:.4f} | F1={f1:.4f} | {elapsed:.1f}s", flush=True)

    return score(mode, labels, page_rows)


def main() -> None:
    import paddle
    from paddleocr import PaddleOCRVL

    print(f"PaddlePaddle version : {paddle.__version__}", flush=True)
    print(f"CUDA available       : {paddle.is_compiled_with_cuda()}", flush=True)

    labels         = ensure_repository()
    layout_regions = load_regions()

    device = "gpu:0" if paddle.is_compiled_with_cuda() else "cpu"
    print(f"Loading PaddleOCRVL pipeline (device={device}) ...", flush=True)
    pipeline = PaddleOCRVL(pipeline_version="v1.6", device=device)
    print("PaddleOCRVL pipeline loaded.", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)

    # ── Mode 1: full-page ──────────────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("RUNNING MODE 1: DIRECT FULL-PAGE PADDLEOCR-VL", flush=True)
    print("=" * 80, flush=True)
    m1 = run_mode(pipeline, "full-page", labels, layout_regions)

    # ── Mode 2: PP-DocLayoutV3 crops ──────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("RUNNING MODE 2: PP-DOCLAYOUT-V3 CROPS PADDLEOCR-VL", flush=True)
    print("=" * 80, flush=True)
    m2 = run_mode(pipeline, "ppdoclayout-v3", labels, layout_regions)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("PADDLEOCR-VL-1.6 BENCHMARK SUMMARY", flush=True)
    print("=" * 80, flush=True)
    for label, m in [("Full-Page", m1), ("PP-DocLayoutV3", m2)]:
        print(f"  {label:15s}: micro_CER={m['micro_cer']:.4f} | micro_WER={m['micro_wer']:.4f} | macro_Word_F1={m['macro_word_f1']:.4f}", flush=True)

    archive = shutil.make_archive("/kaggle/working/paddleocr-vl-benchmark", "zip", root_dir=OUT)
    print(f"\nDownload: {archive}", flush=True)


if __name__ == "__main__":
    main()
