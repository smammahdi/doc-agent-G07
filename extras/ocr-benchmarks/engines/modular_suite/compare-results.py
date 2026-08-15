#!/usr/bin/env python3
"""Compare saved OCR page text with one reusable, explicit scorer.

The script deliberately scores *saved text*, not model-specific temporary
variables. It accepts ordinary page JSONL files (``page_id`` + ``text``),
the Chandra block JSONL (``book_page`` + ``content``), and the raw Qwen
chunk-text export (``--qwen-raw NAME=TXT``). Chandra image-like blocks are
excluded before the remaining block text is joined in source order. Qwen
chunks are grouped by page and their repeated overlap is removed before
scoring.

Example::

    python extras/ocr-benchmarks/engines/modular_suite/compare-results.py \
      --labels grading_kit/labels.jsonl \
      --engine "Chandra=extras/ocr-benchmarks/outputs/full-book/chandra/chunks.jsonl" \
      --engine "MinerU 2605=extras/ocr-benchmarks/outputs/full-book/mineru/full-page/pages.jsonl" \
      --engine "Tesseract layout=extras/ocr-benchmarks/outputs/heldout/tesseract/layout_results.jsonl" \
      --engine "TrOCR layout=extras/ocr-benchmarks/outputs/heldout/trocr/ppdoclayout-v3/pages.jsonl" \
      --engine "Qwen3.5=extras/ocr-benchmarks/outputs/heldout/qwen/pages.jsonl" \
      --json /tmp/ocr-comparison.json \
      --markdown /tmp/ocr-comparison.md

Primary metrics use the same transparent normalization for every engine:
HTML unescape, HTML-tag removal, Unicode NFKC, case-folding, replacement of
non-alphanumeric characters with spaces, and whitespace collapse.  CER/WER
are edit-distance rates; Word-F1 is multiset token F1.  Both macro (mean of
page scores) and micro (pooled error counts) rates are reported.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

try:
    from rapidfuzz.distance import Levenshtein as _RapidLevenshtein
except ImportError:  # pragma: no cover - portable fallback for minimal runtimes
    _RapidLevenshtein = None

NORMALIZATION = (
    "HTML-unescape, strip-tags, Unicode NFKC, casefold, "
    "letters/numbers only, collapsed whitespace"
)
FIGURE_LABELS = {"image", "figure", "diagram"}
QWEN_RAW_HEADER = re.compile(r"^\[pcma_(p\d{4})_c(\d+)\]\s+\(([^)]*)\)\s+.*$", re.MULTILINE)


def normalize_text(text: str) -> str:
    """Return the canonical comparison form used for every OCR engine."""

    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(character if character.isalnum() else " " for character in text)
    return " ".join(text.split())


def levenshtein(left: Sequence[Any], right: Sequence[Any]) -> int:
    """Compute Levenshtein distance with an optional fast accelerator."""

    if _RapidLevenshtein is not None:
        return int(_RapidLevenshtein.distance(left, right))
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index, value in enumerate(left, 1):
        current = [index]
        for other_index, other in enumerate(right, 1):
            current.append(
                min(
                    previous[other_index] + 1,
                    current[-1] + 1,
                    previous[other_index - 1] + (value != other),
                )
            )
        previous = current
    return previous[-1]


def word_f1(hypothesis: str, reference: str) -> float:
    """Return multiset token F1 after canonical normalization."""

    hypothesis_words = Counter(normalize_text(hypothesis).split())
    reference_words = Counter(normalize_text(reference).split())
    overlap = sum((hypothesis_words & reference_words).values())
    if not hypothesis_words and not reference_words:
        return 1.0
    if not overlap:
        return 0.0
    precision = overlap / sum(hypothesis_words.values())
    recall = overlap / sum(reference_words.values())
    return 2 * precision * recall / (precision + recall)


def score_text(hypothesis: str, reference: str) -> dict[str, float | int]:
    """Score one saved page transcript against one reference transcript."""

    normalized_hypothesis = normalize_text(hypothesis)
    normalized_reference = normalize_text(reference)
    hypothesis_words = normalized_hypothesis.split()
    reference_words = normalized_reference.split()
    return {
        "cer": levenshtein(normalized_hypothesis, normalized_reference)
        / max(1, len(normalized_reference)),
        "wer": levenshtein(hypothesis_words, reference_words) / max(1, len(reference_words)),
        "word_f1": word_f1(hypothesis, reference),
        "reference_chars": len(normalized_reference),
        "reference_words": len(reference_words),
        "hypothesis_chars": len(normalized_hypothesis),
        "hypothesis_words": len(hypothesis_words),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL rows and fail with the source path on bad JSON."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected an object in {path}:{line_number}")
            rows.append(row)
    return rows


def read_qwen_raw_chunks(path: Path) -> list[tuple[str, int, str, str]]:
    """Read the raw Qwen readable export as page/chunk records.

    The export is a plain-text stream whose headers contain the stable page and
    chunk IDs.  Content is retained verbatim apart from surrounding whitespace;
    the caller performs the page grouping and overlap repair.
    """

    records: list[tuple[str, int, str, str]] = []
    raw_text = path.read_text(encoding="utf-8")
    matches = list(QWEN_RAW_HEADER.finditer(raw_text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else None
        content = raw_text[match.end() : end].strip()
        if content:
            records.append((match.group(1), int(match.group(2)), match.group(3), content))
    if not records:
        raise ValueError(f"no Qwen raw chunk headers found in {path}")
    return records


def merge_qwen_chunks(chunks: Sequence[str]) -> str:
    """Join ordered Qwen chunks while removing repeated chunk overlap.

    Qwen's readable export is made from overlapping retrieval chunks rather than
    page records.  A later chunk may begin inside the final part of the previous
    chunk, so the longest repeated prefix found near the previous tail is replaced
    instead of appended a second time.  Chunks without a reliable overlap are
    separated by one newline and preserved.
    """

    merged = ""
    for chunk in chunks:
        current = " ".join(chunk.split())
        if not current:
            continue
        if not merged:
            merged = current
            continue
        overlap: tuple[int, int] | None = None
        for length in range(min(512, len(current)), 31, -1):
            prefix = current[:length]
            position = merged.rfind(prefix, max(0, len(merged) - 1024))
            if position >= max(0, len(merged) - 512):
                overlap = position, length
                break
        if overlap is None:
            merged = f"{merged}\n{current}"
        else:
            position, _ = overlap
            merged = merged[:position] + current
    return merged


def load_qwen_raw_texts(path: Path) -> dict[str, str]:
    """Load page transcripts from Qwen's raw chunk-text export."""

    grouped: defaultdict[str, list[tuple[int, str]]] = defaultdict(list)
    for page_id, chunk_id, modality, content in read_qwen_raw_chunks(path):
        if modality.casefold() == "text":
            grouped[page_id].append((chunk_id, content))
    page_texts = {
        page_id: merge_qwen_chunks([content for _, content in sorted(chunks)])
        for page_id, chunks in grouped.items()
    }
    if not page_texts:
        raise ValueError(f"no text-mode Qwen chunks found in {path}")
    return page_texts


def read_labels(path: Path) -> tuple[list[str], dict[str, str]]:
    """Read ordered page labels and reject duplicate page IDs."""

    rows = read_jsonl(path)
    page_ids: list[str] = []
    labels: dict[str, str] = {}
    for row in rows:
        page_id = row.get("page_id")
        text = row.get("text")
        if not isinstance(page_id, str) or not isinstance(text, str):
            raise ValueError(f"labels require string page_id/text rows: {path}")
        if page_id in labels:
            raise ValueError(f"duplicate label page_id {page_id} in {path}")
        page_ids.append(page_id)
        labels[page_id] = text
    if not page_ids:
        raise ValueError(f"no labels found in {path}")
    return page_ids, labels


def page_id_from_row(row: dict[str, Any]) -> str | None:
    """Resolve the page ID used by page and Chandra block records."""

    page_id = row.get("page_id")
    if isinstance(page_id, str):
        return page_id
    book_page = row.get("book_page", row.get("page_index"))
    if isinstance(book_page, int):
        return f"p{book_page:04d}"
    return None


def load_page_texts(path: Path) -> dict[str, str]:
    """Load page text from page JSONL or Chandra block JSONL.

    Chandra rows are recognized by ``content`` plus ``book_page``/``page_index``.
    Image-like blocks are omitted because they are intentionally not OCR text.
    For repeated page IDs (regions or Chandra blocks), text is joined in file
    order; ordinary page JSONL therefore remains unchanged.
    """

    rows = read_jsonl(path)
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        page_id = page_id_from_row(row)
        if page_id is None:
            continue
        if "content" in row and ("book_page" in row or "page_index" in row):
            label = str(row.get("label", "")).casefold()
            if label in FIGURE_LABELS:
                continue
            text = row.get("content", "")
        else:
            text = row.get("text", "")
        if isinstance(text, str) and text:
            grouped[page_id].append(text)
    if not grouped:
        raise ValueError(f"no page text rows found in {path}")
    return {page_id: "\n".join(parts) for page_id, parts in grouped.items()}


def score_engine(
    name: str,
    source: Path,
    page_ids: Iterable[str],
    labels: dict[str, str],
    loader: Any = load_page_texts,
) -> dict[str, Any]:
    """Score one source and return reproducible per-page plus summary metrics."""

    page_texts = loader(source)
    ordered_page_ids = list(page_ids)
    missing = [page_id for page_id in ordered_page_ids if page_id not in page_texts]
    if missing:
        raise ValueError(f"{name} is missing {len(missing)} labelled pages: {missing}")

    per_page: list[dict[str, Any]] = []
    total_char_errors = total_word_errors = total_chars = total_words = 0
    for page_id in ordered_page_ids:
        result = score_text(page_texts[page_id], labels[page_id])
        per_page.append({"page_id": page_id, **result})
        total_char_errors += float(result["cer"]) * int(result["reference_chars"])
        total_word_errors += float(result["wer"]) * int(result["reference_words"])
        total_chars += int(result["reference_chars"])
        total_words += int(result["reference_words"])

    return {
        "engine": name,
        "source": str(source),
        "pages": len(per_page),
        "normalization": NORMALIZATION,
        "micro_cer": total_char_errors / max(1, total_chars),
        "micro_wer": total_word_errors / max(1, total_words),
        "macro_cer": sum(float(row["cer"]) for row in per_page) / len(per_page),
        "macro_wer": sum(float(row["wer"]) for row in per_page) / len(per_page),
        "macro_word_f1": sum(float(row["word_f1"]) for row in per_page) / len(per_page),
        "per_page": per_page,
    }


def parse_engine_spec(spec: str) -> tuple[str, Path]:
    """Parse the CLI form ``NAME=JSONL_PATH``."""

    if "=" not in spec:
        raise argparse.ArgumentTypeError("engine must use NAME=JSONL_PATH")
    name, source = spec.split("=", 1)
    if not name.strip() or not source.strip():
        raise argparse.ArgumentTypeError("engine name and path must be non-empty")
    path = Path(source).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"engine source does not exist: {path}")
    return name.strip(), path


def parse_qwen_raw_spec(spec: str) -> tuple[str, Path]:
    """Parse the CLI form ``NAME=QWEN_RAW_TEXT_PATH``."""

    return parse_engine_spec(spec)


def markdown_report(results: list[dict[str, Any]], excluded_pages: Iterable[str] = ()) -> str:
    """Render a compact summary table suitable for a report or review."""

    lines = [
        "# OCR comparison",
        "",
        f"Normalization: {NORMALIZATION}.",
        "CER/WER are lower-is-better; Word-F1 is higher-is-better.",
        "",
        "| Engine | Pages | Macro CER | Macro WER | Macro Word-F1 | Micro CER | Micro WER |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in sorted(results, key=lambda item: float(item["macro_cer"])):
        lines.append(
            f"| {result['engine']} | {result['pages']} | "
            f"{float(result['macro_cer']):.4f} | {float(result['macro_wer']):.4f} | "
            f"{float(result['macro_word_f1']):.4f} | {float(result['micro_cer']):.4f} | "
            f"{float(result['micro_wer']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Scores are computed from the saved JSONL text. They are not claims of",
            "human OCR accuracy unless the reference labels are manually verified.",
            "Ranking is by macro CER ascending; no composite average of CER, WER,",
            "and Word-F1 is used because CER/WER are lower-is-better and Word-F1",
            "is higher-is-better.",
        ]
    )
    excluded = sorted(excluded_pages)
    if excluded:
        lines.insert(4, f"Excluded pages: {', '.join(excluded)}.")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="ordered reference labels JSONL")
    parser.add_argument(
        "--engine",
        action="append",
        required=True,
        metavar="NAME=JSONL",
        help="saved page/block JSONL; repeat for every engine or mode",
    )
    parser.add_argument(
        "--qwen-raw",
        action="append",
        default=[],
        metavar="NAME=TXT",
        help="raw Qwen readable chunk export; repeat for Qwen variants",
    )
    parser.add_argument(
        "--exclude-page",
        action="append",
        default=[],
        metavar="PAGE_ID",
        help="exclude a labelled page from scoring; repeat as needed",
    )
    parser.add_argument("--json", type=Path, help="optional JSON report destination")
    parser.add_argument("--markdown", type=Path, help="optional Markdown report destination")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    page_ids, labels = read_labels(args.labels)
    excluded_pages = set(args.exclude_page)
    unknown_pages = sorted(excluded_pages - set(page_ids))
    if unknown_pages:
        raise ValueError(f"cannot exclude unlabelled pages: {unknown_pages}")
    scored_page_ids = [page_id for page_id in page_ids if page_id not in excluded_pages]
    if not scored_page_ids:
        raise ValueError("page exclusion removed every labelled page")
    specs = [(name, path, load_page_texts) for name, path in map(parse_engine_spec, args.engine)]
    specs.extend(
        (name, path, load_qwen_raw_texts) for name, path in map(parse_qwen_raw_spec, args.qwen_raw)
    )
    names = [name for name, _, _ in specs]
    if len(names) != len(set(names)):
        raise ValueError("engine names must be unique")
    results = [
        score_engine(name, path, scored_page_ids, labels, loader) for name, path, loader in specs
    ]
    payload = {
        "labels": str(args.labels),
        "pages": scored_page_ids,
        "excluded_pages": sorted(excluded_pages),
        "normalization": NORMALIZATION,
        "ranking": {
            "metric": "macro_cer",
            "direction": "ascending",
            "composite_average_used": False,
            "macro_definition": "mean of per-page scores",
        },
        "engines": results,
    }
    report = markdown_report(results, excluded_pages)
    print(report, end="")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
