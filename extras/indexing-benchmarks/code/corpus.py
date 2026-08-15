from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CanonicalPage:
    doc_id: str
    page_id: str
    page_num: int
    text: str
    word_count: int
    char_count: int
    ocr_source: str


def load_canonical_corpus(search_root: Path | None = None) -> list[CanonicalPage]:
    """Loads the full 1,034-page book with explicit missing pages policy."""
    roots = [search_root] if search_root else []
    roots.extend([
        Path("extras/indexing-benchmarks/data"),
        Path("extras/indexing-benchmarks"),
        Path("/kaggle/input"),
        Path("extras/ocr-benchmarks/extractions/chandra (full book)"),
        Path("extras/ocr-benchmarks"),
    ])

    # 1. Search for pre-generated canonical-pages.jsonl or canonical_pages.jsonl
    for r in roots:
        if not r or not r.exists():
            continue
        candidates = [r / "canonical-pages.jsonl", r / "canonical_pages.jsonl", *r.rglob("canonical*pages.jsonl")]
        for p in candidates:
            if p.is_file():
                pages = []
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip() and not line.startswith("#"):
                        item = json.loads(line)
                        pages.append(
                            CanonicalPage(
                                doc_id=item["doc_id"],
                                page_id=item["page_id"],
                                page_num=item["page_num"],
                                text=item["text"],
                                word_count=item["word_count"],
                                char_count=item["char_count"],
                                ocr_source=item["ocr_source"],
                            )
                        )
                if len(pages) == 1034:
                    return pages

    # 2. Reconstruct from Chandra chunks.jsonl if canonical file not found
    chandra_file = None
    for r in roots:
        if not r or not r.exists():
            continue
        candidates = [r / "chunks.jsonl", *r.rglob("chunks.jsonl")]
        for p in candidates:
            if p.is_file():
                chandra_file = p
                break
        if chandra_file:
            break

    if chandra_file is None:
        raise FileNotFoundError("Could not locate canonical-pages.jsonl or chunks.jsonl")

    page_blocks: dict[int, list[str]] = {}
    with open(chandra_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            bp = item.get("book_page")
            lbl = item.get("label", "")
            cnt = item.get("content", "")
            if lbl in ["Image", "Figure", "Diagram"]:
                continue
            clean_text = re.sub(r"<[^>]+>", " ", cnt)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
            if clean_text:
                page_blocks.setdefault(bp, []).append(clean_text)

    missing_pages = {2, 3, 4, 6, 1031, 1033}
    pages = []
    for p_num in range(1, 1035):
        pid = f"p{p_num:04d}"
        if p_num in missing_pages:
            pages.append(
                CanonicalPage(
                    doc_id="pierce-1890",
                    page_id=pid,
                    page_num=p_num,
                    text="",
                    word_count=0,
                    char_count=0,
                    ocr_source="ocr_missing_blank_flyleaf",
                )
            )
            continue
        blocks = page_blocks.get(p_num, [])
        page_text = "\n\n".join(blocks).strip()
        w_count = len(page_text.split()) if page_text else 0
        pages.append(
            CanonicalPage(
                doc_id="pierce-1890",
                page_id=pid,
                page_num=p_num,
                text=page_text,
                word_count=w_count,
                char_count=len(page_text),
                ocr_source="chandra" if w_count > 0 else "ocr_empty_illustration_only",
            )
        )
    return pages
