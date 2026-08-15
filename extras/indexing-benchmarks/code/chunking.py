from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .corpus import CanonicalPage


@dataclass
class BenchmarkChunk:
    chunk_id: str
    doc_id: str
    page_id: str
    text: str
    word_count: int
    strategy: str
    parent_id: str | None = None
    parent_text: str | None = None
    section_title: str | None = None


def fixed_window_word_chunking(
    pages: list[CanonicalPage], chunk_size: int = 256, overlap: int = 32
) -> list[BenchmarkChunk]:
    """Generates chunks using fixed whitespace word windowing."""
    chunks: list[BenchmarkChunk] = []
    step = max(1, chunk_size - overlap)
    for page in pages:
        words = page.text.split()
        if not words:
            continue
        for i in range(0, len(words), step):
            win = words[i : i + chunk_size]
            chunks.append(
                BenchmarkChunk(
                    chunk_id=f"{page.doc_id}_{page.page_id}_c{len(chunks):04d}",
                    doc_id=page.doc_id,
                    page_id=page.page_id,
                    text=" ".join(win),
                    word_count=len(win),
                    strategy=f"fixed_{chunk_size}_{overlap}",
                )
            )
            if i + chunk_size >= len(words):
                break
    return chunks


def hierarchical_parent_child_chunking(
    pages: list[CanonicalPage],
    parent_size: int = 512,
    child_size: int = 128,
    child_overlap: int = 16,
) -> list[BenchmarkChunk]:
    """Indexes small child word chunks while associating large parent word context."""
    chunks: list[BenchmarkChunk] = []
    child_step = max(1, child_size - child_overlap)
    for page in pages:
        words = page.text.split()
        if not words:
            continue
        for p_idx, i in enumerate(range(0, len(words), parent_size)):
            p_win = words[i : i + parent_size]
            p_id_str = f"{page.doc_id}_{page.page_id}_p{p_idx:03d}"
            p_text = " ".join(p_win)
            for j in range(0, len(p_win), child_step):
                c_win = p_win[j : j + child_size]
                chunks.append(
                    BenchmarkChunk(
                        chunk_id=f"{page.doc_id}_{page.page_id}_c{len(chunks):04d}",
                        doc_id=page.doc_id,
                        page_id=page.page_id,
                        text=" ".join(c_win),
                        word_count=len(c_win),
                        strategy="parent_child_128_512",
                        parent_id=p_id_str,
                        parent_text=p_text,
                    )
                )
                if j + child_size >= len(p_win):
                    break
            if i + parent_size >= len(words):
                break
    return chunks


def paragraph_header_aware_chunking(
    pages: list[CanonicalPage], target_chunk_size: int = 300
) -> list[BenchmarkChunk]:
    """Splits strictly along paragraph breaks and section headers, respecting word limits."""
    chunks: list[BenchmarkChunk] = []
    header_re = re.compile(r"^[A-Z0-9\s,\.\-\(\)]{4,60}$")
    for page in pages:
        paras = [p.strip() for p in page.text.split("\n\n") if p.strip()]
        cur_sec = f"Page {page.page_id}"
        cur_words: list[str] = []
        for para in paras:
            lines = para.split("\n")
            if len(lines) == 1 and header_re.match(lines[0].strip()):
                if cur_words:
                    chunks.append(
                        BenchmarkChunk(
                            chunk_id=f"{page.doc_id}_{page.page_id}_c{len(chunks):04d}",
                            doc_id=page.doc_id,
                            page_id=page.page_id,
                            text=" ".join(cur_words),
                            word_count=len(cur_words),
                            strategy="paragraph_header_aware",
                            section_title=cur_sec,
                        )
                    )
                    cur_words = []
                cur_sec = lines[0].strip()
                continue
            p_words = para.split()
            if len(cur_words) + len(p_words) > target_chunk_size and cur_words:
                chunks.append(
                    BenchmarkChunk(
                        chunk_id=f"{page.doc_id}_{page.page_id}_c{len(chunks):04d}",
                        doc_id=page.doc_id,
                        page_id=page.page_id,
                        text=" ".join(cur_words),
                        word_count=len(cur_words),
                        strategy="paragraph_header_aware",
                        section_title=cur_sec,
                    )
                )
                cur_words = []
            cur_words.extend(p_words)
        if cur_words:
            chunks.append(
                BenchmarkChunk(
                    chunk_id=f"{page.doc_id}_{page.page_id}_c{len(chunks):04d}",
                    doc_id=page.doc_id,
                    page_id=page.page_id,
                    text=" ".join(cur_words),
                    word_count=len(cur_words),
                    strategy="paragraph_header_aware",
                    section_title=cur_sec,
                )
            )
    return chunks


def build_chunk_suites(pages: list[CanonicalPage]) -> dict[str, list[BenchmarkChunk]]:
    """Builds all 5 candidate chunking suites."""
    return {
        "fixed_128_16": fixed_window_word_chunking(pages, 128, 16),
        "fixed_256_32": fixed_window_word_chunking(pages, 256, 32),
        "fixed_512_64": fixed_window_word_chunking(pages, 512, 64),
        "paragraph_header_aware": paragraph_header_aware_chunking(pages, 300),
        "parent_child_128_512": hierarchical_parent_child_chunking(pages, 512, 128, 16),
    }
