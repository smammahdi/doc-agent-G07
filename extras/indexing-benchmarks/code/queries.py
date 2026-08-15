from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RetrievalQuery:
    query_id: str
    split: str  # "dev" | "test" | "out_of_corpus"
    type: str   # "single_page" | "multi_page" | "out_of_corpus"
    region: str
    category: str
    question: str
    page_ids: list[str]
    exact_answer_span: str
    span_start: int
    span_end: int
    manually_verified: bool


def load_retrieval_queries(search_root: Path | None = None) -> list[RetrievalQuery]:
    """Loads verified retrieval queries from dataset files."""
    roots = [search_root] if search_root else []
    roots.extend([
        Path("extras/indexing-benchmarks/data"),
        Path("extras/indexing-benchmarks"),
        Path("/kaggle/input"),
        Path("grading_kit"),
    ])

    for r in roots:
        if not r or not r.exists():
            continue
        candidates = [
            r / "retrieval-queries.jsonl",
            r / "retrieval_queries.jsonl",
            r / "tasks.jsonl",
            *r.rglob("retrieval*queries.jsonl"),
        ]
        for p in candidates:
            if p.is_file():
                queries = []
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip() and not line.startswith("#"):
                        item = json.loads(line)
                        pids = item.get("page_ids") or item.get("target_pages") or item.get("gold_pages") or []
                        queries.append(
                            RetrievalQuery(
                                query_id=item.get("query_id") or item.get("id"),
                                split=item.get("split", "dev"),
                                type=item.get("type", "single_page" if len(pids) == 1 else ("multi_page" if len(pids) > 1 else "out_of_corpus")),
                                region=item.get("region", "general"),
                                category=item.get("category", "General"),
                                question=item["question"],
                                page_ids=pids,
                                exact_answer_span=item.get("exact_answer_span") or item.get("gold", ""),
                                span_start=item.get("span_start", 0),
                                span_end=item.get("span_end", 0),
                                manually_verified=item.get("manually_verified", False),
                            )
                        )
                if len(queries) >= 50:
                    return queries

    raise FileNotFoundError("Could not find retrieval-queries.jsonl")
