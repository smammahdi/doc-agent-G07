# ruff: noqa: N806
"""Recover `Fig. N` labels from the Document OCR word stream.

The book's figure numbers are the one thing every approach so far has been short on:
the PDF's own text layer yields 156 distinct numbers over a 1..289 series, the best single
detector 142, and the fused consensus 188. Those all read a cropped label strip. This reads
the whole page at 0.98 mean confidence instead, so it is a genuinely different measurement.

A label is two adjacent tokens -- `Fig.` then `45.` -- so tokens are re-assembled in reading
order (top-to-bottom, left-to-right within a line band) before matching.
"""

import json
import re
from collections import defaultdict

# Document OCR tokenizes punctuation separately: `Fig` `.` `1` `.` is four tokens, so the
# number is not the adjacent one. It also sometimes glues them: `Fig.2`.
FIG = re.compile(r"^f[il1]gs?[.,]?$", re.I)  # Fig / Figs / Flg / F1g, with OCR noise
GLUED = re.compile(r"^f[il1]gs?[.,]\s*(\d{1,3})[.,;:]?$", re.I)
NUM = re.compile(r"^(\d{1,3})[.,;:]?$")
PUNCT = re.compile(r"^[.,;:\-—–]+$")
SERIES_MAX = 289


def load_words(path):
    by_page = defaultdict(list)
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_page[r["page_id"]].append(r)
    return by_page


def reading_order(words, line_tol=0.006):
    """Sort into lines by y, then by x within a line. Tolerance is in normalized units."""
    ws = sorted(words, key=lambda r: (r["bbox_norm"][1], r["bbox_norm"][0]))
    lines, cur, y = [], [], None
    for w in ws:
        wy = w["bbox_norm"][1]
        if y is None or abs(wy - y) <= line_tol:
            cur.append(w)
            y = wy if y is None else y
        else:
            lines.append(sorted(cur, key=lambda r: r["bbox_norm"][0]))
            cur, y = [w], wy
    if cur:
        lines.append(sorted(cur, key=lambda r: r["bbox_norm"][0]))
    return [w for line in lines for w in line]


def labels_on_page(words):
    """-> [(number, bbox_norm of the 'Fig. N' pair)] for each label found on the page."""
    seq = reading_order(words)
    out = []

    def emit(n, a, b):
        if 1 <= n <= SERIES_MAX:
            out.append((n, [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]))

    for i, w in enumerate(seq):
        g = GLUED.match(w["text"])
        if g:
            emit(int(g.group(1)), w["bbox_norm"], w["bbox_norm"])
            continue
        if not FIG.match(w["text"]):
            continue
        # look ahead past separator tokens for the number
        for nxt in seq[i + 1 : i + 4]:
            if PUNCT.match(nxt["text"]):
                continue
            m = NUM.match(nxt["text"])
            if m:
                emit(int(m.group(1)), w["bbox_norm"], nxt["bbox_norm"])
            break
    return out


def report(words_path, out_path=None):
    by_page = load_words(words_path)
    found = {}
    for pid, ws in by_page.items():
        for n, box in labels_on_page(ws):
            found.setdefault(n, []).append((pid, box))
    nums = sorted(found)
    missing = [k for k in range(1, SERIES_MAX + 1) if k not in found]
    dup = {n: sorted({p for p, _ in v}) for n, v in found.items() if len({p for p, _ in v}) > 1}

    L = [
        "# `Fig. N` recovery from Document OCR",
        "",
        f"{len(nums)} distinct numbers of 1..{SERIES_MAX}, over {len(by_page)} pages "
        f"with text.",
        "",
        "| source | distinct Fig. numbers |",
        "|---|---|",
        "| embedded PDF text layer | 156 |",
        "| best single detector (doclayout_yolo) | 142 |",
        "| fused consensus of 3 detectors | 188 |",
        f"| **Document OCR word stream** | **{len(nums)}** |",
        "",
        f"Unrecovered: {len(missing)} numbers.",
        f"Numbers appearing on more than one page (cross-references, or a misread): "
        f"{len(dup)}.",
        "",
    ]
    txt = "\n".join(L) + "\n"
    if out_path:
        open(out_path, "w").write(txt)
    print(txt)
    return {"distinct": len(nums), "numbers": nums, "missing": missing, "multi_page": dup}
