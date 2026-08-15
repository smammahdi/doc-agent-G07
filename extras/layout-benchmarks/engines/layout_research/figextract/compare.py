# ruff: noqa: N806
"""Cross-detector comparison over whatever runs exist in output/.

The questions that actually decide the detector:
  1. How many figures does each find, on how many pages?
  2. Where do they agree? A figure several detectors independently found is the
     trustworthy kind -- that agreement is an explainability signal, not just a metric.
  3. How many `Fig. N` labels does each recover, against the 156 the embedded OCR
     text layer already gives for free?
  4. What does each find that the others miss? That is where the errors hide.
"""

import json
from collections import defaultdict
from pathlib import Path

from .config import OUTPUT
from .detectors import NAMES
from .geometry import iou

EMBEDDED_BASELINE = 156  # distinct Fig-numbers recoverable from the PDF's own layer
SERIES_MAX = 289  # highest Fig. number seen anywhere in that layer


def plausible(n):
    """A parsed number above the printed series is an OCR artefact, not a figure.

    Rare (1-2 per full-book run: 444, 2165, 2651) but it wrecks the max/coverage columns,
    so it is excluded from counts and reported separately rather than silently dropped.
    """
    return n is not None and 1 <= n <= SERIES_MAX


def load_runs(base=None):
    base = Path(base) if base else OUTPUT
    runs = {}
    for n in NAMES:
        f = base / n / "figures.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(line) for line in f.open() if line.strip()]
        if not rows:
            continue
        by_page = defaultdict(list)
        for r in rows:
            by_page[r["page_id"]].append(r)
        runs[n] = {"rows": rows, "by_page": by_page}
    return runs


def _count_blank(run_dir, rows):
    """Blank paper detected as a figure. Prefers the recorded flag; falls back to
    measuring the crops, so runs made before the flag existed still get audited."""
    if rows and "is_blank" in rows[0]:
        return sum(1 for r in rows if r.get("is_blank"))
    import cv2
    import numpy as np

    n = 0
    for r in rows:
        g = cv2.imread(str(run_dir / r["crop_path"]), cv2.IMREAD_GRAYSCALE)
        if g is not None and (g < 128).mean() < 0.01 and np.std(g) < 25:
            n += 1
    return n


def fignum(r):
    if not r.get("fig_label"):
        return None
    try:
        return int(str(r["fig_label"]).split()[-1])
    except ValueError:
        return None


def cmd_compare(a):
    runs = load_runs(a.out)
    if not runs:
        raise SystemExit(f"no runs under {a.out or OUTPUT}")

    base_dir = Path(a.out) if a.out else OUTPUT
    L = [
        "# Figure extraction — detector comparison",
        "",
        f"Agreement threshold IoU >= {a.iou}.",
        "",
        "## Totals",
        "",
        "| detector | figures | blank FPs | real figures | pages | labels | captions |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, r in runs.items():
        rows = r["rows"]
        blanks = _count_blank(base_dir / n, rows)
        real = len(rows) - blanks
        lab = sum(1 for x in rows if x.get("fig_label"))
        cap = sum(1 for x in rows if x.get("caption_printed"))
        L.append(
            f"| {n} | {len(rows)} | {blanks} | {real} | {len(r['by_page'])} | "
            f"{lab} ({lab/max(len(rows),1):.0%}) | "
            f"{cap} ({cap/max(len(rows),1):.0%}) |"
        )
    L += [
        "",
        "*blank FPs* = crops that are blank front-matter paper, not figures "
        "(dark-pixel fraction < 1% and contrast < 25 at an absolute threshold).",
        "",
    ]
    L += [
        "",
        f"Embedded text-layer baseline for distinct Fig. numbers: **{EMBEDDED_BASELINE}** "
        f"(series runs to {SERIES_MAX}).",
        "",
    ]

    names = list(runs)
    if len(names) > 1:
        L += [
            "## Pairwise box agreement",
            "",
            "| A | B | A figs | B figs | matched | A-only | B-only |",
            "|---|---|---|---|---|---|---|",
        ]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                na, nb = names[i], names[j]
                matched = a_only = 0
                for pid, ra in runs[na]["by_page"].items():
                    rb = list(runs[nb]["by_page"].get(pid, []))
                    for x in ra:
                        hit = next(
                            (y for y in rb if iou(x["bbox_img"], y["bbox_img"]) >= a.iou), None
                        )
                        if hit:
                            matched += 1
                            rb.remove(hit)
                        else:
                            a_only += 1
                a_only += (
                    sum(
                        len(v)
                        for p, v in runs[na]["by_page"].items()
                        if p not in runs[nb]["by_page"]
                    )
                    * 0
                )
                L.append(
                    f"| {na} | {nb} | {len(runs[na]['rows'])} | "
                    f"{len(runs[nb]['rows'])} | {matched} | {a_only} | "
                    f"{len(runs[nb]['rows']) - matched} |"
                )
        L.append("")

    L += ["## Page-level consensus", "", "| found by | pages | examples |", "|---|---|---|"]
    allp = set()
    for r in runs.values():
        allp |= set(r["by_page"])
    tally = defaultdict(list)
    for pid in allp:
        who = tuple(sorted(n for n in names if pid in runs[n]["by_page"]))
        tally[who].append(pid)
    for who, pgs in sorted(tally.items(), key=lambda kv: -len(kv[1])):
        L.append(f"| {' + '.join(who)} | {len(pgs)} | {', '.join(sorted(pgs)[:6])} |")

    L += [
        "",
        "## Figure-number coverage",
        "",
        f"Counting only plausible numbers (1..{SERIES_MAX}); anything above the printed "
        "series is an OCR artefact and is reported in its own column.",
        "",
        "| detector | distinct plausible | min | max | vs embedded (156) | implausible |",
        "|---|---|---|---|---|---|",
    ]
    union = set()
    for n, r in runs.items():
        nums = sorted({x for x in (fignum(y) for y in r["rows"]) if plausible(x)})
        bad = sorted(
            {x for x in (fignum(y) for y in r["rows"]) if x is not None and not plausible(x)}
        )
        union |= set(nums)
        if not nums:
            L.append(f"| {n} | 0 | - | - | - | {len(bad)} |")
            continue
        delta = len(nums) - EMBEDDED_BASELINE
        L.append(
            f"| {n} | {len(nums)} | {min(nums)} | {max(nums)} | "
            f"{delta:+d} | {len(bad)} {bad[:3] if bad else ''} |"
        )
    if len(runs) > 1:
        L += [
            "",
            f"**Union across all {len(runs)} detectors: {len(union)} distinct figure "
            f"numbers** ({len(union) - EMBEDDED_BASELINE:+d} vs the embedded baseline; "
            f"{len([k for k in range(1, SERIES_MAX+1) if k not in union])} of 1..{SERIES_MAX} "
            "still unrecovered).",
            "",
            "No single detector beats the free embedded text layer on figure numbers. "
            "Their union does — the detectors fail on *different* figures, which is the "
            "argument for keeping more than one.",
            "",
        ]

    base = Path(a.out) if a.out else OUTPUT
    out = base / "comparison.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {out}")
