# ruff: noqa: N806
"""Consensus fusion of the per-detector runs.

Each detector fails on *different* figures (see output/comparison.md: best single run
recovers 142 distinct Fig. numbers, the union of three reaches 188). Fusion turns that
into one record set, and the vote count -- how many independent detectors found the same
box -- becomes an auditable confidence tier rather than an opaque model score.

Boxes are clustered on `bbox_norm`, not `bbox_img`: page pixel sizes vary across the
book (1335..1503 px wide at 300 dpi), so normalised coords are the only scale-free key.
No re-cropping: the fused row points at the representative detector's existing crop.
"""

import json
from collections import defaultdict
from pathlib import Path

from .compare import EMBEDDED_BASELINE, SERIES_MAX, fignum, load_runs, plausible
from .config import OUTPUT
from .geometry import iou

# Order in which a cluster's fields are taken. PP-DocLayoutV3 leads because it is the
# measured precision winner (0 blank-paper false positives vs 12-13 for the others);
# orphan_ink trails because it is the zero-ML baseline.
PRECEDENCE = ["ppdoclayout_v3", "doclayout_yolo", "orphan_ink"]
TIERS = {3: "high", 2: "medium", 1: "low"}


def _rank(name):
    return PRECEDENCE.index(name) if name in PRECEDENCE else len(PRECEDENCE)


def cluster(rows, thr=0.5):
    """Group rows whose bbox_norm overlap at IoU >= thr. Union-find, single linkage.

    Single linkage on purpose: a detector that splits one engraving into two boxes should
    still land in the same cluster as the detector that boxed it whole.
    """
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if iou(rows[i]["bbox_norm"], rows[j]["bbox_norm"]) >= thr:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[find(i)].append(r)
    return list(groups.values())


def _first(members, field):
    """First non-empty value of `field` in precedence order, plus who supplied it.

    This is where the union gain lives: a figure whose label only ppdoclayout_v3 read and
    whose caption only doclayout_yolo read ends up with both.
    """
    for m in sorted(members, key=lambda r: _rank(r["detector"])):
        if m.get(field):
            return m[field], m["detector"]
    return None, None


def fuse_cluster(members, page_id, seq):
    rep = min(members, key=lambda r: (_rank(r["detector"]), -r.get("score", 0.0)))
    voters = sorted({m["detector"] for m in members})
    label, label_src = _first(members, "fig_label")
    caption, caption_src = _first(members, "caption_printed")
    # Two detectors can read *different* numbers off the same figure. Precedence picks one;
    # the rest are kept rather than dropped -- disagreement is itself an audit signal, and
    # discarding them would lose figure numbers the union of the runs did recover.
    alts = sorted(
        {m["fig_label"] for m in members if m.get("fig_label") and m["fig_label"] != label}
    )
    row = dict(rep)
    row.update(
        figure_id=f"fused_{page_id}_{seq:02d}",
        detector="fused",
        impl="consensus/" + "+".join(voters),
        votes=len(voters),
        voters=voters,
        confidence=TIERS.get(len(voters), "low"),
        source_detector=rep["detector"],
        source_figure_id=rep["figure_id"],
        # crop_path stays relative to the run dir, as in every other run -- it just points
        # out of output/fused/ into the detector's own crops/, so nothing is re-rendered.
        crop_path=f"../{rep['detector']}/{rep['crop_path']}",
        fig_label=label,
        fig_label_from=label_src,
        fig_label_alts=alts,
        caption_printed=caption,
        caption_from=caption_src,
        member_boxes={m["detector"]: m["bbox_norm"] for m in members},
    )
    return row


def build_fused(runs, thr=0.5):
    by_page = defaultdict(list)
    for _name, r in runs.items():
        for row in r["rows"]:
            by_page[row["page_id"]].append(row)
    out = []
    for page_id in sorted(by_page):
        groups = cluster(by_page[page_id], thr)
        groups.sort(key=lambda g: min(m["bbox_norm"][1] for m in g))  # top of page first
        for k, g in enumerate(groups, 1):
            out.append(fuse_cluster(g, page_id, k))
    return out


def _mark_blank(base, rows):
    """The existing runs predate the `is_blank` flag, so measure the representative crop
    with the same absolute-threshold test compare.py falls back to. Flagged, never dropped.
    """
    import cv2
    import numpy as np

    for r in rows:
        if "is_blank" in r and r["is_blank"] is not None:
            continue
        g = cv2.imread(str((base / "fused" / r["crop_path"]).resolve()), cv2.IMREAD_GRAYSCALE)
        r["is_blank"] = bool(g is not None and (g < 128).mean() < 0.01 and np.std(g) < 25)


def embedded_check(rows, pdf_path):
    """How many fused labels does the PDF's own text layer corroborate?

    Weak oracle on purpose: an embedded `Fig. N` on a page may be a body-text cross-reference
    rather than the printed caption, so this bounds the label error rate, it does not measure
    it. Still the only external signal available before gt_figures.json exists.
    """
    import re

    from .pdf import open_book

    pat = re.compile(r"Fig\.?\s*\n?\s*(\d{1,3})")
    doc = open_book(pdf_path)
    seen = {}
    agree = disagree = no_signal = 0
    for r in rows:
        labs = {
            n
            for n in (
                fignum({"fig_label": x})
                for x in [r.get("fig_label")] + list(r.get("fig_label_alts") or [])
            )
            if n is not None
        }
        if not labs:
            continue
        i = r["pdf_index"]
        if i not in seen:
            seen[i] = {int(n) for n in pat.findall(doc[i].get_text())}
        if not seen[i]:
            no_signal += 1
        elif labs & seen[i]:
            agree += 1
        else:
            disagree += 1
    return agree, disagree, no_signal


def cmd_fuse(a):
    runs = load_runs(a.out)
    if len(runs) < 2:
        raise SystemExit(f"need >=2 detector runs to fuse; found {list(runs)}")

    base = Path(a.out) if a.out else OUTPUT
    rows = build_fused(runs, a.iou)
    (base / "fused").mkdir(parents=True, exist_ok=True)
    _mark_blank(base, rows)
    kept = [r for r in rows if r["votes"] >= a.min_votes]
    d = base / "fused"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "figures.jsonl").open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    def nums(rs):
        """Every plausible number a cluster carries, chosen label plus its alternatives."""
        out = set()
        for r in rs:
            for lbl in [r.get("fig_label")] + list(r.get("fig_label_alts") or []):
                n = fignum({"fig_label": lbl})
                if plausible(n):
                    out.add(n)
        return out

    L = [
        "# Consensus fusion",
        "",
        f"Clustered on `bbox_norm` at IoU >= {a.iou}, single linkage, over "
        f"{len(runs)} runs: {', '.join(runs)}.",
        f"Field precedence: {' > '.join(PRECEDENCE)}. "
        f"Written with `--min-votes {a.min_votes}`.",
        "",
        "## Confidence tiers",
        "",
        "| votes | tier | figures | pages | blank-flagged | labels | captions "
        "| distinct Fig. nos |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v in (3, 2, 1):
        g = [r for r in rows if r["votes"] == v]
        if not g:
            continue
        L.append(
            f"| {v} | {TIERS[v]} | {len(g)} | {len({r['page_id'] for r in g})} | "
            f"{sum(1 for r in g if r.get('is_blank'))} | "
            f"{sum(1 for r in g if r.get('fig_label'))} | "
            f"{sum(1 for r in g if r.get('caption_printed'))} | {len(nums(g))} |"
        )
    L.append(
        f"| **all** | | **{len(rows)}** | {len({r['page_id'] for r in rows})} | "
        f"{sum(1 for r in rows if r.get('is_blank'))} | "
        f"{sum(1 for r in rows if r.get('fig_label'))} | "
        f"{sum(1 for r in rows if r.get('caption_printed'))} | "
        f"**{len(nums(rows))}** |"
    )

    L += [
        "",
        "## Against the single best detector",
        "",
        "| run | figures | distinct Fig. nos |",
        "|---|---|---|",
    ]
    for n, r in runs.items():
        L.append(f"| {n} | {len(r['rows'])} | {len(nums(r['rows']))} |")
    L.append(f"| embedded text layer | - | {EMBEDDED_BASELINE} |")
    L.append(f"| **fused** | **{len(rows)}** | **{len(nums(rows))}** |")

    missing = [k for k in range(1, SERIES_MAX + 1) if k not in nums(rows)]
    disputed = sum(1 for r in rows if r.get("fig_label_alts"))
    L += [
        "",
        f"{len(missing)} of 1..{SERIES_MAX} still unrecovered. "
        f"{disputed} clusters carry a disputed label (detectors read different "
        "numbers off the same figure); all readings are kept in `fig_label_alts`. "
        "Alternatives count toward the distinct total, so a disputed pair such as "
        "`Fig. 68` / `Fig. 45` contributes both — exactly as the per-detector union "
        "in `comparison.md` does. That number is coverage, not correctness.",
        "",
        "## Where each field came from",
        "",
        "| field | " + " | ".join(PRECEDENCE) + " | none |",
        "|---|" + "---|" * 4,
    ]
    for field in ("fig_label_from", "caption_from"):
        c = defaultdict(int)
        for r in rows:
            c[r.get(field)] += 1
        L.append(
            f"| {field.replace('_from', '')} | "
            + " | ".join(str(c[n]) for n in PRECEDENCE)
            + f" | {c[None]} |"
        )
    try:
        from .config import find_pdf

        ag, dis_, none_ = embedded_check(rows, find_pdf(a.pdf))
        L += [
            "",
            "## Corroboration by the PDF's own text layer",
            "",
            f"Of the {ag + dis_ + none_} labelled fused rows, {ag + dis_} sit on a page "
            f"whose embedded text contains at least one `Fig. N`: **{ag} agree, {dis_} "
            f"disagree**. The remaining {none_} have no embedded number to check against.",
            "",
            "Weak oracle: an embedded number can be a body-text cross-reference rather "
            "than the printed caption, so this bounds the label error rate rather than "
            "measuring it.",
            "",
        ]
    except Exception as e:  # a missing PDF must not cost the whole report
        L += ["", f"*(embedded-layer corroboration skipped: {type(e).__name__}: {e})*", ""]

    L += [
        "",
        "Votes are a confidence tier, not an accuracy measure. Three detectors "
        "agreeing is still three out-of-domain models agreeing; only hand-checked "
        "boxes in `gt_figures.json` can turn this into precision/recall.",
        "",
    ]

    (base / "fusion.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n{len(kept)}/{len(rows)} rows -> {d / 'figures.jsonl'}\n-> {base / 'fusion.md'}")
