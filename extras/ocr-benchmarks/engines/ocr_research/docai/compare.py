# ruff: noqa: N806
"""Document AI vs our fused consensus, on whatever pages both have seen.

This is agreement with an *independent* system, not accuracy. Document AI is out-of-domain
for an 1890 engraved book too. It earns the word "ground truth" only after someone eyeballs
the crops -- which is exactly what the contact sheet is for.
"""

import json
from collections import defaultdict
from pathlib import Path

from .config import ROOT, out_dir

FUSED = ROOT / "extras/layout_research/output/fused/figures.jsonl"


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def load(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"missing {p}")
    by_page = defaultdict(list)
    for line in p.open():
        if line.strip():
            r = json.loads(line)
            by_page[r["page_id"]].append(r)
    return by_page


def report(docai_jsonl=None, fused_jsonl=None, thr=0.5):
    out = out_dir()
    dc = load(docai_jsonl or out / "figures.jsonl")
    fu = load(fused_jsonl or FUSED)
    pages = sorted(dc)  # only pages Document AI actually saw

    matched = docai_only = fused_only = 0
    rows = []
    for pid in pages:
        d, f = dc[pid], list(fu.get(pid, []))
        m = 0
        for x in d:
            hit = next((y for y in f if iou(x["bbox_norm"], y["bbox_norm"]) >= thr), None)
            if hit:
                m += 1
                f.remove(hit)
        matched += m
        docai_only += len(d) - m
        fused_only += len(f)
        votes = sorted({y["votes"] for y in fu.get(pid, [])}) or ["-"]
        rows.append((pid, len(d), len(fu.get(pid, [])), m, votes))

    L = [
        "# Document AI Layout Parser vs fused consensus",
        "",
        f"IoU >= {thr}, on the {len(pages)} pages Document AI has processed.",
        "",
        "| page | docai | fused | matched | fused vote tiers |",
        "|---|---|---|---|---|",
    ]
    for pid, nd, nf, m, v in rows:
        L.append(f"| {pid} | {nd} | {nf} | {m} | {','.join(str(x) for x in v)} |")
    L += [
        "",
        f"**matched {matched}**, docai-only {docai_only}, fused-only {fused_only}.",
        "",
        "Agreement with an independent system, not accuracy — Document AI is also "
        "out-of-domain for an 1890 engraved book. Verify the crops before calling any of "
        "this ground truth.",
        "",
    ]
    p = out / "vs_fused.md"
    p.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"-> {p}")
    return p
