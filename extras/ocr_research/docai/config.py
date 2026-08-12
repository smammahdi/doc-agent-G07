"""Paths, project settings, and the .env.vertex loader. Nothing else reads the environment."""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent  # .../ocr_research/docai
TASK = HERE.parent  # .../extras/ocr_research
ROOT = TASK.parents[1]  # repository root

OUTPUT = TASK / "output"
DPI = 300  # must match figextract, or boxes won't compare
ONLINE_MAX_PAGES = 15  # Document AI sync limit (20 MB / 15 pages)

# Document AI runs in multi-region `us` or `eu` -- NOT a Vertex region like us-central1.
# .env.vertex's GOOGLE_CLOUD_LOCATION is for Vertex and is deliberately not reused here.
DEFAULT_LOCATION = "us"
PROCESSOR_TYPE_MATCH = "layout"  # matched against fetch_processor_types()


def load_env(path=None):
    """Read .env.vertex into os.environ. A five-line parser beats a dependency.

    Only sets keys that are not already exported, so a shell `export` still wins.
    """
    p = Path(path) if path else ROOT / ".env.vertex"
    if not p.exists():
        return {}
    got = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        got[k] = v
        os.environ.setdefault(k, v)
    return got


def settings():
    load_env()
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if creds and not Path(creds).exists():
        raise SystemExit(f"GOOGLE_APPLICATION_CREDENTIALS points at a missing file: {creds}")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise SystemExit("GOOGLE_CLOUD_PROJECT unset (export it or use untracked .env.vertex)")
    return {
        "project": project,
        "location": os.environ.get("DOCAI_LOCATION", DEFAULT_LOCATION),
        "processor_id": os.environ.get("DOCAI_PROCESSOR_ID"),  # set to reuse one
        "credentials": creds,
    }


def find_pdf(explicit=None):
    """The Pierce book. `data/` also holds Gray's Anatomy -- never glob blindly here."""
    if explicit:
        return Path(explicit)
    hits = sorted((ROOT / "data/raw").glob("*ierce*.pdf"))
    if not hits:
        raise FileNotFoundError(
            "Pierce PDF not found under data/raw; run scripts/get_data.sh or pass --pdf"
        )
    return hits[0]


def out_dir(name="layout_parser"):
    """Only the run directory itself. `crops/` and `raw/` are made by whoever writes into
    them -- creating them up front left empty directories behind on every run that had no
    crops to save, which is most of them."""
    d = OUTPUT / name
    d.mkdir(parents=True, exist_ok=True)
    return d
