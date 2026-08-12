"""Paths and tunables. Everything env-dependent resolves here, nowhere else."""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent  # .../layout_research/figextract
TASK = HERE.parent  # .../extras/layout_research
ROOT = TASK.parents[1]  # repository root

VENDOR = TASK / "vendor"  # cloned official repos
ASSETS = TASK / "assets"  # downloaded weights
OUTPUT = TASK / "output"  # one subdir per detector

DPI = 300  # render resolution for detection + crops
BOOK_GLOB = "*ierce*.pdf"  # the 1890 Pierce medical adviser

# Where each detector's weights live. Override with env vars for other machines.
DOCLAYOUT_ONNX = ASSETS / "doclayout_yolo_docstructbench_imgsz1024.onnx"
DOCLAYOUT_PT = Path(
    os.environ.get("DOCLAYOUT_PT", ASSETS / "doclayout_yolo_docstructbench_imgsz1024.pt")
)
PPV3_DIR = Path(os.environ.get("PPV3_DIR", ASSETS / "PP-DocLayoutV3"))
EYNOLLAH_MODELS = Path(os.environ.get("EYNOLLAH_MODELS", ASSETS / "eynollah-models"))


def find_pdf(explicit=None):
    """Kaggle mounts datasets at /kaggle/input; locally the book sits in data/."""
    if explicit:
        return Path(explicit)
    for root in (Path("/kaggle/input"), ROOT / "data", Path.cwd()):
        if root.exists():
            hits = sorted(root.rglob(BOOK_GLOB)) or sorted(root.rglob("*.pdf"))
            if hits:
                return hits[0]
    raise FileNotFoundError("no Pierce PDF found; run scripts/get_data.sh or pass --pdf")


def run_dir(detector, base=None):
    """output/<detector>/ -- figures.jsonl + crops/ live here."""
    d = Path(base) if base else OUTPUT
    d = d / detector
    (d / "crops").mkdir(parents=True, exist_ok=True)
    return d
