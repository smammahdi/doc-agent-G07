"""Local step 1: build the small PDF that gets uploaded to Kaggle as a dataset.

Defaults to the same 10 pages the Document AI smoke test used, so Chandra's figure boxes
land next to numbers we already have for those exact pages instead of a fresh sample nobody
can compare against.

    python3 make_sample.py                  # 10 pages -> sample/chandra_sample.pdf
    python3 make_sample.py --pages 21,74    # explicit
    python3 make_sample.py --count 50       # 10 chosen + 40 spread through the book
"""

import argparse
import json
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# Same ten as docai_layout: agreed plates, single-detector finds, a blank-page false
# positive, and two disputed Fig. labels.
SMOKE_PAGES = [21, 32, 36, 74, 2, 5, 84, 261, 138, 312]


def find_pdf(explicit=None):
    if explicit:
        return Path(explicit)
    hits = sorted((ROOT / "data/raw").glob("*ierce*.pdf"))
    if not hits:
        raise SystemExit(
            "Pierce PDF not found under data/raw; run scripts/get_data.sh or pass --pdf"
        )
    return hits[0]


def build(pages, out_dir, pdf=None):
    source_pdf = find_pdf(pdf)
    src = fitz.open(source_pdf)
    pages = [p for p in pages if 1 <= p <= src.page_count]
    dst = fitz.open()
    for p in pages:
        dst.insert_pdf(src, from_page=p - 1, to_page=p - 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "chandra_sample.pdf"
    dst.save(pdf_path)

    # Chandra sees a 1..N document; this maps its page numbers back to book pages.
    manifest = {
        "book_pages": pages,
        "sample_index_to_book_page": {str(i + 1): p for i, p in enumerate(pages)},
        "source_pdf": source_pdf.name,
        "dpi_note": "render at 300 dpi to match figextract",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    size_mb = pdf_path.stat().st_size / 1e6
    print(f"{len(pages)} pages -> {pdf_path} ({size_mb:.1f} MB)")
    print(f"pages: {pages}")
    return pdf_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", help="1-indexed, comma separated")
    ap.add_argument("--count", type=int, help="pad the default 10 up to this many, spread evenly")
    ap.add_argument("--out", default=str(HERE / "sample"))
    ap.add_argument("--pdf")
    a = ap.parse_args()

    if a.pages:
        pages = [int(x) for x in a.pages.split(",")]
    else:
        pages = list(SMOKE_PAGES)
        if a.count and a.count > len(pages):
            total = fitz.open(find_pdf(a.pdf)).page_count
            step = max(1, total // (a.count - len(pages)))
            extra = [p for p in range(1, total + 1, step) if p not in pages]
            pages += extra[: a.count - len(pages)]
    build(sorted(set(pages)), Path(a.out), a.pdf)


if __name__ == "__main__":
    main()
