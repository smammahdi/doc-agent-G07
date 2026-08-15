# Tesseract Layout-Aware OCR Benchmark

Layout-aware Tesseract OCR benchmark for the Pierce 1890 corpus.
**Run on Kaggle GPU/CPU** — not on your local machine.

## What this does

Rather than running Tesseract blindly on the full page image (which forces it
to read figures, rules, and plates as body text), this benchmark:

1. Loads **Chandra `chunks.jsonl`** layout blocks and filters to text-only
   regions (Image / Figure / Diagram blocks are excluded).
2. Renders each PDF page at **300 DPI** via PyMuPDF.
3. **Crops each Chandra text region** and runs Tesseract on the crop
   individually — isolating OCR capability from layout detection errors.
4. Reassembles the page transcript from block texts in reading order.
5. **Scores against hand-verified labels** (`grading_kit/labels.jsonl`,
   pages `p0024`–`p0037`) using CER, WER, and Word F1.
6. **Also benchmarks Chandra's own text** against the same ground truth, so
   you can see exactly how much Tesseract lags the pseudo-GT source.
7. Writes `page_transcripts.jsonl`, `heldout_scores.csv`, and `report.md`
   to `/kaggle/working/tesseract_layout_bench/`.

## Kaggle dataset inputs required

| Kaggle dataset slug | Expected path |
|---|---|
| `kmazd1110/dl-peoples-common-sense-med-advisor` | `/kaggle/input/datasets/kmazd1110/dl-peoples-common-sense-med-advisor/EN_The-Peoples-Common-Sense-Medical-Adviser.pdf` |
| `cruelangelssprint/pierce-1890-figure-and-ocr-outputs` | `/kaggle/input/pierce-1890-figure-and-ocr-outputs/chandra/chunks.jsonl` |
| *(your repo dataset with grading_kit/)* | `/kaggle/input/doc-agent-7/grading_kit/labels.jsonl` |

## How to run on Kaggle

1. Create a new **Kaggle Notebook** (Python, CPU is fine for Tesseract).
2. Add the three datasets above under **+ Add Data**.
3. Upload `tesseract_layout_bench.py` to the notebook session or add this
   folder as a dataset.
4. In the first notebook cell:
   ```python
   import subprocess
   subprocess.run(["python", "/kaggle/input/<your-repo>/extras/tesseract_layout_bench/tesseract_layout_bench.py"])
   ```
   Or paste the script content directly into notebook cells.
5. Run all cells. The benchmark takes ~30–90 min for all 1,034 pages on CPU.

## Outputs

| File | Contents |
|---|---|
| `page_transcripts.jsonl` | `{page_id, text, n_blocks, elapsed_s}` for every page |
| `heldout_scores.csv` | Per-page CER / WER / Word-F1 for p0024–p0037 |
| `report.md` | Summary table + worst failure + Chandra vs Tesseract comparison |

## Why layout masking matters

Running Tesseract on the full page of a 19th-century illustrated medical book
inflates CER because:
- Engraved plates produce garbled character strings.
- Figure captions and running headers get mixed into body text.
- Two-column pages where Tesseract loses reading order.

By cropping Chandra's text-only regions first, we benchmark **pure OCR
recognition quality** independently from layout quality.
