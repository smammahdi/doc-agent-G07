# Layout research

This directory keeps the reproducible layout experiments that informed the
runtime adapters in `src/doc_agent/vision/layout.py`. It is supplemental A2
research code: the production pipeline does not import it.

## What is here

- `figextract/`: the original figure extraction package. It renders Pierce
  pages, runs heuristic or pretrained detectors, binds printed labels and
  captions, writes provenance-rich JSONL, compares detector runs, and fuses
  their boxes.
- `run_layout_comparison.py`: the canonical full-book comparison runner used
  for the Chandra-reference table. It writes four files per detector:
  `pages.jsonl`, `detections.jsonl`, `summary.json`, and `run.log`.
The repository intentionally does not contain the book, model weights, crops,
or generated run output. Those are data artifacts, not source code. The output
directory is ignored by Git. A curated copy of the outputs is packaged in
[Kaggle dataset version 3](https://www.kaggle.com/datasets/cruelangelssprint/pierce-1890-figure-and-ocr-outputs)
for partner access.

## Published artifact layout

The external package contains:

- `chandra/chunks.jsonl`: 8,544 Chandra blocks on 1,028 observed pages;
- `figure-extraction/{orphan_ink,doclayout_yolo,ppdoclayout_v3}/figures.jsonl`;
- `figure-extraction/rerun_150dpi/<detector>/pages.jsonl`, with one row for
  each of the 1,034 pages;
- matching `detections.jsonl`, `summary.json`, and `run.log` files for each
  page-complete rerun; and
- the existing comparison and fusion summaries.

The Chandra rows use their own per-page `page_box`; detector rerun boxes use
the canonical pixel and normalized coordinates recorded in each JSONL file.
Chandra remains a provisional reference for agreement analysis, not human
ground truth. The package is for analysis and partner access only; the runtime
does not download it automatically.

## Canonical comparison

Fetch the book first:

```bash
bash scripts/get_data.sh
```

Run the no-weight heuristic over all pages:

```bash
python extras/layout_research/run_layout_comparison.py \
  --mode orphan_ink \
  --pages all \
  --chandra /path/to/chandra/chunks.jsonl
```

Run a pretrained detector by supplying its local checkpoint explicitly:

```bash
python extras/layout_research/run_layout_comparison.py \
  --mode doclayout_yolo \
  --pages all \
  --chandra /path/to/chandra/chunks.jsonl \
  --doclayout-yolo-weights /path/to/doclayout_yolo.onnx
```

PP-DocLayoutV3 uses `--mode ppdoclayout_v3` and
`--ppdoclayout-v3-weights /path/to/model-directory`. It needs an isolated
Transformers runtime because its published version conflicts with the pinned
starter environment.

The comparison treats Chandra's `Image`, `Figure`, and `Diagram` boxes as a
provisional reference. The program infers which pages Chandra actually
observed and excludes missing pages. These are agreement measurements, not
human-verified layout accuracy.

## Original figure pipeline

The richer original pipeline remains available when crops, printed `Fig. N`
labels, captions, detector comparison, or consensus fusion are needed:

```bash
PYTHONPATH=extras/layout_research python -m figextract probe
PYTHONPATH=extras/layout_research python -m figextract run \
  --detector orphan_ink --pdf data/raw/pierce-peoples-common-sense-medical-adviser-1890.pdf
PYTHONPATH=extras/layout_research python -m figextract compare
PYTHONPATH=extras/layout_research python -m figextract fuse
```

Eynollah remains in the package as a documented attempted option, but it was
not a successful Pierce run on macOS. Do not include it in measured tables.
