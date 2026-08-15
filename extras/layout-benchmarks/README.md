# Layout Benchmarks Workspace (`extras/layout-benchmarks/`)

This directory contains the experimental harnesses, models, predictions, and evaluation summaries for document layout analysis, visual element detection, and figure extraction across the 1,034-page historical volume (*People's Common Sense Medical Adviser*, Pierce, 1890).

---

## Directory Structure

```
extras/layout-benchmarks/
├── README.md                           # Master layout benchmark documentation
├── engines/                            # Execution code and detector drivers:
│   └── layout_research/                # Figure extraction pipelines and multi-detector harnesses
├── outputs/                            # Layout predictions across 1,034 pages:
│   ├── doclayout-yolo/                 # DocLayout-YOLO detections and summary
│   ├── orphan-ink/                     # Pixel connected components detections and summary
│   ├── picodet-s/                      # PicoDet-S layout detections and reference evaluation
│   ├── ppdoclayout-plus-l/             # PP-DocLayout+L detections and reference evaluation
│   ├── ppdoclayout-v3/                 # PP-DocLayoutV3 detections and summary
│   └── heldout-visualizations/         # Qualitative overlay PDFs across held-out pages
└── reports/                            # Empirical evaluation reports:
    └── layout_benchmark.md             # Canonical evaluation report against silver reference
```

---

## Evaluated Detectors & Provisional Evaluation

- **Provisional Silver Reference**: Chandra's 353 figure-like blocks detected across 254 illustrated pages (excluding 6 missing Chandra pages: `p0002`, `p0003`, `p0004`, `p0006`, `p1031`, `p1033`).
- **Full Analysis Report**: See [`reports/layout_benchmark.md`](reports/layout_benchmark.md) or [`../research-notes/layout_benchmark.md`](../research-notes/layout_benchmark.md).
