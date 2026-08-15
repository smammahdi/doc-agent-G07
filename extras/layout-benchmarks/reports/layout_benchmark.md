# Milestone A2: Document Layout Segmentation & Figure Detection Benchmark

**Document Scope**: Historical Medical Document Agent (*People's Common Sense Medical Adviser*, R. V. Pierce, M.D., 1890).
**Evaluation Scope**: Quantitative detection and bounding-box agreement for figure-like elements across the 1890 medical volume.
**Provisional Reference**: Chandra's 353 extracted figure-like bounding boxes across 254 illustrated pages (treated as a provisional silver reference, not human gold truth).
**Observed Page Scope**: All detectors were executed over all 1,034 PDF pages; quantitative agreement is scored over the **1,028 Chandra-observed pages** (excluding six unobserved pages: `p0002`, `p0003`, `p0004`, `p0006`, `p1031`, and `p1033`).

> **Important Scope Limitation**: The quantitative metrics in this benchmark evaluate **figure-like bounding boxes only** (`Figure`, `Image`, `Diagram`). They do not measure text block segmentation accuracy, multi-column reading-order correctness, or table cell precision, which are evaluated downstream via OCR string error rates.

---

## 1. Empirical Figure Detection Benchmark

Metrics are calculated using greedy 1-to-1 bounding box matching at $\text{IoU} \ge 0.5$ on normalized page coordinates against the provisional Chandra reference:

| Detector | Box Precision | Box Recall | Box F1 | Mean Matched IoU | Page Presence F1 | Source Output Directory |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **PP-DocLayoutV3** | **.9412** | **.8159** | **.8741** | **.9270** | .9697 | [`../outputs/ppdoclayout-v3/`](../outputs/ppdoclayout-v3/) |
| **Orphan ink** | .8667 | .8102 | .8375 | .9101 | .9781 | [`../outputs/orphan-ink/`](../outputs/orphan-ink/) |
| **PP-DocLayout-plus-L** | .9013 | .7762 | .8341 | .9149 | **.9800** | [`../outputs/ppdoclayout-plus-l/`](../outputs/ppdoclayout-plus-l/) |
| **DocLayout-YOLO** | .9228 | .7450 | .8245 | .9059 | .9574 | [`../outputs/doclayout-yolo/`](../outputs/doclayout-yolo/) |
| **PicoDet-S** | .6794 | .5042 | .5789 | .8648 | .8595 | [`../outputs/picodet-s/`](../outputs/picodet-s/) |

---

## 2. Objective-Dependent Model Assessment

No single detector universally dominates across all evaluation criteria; selection depends on the target operational objective:

1. **Box-Level Overlap & Boundary Accuracy Objective**:
   - **Winner**: **`PP-DocLayoutV3`** achieves the highest **Box F1 (.8741)**, highest **Box Precision (.9412)**, highest **Box Recall (.8159)**, and highest **Mean Matched IoU (.9270)**.
   - It provides the tightest and most consistent bounding boxes around historical anatomical woodcuts, minimizing accidental inclusion of surrounding caption text.

2. **Page-Level Figure Presence Detection Objective**:
   - **Winner**: **`PP-DocLayout-plus-L`** achieves the highest **Page Presence F1 (.9800)** with $99.59\%$ page precision (245 true positive pages vs. only 1 false positive across 1,028 pages).
   - It is optimal when the goal is binary classification of whether a page contains visual/anatomical plates requiring figure suppression.

3. **Lightweight / Baseline Alternatives**:
   - **Orphan-Ink**: Rule-based pixel connected-component clustering achieves solid agreement (**Box F1 .8375**, **Page F1 .9781**) without deep learning weights, though with lower box precision (.8667).
   - **DocLayout-YOLO**: Demonstrates high box precision (.9228) but lower recall (.7450) on complex 19th-century engravings.
   - **PicoDet-S**: Exhibits high false-negative rates on delicate historical linework (Box Recall .5042, Box F1 .5789).

---

## 3. Qualitative Visual Observations (Held-Out PDF Inspections)

The visual overlay PDFs in [`extras/layout-benchmarks/outputs/heldout-visualizations/`](../outputs/heldout-visualizations/) provide qualitative confirmation of model behavior across the 24 held-out pages:
- *Observation 1 (Caption Handling)*: PP-DocLayoutV3 cleanly separates figure graphics from multi-line woodcut captions (`Fig. 1`, `Fig. 2`), whereas raw bounding boxes occasionally merge captions into the illustration box.
- *Observation 2 (Border Suppression)*: Connected-component analysis and PP-DocLayout models successfully avoid classifying ornamental section dividers as standalone figures.
- *Observation 3 (Composite Woodcuts)*: Pages with multiple miniature instrument sketches (e.g., `p0463`, `p0464`) represent the primary failure mode across all detectors, frequently resulting in fragmented multi-box predictions.
