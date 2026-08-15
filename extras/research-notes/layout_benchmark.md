# Milestone A2: Document Layout Segmentation & Figure Detection Benchmark

> **Canonical Document Pointer**: This research note is synchronized with the canonical layout benchmark report at [`extras/layout-benchmarks/reports/layout_benchmark.md`](../layout-benchmarks/reports/layout_benchmark.md).

**Document Scope**: Historical Medical Document Agent (*People's Common Sense Medical Adviser*, R. V. Pierce, M.D., 1890).
**Evaluation Scope**: Quantitative detection and bounding-box agreement for figure-like elements across the 1890 medical volume.
**Provisional Reference**: Chandra's 353 extracted figure-like bounding boxes across 254 illustrated pages (treated as a provisional silver reference, not human gold truth).
**Observed Page Scope**: All detectors were executed over all 1,034 PDF pages; quantitative agreement is scored over the **1,028 Chandra-observed pages** (excluding six unobserved pages: `p0002`, `p0003`, `p0004`, `p0006`, `p1031`, and `p1033`).

> **Important Scope Limitation**: The quantitative metrics in this benchmark evaluate **figure-like bounding boxes only** (`Figure`, `Image`, `Diagram`). They do not measure text block segmentation accuracy, multi-column reading-order correctness, or table cell precision, which are evaluated downstream via OCR string error rates.

---

## 1. Empirical Figure Detection Benchmark

Metrics are calculated using the canonical Chandra-reference evaluator [`extras/layout-benchmarks/engines/layout_research/run_layout_comparison.py`](../layout-benchmarks/engines/layout_research/run_layout_comparison.py) using greedy 1-to-1 bounding box matching at $\text{IoU} \ge 0.5$ on normalized page coordinates:

| Detector | Box Precision | Box Recall | Box F1 | Mean Matched IoU | Page Presence F1 | Source Output Directory |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **PP-DocLayoutV3** | **.9412** | **.8159** | **.8741** | **.9270** | .9697 | [`../layout-benchmarks/outputs/ppdoclayout-v3/`](../layout-benchmarks/outputs/ppdoclayout-v3/) |
| **Orphan ink** | .8667 | .8102 | .8375 | .9101 | .9781 | [`../layout-benchmarks/outputs/orphan-ink/`](../layout-benchmarks/outputs/orphan-ink/) |
| **PP-DocLayout-plus-L** | .9013 | .7762 | .8341 | .9149 | **.9800** | [`../layout-benchmarks/outputs/ppdoclayout-plus-l/`](../layout-benchmarks/outputs/ppdoclayout-plus-l/) |
| **DocLayout-YOLO** | .9228 | .7450 | .8245 | .9059 | .9574 | [`../layout-benchmarks/outputs/doclayout-yolo/`](../layout-benchmarks/outputs/doclayout-yolo/) |
| **PicoDet-S** | .6794 | .5042 | .5789 | .8648 | .8595 | [`../layout-benchmarks/outputs/picodet-s/`](../layout-benchmarks/outputs/picodet-s/) |

---

## 2. Automated Evaluation Methodology & Matching Arithmetic

The canonical evaluation is implemented in [`extras/layout-benchmarks/engines/layout_research/run_layout_comparison.py`](../layout-benchmarks/engines/layout_research/run_layout_comparison.py). *(Note: `figextract/compare.py` is the older cross-detector agreement utility, whereas `run_layout_comparison.py` is the canonical evaluator against the Chandra reference).*

### A. Coordinate Normalization & 2D Intersection over Union (IoU)
For each page, bounding box coordinates are normalized to the unit square $[0, 1] \times [0, 1]$ using page dimensions:

$$\text{IoU}(\text{Box}_A, \text{Box}_B) = \frac{\text{Area}(\text{Box}_A \cap \text{Box}_B)}{\text{Area}(\text{Box}_A \cup \text{Box}_B)}$$

### B. Pairwise IoU Sorting & Greedy Bipartite Matching ($\text{IoU} \ge 0.5$)
1. On each page, the evaluator constructs all possible (prediction, reference) candidate pairs and computes their 2D IoU.
2. Candidate pairs are **sorted by IoU in descending order**.
3. The evaluator greedily matches candidate pairs with $\text{IoU} \ge 0.5$, ensuring each predicted box and each reference box is matched at most once:
   - **True Positive ($\text{TP}$)**: A predicted box matched to an unmatched reference box with $\text{IoU} \ge 0.5$.
   - **False Positive ($\text{FP}$)**: A predicted box that remains unmatched ($\text{IoU} < 0.5$).
   - **False Negative ($\text{FN}$)**: A reference box with no matching predicted box.
4. **Metric Definitions**:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{Box F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
   $$\text{Mean Matched IoU} = \frac{1}{\text{TP}} \sum_{i=1}^{\text{TP}} \text{IoU}_i$$

### C. Detector-by-Detector Breakdown

| Detector | Predicted Boxes | Matched TP | False Positives (FP) | False Negatives (FN) | Matched Pages (TP / FP / FN) |
|---|:---:|:---:|:---:|:---:|:---:|
| **PP-DocLayoutV3** | 306 | 288 | 18 | 65 | 240 / 1 / 14 |
| **Orphan ink** | 330 | 286 | 44 | 67 | 246 / 3 / 8 |
| **PP-DocLayout-plus-L** | 304 | 274 | 30 | 79 | 245 / 1 / 9 |
| **DocLayout-YOLO** | 285 | 263 | 22 | 90 | 236 / 3 / 18 |
| **PicoDet-S** | 262 | 178 | 84 | 175 | 208 / 22 / 46 |

---

## 3. Objective-Dependent Model Assessment

1. **Box-Level Overlap & Boundary Accuracy Objective**:
   - **Winner**: **`PP-DocLayoutV3`** achieves the highest **Box F1 (.8741)**, highest **Box Precision (.9412)**, highest **Box Recall (.8159)**, and highest **Mean Matched IoU (.9270)**.
2. **Page-Level Figure Presence Detection Objective**:
   - **Winner**: **`PP-DocLayout-plus-L`** achieves the highest **Page Presence F1 (.9800)** with $99.59\%$ page precision.

---

## 4. Qualitative Visual Observations (Held-Out PDF Inspections)

Visual overlay PDFs in [`extras/layout-benchmarks/outputs/heldout-visualizations/`](../layout-benchmarks/outputs/heldout-visualizations/) provide qualitative confirmation across held-out pages:
- *Caption Handling*: PP-DocLayoutV3 separates figure graphics from multi-line woodcut captions.
- *Border Suppression*: Ornamental section dividers are avoided.
- *Composite Woodcuts*: Dense composite sketches (`p0463`, `p0464`) remain the primary failure mode.
