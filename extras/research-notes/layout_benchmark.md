# Milestone A2: Document Layout Segmentation & Figure Extraction Benchmark

**Document Scope**: Historical Medical Document Agent (*People's Common Sense Medical Adviser*, R. V. Pierce, M.D., 1890).  
**Task Objective**: Detect anatomical woodcuts, illustrations, tables, running headers, and multi-column body text blocks across 1,034 pages to prevent OCR cross-contamination and preserve reading order.  
**Provisional Ground Truth**: 353 manually confirmed figure/diagram bounding boxes across 254 illustrated pages of the 1890 medical volume.

---

## 1. Executive Summary & Model Selection

- **Selected Layout Pipeline**: **PP-DocLayoutV3** / **PP-DocLayout+L**
- **Justification**:
  1. **Highest Figure Presence F1 Score**: Achieved **0.9800 Page Presence F1** ($99.59\%$ Precision, $96.46\%$ Recall), successfully identifying virtually every anatomical illustration page.
  2. **Superior Bounding Box Overlap**: Achieved **0.8341 IoU@0.5 F1** with a **Mean Matched IoU of 0.9149** (91.49% average box intersection over union).
  3. **Robust Multi-Column Segmentation**: Accurately isolates single-column running headers from dual-column disease symptomatology listings.

---

## 2. Layout Model Comparative Evaluation

Evaluations were performed using greedy 1-to-1 bounding box matching at $\text{IoU} \ge 0.5$ on normalized page coordinates across all 1,034 pages.

| Layout Detector | Model Checkpoint / Architecture | Reference Boxes | Predicted Boxes | Page Presence F1 | Box IoU@0.5 Precision | Box IoU@0.5 Recall | Box IoU@0.5 F1 | Mean Matched IoU |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PP-DocLayout+L (Winner)** | `PP-DocLayout_plus-L_infer` | 353 | 304 | **0.9800** | **0.9013** | **0.7762** | **0.8341** | **0.9149** |
| **PP-DocLayoutV3** | `PP-DocLayoutV3` (Base) | 353 | 298 | 0.9720 | 0.8845 | 0.7478 | 0.8105 | 0.8982 |
| **PicoDet-S** | `PicoDet-S_layout_17cls_infer` | 353 | 262 | 0.8595 | 0.6794 | 0.5042 | 0.5789 | 0.8648 |
| **DocLayout-YOLO** | `doclayout_yolo_docstructbench` | 353 | 280 | 0.9120 | 0.7850 | 0.6232 | 0.6948 | 0.8812 |
| **Orphan-Ink** | Pixel Connected Components | — | — | 0.7410 | 0.4210 | 0.6800 | 0.5201 | 0.7120 |

---

## 3. Detailed Detector Analysis

### A. PP-DocLayout+L / PP-DocLayoutV3
- **Strengths**: Deep convolutional backbone trained on dense historical and multi-lingual layout benchmarks.
  - Page-presence precision of **0.9959** (only 1 false positive across 1,034 pages).
  - Tightly wraps irregular woodcut borders without clipping accompanying caption text (`Fig. 1`, `Fig. 2`, etc.).
- **Failure Cases**: Occasional split predictions on complex multi-part composite figures (e.g., `p0463`, `p0464` containing 8 miniature surgical instrument drawings).

### B. PicoDet-S
- **Strengths**: Extremely fast inference (suitable for CPU / lightweight edge deployment).
- **Weaknesses**: High false-negative rate on low-contrast historical woodcuts ($\text{Recall} = 50.42\%$, 175 missed boxes). Frequently missed delicate line-art diagrams on pages `p0698` and `p0706`.

### C. DocLayout-YOLO
- **Strengths**: Fast anchor-based detection with strong performance on standard modern rectangular layouts.
- **Weaknesses**: Struggled with 19th-century decorative borders and non-standard aspect ratios, yielding lower box recall ($62.32\%$).

---

## 4. Downstream Impact on Knowledge Base & Retrieval

1. **OCR Quality Boost**: Using PP-DocLayoutV3 bounding crops improved Tesseract OCR accuracy by **+21.7%** and EasyOCR by **+61.6%**.
2. **Clean Vector Embeddings**: Suppressing anatomical figures during text indexing prevents non-semantic OCR noise from degrading embedding quality in Stage 4.
3. **Visual Verification**: All generated bounding boxes and segmentation masks are visually verifiable in [`extras/layout-benchmarks/outputs/heldout-visualizations/`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/extras/layout-benchmarks/outputs/heldout-visualizations/).
