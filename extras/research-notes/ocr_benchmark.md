# Milestone A2: OCR Engine Benchmark & Comparative Analysis

**Document Scope**: Historical Medical Document Agent (*People's Common Sense Medical Adviser*, R. V. Pierce, M.D., 1890).
**Evaluation Set**: 22 human-transcribed held-out text pages (`p0024–p0047`, excluding illustration-only pages `p0041` and `p0043`) evaluated against reference labels in [`grading_kit/labels.jsonl`](../../grading_kit/labels.jsonl).
**Evaluation Harness**: [`extras/ocr-benchmarks/engines/modular_suite/compare-results.py`](../ocr-benchmarks/engines/modular_suite/compare-results.py) operating directly on saved prediction files.
**Normalization Protocol**: HTML-unescape, HTML tag stripping, Unicode NFKC normalization, case-folding, non-alphanumeric character replacement with whitespace, and whitespace collapsing.

> **Methodological Caveat**: All reported CER, WER, and Word-F1 scores measure string agreement against the reference transcripts in [`grading_kit/labels.jsonl`](../../grading_kit/labels.jsonl). These numbers reflect true historical OCR accuracy only to the extent that the reference labels have been independently verified against the physical scan images.

---

## 1. Multi-Engine Empirical Benchmark (22 Held-Out Pages)

The table below reproduces the exact numbers from [`extras/ocr-benchmarks/reports/output_reports/ocr-benchmark-comparison-22-pages.json`](../ocr-benchmarks/reports/output_reports/ocr-benchmark-comparison-22-pages.json) across all 18 evaluated OCR configurations:

| Rank | OCR Engine / Configuration | Checkpoint / Source | Scored Pages | Macro CER ↓ | Macro WER ↓ | Macro Word-F1 ↑ | Micro CER ↓ | Micro WER ↓ |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Qwen3.5** | `Qwen3.5-9B (8q_k_xl)` | 22 | **0.1215** | **0.1432** | 0.9540 | **0.1109** | **0.1322** |
| 2 | **Chandra** | `Chandra-OCR` (layout blocks) | 22 | 0.1232 | 0.1507 | 0.9872 | 0.1147 | 0.1446 |
| 3 | **MinerU full-page** | `opendatalab/MinerU2.5-Pro-2604-1.2B` | 22 | 0.1289 | 0.1550 | **0.9887** | 0.1196 | 0.1467 |
| 4 | **GLM-OCR full-page** | `zai-org/GLM-OCR` | 22 | 0.1348 | 0.1501 | 0.9347 | 0.1232 | 0.1396 |
| 5 | **MinerU layout** | `MinerU` + `PP-DocLayoutV3` | 22 | 0.1362 | 0.1610 | 0.9705 | 0.1189 | 0.1420 |
| 6 | **GLM-OCR layout** | `GLM-OCR` + `PP-DocLayoutV3` | 22 | 0.1373 | 0.1613 | 0.9705 | 0.1199 | 0.1420 |
| 7 | **PaddleOCR layout** | `PP-OCRv6_medium` + `PP-DocLayoutV3` | 22 | 0.1392 | 0.1891 | 0.9465 | 0.1219 | 0.1692 |
| 8 | **Tesseract layout** | `Tesseract 5.x` + `PP-DocLayoutV3` | 22 | 0.1410 | 0.1980 | 0.9349 | 0.1236 | 0.1780 |
| 9 | **EasyOCR layout** | `EasyOCR` + `PP-DocLayoutV3` | 22 | 0.1583 | 0.2759 | 0.8549 | 0.1407 | 0.2541 |
| 10 | **PaddleOCR full-page** | `PP-OCRv6_medium` (full page) | 22 | 0.1607 | 0.2174 | 0.9622 | 0.1449 | 0.1996 |
| 11 | **Tesseract full-page** | `Tesseract 5.x` (full page) | 22 | 0.1801 | 0.2759 | 0.9106 | 0.1594 | 0.2441 |
| 12 | **TrOCR layout** | `microsoft/trocr-large-printed` + Layout | 22 | 0.1846 | 0.2745 | 0.8639 | 0.1708 | 0.2585 |
| 13 | **Florence-2 layout** | `microsoft/Florence-2-base` + Layout | 22 | 0.1893 | 0.3672 | 0.7865 | 0.1764 | 0.3526 |
| 14 | **Florence-2 full-page** | `microsoft/Florence-2-base` (full page) | 22 | 0.2270 | 0.4066 | 0.7674 | 0.2178 | 0.3993 |
| 15 | **DeepSeek-OCR full-page**| `deepseek-ai/DeepSeek-OCR-2` | 22 | 0.2548 | 0.3133 | 0.8567 | 0.2364 | 0.2959 |
| 16 | **TrOCR full-page** | `microsoft/trocr-large-printed` (full page)| 22 | 0.3550 | 0.4698 | 0.6241 | 0.3340 | 0.4529 |
| 17 | **EasyOCR full-page** | `EasyOCR` (full page) | 22 | 0.4117 | 0.5634 | 0.8651 | 0.4036 | 0.5486 |
| 18 | **DeepSeek-OCR layout** | `DeepSeek-OCR-2` + Layout | 22 | 0.7495 | 0.9584 | 0.7050 | 0.7259 | 0.9143 |

---

## 2. Key Findings & Empirical Observations

1. **Best Macro CER/WER**: **`Qwen3.5-9B (8q_k_xl)`** achieved the lowest Macro Character Error Rate (**0.1215**) and lowest Macro Word Error Rate (**0.1432**).
2. **Highest Macro Word-F1**: **`MinerU full-page`** achieved the highest token-level Macro Word-F1 score (**0.9887**), followed closely by **Chandra** (**0.9872**).
3. **Selective Impact of Layout Guidance**:
   - Applying `PP-DocLayoutV3` bounding crops significantly helped line-based and traditional OCR models by isolating text columns and suppressing non-text borders:
     - **Tesseract**: Full-page CER $0.1801 \to 0.1410$ (-21.7% relative error).
     - **PaddleOCR**: Full-page CER $0.1607 \to 0.1392$ (-13.4% relative error).
     - **EasyOCR**: Full-page CER $0.4117 \to 0.1583$ (-61.6% relative error).
     - **TrOCR**: Full-page CER $0.3550 \to 0.1846$ (-48.0% relative error).
     - **Florence-2**: Full-page CER $0.2270 \to 0.1893$ (-16.6% relative error).
   - In contrast, layout cropping did **not** benefit native end-to-end vision-language document models, which perform their own internal full-page spatial reasoning:
     - **MinerU**: Full-page CER **0.1289** was superior to layout crop CER **0.1362**.
     - **GLM-OCR**: Full-page CER **0.1348** was superior to layout crop CER **0.1373**.
     - **DeepSeek-OCR**: Cropping caused severe degradation ($0.2548 \to 0.7495$) due to tile scaling and context truncation.

---

## 3. Production Selection Rationale

**Selected Engine**: **Chandra OCR** (structured layout block parsing).

Chandra is selected as the practical, corpus-wide transcription source for the Milestone A2 Knowledge Base, balancing quantitative accuracy with structured block preservation:
- **Numerical Competitiveness**: With a Macro CER of **0.1232** (within 0.0017 of Qwen3.5) and a Macro Word-F1 of **0.9872** (second only to MinerU), Chandra matches top-tier accuracy.
- **Structured Block Output**: Unlike flat-text exporters, Chandra outputs **8,544 typed blocks** (`Text`, `Section-Header`, `Page-Header`, `Table`, `Figure`) with explicit bounding boxes (`bbox` and `page_box`). This structured hierarchy enables paragraph-aware chunking and bounding-box level citation verification in Stage 4.
- **Corpus Coverage & Missing Pages Accounting**:
  - Chandra processed **1,028 observed pages** out of the 1,034 PDF pages.
  - **Six unobserved pages** were omitted by the Chandra run: `p0002`, `p0003`, `p0004`, `p0006`, `p1031`, and `p1033`.
  - In the canonical knowledge base ([`extras/indexing-benchmarks/data/canonical-pages.jsonl`](../indexing-benchmarks/data/canonical-pages.jsonl)), these six pages are recorded with `ocr_source: "ocr_missing_unobserved"` and empty text rather than synthetic fallbacks.

---

## 4. Commercial Silver Reference: Google Cloud Document AI

Google Cloud Document AI (`document-ai`) extractions are preserved as a commercial cloud reference baseline across both scopes:
- **Full Book Asset**: [`extras/ocr-benchmarks/outputs/full-book/document-ai/words.jsonl`](../ocr-benchmarks/outputs/full-book/document-ai/words.jsonl) containing 419,565 word-level bounding boxes across 1,016 observed non-empty pages.
- **Held-Out Assembly**: Assembled source-order reading sequence in [`extras/ocr-benchmarks/outputs/heldout/document-ai/pages.jsonl`](../ocr-benchmarks/outputs/heldout/document-ai/pages.jsonl).
- **Held-Out Performance (22 Scored Pages)**:
  - **Macro CER**: `0.1508` ($15.08\%$)
  - **Macro WER**: `0.2045` ($20.45\%$)
  - **Macro Word-F1**: `0.9618` ($96.18\%$)
  - **Micro CER**: `0.1318`, **Micro WER**: `0.1849`
- **Assessment**: Document AI provides a strong commercial baseline with high word accuracy, but outputs unsegmented word-level bounding boxes rather than native structural layout blocks (headers, captions, paragraphs). It is retained as a silver reference for lexical verification.

---

## 5. Summary Reference for Milestone Form (A2)

- **Options Compared**: 10 distinct OCR engines (Qwen3.5-9B, Chandra, MinerU, GLM-OCR, PaddleOCR PP-OCRv6, Tesseract 5.x, EasyOCR, Florence-2-base, DeepSeek-OCR-2, TrOCR-large) across 18 full-page and layout-guided configurations.
- **Selected Choice**: **Chandra OCR** with structured block parsing.
- **22-Page Evaluation Benchmark**:
  - Best Macro CER: Qwen3.5 (0.1215)
  - Best Macro Word-F1: MinerU full-page (0.9887)
  - Selected (Chandra): Macro CER = 0.1232 (12.32%), Macro WER = 0.1507 (15.07%), Macro Word-F1 = 0.9872 (98.72%).
- **Full Corpus Coverage**: 1,034 PDF pages, 1,016 non-empty indexed pages, 18 empty/unobserved pages (including 6 Chandra unobserved: `p0002`, `p0003`, `p0004`, `p0006`, `p1031`, `p1033`), 364,824 words across 8,544 structured layout blocks.
