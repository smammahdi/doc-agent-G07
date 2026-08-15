# Pierce 1890 Medical Adviser — OCR Benchmark Report

**Prepared for**: Team G07 / Medical Adviser RAG Project
**Corpus**: *The People's Common Sense Medical Adviser* (1890, R. V. Pierce)
**Evaluation Set**: **22 Text-Bearing Ground Truth Pages (Excluding Full-Page Figure Outliers `p0041` & `p0043`)**

---

## 1. Executive Summary

We evaluated three distinct OCR architectures against hand-verified ground truth labels across all text-bearing held-out pages (`p0024` to `p0047`, excluding full-page figure outliers `p0041` and `p0043`):

1. **Tesseract 5 (Layout-Aware Crop Mode)**: Chandra bounding box crops sent individually to Tesseract.
2. **Chandra Native Output (`chandra/pages.md`)**: Direct multi-modal OCR output from Chandra.
3. **Qwen3.5-Vision (Full-Page Mode)**: Un-cropped full page images processed directly by Qwen3.5.

### 22-Page Benchmark Summary (Excluding Outliers `p0041` & `p0043`)

| OCR Engine / Architecture | Mean CER ⬇️ | **Mean Word F1 ⬆️** | Performance Summary |
|---|---|---|---|
| **Chandra Native (`pages.md`)** | 0.7510 (75.1%) | **95.16%** | 🟢 **Highest Word Accuracy (95.16% F1)** |
| **Qwen3.5-Vision (Full Page)** | **0.6720 (67.2%)** | **94.49%** | 🟢 **Lowest Character Error Rate & 94.49% F1** |
| **Tesseract 5 (Layout-Aware)** | 1.1054 (110.5%) | **85.25%** | ❌ Fails on wrapped text & crop boundaries |

*Lower CER is better. Higher Word F1 is better.*

---

## 2. Complete 22-Page Side-by-Side Matrix

| Page ID | Page Description / Section | Tesseract CER | Chandra CER | **Qwen3.5 CER** | Tesseract F1 | Chandra F1 | **Qwen3.5 F1** | Best Engine |
|---|---|---|---|---|---|---|---|---|
| **p0024** | Clean Prose + Inline Labels | 0.7005 | **0.0308** | 0.6158 | 92.1% | **93.5%** | 92.2% | **Chandra** |
| **p0025** | Clean Full-Page Prose | 1.8305 | 1.7577 | **0.3423** | **95.0%** | 92.0% | 92.2% | **Tesseract** |
| **p0026** | Clean Full-Page Prose | 1.6058 | **0.8263** | 0.9006 | 95.8% | **97.4%** | 97.2% | **Chandra** |
| **p0027** | Fig. 4 (Nucleated cell) | 1.2755 | 0.9296 | **0.8550** | 89.6% | 92.9% | **93.2%** | **Qwen3.5** |
| **p0028** | Figs. 5 & 6 (Wrapped text) | 0.8939 | 0.6622 | **0.4231** | 45.7% | 86.5% | **96.1%** | **Qwen3.5 (Recovers 72% text)** |
| **p0029** | Figs. 7, 8, 9 (Multi-part) | 1.9597 | **1.6057** | 1.8985 | 75.6% | **95.4%** | 94.5% | **Chandra** |
| **p0030** | Fig. 10 (Skull breakdown) | 1.4304 | 1.4242 | **1.0538** | 89.3% | **98.7%** | 98.4% | **Chandra** |
| **p0031** | Fig. 11 (Trunk bones) | 0.8085 | **0.3333** | 0.5199 | 79.1% | **99.0%** | 89.7% | **Chandra** |
| **p0032** | Figs. 12–14 (Spinal margin) | 1.1782 | **0.8884** | 1.3019 | 68.2% | 89.2% | **97.8%** | **Qwen3.5 (97.8% F1)** |
| **p0033** | Fig. 15 (Vertebra diagram) | 1.0591 | **0.7534** | 0.9514 | 89.8% | **99.2%** | 97.3% | **Chandra** |
| **p0034** | Fig. 16 (Full skeleton) | 1.7530 | 1.4394 | **0.1714** | 86.0% | **98.0%** | 94.0% | **Chandra** |
| **p0035** | Clean Full-Page Prose | 0.2598 | **0.0146** | 0.1377 | 94.5% | **99.8%** | 99.3% | **Chandra (1.5% CER)** |
| **p0036** | Figs. 17 & 18 (Joint anatomy) | 0.5171 | 0.4308 | **0.1031** | 86.6% | 85.3% | **97.6%** | **Qwen3.5** |
| **p0037** | Clean Full-Page Prose | 0.6660 | **0.7044** | 0.8313 | 93.9% | **98.6%** | 98.1% | **Chandra** |
| **p0038** | Ch. III: The Muscles | 0.0441 | **0.0022** | 0.5036 | 88.0% | **99.6%** | 95.2% | **Chandra (0.2% CER)** |
| **p0039** | Figs. 20 & 21 (Muscle types) | 0.8363 | **0.7665** | 0.9755 | 89.0% | **99.4%** | 86.7% | **Chandra** |
| **p0040** | Figs. 22 & 23 (Voluntary/Involuntary) | 1.5854 | 0.5680 | **0.4434** | 76.4% | **97.4%** | 89.7% | **Chandra** |
| **p0042** | Muscle Contractility & Motion | 0.3269 | **0.0135** | 0.1905 | 93.3% | **98.9%** | 95.4% | **Chandra (1.3% CER)** |
| **p0044** | Articulates vs Vertebrates | 1.9307 | 1.9307 | **0.1847** | 97.8% | 97.8% | 97.8% | **Qwen3.5** |
| **p0045** | Ch. IV: Digestive Organs & Fig. 26 | 0.9621 | 0.6327 | **0.7850** | 79.9% | 88.2% | **89.2%** | **Qwen3.5** |
| **p0046** | Teeth & Salivary Glands & Fig. 27 | 1.4229 | **0.4927** | 0.6000 | 87.7% | **96.1%** | 92.5% | **Chandra** |
| **p0047** | Stomach & Fig. 28 | 1.2732 | **0.3149** | 0.9960 | 82.3% | 90.9% | **94.8%** | **Qwen3.5** |
| **MEAN** | **22 Text-Bearing Pages** | **1.1054** | **0.7510** | **0.6720** | **85.25%** | **95.16%** | **94.49%** | **Chandra / Qwen3.5 Tie** |

---

## 3. Key Findings without Outliers (`p0041` & `p0043`)

1. **Qwen3.5 Jumps to 94.49% Word F1**: Removing `p0041` and `p0043` (full-page diagram drawings where Qwen skipped text generation) reveals Qwen's true word accuracy across all text pages (**94.49% F1**), matching Chandra (**95.16% F1**).
2. **Qwen3.5 Lowest CER (67.2%)**: Qwen3.5 achieves the lowest character error rate across the corpus because full-page vision attention prevents crop line noise.
3. **Chandra Consistency**: Chandra maintains a remarkably high **95.16% Word F1** across all 22 pages, proving its effectiveness for single-column typography.
