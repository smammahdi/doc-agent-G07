# OCR comparison (24 Pages)

> [!WARNING]
> **SUPERSEDED REPORT**: This 24-page report includes the full-page illustration outliers `p0041` and `p0043`. It has been superseded by the canonical 22-page benchmark in [`ocr-benchmark-comparison-22-pages.md`](ocr-benchmark-comparison-22-pages.md) and [`extras/research-notes/ocr_benchmark.md`](../../../research-notes/ocr_benchmark.md).

Normalization: HTML-unescape, strip-tags, Unicode NFKC, casefold, letters/numbers only, collapsed whitespace.
CER/WER are lower-is-better; Word-F1 is higher-is-better.

| Engine | Pages | Macro CER | Macro WER | Macro Word-F1 | Micro CER | Micro WER |
|---|---:|---:|---:|---:|---:|---:|
| Chandra | 24 | 0.1230 | 0.1567 | 0.9827 | 0.1147 | 0.1450 |
| MinerU 2605 PP-DocLayout | 24 | 0.1248 | 0.1475 | 0.9730 | 0.1183 | 0.1412 |
| GLM PP-DocLayout | 24 | 0.1259 | 0.1479 | 0.9730 | 0.1193 | 0.1412 |
| Paddle PP-DocLayout | 24 | 0.1276 | 0.1734 | 0.9510 | 0.1213 | 0.1683 |
| MinerU 2605 full | 24 | 0.1286 | 0.1606 | 0.9853 | 0.1196 | 0.1471 |
| Tesseract PP-DocLayout | 24 | 0.1293 | 0.1815 | 0.9403 | 0.1229 | 0.1771 |
| MinerU 2604 full | 24 | 0.1330 | 0.1667 | 0.9840 | 0.1243 | 0.1530 |
| GLM full | 24 | 0.1340 | 0.1515 | 0.9283 | 0.1232 | 0.1397 |
| MinerU 2604 PP-DocLayout | 24 | 0.1343 | 0.1606 | 0.9606 | 0.1257 | 0.1505 |
| Document AI | 24 | 0.1487 | 0.2060 | 0.9606 | 0.1317 | 0.1851 |
| Paddle full | 24 | 0.1529 | 0.2085 | 0.9609 | 0.1445 | 0.1992 |
| Tesseract full | 24 | 0.1731 | 0.2668 | 0.9115 | 0.1591 | 0.2437 |
| TrOCR PP-DocLayout | 24 | 0.1869 | 0.2840 | 0.8489 | 0.1710 | 0.2592 |
| DeepSeek full | 24 | 0.2855 | 0.3659 | 0.8353 | 0.2385 | 0.2993 |
| TrOCR full | 24 | 0.4194 | 0.5510 | 0.6086 | 0.3382 | 0.4582 |
| DeepSeek PP-DocLayout | 24 | 0.6931 | 0.8924 | 0.7183 | 0.7224 | 0.9103 |

Scores are computed from the saved JSONL text. They are not claims of
human OCR accuracy unless the reference labels are manually verified.

## Included and unavailable sources

The table includes the saved 24-page outputs for Chandra, Document AI,
MinerU 2.5 Pro 2604/2605, GLM-OCR, DeepSeek-OCR-2, PaddleOCR, TrOCR, and
Tesseract.  Florence-2 is not scored because the repository contains its
runner and notebook but no executed prediction JSONL.  The Document AI row
is assembled from its saved word boxes in file order; it is included as a
separate OCR export, not treated as manually verified ground truth.
