# OCR comparison

Normalization: HTML-unescape, strip-tags, Unicode NFKC, casefold, letters/numbers only, collapsed whitespace.
CER/WER are lower-is-better; Word-F1 is higher-is-better.
Excluded pages: p0041, p0043.

| Engine | Pages | Macro CER | Macro WER | Macro Word-F1 | Micro CER | Micro WER |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5 | 22 | 0.1215 | 0.1432 | 0.9540 | 0.1109 | 0.1322 |
| Chandra | 22 | 0.1232 | 0.1507 | 0.9872 | 0.1147 | 0.1446 |
| MinerU full-page | 22 | 0.1289 | 0.1550 | 0.9887 | 0.1196 | 0.1467 |
| GLM-OCR full-page | 22 | 0.1348 | 0.1501 | 0.9347 | 0.1232 | 0.1396 |
| MinerU layout | 22 | 0.1362 | 0.1610 | 0.9705 | 0.1189 | 0.1420 |
| GLM-OCR layout | 22 | 0.1373 | 0.1613 | 0.9705 | 0.1199 | 0.1420 |
| PaddleOCR layout | 22 | 0.1392 | 0.1891 | 0.9465 | 0.1219 | 0.1692 |
| Tesseract layout | 22 | 0.1410 | 0.1980 | 0.9349 | 0.1236 | 0.1780 |
| EasyOCR layout | 22 | 0.1583 | 0.2759 | 0.8549 | 0.1407 | 0.2541 |
| PaddleOCR full-page | 22 | 0.1607 | 0.2174 | 0.9622 | 0.1449 | 0.1996 |
| Tesseract full-page | 22 | 0.1801 | 0.2759 | 0.9106 | 0.1594 | 0.2441 |
| TrOCR layout | 22 | 0.1846 | 0.2745 | 0.8639 | 0.1708 | 0.2585 |
| Florence-2 layout | 22 | 0.1893 | 0.3672 | 0.7865 | 0.1764 | 0.3526 |
| Florence-2 full-page | 22 | 0.2270 | 0.4066 | 0.7674 | 0.2178 | 0.3993 |
| DeepSeek-OCR full-page | 22 | 0.2548 | 0.3133 | 0.8567 | 0.2364 | 0.2959 |
| TrOCR full-page | 22 | 0.3550 | 0.4698 | 0.6241 | 0.3340 | 0.4529 |
| EasyOCR full-page | 22 | 0.4117 | 0.5634 | 0.8651 | 0.4036 | 0.5486 |
| DeepSeek-OCR layout | 22 | 0.7495 | 0.9584 | 0.7050 | 0.7259 | 0.9143 |

Scores are computed from the saved JSONL text. They are not claims of
human OCR accuracy unless the reference labels are manually verified.
Ranking is by macro CER ascending; no composite average of CER, WER,
and Word-F1 is used because CER/WER are lower-is-better and Word-F1
is higher-is-better.
