# OCR comparison

Normalization: HTML-unescape, strip-tags, Unicode NFKC, casefold, letters/numbers only, collapsed whitespace.
CER/WER are lower-is-better; Word-F1 is higher-is-better.
Excluded pages: p0041, p0043.

| Engine | Pages | Macro CER | Macro WER | Macro Word-F1 | Micro CER | Micro WER |
|---|---:|---:|---:|---:|---:|---:|
| Chandra | 22 | 0.1232 | 0.1507 | 0.9872 | 0.1147 | 0.1446 |
| MinerU 2605 full | 22 | 0.1289 | 0.1550 | 0.9887 | 0.1196 | 0.1467 |
| MinerU 2604 full | 22 | 0.1337 | 0.1617 | 0.9873 | 0.1243 | 0.1527 |
| GLM full | 22 | 0.1348 | 0.1501 | 0.9347 | 0.1232 | 0.1396 |
| MinerU 2605 PP-DocLayout | 22 | 0.1362 | 0.1610 | 0.9705 | 0.1189 | 0.1420 |
| GLM PP-DocLayout | 22 | 0.1373 | 0.1613 | 0.9705 | 0.1199 | 0.1420 |
| Paddle PP-DocLayout | 22 | 0.1392 | 0.1891 | 0.9465 | 0.1219 | 0.1692 |
| Tesseract PP-DocLayout | 22 | 0.1410 | 0.1980 | 0.9349 | 0.1236 | 0.1780 |
| MinerU 2604 PP-DocLayout | 22 | 0.1439 | 0.1701 | 0.9597 | 0.1262 | 0.1510 |
| Document AI | 22 | 0.1508 | 0.2045 | 0.9618 | 0.1318 | 0.1849 |
| Paddle full | 22 | 0.1607 | 0.2174 | 0.9622 | 0.1449 | 0.1996 |
| Tesseract full | 22 | 0.1801 | 0.2759 | 0.9106 | 0.1594 | 0.2441 |
| TrOCR PP-DocLayout | 22 | 0.1846 | 0.2745 | 0.8639 | 0.1708 | 0.2585 |
| DeepSeek full | 22 | 0.2548 | 0.3133 | 0.8567 | 0.2364 | 0.2959 |
| TrOCR full | 22 | 0.3550 | 0.4698 | 0.6241 | 0.3340 | 0.4529 |
| DeepSeek PP-DocLayout | 22 | 0.7495 | 0.9584 | 0.7050 | 0.7259 | 0.9143 |

Scores are computed from the saved JSONL text. They are not claims of
human OCR accuracy unless the reference labels are manually verified.
Ranking is by macro CER ascending; no composite average of CER, WER,
and Word-F1 is used because CER/WER are lower-is-better and Word-F1
is higher-is-better.
