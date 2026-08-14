# Held-out OCR benchmark comparison

This report compares four OCR engines on the same 24 held-out Pierce pages
(`p0024`–`p0047`). Each engine was evaluated in two modes:

- **Full page:** one OCR input per page.
- **PP-DocLayoutV3:** OCR over the same 188 stored non-figure regions, ordered
  top-to-bottom and then left-to-right.

CER, WER, and word F1 are macro averages of the 24 per-page scores. Lower CER
and WER are better; higher word F1 is better. All four engines were rescored
from their saved page text with the same normalization: Unicode NFKC,
case-folding, letters/numbers only, and collapsed whitespace.

## Full-page OCR

| Rank by CER | Engine | Pages | Regions | CER | WER | Word F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU2.5-Pro-2604-1.2B | 24 | 24 | **0.1330** | 0.1667 | **0.9840** |
| 2 | GLM-OCR | 24 | 24 | 0.1391 | **0.1581** | 0.9247 |
| 3 | DeepSeek-OCR-2 | 24 | 24 | 0.2058 | 0.2436 | 0.8928 |
| 4 | TrOCR large-printed | 24 | 24 | 0.4194 | 0.5510 | 0.6086 |

## PP-DocLayoutV3 region OCR

| Rank by CER | Engine | Pages | Regions | CER | WER | Word F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU2.5-Pro-2604-1.2B | 24 | 188 | **0.1343** | **0.1606** | **0.9606** |
| 2 | GLM-OCR | 24 | 188 | 0.1535 | 0.1834 | 0.9540 |
| 3 | TrOCR large-printed | 24 | 188 | 0.1869 | 0.2840 | 0.8489 |
| 4 | DeepSeek-OCR-2 | 24 | 188 | 1.0085 | 1.4950 | 0.6747 |

## Interpretation

- **MinerU is best overall on this held-out set.** It has the lowest CER in
  both modes and the highest word F1.
- **GLM-OCR is a close second.** Full-page GLM has the lowest WER, while its
  CER is only 0.0061 above MinerU.
- **Layout cropping helps TrOCR substantially.** Its normalized CER falls
  from 0.4194 to 0.1869.
- **The DeepSeek region result is a failed pipeline configuration, not a fair
  model-quality result.** Overlapping PP-DocLayoutV3 boxes and hallucinated
  crop outputs inflate some pages to several times the reference length.
- **MinerU does not need the external layout to perform well.** Its full-page
  and region CER are nearly identical. Its full-page word F1 is higher,
  suggesting that splitting text into PP-DocLayoutV3 regions can lose useful
  context or alter reading order.

## Method correspondence

The scoring follows the same core procedure as the supplied Tesseract example:
calculate CER, WER, and word F1 per page, then average the 24 page scores. The
benchmark runners additionally save micro CER/WER and both full-page and
PP-DocLayoutV3 outputs. The `regions` count corresponds to the example's
`BLOCKS` concept: 1 parent region per page in full-page mode and 188 total
non-figure regions in PP-DocLayoutV3 mode.

These are held-out benchmark results, not whole-book OCR accuracy claims.
See `ocr-page-by-page-comparison.md` and its wide-canvas PDF companion for the
per-page metrics and actual side-by-side text.
