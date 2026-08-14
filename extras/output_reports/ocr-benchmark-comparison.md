# Held-out OCR benchmark comparison

This report compares four OCR engines on the same 24 held-out Pierce pages
(`p0024`–`p0047`). Each engine was evaluated in two modes:

- **Full page:** one OCR input per page.
- **PP-DocLayoutV3:** OCR over the same 188 stored non-figure regions, ordered
  top-to-bottom and then left-to-right.

CER, WER, and word F1 are macro averages of the 24 per-page scores. Lower CER
and WER are better; higher word F1 is better. Values below come from each
run's saved page text and held-out labels.

## Full-page OCR

| Rank by CER | Engine | Pages | Regions | CER | WER | Word F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU2.5-Pro-2604-1.2B | 24 | 24 | **0.1416** | 0.1660 | **0.9812** |
| 2 | GLM-OCR | 24 | 24 | 0.1478 | **0.1607** | 0.9185 |
| 3 | TrOCR large-printed | 24 | 24 | 0.9272 | 1.0958 | 0.0226 |
| 4 | DeepSeek-OCR-2 | 24 | 24 | 1.1823 | 1.3446 | 0.6174 |

## PP-DocLayoutV3 region OCR

| Rank by CER | Engine | Pages | Regions | CER | WER | Word F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU2.5-Pro-2604-1.2B | 24 | 188 | **0.1396** | **0.1657** | **0.9535** |
| 2 | GLM-OCR | 24 | 188 | 0.1778 | 0.1852 | 0.9290 |
| 3 | DeepSeek-OCR-2 | 24 | 188 | 0.5028 | 0.6013 | 0.6665 |
| 4 | TrOCR large-printed | 24 | 188 | 0.8288 | 1.0286 | 0.0451 |

## Interpretation

- **MinerU is best overall on this held-out set.** It has the lowest CER in
  both modes, the best full-page word F1, and the best region-mode WER/F1.
- **GLM-OCR is a close second.** Full-page GLM has the lowest WER, while its
  CER is only 0.0062 above MinerU.
- **Layout cropping helps DeepSeek and TrOCR substantially.** DeepSeek CER
  falls from 1.1823 to 0.5028; TrOCR CER falls from 0.9272 to 0.8288.
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
