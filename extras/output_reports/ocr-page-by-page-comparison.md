# OCR page-by-page comparison

This report compares four OCR engines against the same 24 manually verified
held-out labels (`p0024`-`p0047`) in full-page and PP-DocLayoutV3-region modes.
All reported metrics are recomputed from the saved page text with one common
normalization: Unicode NFKC, case-folding, letters/numbers only, and collapsed
whitespace. DeepSeek grounding coordinates are removed before display and scoring.

## Overall macro averages

| Mode | Engine | CER | WER | Word F1 |
|---|---|---:|---:|---:|
| Full-page OCR | MinerU 2.5 Pro | 0.1330 | 0.1667 | 0.9840 |
| Full-page OCR | GLM-OCR | 0.1391 | 0.1581 | 0.9247 |
| Full-page OCR | DeepSeek-OCR-2 | 0.2058 | 0.2436 | 0.8928 |
| Full-page OCR | TrOCR large-printed | 0.4194 | 0.5510 | 0.6086 |
| PP-DocLayoutV3 region OCR | MinerU 2.5 Pro | 0.1343 | 0.1606 | 0.9606 |
| PP-DocLayoutV3 region OCR | GLM-OCR | 0.1535 | 0.1834 | 0.9540 |
| PP-DocLayoutV3 region OCR | TrOCR large-printed | 0.1869 | 0.2840 | 0.8489 |
| PP-DocLayoutV3 region OCR | DeepSeek-OCR-2 | 1.0085 | 1.4950 | 0.6747 |

## Page-by-page scores

Lower CER/WER is better; higher word F1 is better. `Length ratio` is normalized
OCR characters divided by normalized reference characters; unusually large values
usually indicate duplicated or hallucinated output.

### p0024

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 1 | 0.0087 | 0.0356 | 0.9755 | 0.99x |
| 2 | GLM-OCR | 1 | 0.0218 | 0.0388 | 0.9755 | 0.98x |
| 3 | TrOCR large-printed | 1 | 0.0463 | 0.1456 | 0.8631 | 1.01x |
| 4 | DeepSeek-OCR-2 | 1 | 0.0484 | 0.0777 | 0.9486 | 1.02x |

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 10 | 0.0082 | 0.0291 | 0.9804 | 0.99x |
| 2 | TrOCR large-printed | 10 | 0.0185 | 0.0809 | 0.9256 | 1.00x |
| 3 | GLM-OCR | 10 | 0.0299 | 0.0583 | 0.9696 | 1.03x |
| 4 | DeepSeek-OCR-2 | 10 | 0.3500 | 0.5922 | 0.7484 | 1.33x |

### p0025

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 1 | 0.0056 | 0.0410 | 0.9697 | 1.00x |
| 2 | GLM-OCR | 1 | 0.0183 | 0.0574 | 0.9612 | 0.99x |
| 3 | DeepSeek-OCR-2 | 1 | 0.0296 | 0.0765 | 0.9478 | 1.00x |
| 4 | TrOCR large-printed | 1 | 0.0880 | 0.1913 | 0.8333 | 1.03x |

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 9 | 0.0085 | 0.0464 | 0.9697 | 1.00x |
| 2 | TrOCR large-printed | 9 | 0.0193 | 0.0683 | 0.9384 | 1.00x |
| 3 | GLM-OCR | 9 | 0.0278 | 0.0683 | 0.9578 | 1.02x |
| 4 | DeepSeek-OCR-2 | 9 | 0.0301 | 0.0546 | 0.9595 | 1.01x |

### p0026

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 1 | 0.0000 | 0.0000 | 1.0000 | 1.00x |
| 2 | DeepSeek-OCR-2 | 1 | 0.0156 | 0.0181 | 0.9878 | 0.99x |
| 3 | GLM-OCR | 1 | 0.0187 | 0.0181 | 0.9909 | 0.99x |
| 4 | TrOCR large-printed | 1 | 0.0374 | 0.1239 | 0.8795 | 1.00x |

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 3 | 0.0015 | 0.0030 | 0.9985 | 1.00x |
| 2 | GLM-OCR | 3 | 0.0076 | 0.0091 | 0.9955 | 1.01x |
| 3 | TrOCR large-printed | 3 | 0.0202 | 0.0846 | 0.9187 | 1.00x |
| 4 | DeepSeek-OCR-2 | 3 | 1.1580 | 1.7311 | 0.4460 | 2.00x |

Flags: DeepSeek-OCR-2 output is inflated to 2.00x reference length.

### p0027

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GLM-OCR | 1 | 0.0803 | 0.0921 | 0.9520 | 0.93x |
| 2 | MinerU 2.5 Pro | 1 | 0.1556 | 0.1799 | 0.9979 | 1.00x |
| 3 | DeepSeek-OCR-2 | 1 | 0.1910 | 0.2176 | 0.9775 | 1.04x |
| 4 | TrOCR large-printed | 1 | 0.4255 | 0.5063 | 0.6326 | 0.69x |

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 11 | 0.4609 | 0.5649 | 0.8246 | 1.35x |
| 2 | GLM-OCR | 11 | 0.4971 | 0.6109 | 0.8090 | 1.39x |
| 3 | TrOCR large-printed | 11 | 0.5072 | 0.6820 | 0.7414 | 1.38x |
| 4 | DeepSeek-OCR-2 | 11 | 0.5362 | 0.7155 | 0.7368 | 1.38x |

### p0028

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GLM-OCR | 1 | 0.0767 | 0.0786 | 0.9573 | 0.92x |
| 2 | MinerU 2.5 Pro | 1 | 0.1213 | 0.1357 | 0.9893 | 1.00x |
| 3 | DeepSeek-OCR-2 | 1 | 0.1766 | 0.2000 | 0.9510 | 1.03x |
| 4 | TrOCR large-printed | 1 | 0.4411 | 0.5321 | 0.6054 | 0.58x |

Flags: TrOCR large-printed output is truncated to 0.58x reference length.

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | TrOCR large-printed | 12 | 0.1219 | 0.1893 | 0.8905 | 1.00x |
| 2 | MinerU 2.5 Pro | 12 | 0.1332 | 0.1393 | 0.9751 | 1.01x |
| 3 | GLM-OCR | 12 | 0.1492 | 0.1714 | 0.9549 | 1.05x |
| 4 | DeepSeek-OCR-2 | 12 | 1.2973 | 1.3964 | 0.5698 | 2.14x |

Flags: DeepSeek-OCR-2 output is inflated to 2.14x reference length.

### p0029

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 1 | 0.1667 | 0.2589 | 0.9585 | 0.99x |
| 2 | DeepSeek-OCR-2 | 1 | 0.3979 | 0.4873 | 0.7267 | 0.69x |
| 3 | GLM-OCR | 1 | 0.6640 | 0.7462 | 0.9534 | 0.98x |
| 4 | TrOCR large-printed | 1 | 0.7513 | 0.8325 | 0.3125 | 0.28x |

Flags: TrOCR large-printed output is truncated to 0.28x reference length.

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | TrOCR large-printed | 11 | 0.6806 | 0.8528 | 0.8196 | 0.95x |
| 2 | MinerU 2.5 Pro | 11 | 0.6911 | 0.7411 | 0.8871 | 0.89x |
| 3 | GLM-OCR | 11 | 0.7094 | 0.8223 | 0.8957 | 1.01x |
| 4 | DeepSeek-OCR-2 | 11 | 0.7417 | 1.0000 | 0.6780 | 1.03x |

### p0030

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GLM-OCR | 1 | 0.2748 | 0.2997 | 0.8244 | 0.73x |
| 2 | DeepSeek-OCR-2 | 1 | 0.2770 | 0.2905 | 0.8246 | 0.73x |
| 3 | MinerU 2.5 Pro | 1 | 0.3488 | 0.3731 | 0.9954 | 1.00x |
| 4 | TrOCR large-printed | 1 | 0.6258 | 0.7187 | 0.4279 | 0.41x |

Flags: TrOCR large-printed output is truncated to 0.41x reference length.

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 10 | 0.0059 | 0.0153 | 0.9908 | 1.00x |
| 2 | GLM-OCR | 10 | 0.0308 | 0.0428 | 0.9805 | 1.03x |
| 3 | DeepSeek-OCR-2 | 10 | 0.2538 | 0.3425 | 0.7760 | 1.10x |
| 4 | TrOCR large-printed | 10 | 0.3790 | 0.4709 | 0.6703 | 0.66x |

### p0031

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 1 | 0.1690 | 0.1831 | 0.9831 | 1.00x |
| 2 | GLM-OCR | 1 | 0.1970 | 0.2282 | 0.8703 | 0.81x |
| 3 | DeepSeek-OCR-2 | 1 | 0.2317 | 0.3465 | 0.9790 | 1.01x |
| 4 | TrOCR large-printed | 1 | 0.4221 | 0.5521 | 0.5958 | 0.64x |

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 7 | 0.0158 | 0.0254 | 0.9845 | 1.01x |
| 2 | GLM-OCR | 7 | 0.0204 | 0.0310 | 0.9833 | 1.02x |
| 3 | TrOCR large-printed | 7 | 0.0947 | 0.1634 | 0.8806 | 0.95x |
| 4 | DeepSeek-OCR-2 | 7 | 0.1079 | 0.1521 | 0.9161 | 1.10x |

### p0032

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 1 | 0.2674 | 0.3130 | 0.9592 | 0.99x |
| 2 | GLM-OCR | 1 | 0.3788 | 0.4174 | 0.8255 | 0.74x |
| 3 | DeepSeek-OCR-2 | 1 | 0.4164 | 0.4812 | 0.7534 | 0.70x |
| 4 | TrOCR large-printed | 1 | 0.9215 | 0.9449 | 0.1024 | 0.08x |

Flags: TrOCR large-printed output is truncated to 0.08x reference length.

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 8 | 0.2562 | 0.3304 | 0.8707 | 0.86x |
| 2 | GLM-OCR | 8 | 0.2759 | 0.3391 | 0.8701 | 0.86x |
| 3 | TrOCR large-printed | 8 | 0.4000 | 0.4725 | 0.6869 | 0.62x |
| 4 | DeepSeek-OCR-2 | 8 | 0.6355 | 0.8638 | 0.6732 | 1.26x |

### p0033

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GLM-OCR | 1 | 0.0896 | 0.1019 | 0.9433 | 0.92x |
| 2 | DeepSeek-OCR-2 | 1 | 0.0950 | 0.1142 | 0.9355 | 0.93x |
| 3 | MinerU 2.5 Pro | 1 | 0.1376 | 0.1698 | 0.9877 | 1.00x |
| 4 | TrOCR large-printed | 1 | 0.2168 | 0.3673 | 0.7243 | 0.89x |

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 10 | 0.1365 | 0.1698 | 0.9814 | 1.00x |
| 2 | TrOCR large-printed | 10 | 0.1567 | 0.2469 | 0.9205 | 1.00x |
| 3 | GLM-OCR | 10 | 0.1606 | 0.1975 | 0.9712 | 1.03x |
| 4 | DeepSeek-OCR-2 | 10 | 0.2458 | 0.3395 | 0.8497 | 1.01x |

### p0034

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GLM-OCR | 1 | 0.2738 | 0.3088 | 0.8157 | 0.73x |
| 2 | MinerU 2.5 Pro | 1 | 0.5064 | 0.5754 | 0.9965 | 1.00x |
| 3 | DeepSeek-OCR-2 | 1 | 0.7229 | 0.7298 | 0.5115 | 0.36x |
| 4 | TrOCR large-printed | 1 | 0.7276 | 0.8456 | 0.3093 | 0.33x |

Flags: DeepSeek-OCR-2 output is truncated to 0.36x reference length; TrOCR large-printed output is truncated to 0.33x reference length.

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 6 | 0.0276 | 0.0316 | 0.9840 | 0.97x |
| 2 | TrOCR large-printed | 6 | 0.0627 | 0.1298 | 0.9231 | 0.99x |
| 3 | GLM-OCR | 6 | 0.0641 | 0.0807 | 0.9755 | 1.01x |
| 4 | DeepSeek-OCR-2 | 6 | 1.1457 | 1.6105 | 0.5202 | 2.05x |

Flags: DeepSeek-OCR-2 output is inflated to 2.05x reference length.

### p0035

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 1 | 0.0000 | 0.0000 | 1.0000 | 1.00x |
| 2 | DeepSeek-OCR-2 | 1 | 0.0168 | 0.0196 | 0.9890 | 0.99x |
| 3 | GLM-OCR | 1 | 0.0168 | 0.0171 | 0.9914 | 0.99x |
| 4 | TrOCR large-printed | 1 | 0.0552 | 0.1663 | 0.8527 | 1.01x |

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GLM-OCR | 6 | 0.0129 | 0.0147 | 0.9927 | 1.01x |
| 2 | TrOCR large-printed | 6 | 0.0181 | 0.0954 | 0.9126 | 1.00x |
| 3 | MinerU 2.5 Pro | 6 | 0.0254 | 0.0318 | 0.9839 | 0.97x |
| 4 | DeepSeek-OCR-2 | 6 | 0.0849 | 0.1198 | 0.9231 | 0.93x |

### p0036

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 1 | 0.0059 | 0.0132 | 0.9901 | 1.00x |
| 2 | GLM-OCR | 1 | 0.0240 | 0.0263 | 0.9851 | 0.98x |
| 3 | DeepSeek-OCR-2 | 1 | 0.0727 | 0.0954 | 0.9496 | 1.03x |
| 4 | TrOCR large-printed | 1 | 0.1307 | 0.2961 | 0.7629 | 0.98x |

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GLM-OCR | 12 | 0.1219 | 0.1743 | 0.9806 | 1.04x |
| 2 | TrOCR large-printed | 12 | 0.1307 | 0.2401 | 0.8969 | 1.00x |
| 3 | MinerU 2.5 Pro | 12 | 0.1348 | 0.1842 | 0.9646 | 0.96x |
| 4 | DeepSeek-OCR-2 | 12 | 1.4560 | 2.1776 | 0.3922 | 2.16x |

Flags: DeepSeek-OCR-2 output is inflated to 2.16x reference length.

### p0037

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 1 | 0.0000 | 0.0000 | 1.0000 | 1.00x |
| 2 | DeepSeek-OCR-2 | 1 | 0.0188 | 0.0258 | 0.9831 | 0.99x |
| 3 | GLM-OCR | 1 | 0.0188 | 0.0233 | 0.9857 | 0.99x |
| 4 | TrOCR large-printed | 1 | 0.1967 | 0.3307 | 0.7585 | 1.13x |

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 5 | 0.0013 | 0.0026 | 0.9987 | 1.00x |
| 2 | GLM-OCR | 5 | 0.0135 | 0.0207 | 0.9884 | 1.01x |
| 3 | DeepSeek-OCR-2 | 5 | 0.0516 | 0.0672 | 0.9527 | 1.02x |
| 4 | TrOCR large-printed | 5 | 0.0612 | 0.1421 | 0.8872 | 1.03x |

### p0038

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GLM-OCR | 1 | 0.0314 | 0.0390 | 0.9737 | 0.97x |
| 2 | MinerU 2.5 Pro | 1 | 0.0598 | 0.0519 | 1.0000 | 1.00x |
| 3 | DeepSeek-OCR-2 | 1 | 0.0860 | 0.0952 | 0.9595 | 1.02x |
| 4 | TrOCR large-printed | 1 | 0.4390 | 0.5801 | 0.5564 | 0.64x |

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 9 | 0.0643 | 0.0693 | 0.9828 | 1.00x |
| 2 | GLM-OCR | 9 | 0.0823 | 0.0866 | 0.9809 | 1.03x |
| 3 | TrOCR large-printed | 9 | 0.0957 | 0.1818 | 0.8639 | 0.99x |
| 4 | DeepSeek-OCR-2 | 9 | 0.3411 | 0.5281 | 0.7362 | 1.13x |

### p0039

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GLM-OCR | 1 | 0.2417 | 0.2286 | 0.8675 | 0.76x |
| 2 | MinerU 2.5 Pro | 1 | 0.3710 | 0.4179 | 0.9929 | 1.00x |
| 3 | DeepSeek-OCR-2 | 1 | 0.4054 | 0.4607 | 0.9719 | 1.03x |
| 4 | TrOCR large-printed | 1 | 0.5674 | 0.7071 | 0.4522 | 0.54x |

Flags: TrOCR large-printed output is truncated to 0.54x reference length.

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 10 | 0.3988 | 0.4500 | 0.9820 | 0.99x |
| 2 | GLM-OCR | 10 | 0.4073 | 0.4714 | 0.9648 | 1.02x |
| 3 | TrOCR large-printed | 10 | 0.4109 | 0.5286 | 0.8873 | 1.00x |
| 4 | DeepSeek-OCR-2 | 10 | 0.4133 | 0.5107 | 0.9247 | 1.02x |

### p0040

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GLM-OCR | 1 | 0.1404 | 0.1626 | 0.9099 | 0.87x |
| 2 | MinerU 2.5 Pro | 1 | 0.2206 | 0.2886 | 0.9634 | 0.99x |
| 3 | DeepSeek-OCR-2 | 1 | 0.6279 | 0.7236 | 0.5946 | 0.48x |
| 4 | TrOCR large-printed | 1 | 0.6846 | 0.7724 | 0.3988 | 0.37x |

Flags: DeepSeek-OCR-2 output is truncated to 0.48x reference length; TrOCR large-printed output is truncated to 0.37x reference length.

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GLM-OCR | 6 | 0.1425 | 0.1829 | 0.8889 | 0.88x |
| 2 | MinerU 2.5 Pro | 6 | 0.1549 | 0.2236 | 0.8659 | 0.85x |
| 3 | TrOCR large-printed | 6 | 0.1660 | 0.3008 | 0.7905 | 0.86x |
| 4 | DeepSeek-OCR-2 | 6 | 2.0975 | 4.8415 | 0.0863 | 2.49x |

Flags: DeepSeek-OCR-2 output is inflated to 2.49x reference length.

### p0041

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 1 | 0.1262 | 0.2222 | 0.9474 | 1.07x |
| 2 | DeepSeek-OCR-2 | 1 | 0.1748 | 0.2778 | 0.7895 | 1.08x |
| 3 | GLM-OCR | 1 | 0.1748 | 0.2222 | 0.8333 | 0.97x |
| 4 | TrOCR large-printed | 1 | 1.4369 | 1.8333 | 0.3279 | 2.31x |

Flags: TrOCR large-printed output is inflated to 2.31x reference length.

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | DeepSeek-OCR-2 | 3 | 0.0291 | 0.1111 | 0.9189 | 1.01x |
| 2 | MinerU 2.5 Pro | 3 | 0.0291 | 0.0556 | 0.9714 | 0.97x |
| 3 | GLM-OCR | 3 | 0.1456 | 0.1667 | 0.9231 | 1.15x |
| 4 | TrOCR large-printed | 3 | 0.2039 | 0.3333 | 0.7368 | 1.10x |

### p0042

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 1 | 0.0000 | 0.0000 | 1.0000 | 1.00x |
| 2 | GLM-OCR | 1 | 0.0156 | 0.0147 | 0.9926 | 0.99x |
| 3 | DeepSeek-OCR-2 | 1 | 0.0168 | 0.0172 | 0.9902 | 0.99x |
| 4 | TrOCR large-printed | 1 | 0.0803 | 0.2255 | 0.7887 | 0.97x |

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 5 | 0.0013 | 0.0025 | 0.9988 | 1.00x |
| 2 | GLM-OCR | 5 | 0.0105 | 0.0123 | 0.9939 | 1.01x |
| 3 | TrOCR large-printed | 5 | 0.0790 | 0.1740 | 0.8497 | 0.95x |
| 4 | DeepSeek-OCR-2 | 5 | 2.5797 | 2.7917 | 0.3903 | 3.56x |

Flags: DeepSeek-OCR-2 output is inflated to 3.56x reference length.

### p0043

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 1 | 0.1250 | 0.2222 | 0.9474 | 1.07x |
| 2 | DeepSeek-OCR-2 | 1 | 0.1731 | 0.2778 | 0.7895 | 1.08x |
| 3 | GLM-OCR | 1 | 0.1731 | 0.2222 | 0.8333 | 0.97x |
| 4 | TrOCR large-printed | 1 | 0.8173 | 1.0556 | 0.5490 | 1.73x |

Flags: TrOCR large-printed output is inflated to 1.73x reference length.

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 3 | 0.0288 | 0.0556 | 0.9714 | 0.97x |
| 2 | DeepSeek-OCR-2 | 3 | 0.1154 | 0.2222 | 0.8108 | 1.03x |
| 3 | GLM-OCR | 3 | 0.1442 | 0.1667 | 0.9231 | 1.14x |
| 4 | TrOCR large-printed | 3 | 0.2212 | 0.4444 | 0.6316 | 1.10x |

### p0044

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 1 | 0.0000 | 0.0000 | 1.0000 | 1.00x |
| 2 | GLM-OCR | 1 | 0.0379 | 0.0361 | 0.9817 | 0.97x |
| 3 | TrOCR large-printed | 1 | 0.0390 | 0.1566 | 0.8434 | 0.99x |
| 4 | DeepSeek-OCR-2 | 1 | 0.0410 | 0.0422 | 0.9726 | 0.98x |

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 4 | 0.0031 | 0.0060 | 0.9970 | 1.00x |
| 2 | GLM-OCR | 4 | 0.0205 | 0.0241 | 0.9881 | 1.02x |
| 3 | DeepSeek-OCR-2 | 4 | 0.0226 | 0.0482 | 0.9581 | 0.99x |
| 4 | TrOCR large-printed | 4 | 0.0226 | 0.0904 | 0.9096 | 1.00x |

### p0045

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GLM-OCR | 1 | 0.1111 | 0.1387 | 0.9258 | 0.90x |
| 2 | DeepSeek-OCR-2 | 1 | 0.1796 | 0.2101 | 0.9592 | 1.05x |
| 3 | MinerU 2.5 Pro | 1 | 0.1796 | 0.2647 | 0.9937 | 1.00x |
| 4 | TrOCR large-printed | 1 | 0.3706 | 0.5252 | 0.6308 | 0.75x |

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | TrOCR large-printed | 9 | 0.1918 | 0.2521 | 0.8983 | 0.97x |
| 2 | GLM-OCR | 9 | 0.1963 | 0.2185 | 0.9558 | 0.99x |
| 3 | MinerU 2.5 Pro | 9 | 0.2298 | 0.2395 | 0.9474 | 0.91x |
| 4 | DeepSeek-OCR-2 | 9 | 8.2169 | 12.8025 | 0.0978 | 9.06x |

Flags: DeepSeek-OCR-2 output is inflated to 9.06x reference length.

### p0046

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GLM-OCR | 1 | 0.1276 | 0.1353 | 0.9212 | 0.88x |
| 2 | MinerU 2.5 Pro | 1 | 0.2114 | 0.2244 | 0.9885 | 1.00x |
| 3 | DeepSeek-OCR-2 | 1 | 0.2497 | 0.2607 | 0.9673 | 1.01x |
| 4 | TrOCR large-printed | 1 | 0.2691 | 0.4059 | 0.6897 | 0.83x |

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | GLM-OCR | 9 | 0.1264 | 0.1221 | 0.9838 | 1.03x |
| 2 | TrOCR large-printed | 9 | 0.1306 | 0.1881 | 0.9067 | 1.00x |
| 3 | MinerU 2.5 Pro | 9 | 0.1428 | 0.1452 | 0.9645 | 0.96x |
| 4 | DeepSeek-OCR-2 | 9 | 1.4277 | 1.9043 | 0.4694 | 2.25x |

Flags: DeepSeek-OCR-2 output is inflated to 2.25x reference length.

### p0047

#### Full-page OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 1 | 0.0057 | 0.0301 | 0.9789 | 1.00x |
| 2 | GLM-OCR | 1 | 0.1302 | 0.1416 | 0.9223 | 0.87x |
| 3 | DeepSeek-OCR-2 | 1 | 0.2750 | 0.3012 | 0.9686 | 1.01x |
| 4 | TrOCR large-printed | 1 | 0.2750 | 0.4096 | 0.7093 | 0.87x |

#### PP-DocLayoutV3 region OCR

| Rank | Engine | Regions | CER | WER | Word F1 | Length ratio |
|---:|---|---:|---:|---:|---:|---:|
| 1 | MinerU 2.5 Pro | 10 | 0.2641 | 0.2922 | 0.9806 | 1.02x |
| 2 | GLM-OCR | 10 | 0.2865 | 0.3102 | 0.9692 | 1.05x |
| 3 | TrOCR large-printed | 10 | 0.2932 | 0.4036 | 0.8869 | 1.02x |
| 4 | DeepSeek-OCR-2 | 10 | 0.8672 | 0.9578 | 0.6580 | 1.77x |

Flags: DeepSeek-OCR-2 output is inflated to 1.77x reference length.

## Evidence boundary

These values describe this 24-page held-out set only. The PDF companion shows the
actual saved OCR text beside each verified label. PP-DocLayoutV3 detections can
overlap, so its region-mode text should not be treated as an independent partition
without checking the side-by-side evidence.
