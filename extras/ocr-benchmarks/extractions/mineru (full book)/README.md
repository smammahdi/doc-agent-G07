# MinerU full-book OCR output

This directory contains real OCR output from
`opendatalab/MinerU2.5-Pro-2605-1.2B` over all 1,034 Pierce pages in two modes:

- `full-page/`: MinerU's two-step full-page document extraction;
- `ppdoclayout-v3/`: direct content recognition over existing non-figure
  PP-DocLayoutV3 regions.

Each mode contains ordered `pages.jsonl`, `regions.jsonl`, and `metrics.json`.
`comparison.json` summarizes both runs. All 1,034 pages have output records, but
CER, WER, and word-F1 are calculated only on the 24 manually labelled held-out
pages (`p0024` through `p0047`). The remaining pages are unscored model output.

These results are research evidence, not ground truth. Model weights, rendered
images, the source PDF, private runtime paths, and credentials are not included.
