# Held-out OCR benchmark outputs

These folders contain saved OCR text and metrics for the same 24 manually
labelled Pierce pages (`p0024` through `p0047`). Each engine was run in two
modes: one full-page input per page and OCR over 188 stored non-figure
PP-DocLayoutV3 regions.

| Folder | Engine | Status |
|---|---|---|
| `deepseek-ocr-2/` | `deepseek-ai/DeepSeek-OCR-2` | Complete 24-page historical run |
| `glm-ocr/` | `zai-org/GLM-OCR` | Complete 24-page run |
| `trocr-large-printed/` | `microsoft/trocr-large-printed` | Complete but poor-performing run |
| `mineru-2604-heldout/` | `opendatalab/MinerU2.5-Pro-2604-1.2B` | Complete historical run; superseded by the 2605 full-book output |

Each mode contains `pages.jsonl`, `regions.jsonl`, and `metrics.json`; each
engine root contains `comparison.json`. Metrics are recomputed from the saved
page text and the held-out labels.

The old PaddleOCR output is intentionally excluded because it produced
foreign-script garbage and is not a valid PP-OCRv6 benchmark. It will be added
only after the corrected runner succeeds. Large layout/runtime ZIP files are
also excluded because their contents already live elsewhere or are external
runtime assets.

These model outputs are research evidence, not ground truth. The 24 held-out
labels provide the evaluation reference.
