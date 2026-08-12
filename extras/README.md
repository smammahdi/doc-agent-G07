# A2 team workspace

`extras/` holds team experiments and reproducibility tools that support the
fixed starter implementation. Production code remains under `src/doc_agent/`;
these files are not imported by the runtime pipeline.

## Contents

| Path | Purpose |
|---|---|
| `implementation-plan.md` | Partner-owned A2 held-out OCR and index plan |
| `kaggle_heldout_score.ipynb` | Partner-owned held-out scoring notebook |
| `kaggle_ocr_comparison.ipynb` | Partner-owned OCR comparison notebook |
| `layout_research/` | Mahdi's figure extraction, detector comparison, and Chandra-reference evaluation code |
| `ocr_research/` | Mahdi's OCR bake-off and offline Document AI reference-generation code |
| `chandra_research/` | Mahdi's direct Kaggle Chandra notebook and output-normalization utilities |

Each research directory has its own README with commands and evidence limits.
The original partner files above are preserved unchanged.

## External layout artifact dataset

The generated full-book layout records are packaged outside Git in the Kaggle
[dataset `cruelangelssprint/pierce-1890-figure-and-ocr-outputs`](https://www.kaggle.com/datasets/cruelangelssprint/pierce-1890-figure-and-ocr-outputs),
currently at version 3. The package contains the Chandra block output used as
the provisional reference, the crop-oriented detector outputs, and the
page-complete 150-DPI rerun records. The dataset is an artifact release, not a
runtime dependency.

## Repository boundary

Keep source code, small notebooks, and documentation here. Do not commit:

- the 1,034-page source PDF;
- rendered pages, crops, JSONL run outputs, or vector indexes;
- model checkpoints, Kaggle bundles, wheels, or virtual environments;
- `.env` files, cloud credentials, Kaggle keys, or GitHub tokens.

Those artifacts should be regenerated with the documented tools or shared as
an external dataset. Chandra and Document AI outputs are provisional
references until a person verifies the selected held-out pages. The Kaggle
package does not change that evidence boundary.
