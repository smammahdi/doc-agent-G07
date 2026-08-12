# Chandra layout research

This directory contains the direct Kaggle experiment used to evaluate Chandra
on the Pierce book and the small utilities needed to prepare a real-page sample
or normalize Chandra output.

Included:

- `chandra_kaggle.ipynb`: the executed direct Kaggle run notebook.
- `make_sample.py`: builds a real-page PDF and a page-number manifest.
- `parse_chandra.py`: converts Chandra figure-like blocks into the normalized
  figure schema used by the layout experiments.

Not included are the offline model bundle, offline installation notebooks,
downloaded weights, wheels, synthetic notebook tests, or generated Chandra
outputs. Those are environment setup or data artifacts, not repository source.

Build a sample from the canonical corpus:

```bash
bash scripts/get_data.sh
python extras/chandra_research/make_sample.py --pages 34,74
```

The sample PDF and manifest are generated under `sample/` and ignored by Git.
The notebook can be uploaded to Kaggle and pointed at the Pierce dataset.

Normalize an unzipped Chandra CLI metadata directory:

```bash
python extras/chandra_research/parse_chandra.py /path/to/chandra-output \
  --pdf data/raw/pierce-peoples-common-sense-medical-adviser-1890.pdf \
  --out /tmp/chandra-figures.jsonl
```

The direct full-book artifact used by the runtime has a different,
block-preserving `chunks.jsonl` schema (`book_page`, `page_box`, `bbox`,
`label`, `content`). `src/doc_agent/vision/layout.py` reads that schema
directly. Always normalize each block by its own `page_box`; the fold-out pages
do not share the common portrait dimensions.

Chandra is currently a provisional teacher/reference. Its 353 figure-like
blocks are useful for detector agreement experiments, but they are not a
substitute for human-checked layout boxes.
