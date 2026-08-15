# Document Agent R&D Workspace (`extras/`)

This directory contains the experimental benchmarks, evaluation harnesses, and research notes for the **Historical Medical Document Agent** project (*People's Common Sense Medical Adviser*, Pierce, 1890).

---

## 1. Directory Overview & Navigation

```
extras/
├── README.md                           # Master index and navigation guide
│
├── ocr-benchmarks/                     # Stages 1–3: OCR Engines, Multimodal Text Recognition, & Evaluation
│   ├── notebooks/                      # Interactive Kaggle evaluation notebooks (Qwen, Heldout)
│   ├── engines/                        # Isolated benchmark harnesses for OCR/layout engines
│   │   ├── chandra/                    # Chandra OCR extraction & parsing scripts
│   │   ├── easyocr/                    # EasyOCR benchmark runner
│   │   ├── florence2_layout/           # Florence-2 vision model benchmark
│   │   ├── mineru/                     # MinerU PDF parser benchmark
│   │   ├── paddleocr_vl/               # PaddleOCR vision-language harness
│   │   ├── tesseract_fullpage/         # Raw full-page Tesseract baseline
│   │   ├── tesseract_layout/           # Layout-guided Tesseract pipeline
│   │   ├── layout_research/            # Figure extraction & detector research
│   │   ├── ocr_research/               # Google Cloud Document AI research
│   │   └── modular_suite/              # Multi-engine comparative runner scripts (compare-results.py)
│   ├── outputs/                        # Standardized OCR predictions by scope:
│   │   ├── heldout/                    # 22-page test set predictions (DeepSeek, EasyOCR, Florence-2, GLM, PaddleOCR, Tesseract, TrOCR, Qwen)
│   │   └── full-book/                  # 1,034-page book extractions (Chandra chunks.jsonl, Document AI, MinerU)
│   └── reports/                        # Quantitative CER/WER/Word-F1 comparison reports & JSON outputs
│
├── layout-benchmarks/                  # Document Layout Vision & Figure Extraction Benchmarks
│   └── outputs/                        # Layout detector bounding boxes & segmentations:
│       ├── doclayout-yolo/             # DocLayout-YOLO detections
│       ├── orphan-ink/                 # Pixel-level orphan ink detections
│       ├── picodet-s/                  # PicoDet-S layout detections
│       ├── ppdoclayout-plus-l/         # PP-DocLayout+L detections
│       ├── ppdoclayout-v3/             # PP-DocLayoutV3 detections
│       └── heldout-visualizations/     # Bounding box visual overlay PDFs across all layout models
│
├── indexing-benchmarks/                # Stage 4: Chunking, Embedding Models, & Vector Search
│   ├── code/                           # Reusable Python package (corpus, queries, models, runner)
│   ├── data/                           # Canonical 1,034-page corpus & 110-query benchmark suite
│   ├── notebooks/                      # Dev selection (`stage4-dev-selection.ipynb`) & Final evidence notebooks
│   ├── bundles/                        # Packaged Kaggle input ZIPs (indexing-benchmark-data.zip)
│   └── results/                        # Validated output reports
│
└── research-notes/                     # Internal developer guides and handoffs
    ├── implementation-plan.md
    ├── gold_dataset_generation_handoff.md
    ├── stage4_indexing_and_retrieval_guide.md
    └── stage4_options_tradeoffs_and_alternatives.md
```

---

## 2. Key Experimental Stages

### Stages 1–3: Vision, Layout, & OCR Benchmarks ([`ocr-benchmarks/`](ocr-benchmarks/))
- Evaluates full-page vs layout-guided OCR on the 22-page human-transcribed held-out test set (`p0024–p0047` excluding illustration-only pages `p0041`, `p0043`).
- Engines compared: **Chandra OCR**, **Qwen3.5**, **MinerU**, **GLM-OCR**, **PaddleOCR**, **Tesseract**, **EasyOCR**, **Florence-2**, **DeepSeek-OCR**, and **TrOCR**.
- Winning pipeline: **Chandra OCR** with structured layout block parsing (0.1232 Macro CER, 0.9872 Macro Word-F1).

### Stage 4: Chunking, Embedding Models, & FAISS Vector Indexing ([`indexing-benchmarks/`](indexing-benchmarks/))
- Evaluates a **5x5 Factorial Grid** (5 Embedding Models x 5 Chunking Strategies) across the entire 1,034-page canonical corpus.
- Grounded on 110 verified queries (80 Dev across 10 regions + 20 Final Test + 10 Out-of-Corpus).
- Vector search comparison: **`IndexFlatIP`** vs **`IndexHNSWFlat`** vs **`IndexIVFFlat`** using inner product on normalized embeddings.
