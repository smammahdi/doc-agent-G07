# Document Agent R&D Workspace (`extras/`)

This directory contains the experimental benchmarks, evaluation harnesses, and research notes for the **Historical Medical Document Agent** project (*People's Common Sense Medical Adviser*, Pierce, 1890).

---

## 1. Directory Overview & Navigation

```
extras/
├── README.md                           # Master index and navigation guide
│
├── ocr-benchmarks/                     # Stages 1–3: OCR Engines, Layout Vision, & Evaluation
│   ├── notebooks/                      # Interactive Kaggle evaluation notebooks (Qwen, Heldout)
│   ├── engines/                        # Isolated benchmark harnesses for OCR/layout engines
│   │   ├── chandra/                    # Chandra OCR extraction & parsing scripts
│   │   ├── easyocr/                    # EasyOCR layout benchmark
│   │   ├── florence2_layout/           # Florence-2 vision model benchmark
│   │   ├── mineru/                     # MinerU PDF parser benchmark
│   │   ├── paddleocr_vl/               # PaddleOCR vision-language harness
│   │   ├── tesseract_fullpage/         # Raw full-page Tesseract baseline
│   │   ├── tesseract_layout/           # Layout-guided Tesseract pipeline
│   │   ├── layout_research/            # Figure extraction & detector research
│   │   ├── ocr_research/               # Google Cloud Document AI research
│   │   └── modular_suite/              # Multi-engine comparative runner scripts
│   ├── reports/                        # Quantitative CER/WER comparison reports & JSON outputs
│   └── extractions/                    # Frozen layout & OCR transcripts (labeled by scope):
│       ├── chandra (full book)/        # 1,034-page Chandra layout/OCR blocks (chunks.jsonl)
│       ├── mineru (full book)/         # 1,034-page MinerU OCR extraction
│       ├── ppdoclayout-v3 (heldout set)/ # PP-DocLayoutV3 on held-out pages
│       ├── doclayout-yolo (heldout set)/ # DocLayout-YOLO on held-out pages
│       ├── ppdoclayout-plus-l (heldout set)/ # PP-DocLayout+L on held-out pages
│       ├── picodet-s (heldout set)/    # PicoDet-S on held-out pages
│       ├── orphan-ink (heldout set)/   # Orphan ink segmentation on held-out pages
│       ├── document-ai (heldout set)/  # Document AI on held-out pages
│       └── layout-pdfs (heldout set)/  # Layout PDFs on held-out pages
│
├── indexing-benchmarks/                # Stage 4: Chunking, Embedding Models, & Vector Search
│   ├── code/                           # Reusable Python package (corpus, queries, models, runner)
│   ├── data/                           # Canonical 1,034-page corpus & 60-query benchmark suite
│   ├── notebooks/                      # Dev selection & untouched Final evidence notebooks
│   ├── bundles/                        # Packaged Kaggle input ZIPs (indexing-benchmark-data.zip)
│   └── results/                        # Validated output reports
│
└── research-notes/                     # Internal developer guides, handoffs, & offline builders
    ├── implementation-plan.md
    ├── gold_dataset_generation_handoff.md
    ├── stage4_indexing_and_retrieval_guide.md
    ├── stage4_options_tradeoffs_and_alternatives.md
    └── offline-asset-builders/         # Git-ignored wheel packagers & model downloaders
```

---

## 2. Key Experimental Stages

### Stages 1–3: Vision, Layout, & OCR Benchmarks ([`ocr-benchmarks/`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/extras/ocr-benchmarks/))
- Evaluates full-page vs layout-guided OCR on the 24-page human-transcribed held-out test set (`p0024–p0047`).
- Engines compared: **Chandra OCR**, **PaddleOCR**, **Tesseract**, **EasyOCR**, **Florence-2**, and **MinerU**.
- Winning pipeline: **Chandra OCR** with structured layout block parsing.

### Stage 4: Chunking, Embedding Models, & FAISS Vector Indexing ([`indexing-benchmarks/`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/extras/indexing-benchmarks/))
- Evaluates a **5x5 Factorial Grid** (5 Embedding Models x 5 Chunking Strategies) across the entire 1,034-page canonical corpus.
- Grounded on 60 verified whole-book retrieval queries with exact character answer spans.
- Vector search comparison: **`IndexFlatIP`** vs **`IndexHNSWFlat`** vs **`IndexIVFFlat`** using inner product on normalized embeddings.
