# Document Agent R&D Workspace (`extras/`)

This directory contains the experimental benchmarks, evaluation harnesses, offline asset packagers, and research notes for the **Historical Medical Document Agent** project (*People's Common Sense Medical Adviser*, Pierce, 1890).

---

## 1. Directory Overview & Navigation

```
extras/
├── README.md                   # Master index and navigation guide
├── ocr-benchmarks/             # Stages 1–3: OCR Engines, Layout Segmentation, & Held-Out Evaluations
│   ├── notebooks/              # Interactive Kaggle evaluation notebooks
│   ├── engines/                # Isolated benchmark harnesses for OCR/layout engines (Chandra, Tesseract, etc.)
│   └── reports/                # Quantitative CER/WER comparison reports and benchmark artifacts
├── indexing-benchmarks/        # Stage 4: Chunking Strategies, Embedding Models, & FAISS Vector Search
│   ├── code/                   # Reusable Python benchmark package
│   ├── data/                   # Canonical 1,034-page corpus & 60-query grounded retrieval benchmark
│   ├── notebooks/              # Dev selection (`stage4-dev-selection.ipynb`) & Final evidence notebooks
│   ├── bundles/                # Packaged Kaggle input ZIPs (`indexing-benchmark-data.zip`)
│   └── results/                # Validated output JSON reports and per-query retrieval logs
├── offline-asset-builders/     # Automated wheel packagers & model weight downloaders for offline Kaggle runs
│   └── embedding-indexing/     # Build scripts for FAISS, SentenceTransformers, and embedding weights
├── output/                     # Frozen raw layout and OCR block extractions from document processors
│   └── chandra/                # Full-book 1,034-page Chandra OCR blocks (`chunks.jsonl`)
└── research-notes/             # Internal developer guides, handoff summaries, and design specifications
    ├── implementation-plan.md
    ├── gold_dataset_generation_handoff.md
    ├── stage4_indexing_and_retrieval_guide.md
    └── stage4_options_tradeoffs_and_alternatives.md
```

---

## 2. Key Experimental Stages

### Stage 1–3: Vision, Layout, & OCR Benchmarks ([`ocr-benchmarks/`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/extras/ocr-benchmarks/))
- Evaluates full-page vs layout-guided OCR on the 24-page human-transcribed held-out test set (`p0024–p0047`).
- Engines compared: **Chandra OCR**, **PaddleOCR**, **Tesseract**, **EasyOCR**, **Florence-2**, and **MinerU**.
- Winning pipeline: **Chandra OCR** with structured layout block parsing.

### Stage 4: Chunking, Embedding Models, & FAISS Vector Indexing ([`indexing-benchmarks/`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/extras/indexing-benchmarks/))
- Evaluates a **5x5 Factorial Grid** (5 Embedding Models x 5 Chunking Strategies) across the entire 1,034-page canonical corpus.
- Grounded on 60 verified whole-book retrieval queries with exact character answer spans.
- Vector search comparison: **`IndexFlatIP`** vs **`IndexHNSWFlat`** vs **`IndexIVFFlat`** using inner product on normalized embeddings.

### Offline Deployment ([`offline-asset-builders/`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/extras/offline-asset-builders/))
- Bundles Python `.whl` dependencies and HuggingFace model checkpoints for 100% offline, reproducible Kaggle execution.
