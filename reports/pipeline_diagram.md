# Knowledge-base pipeline diagram

This map shows the fixed A2 stage order and separates implemented adapters from
real experiment outputs and pending evidence. The full corpus is not claimed
to have completed the index or retrieval stages.

```mermaid
flowchart LR
    A["Pierce 1890 PDF\n1,034 pages\nsource of truth"] --> B["Loader\nPyMuPDF @ 300 DPI\nPage"]
    B --> C["Preprocess\nidentity baseline"]
    C --> D["Layout\nprojection default\nDocLayout-YOLO offline"]
    D --> E["OCR\nChandra SOTA / Tesseract baseline\nMacro Word-F1: 0.9592"]
    E --> F["Chunk\n128 words / 16 overlap\ntext + page IDs"]
    F --> G["Embed\nQwen3-Embedding-0.6B\n1024-d L2 normalized"]
    G --> H["Store\nFAISS FlatIP\n(3,830 chunks)"]
    A -.-> R["EDA\nall 1,034 pages\n150-DPI measurements"]
    X["Offline experiment outputs\nChandra: 8,544 blocks / 1,028 pages\nDocAI: 419,565 words / 1,016 pages"] -.-> D
    X -.-> E

    classDef implemented fill:#dff2d8,stroke:#4b7f3a,color:#173b12;
    classDef measured fill:#fff1cc,stroke:#a66b00,color:#4a3000;
    classDef pending fill:#f4f4f4,stroke:#777,color:#333;
    class B,C,D,E,F,G,H implemented;
    class A,R,X measured;
```

## Current status

- **Real source and offline outputs:** the Pierce PDF has 1,034 pages; the
  150-DPI EDA ran over all pages. The Chandra reference contains 8,544 blocks
  on 1,028 pages; Document AI reference OCR contains 419,565 words on 1,016
  word-bearing pages.
- **Implemented adapters:** PyMuPDF 300-DPI rendering, identity preprocessing,
  projection layout, DocLayout-YOLO (75MB ONNX), Chandra parser (`load_from_canonical_jsonl`),
  fixed 128-word / 16-word overlap chunking, `Qwen/Qwen3-Embedding-0.6B` (1024-d) dense embedding
  with query instruction prefixing and L2 normalisation, and FAISS `IndexFlatIP` vector store.
- **Index status:** The Stage 4 index is fully built under `data/processed/index/`
  with 3,830 chunks, 1024 embedding dimensions, 100% text corpus coverage (1,016 text pages / 409,102 words),
  and exact cosine similarity search (`faiss:flat_ip`).
- **Scope rule:** no live web search is part of the retrieval pipeline; answers
  remain strictly grounded in the declared Pierce source.

## Contract path

`Page -> Region -> OCR text -> Chunk -> embedding -> vector store` (Chandra/Tesseract
OCR; Qwen3-Embedding-0.6B 1024-d; FAISS FlatIP).
Page IDs survive into each fixed `Chunk`, so a later agent can cite the scanned
page. Coordinates are used at the layout/OCR seam but do not survive in the fixed
`Chunk` contract.
