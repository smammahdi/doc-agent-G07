# Knowledge-base pipeline diagram

This map shows the fixed A2 stage order and separates implemented adapters from
real experiment outputs and pending evidence. The full corpus is not claimed
to have completed the index or retrieval stages.

```mermaid
flowchart LR
    A["Pierce 1890 PDF\n1,034 pages\nsource of truth"] --> B["Loader\nPyMuPDF @ 300 DPI\nPage"]
    B --> C["Preprocess\nidentity baseline"]
    C --> D["Layout\nprojection default\nChandra blocks offline"]
    D --> E["OCR\nTesseract default\nDocument AI reference offline"]
    E --> F["Chunk\n256 tokens / 32 overlap\ntext + page IDs"]
    F --> G["Embed\nMiniLM 384"]
    G --> H["Store\nFAISS HNSW"]
    A -.-> R["EDA\nall 1,034 pages\n150-DPI measurements"]
    X["Offline experiment outputs\nChandra: 8,544 blocks / 1,028 pages\nDocAI: 419,565 words / 1,016 pages"] -.-> D
    X -.-> E

    classDef implemented fill:#dff2d8,stroke:#4b7f3a,color:#173b12;
    classDef measured fill:#fff1cc,stroke:#a66b00,color:#4a3000;
    classDef pending fill:#f4f4f4,stroke:#777,color:#333;
    class B,C,D,E,F implemented;
    class A,R,X measured;
    class G,H pending;
```

## Current status

- **Real source and offline outputs:** the Pierce PDF has 1,034 pages; the
  150-DPI EDA ran over all pages. The Chandra reference contains 8,544 blocks
  on 1,028 pages; Document AI reference OCR contains 419,565 words on 1,016
  word-bearing pages. Generated layout records are kept in the external
  [Kaggle artifact package](https://www.kaggle.com/datasets/cruelangelssprint/pierce-1890-figure-and-ocr-outputs)
  (version 3); they are not runtime dependencies.
- **Implemented adapters:** PyMuPDF 300-DPI rendering, identity preprocessing,
  projection layout, optional offline Chandra Regions, Tesseract, optional
  Document AI reference mapping, and 256/32 Chunking are implemented. Reference
  assignment covers 417,825/419,565 words; 1,740 words on 36 pages remain outside
  every Chandra Region, so full reference coverage is not claimed.
- **Evidence boundary:** Document AI and Chandra outputs are reference or
  pre-annotation inputs, never ground truth. Word boxes are consumed for Region
  mapping, but fixed `Chunk` keeps text and page IDs only. Hand GT, OCR accuracy,
  full index statistics, and retrieval remain pending.
- **Index status:** MiniLM-384 embedding and FAISS HNSW code exists, but no real
  full-corpus index or retrieval result is claimed.
- **Scope rule:** no live web search is part of the retrieval pipeline; answers
  must remain grounded in the declared Pierce source.

## Contract path

`Page -> Region -> OCR text -> Chunk -> embedding -> vector store` (Tesseract
default; optional offline reference).
Page IDs survive into each fixed `Chunk`, so a later agent can cite the scanned
page. Coordinates are used at the layout/OCR seam but do not survive in the fixed
`Chunk` contract.
