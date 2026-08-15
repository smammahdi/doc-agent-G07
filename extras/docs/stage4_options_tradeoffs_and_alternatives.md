# Stage 4 Architecture: Options Considered, Current Choices & Future Alternatives

This document analyzes the engineering trade-offs for **Chunking**, **Sentence Embeddings**, and **Vector Indexing** for *The People's Common Sense Medical Adviser* (1890) knowledge base. It details what we evaluated, what we implemented for Milestone A2, and advanced alternatives for Milestone A3.

---

## 1. Chunking Strategies

Chunking divides long-form OCR page text into semantic units for dense vector retrieval while preserving verifiable citation links (`page_ids`).

### Options We Considered & Evaluated

| Strategy | Mechanism | Pros | Cons | Decision |
|---|---|---|---|:---:|
| **Fixed-Character Chunking** | Slices text strictly every $N$ characters (e.g., 1000 chars). | Trivial to implement. | Cuts words, anatomical terms, and sentences mid-syllable. | **Rejected** |
| **Page-Level Chunking** | 1 full page = 1 chunk. | 1:1 page citation alignment. | Pages average 400–600 words; dense vectors lose specific paragraph/dosage nuance. | **Rejected** |
| **Sliding-Window Token Chunking** | **256 whitespace tokens with 32-token overlap** (step = 224 tokens). | Captures complete multi-sentence medical prescriptions without truncation; guaranteed overlap prevents boundary fact loss. | Fixed window size does not dynamically adjust to chapter/header boundaries. | **Chosen for A2** |

### Advanced Options That Could Perform Better (A3 Candidates)

1. **Hierarchical / Parent-Child Chunking**:
   * *Mechanism*: Split text into small **128-token child chunks** for vector search, but link each child to a larger **512-token parent chunk** (or complete subsection).
   * *Benefit*: Vector search achieves higher cosine similarity on compact child chunks, while the LLM receives the full parent chunk with complete clinical dosage context.
2. **Semantic / Header-Aware Chunking (`MarkdownHeaderTextSplitter`)**:
   * *Mechanism*: Split text dynamically at Markdown headers (`CHAPTER II`, `THE BONES`, `Fig. X`).
   * *Benefit*: Guarantees that distinct medical conditions or anatomical systems are never blended across chapter borders into the same chunk.
3. **Late Chunking (Contextual Embeddings)**:
   * *Mechanism*: Pass the entire page through a long-context transformer encoder first, then pool tokens into chunk boundaries.
   * *Benefit*: Chunks inherit global document context from the full self-attention matrix before indexing.

---

## 2. Embedding Models (Dense Representation)

Embedding models project textual chunks into a high-dimensional vector space $\mathbb{R}^d$ where cosine distance approximates semantic similarity.

### Options We Considered & Evaluated

| Model | Dimensions | MTEB Score | Pros | Cons | Decision |
|---|:---:|:---:|---|---|:---:|
| **`sentence-transformers/all-MiniLM-L6-v2`** | **384** | **41.95** | **Ultra-lightweight (80MB), fast batch encoding (~30s for full 1,034-page book on CPU/Mac), robust generic semantic retrieval.** | Slightly lower accuracy on obscure 19th-century medical jargon. | **Chosen for A2** |
| **`BAAI/bge-small-en-v1.5`** | 384 | 51.68 | Same compact 384 dimensions with higher MTEB benchmark accuracy. | Requires prepending instruction prefixes to queries (`"Represent this sentence for searching relevant passages:"`). | **Strong A3 Alternative** |
| **`intfloat/e5-small-v2`** | 384 | 50.56 | High retrieval benchmark performance. | Requires strict `"query: "` and `"passage: "` prefix conventions. | **Considered** |

### Advanced Options That Could Perform Better (A3 Candidates)

1. **Biomedical-Specific Embeddings (`BioLinkBERT` / `MedCPT`)**:
   * *Benefit*: Pretrained on biomedical literature; inherently understands historical disease nomenclature (*consumption*, *scrofula*, *cholera infantum*) better than general web models.
2. **Larger Capacity Dense Models (`BAAI/bge-base-en-v1.5` - 768-d)**:
   * *Benefit*: Doubling dimensionality (768 dimensions) captures finer distinctions between subtle symptom presentations and organ subsystems.
3. **Late-Interaction Embeddings (ColBERTv2 / PLAID)**:
   * *Benefit*: Computes token-level `MaxSim` dot products instead of single-vector pooling. Excels at exact matching for rare 19th-century botanical ingredients (*"pleurisy-root"*, *"boneset"*).

---

## 3. Vector Index & Search Strategy

The vector index performs efficient nearest-neighbor search over the chunk embedding matrix.

### Options We Considered & Evaluated

| Index Type | Search Algorithm | Exact Recall? | Latency (1,944 chunks) | Failure Modes / Trade-offs | Decision |
|---|---|:---:|:---:|---|:---:|
| **FAISS `IndexHNSWFlat`** | Graph-based approximate search ($M=32, ef=64$). | No (Approximate) | $\sim 0.5\text{ ms}$ | Causes OpenMP destructor crashes (SIGSEGV 139) on macOS ARM64; overkill for $<100\text{k}$ vectors. | **Rejected** |
| **FAISS `IndexIVFFlat` / `IVFPQ`** | Inverted lists + product quantization. | No (Lossy) | $\sim 0.8\text{ ms}$ | Vector compression introduces quantization noise; unnecessary for 1,944 chunks. | **Rejected** |
| **FAISS `IndexFlatIP`** | **Brute-force inner product on $L_2$-normalized vectors.** | **100% Exact** | **$< 0.3\text{ ms}$** | Scales linearly $\mathcal{O}(N)$; optimal for our 1,944-chunk corpus with zero crash risk. | **Chosen for A2** |

### Mathematical Basis for `IndexFlatIP`:
By normalizing chunk vectors $\mathbf{d}$ and query vectors $\mathbf{q}$ to unit length ($L_2$-norm):
$$\hat{\mathbf{q}} = \frac{\mathbf{q}}{\|\mathbf{q}\|_2}, \quad \hat{\mathbf{d}} = \frac{\mathbf{d}}{\|\mathbf{d}\|_2}$$
The inner product directly equals cosine similarity:
$$\langle \hat{\mathbf{q}}, \hat{\mathbf{d}} \rangle = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2} = \cos(\theta_{\mathbf{q}, \mathbf{d}})$$

---

### Advanced Retrieval Upgrades for Milestone A3

1. **Hybrid Sparse + Dense Search (BM25 + FAISS via Reciprocal Rank Fusion)**:
   * *Why it's needed*: Dense vectors excel at conceptual similarity but can miss exact keyword lookups (e.g. searching exact figure numbers *"Fig. 24"* or Latin botanical names). BM25 handles lexical precision while FAISS handles semantic meaning.
2. **Two-Stage Retrieval with Cross-Encoder Reranker (`cross-encoder/ms-marco-MiniLM-L6-v2`)**:
   * *How it works*: 
     * **Stage 1 (High Recall)**: FAISS retrieves top 20 candidate chunks.
     * **Stage 2 (High Precision)**: Cross-encoder evaluates the full cross-attention between `(query, chunk)` tokens to re-rank the top 3 chunks for the LLM.
   * *Status*: This is the scheduled core feature for **Stage 5 (Retrieval & Reranking) in A3**.

---

## 4. Current Configuration vs. A3 Upgrade Roadmap

```
Current Production Index (Milestone A2 - Implemented & Verified)
├── OCR Engine : MinerU Full-Page SOTA (98.28% Word-F1)
├── Visuals    : Chandra Visual Registry (350 .webp figure crops across 252 pages)
├── Chunking   : Sliding-Window Tokens (256 size, 32 overlap)
├── Embedding  : sentence-transformers/all-MiniLM-L6-v2 (384-d, L2-normalized)
└── Vector DB  : FAISS IndexFlatIP (100% exact recall, <1ms query latency)

Planned Upgrades (Milestone A3 - Agentic RAG)
├── Chunking   : Hierarchical Parent-Child or Section-Aware Header Splitting
├── Embeddings : BGE-small-en-v1.5 / MedCPT
├── Retrieval  : Hybrid Search (BM25 Lexical + FAISS Dense)
└── Reranking  : Stage 5 Cross-Encoder (ms-marco-MiniLM-L6-v2)
```
