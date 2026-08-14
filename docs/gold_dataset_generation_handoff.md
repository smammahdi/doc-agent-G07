# Gold Q&A Dataset Generation — Agent Handoff Specification

## 1. Mission Overview
You are tasked with generating a high-quality, verified **Gold Evaluation Dataset of 20–25 Question-Answer Pairs** for the Group 07 (**G07**) Document Agent pipeline. 

The evaluation dataset must be authored and saved strictly into:
👉 [**`grading_kit/tasks.jsonl`**](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/grading_kit/tasks.jsonl)

This gold dataset directly measures and benchmarks:
1. **Stage 4 & 5 Information Retrieval**: `Recall@1`, `Recall@5`, `Recall@10`, and `MRR`.
2. **Stage 5 Cross-Encoder Reranking**: Rank improvement of gold passages.
3. **Stage 6 Agentic Reasoning & Re-Search**: Evidence-gated widening triggers (`needs_research: true` vs. `false`).
4. **Group 07 Primary NFR (`explainable`)**: 100% accurate page citations verifiable in `<30` seconds.

---

## 2. Source Data & File Locations to Inspect

The source corpus is **R. V. Pierce, *The People's Common Sense Medical Adviser* (1890)** (1,034 pages of historical medical advice, anatomy, physiology, and herbal remedies).

Traverse and read evidence from these locations:
- **Verified Ground-Truth Held-Out Text**: [`grading_kit/labels.jsonl`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/grading_kit/labels.jsonl) (contains clean transcriptions for `p0024` through `p0047`).
- **Scanned Page Images**: [`grading_kit/heldout_pages/`](file:///Users/smammahdi/CSE_stuffs/Project/DL%20Project/doc-agent-starter/grading_kit/heldout_pages) (contains high-resolution page scan JPEGs).
- **Chandra OCR Markdown & Blocks**: `data/raw/` or `development/chandra_kaggle/` (or parsed text in `src/doc_agent/index/chunk.py`).
- **Figure & Diagram Index**: `data/processed/index/image_index.json` or `extras/output/layout-pdfs/` (350 medical illustration boxes across 252 pages, e.g., Fig. 4 cell nucleus, Fig. 10 skull bones, Fig. 15 pelvic structure, Fig. 28 stomach coats).

---

## 3. Strict Output Schema (`grading_kit/tasks.jsonl`)

Every line in `grading_kit/tasks.jsonl` must be a valid, standalone JSON object with **no trailing comments or invalid keys**:

```json
{"id": "t01", "question": "...", "verifiable": true, "judged": false, "gold": "...", "gold_pages": ["p0030"], "needs_research": false}
```

### Field Definitions:
- `id`: Sequential unique string (`"t01"`, `"t02"`, ..., `"t25"`).
- `question`: Natural, unambiguous English query.
- `verifiable`: `true` for factual questions with a concrete answer; `false` for open-ended synthesis.
- `judged`: `true` only if evaluated via LLM-as-judge (typically `false` for verifiable facts).
- `gold`: Ground-truth reference answer extracted directly from the text.
- `gold_pages`: List of page IDs where the evidence lives (e.g., `["p0027"]` or `["p0024", "p0074"]`).
- `needs_research`: 
  - `false` (**Control Item**): Single-hop question; the answer is fully contained in a single top chunk.
  - `true` (**Trigger Item**): Multi-hop / comparative question; requires retrieving multiple disparate pages/chapters to synthesize the answer.

---

## 4. Question Archetype Distribution (Total: 20–25 Items)

To test a variety of pipeline capabilities, author questions across these **5 mandatory categories**:

### A. Single-Hop Factoid & Anatomical Queries (~5 items)
- **Goal**: Measure baseline top-1 and top-3 retrieval precision (`Recall@1`).
- **Characteristics**: Direct, unambiguous anatomical or physiological facts.
- **Example**: 
  ```json
  {"id": "t01", "question": "How many distinct bones are in the human skeleton according to Dr. Pierce, and into what four main divisions are they grouped?", "verifiable": true, "judged": false, "gold": "200 distinct bones (excluding teeth), divided into the Head, Trunk, Upper Extremities, and Lower Extremities.", "gold_pages": ["p0030"], "needs_research": false}
  ```

### B. Multi-Hop Synthesis & Re-Search Triggers (~5 items)
- **Goal**: Test the A3 agentic loop (`needs_research: true`) where the top chunk is insufficient.
- **Characteristics**: Requires connecting two distinct pages (e.g., connecting an anatomical structure on `p0028` with joint dropsy on `p0037`, or connecting a symptom to an herbal tincture).
- **Example**:
  ```json
  {"id": "t06", "question": "What lubricating fluid prevents friction in bone joints, and what medical condition results when it is secreted in excessive quantities?", "verifiable": true, "judged": false, "gold": "Synovia (secreted by the synovial membrane); excessive secretion produces 'dropsy of the joints'.", "gold_pages": ["p0037"], "needs_research": true}
  ```

### C. Multimodal & Figure-Grounded Queries (~5 items)
- **Goal**: Evaluate Vision-Language embeddings (`Qwen3-VL-Embedding-2B`) and diagram linking.
- **Characteristics**: Directly mentions or relies on numbered figures and illustrations (e.g., Fig. 4, Fig. 6, Fig. 10, Fig. 15, Fig. 26, Fig. 28).
- **Example**:
  ```json
  {"id": "t11", "question": "According to the anatomy of the stomach in Fig. 28, what are the two orifices that communicate with the esophagus and the duodenum?", "verifiable": true, "judged": false, "gold": "The cardiac orifice connects with the esophagus, and the pyloric orifice connects with the beginning of the duodenum.", "gold_pages": ["p0047"], "needs_research": false}
  ```

### D. Dirty-OCR & Archaic Medical Terminology (~5 items)
- **Goal**: Test robustness to historical 1890s medical jargon, Latin botanical names, and degraded scan text.
- **Characteristics**: Involves archaic terms like *Haversian canals*, *Hydrastis Canadensis*, *lamellæ*, *sarcolemma*, *alveoli*.
- **Example**:
  ```json
  {"id": "t16", "question": "What is the periosteum, and what role do the Haversian canals serve in bone physiology?", "verifiable": true, "judged": false, "gold": "The periosteum is the external membranous envelope protecting bones; Haversian canals form nutritive channels for blood-vessels to nourish the bone tissue.", "gold_pages": ["p0029", "p0035"], "needs_research": false}
  ```

### E. Negative / Out-of-Corpus Questions (~3–5 items)
- **Goal**: Test **Abstention and Anti-Hallucination Guardrails**.
- **Characteristics**: Questions regarding modern medical treatments that did not exist in 1890 (e.g., Penicillin, MRI scans, modern antibiotics, insulin therapy). The system must abstain.
- **Example**:
  ```json
  {"id": "t21", "question": "What dosage of penicillin or amoxicillin does Dr. Pierce prescribe for bacterial throat infections?", "verifiable": true, "judged": false, "gold": "Abstain: Penicillin and modern antibiotics are not mentioned in the 1890 text.", "gold_pages": [], "needs_research": false}
  ```

---

## 5. Multi-Agent Delegation Strategy

To execute this systematically without collisions or hallucinations, use the following subagent architecture:

```
                          ┌──────────────────────────┐
                          │   MAIN ORCHESTRATOR      │
                          │ • Assigns ID ranges      │
                          │ • Enforces JSONL schema  │
                          │ • Writes tasks.jsonl     │
                          └─────────────┬────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
   ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
   │ SUBAGENT 1        │      │ SUBAGENT 2        │      │ SUBAGENT 3        │
   │ Text & Anatomy    │      │ Visual & Figures  │      │ Critic & Verifier │
   │ Explorer          │      │ Explorer          │      │                   │
   │ • Single-Hop      │      │ • Fig 4, 10, 15,  │      │ • Cross-checks GT │
   │ • Multi-Hop /     │      │   28 diagrams     │      │ • Verifies exact  │
   │   Re-search pairs │      │ • Multimodal QA   │      │   page provenance │
   │ • Archaic terms   │      │ • Image Index     │      │ • Validates <30s  │
   └───────────────────┘      └───────────────────┘      └───────────────────┘
```

1. **Subagent 1 (Text & Anatomy Explorer)**: Reads text chapters (`p0024`–`p0047`), authors factoid, multi-hop (`needs_research: true`), and archaic terminology items.
2. **Subagent 2 (Visual & Multimodal Explorer)**: Inspects figure diagrams, captions, and `image_index.json` to author figure-grounded items.
3. **Subagent 3 (Critic & Grounding Verifier)**: Receives draft Q&A pairs, re-reads the source pages to ensure:
   - Page IDs in `gold_pages` are 100% accurate.
   - The `gold` answer contains only statements directly supported by the text.
   - The question cannot be answered ambiguously.
4. **Main Orchestrator**: Aggregates all validated entries, numbers them sequentially (`t01`..`t25`), verifies JSON formatting, and writes to `grading_kit/tasks.jsonl`.

---

## 6. Verification Checklist Before Completion
- [ ] Exactly 20 to 25 items generated in `grading_kit/tasks.jsonl`.
- [ ] Contains all 5 question archetypes (Factoid, Multi-hop, Multimodal, Archaic, Negative).
- [ ] Contains at least 3 `{needs_research: true}` trigger questions.
- [ ] Every non-negative item has valid `gold_pages` matching real pages (e.g. `p0027`).
- [ ] Python JSON validator passes: `python3 -c "import json; [json.loads(line) for line in open('grading_kit/tasks.jsonl') if line.strip() and not line.startswith('#')]"`
- [ ] `git diff --check` passes with no syntax or lint errors.
