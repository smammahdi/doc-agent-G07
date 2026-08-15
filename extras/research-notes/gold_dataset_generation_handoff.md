# Verified Evaluation Q&A Dataset — Agent Handoff Specification

## 1. Mission Overview
You are tasked with generating a high-quality, independently verified **evaluation dataset of 20–25 question-answer pairs** for the Group 07 (**G07**) Document Agent pipeline.

The evaluation dataset must be authored and saved strictly into:
👉 [**`grading_kit/tasks.jsonl`**](../../grading_kit/tasks.jsonl)

This dataset directly measures and benchmarks:
1. **Stage 4 & 5 Information Retrieval**: `Recall@1`, `Recall@5`, `Recall@10`, and `MRR`.
2. **Stage 5 Cross-Encoder Reranking**: Rank improvement of reference passages.
3. **Stage 6 Agentic Reasoning & Re-Search**: Evidence-gated widening triggers (`needs_research: true` vs. `false`).
4. **Group 07 Primary NFR (`explainable`)**: 100% accurate page citations verifiable in `<30` seconds.

---

## 2. Source Data & File Locations to Inspect

The source corpus is **R. V. Pierce, *The People's Common Sense Medical Adviser* (1890)** (1,034 pages of historical medical advice, anatomy, physiology, and herbal remedies).

Traverse and read evidence from these locations:
- **Verified Ground-Truth Held-Out Text**: [`grading_kit/labels.jsonl`](../../grading_kit/labels.jsonl) (contains clean transcriptions for `p0024` through `p0047`).
- **Scanned Page Images**: [`grading_kit/heldout_pages/`](../../grading_kit/heldout_pages/) (contains high-resolution page scan JPEGs).
- **Chandra OCR Markdown & Blocks**: `data/raw/` or `development/chandra_kaggle/` (or parsed text in `src/doc_agent/index/chunk.py`).
- **Figure & Diagram Index**: `data/processed/index/image_index.json` or `extras/layout-benchmarks/outputs/heldout-visualizations/` (350 medical illustration boxes across 252 pages, e.g., Fig. 4 cell nucleus, Fig. 10 skull bones, Fig. 15 pelvic structure, Fig. 28 stomach coats).

---

## 3. Strict Output Schema (`grading_kit/tasks.jsonl`)

Every line in `grading_kit/tasks.jsonl` must be a valid, standalone JSON object with **no trailing comments or invalid keys**:

```json
{"id": "t01", "question": "...", "image_paths": [], "verifiable": true, "judged": false, "gold": "...", "gold_pages": ["p0030"], "needs_research": false}
```

### Field Definitions:
- `id`: Sequential unique string (`"t01"`, `"t02"`, ..., `"t25"`).
- `question`: Natural, unambiguous English query that a real user could plausibly ask. Do not mention page IDs, the dataset, evaluation instructions, the source author, or hidden provenance. Avoid first-person, third-person, and other meta framing.
- `image_paths`: Repository-relative page-image inputs. Use `[]` for text-only tasks and one or more existing paths for genuinely multimodal tasks.
- `verifiable`: `true` for factual questions with a concrete answer; `false` for open-ended synthesis.
- `judged`: `true` only if evaluated via LLM-as-judge (typically `false` for verifiable facts).
- `gold`: Legacy machine-compatible key containing the verified reference answer. The word “gold” must not leak into the user-facing question or answer.
- `gold_pages`: List of page IDs where the evidence lives (e.g., `["p0027"]` or `["p0024", "p0074"]`).
- `needs_research`:
  - `false` (**Control Item**): Single-hop question; the answer is fully contained in a single top chunk.
  - `true` (**Trigger Item**): Multi-hop / comparative question; requires retrieving multiple disparate pages/chapters to synthesize the answer.

Reference answers must be as complete as their questions require. Do not compress a multi-part synthesis into a factoid-sized response, and do not pad a direct fact with irrelevant detail. As review guidelines rather than fixed quotas, single-hop and OCR items will often need 60–140 words, multimodal items 80–150 words, multi-hop items 120–220 words, and supported abstentions 45–90 words.

---

## 4. Question Archetype Distribution (Total: 20–25 Items)

To test a variety of pipeline capabilities, author questions across these **5 mandatory categories**:

### A. Single-Hop Factoid & Anatomical Queries (~5 items)
- **Goal**: Measure baseline top-1 and top-3 retrieval precision (`Recall@1`).
- **Characteristics**: Direct, unambiguous anatomical or physiological facts.
- **Example**:
  ```json
  {"id": "t01", "question": "How many distinct bones make up the human skeleton, and what are its four main divisions?", "image_paths": [], "verifiable": true, "judged": false, "gold": "Excluding the teeth, the skeleton contains 200 distinct bones. They are grouped into the Head, Trunk, Upper Extremities, and Lower Extremities.", "gold_pages": ["p0030"], "needs_research": false}
  ```

### B. Multi-Hop Synthesis & Re-Search Triggers (~5 items)
- **Goal**: Test the A3 agentic loop (`needs_research: true`) where the top chunk is insufficient.
- **Characteristics**: Requires connecting evidence from two or more distinct pages. The page IDs belong only in `gold_pages`, never in the question.
- **Example**:
  ```json
  {"id": "t06", "question": "How is the hip's ball-and-socket joint formed, and how do cartilage, ligaments, the synovial membrane, and synovia complete and lubricate it?", "image_paths": [], "verifiable": true, "judged": false, "gold": "The acetabulum forms the socket and receives the rounded head of the femur. Cartilage provides smooth, elastic articular surfaces; ligaments bind the bones; and the synovial membrane covers the cartilages and reflects onto the ligaments to form a closed capsule. That membrane secretes synovia, which lubricates the contacting surfaces and lets them move freely over one another.", "gold_pages": ["p0033", "p0035", "p0036", "p0037"], "needs_research": true}
  ```

### C. Multimodal & Figure-Grounded Queries (~5 items)
- **Goal**: Evaluate Vision-Language embeddings (`Qwen3-VL-Embedding-2B`) and diagram linking.
- **Characteristics**: Relies on an attached figure or illustration and cannot be answered completely from the question text alone. Refer naturally to the attached diagram rather than asking the user to know a figure number or page number.
- **Example**:
  ```json
  {"id": "t11", "question": "In the attached stomach illustration, which labeled opening receives the esophagus, and which opening leads into the beginning of the duodenum?", "image_paths": ["grading_kit/heldout_pages/p0047.jpg"], "verifiable": true, "judged": false, "gold": "The esophagus enters through the cardiac orifice, the stomach's inlet. At the other end, the pyloric orifice forms the outlet into the beginning of the duodenum.", "gold_pages": ["p0047"], "needs_research": false}
  ```

### D. Dirty-OCR & Archaic Medical Terminology (~5 items)
- **Goal**: Test robustness to historical 1890s medical jargon, Latin botanical names, and degraded scan text.
- **Characteristics**: Involves archaic terms like *Haversian canals*, *Hydrastis Canadensis*, *lamellæ*, *sarcolemma*, *alveoli*.
- **Example**:
  ```json
  {"id": "t16", "question": "How do Haversian canals, centers of ossification, and the periosteum contribute to a bone's nourishment, formation, and protection?", "image_paths": [], "verifiable": true, "judged": false, "gold": "Haversian canals carry blood-vessels that nourish bone. Bone formation begins at distinct centers of ossification as cartilaginous material changes into bony tissue. The periosteum is the external membranous envelope that covers and protects the bone.", "gold_pages": ["p0035"], "needs_research": false}
  ```

### E. Negative / Out-of-Corpus Questions (~3–5 items)
- **Goal**: Test **Abstention and Anti-Hallucination Guardrails**.
- **Characteristics**: Questions regarding modern medical treatments that did not exist in 1890 (e.g., Penicillin, MRI scans, modern antibiotics, insulin therapy). The system must abstain.
- **Example**:
  ```json
  {"id": "t21", "question": "What amoxicillin dose, dosing frequency, and treatment duration are appropriate for an adult with bacterial sinusitis?", "image_paths": [], "verifiable": true, "judged": false, "gold": "No supported amoxicillin regimen is available because the material contains no mention of amoxicillin, penicillin, or modern antibiotics. A dose, frequency, or duration cannot be supplied without inventing information; current clinical prescribing guidance is required.", "gold_pages": [], "needs_research": false}
  ```

---

## 5. Independent Author-Verifier Workflow

Every question must be created and verified by its own independent agent. Do not have one agent mass-produce a category, and do not accept a question that another agent merely rewrote from a shared template.

1. **Assign one bounded brief per agent**: Give the agent one target archetype and a small evidence scope, but do not supply a finished question or answer.
2. **Run bounded parallel waves**: Use up to five author-verifiers concurrently when the platform permits. If the concurrency cap is lower, use the maximum available and continue in waves without weakening the one-agent-per-question rule.
3. **Require direct inspection**: Each agent must read the relevant `labels.jsonl` transcription and inspect the corresponding scan. A multimodal author must reason from the image itself. A negative-item author must search the full available OCR, not merely the held-out slice.
4. **Require an evidence packet**: Each agent returns one natural question, one appropriately detailed answer, exact page provenance, image paths when applicable, and a concise note explaining what was checked.
5. **Root acceptance review**: The orchestrator rejects unsupported claims, accidental page/source language in questions, shallow answers, fake multi-hop items, missing images, and ambiguous prompts. It then assigns sequential IDs and writes the JSONL file.

The authoring agent is also responsible for verification, so accountability remains one-to-one. The root review is an additional acceptance gate, not a substitute for that independent verification.

---

## 6. Verification Checklist Before Completion
- [ ] Exactly 20 to 25 items generated in `grading_kit/tasks.jsonl`.
- [ ] Contains all 5 question archetypes (Factoid, Multi-hop, Multimodal, Archaic, Negative).
- [ ] Contains at least 3 `{needs_research: true}` trigger questions.
- [ ] Each item was independently authored and source-verified by one agent; no agent mass-authored a category.
- [ ] Questions sound like plausible user queries and contain no page IDs, source-author references, dataset/evaluation language, or first-/third-person meta framing.
- [ ] Reference-answer depth is proportional to the question; multi-part and multi-hop prompts receive complete synthesis rather than terse factoids.
- [ ] Every non-negative item has valid `gold_pages` matching real pages (e.g. `p0027`).
- [ ] Every item contains `image_paths`; multimodal items reference existing held-out page images and text-only items use `[]`.
- [ ] Every multimodal question genuinely depends on its attached image, and every listed image path exists.
- [ ] Python JSON validator passes: `python3 -c "import json; [json.loads(line) for line in open('grading_kit/tasks.jsonl') if line.strip() and not line.startswith('#')]"`
- [ ] `git diff --check` passes with no syntax or lint errors.
