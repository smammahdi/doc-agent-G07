# Stage 4 Final Evidence Summary
## 1. Locked Production Stack
- **Embedding Model**: `Qwen3-Embedding-0.6B` (1024-d)
- **Chunking Strategy**: `fixed_128_16` (3830 chunks, avg 106.8 words)
- **Vector Search Index**: `IndexFlatIP` (15320.0 KB, build time 16.38s)
- **Abstention Threshold**: `0.5500`
## 2. Final Untouched Test Performance
- **Single-Page Recall@1**: 0.8667
- **Single-Page Recall@5**: 1.0000 (95% CI: (1.0, 1.0))
- **Single-Page MRR@10**: 0.9333
- **Multi-Page Coverage@10**: 0.7667
- **Multi-Page All-Found@10**: 0.6000
## 3. Negative Query Abstention Evaluation
- **Abstention Precision**: 1.0000
- **Abstention Recall**: 1.0000
- **Abstention F1**: 1.0000
- **Abstention Accuracy**: 1.0000
