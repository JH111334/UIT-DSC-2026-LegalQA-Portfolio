# Experiment Summary & Ablation Logs — Vietnamese LegalQA

## 1. Retrieval Ablation Results

Offline retrieval benchmarks were conducted on a validation split using diagnostic qrels extracted from ground-truth citations.

| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR | Latency / Query |
|---|---|---|---|---|---|
| BM25 Baseline | 0.462 | 0.691 | 0.748 | 0.541 | ~12 ms |
| Dense E5-small Baseline | 0.514 | 0.732 | 0.795 | 0.598 | ~45 ms |
| BM25 + Dense E5 (Score Summation) | 0.528 | 0.751 | 0.812 | 0.612 | ~58 ms |
| **BM25 + Dense E5 (RRF k=60)** | **0.572** | **0.798** | **0.854** | **0.665** | ~58 ms |
| **RRF + Cross-Encoder Rerank (Top 50 -> 5)** | **0.618** | **0.834** | **0.871** | **0.702** | ~142 ms |

### Key Findings
1. **RRF Superiority over Score Summation:** Raw score summation required tuning arbitrary weights between BM25 unbounded scores and cosine similarity, often collapsing under out-of-distribution queries. RRF proved consistently robust across all question lengths.
2. **Reranker Impact:** Cross-encoder reranking yielded the largest gain in Recall@1 (+4.6 percentage points), which directly reduced generator context dilution.

---

## 2. Generator Training Dynamics (QLoRA)

| Setting | Value | Rationale |
|---|---|---|
| Base Model | Qwen2.5-1.5B-Instruct | Optimal parameter efficiency under single-GPU limits |
| Precision | 4-bit NF4 Quantization (bitsandbytes) | Fits forward/backward passes within 16GB VRAM |
| LoRA Rank / Alpha | $r = 16, \alpha = 32$ | Balances adapter expressiveness and stability |
| Learning Rate | $2 \times 10^{-4}$ (Cosine schedule) | Prevents catastrophic forgetting of base legal knowledge |
| Loss Masking | Answer-only (`labels = -100` on prompt) | Eliminates gradient updates on context/prompt tokens |
| Batch Size / Grad Accum | Effective batch size = 16 | Stable gradient estimation under memory bounds |
