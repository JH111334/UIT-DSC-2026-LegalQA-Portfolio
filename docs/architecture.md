# Architecture Specification — Vietnamese LegalQA

## 1. System Pipeline Stages

The system operates across three decoupled modules:

### Stage 1: Document Processing and Chunking
Statutory legal documents in Vietnam adhere to a hierarchical schema (Bộ luật / Luật -> Chương -> Mục -> Điều -> Khoản -> Điểm). 
- Chunking strategy: Segmenting text while preserving the ancestral hierarchy in metadata (Document Title + Decree Number + Article Number).
- Total chunk count: 316,100 legal text chunks derived from 8,532 official documents.
- Deduplication and boilerplate removal: Eliminating repetitive legal formatting and introductory notices.

### Stage 2: Multi-Modal Hybrid Retrieval
- **Lexical Stream:** BM25 implementation tuned for Vietnamese tokenized legal queries ($k_1 = 1.5, b = 0.75$).
- **Dense Semantic Stream:** Multilingual-E5 (`intfloat/multilingual-e5-small`) generating 384-dimensional dense vectors.
- **Fusion Operator:** Reciprocal Rank Fusion (RRF) with rank penalty $k = 60$:
  $$\text{RRF}(d) = \sum_{m \in M} \frac{1}{60 + \text{rank}_m(d)}$$
- **Reranker:** Cross-attention transformer rescoring the top-50 merged candidates to produce the final top-5 context chunks.

### Stage 3: Evidence-Aware Context Assembly & Generation
- Prompt template enforcing explicit citation constraints:
  `[Context]: {Retrieved Legal Chunks} [Question]: {Legal Query} [Answer Format]: {Direct legal reasoning followed by cited Article and Decree}`
- Base LLM: Qwen2.5-1.5B Instruct.
- Quantization & Adaptation: 4-bit NormalFloat (NF4) with LoRA ($r=16, \alpha=32$) applied to attention projection matrices (`q_proj`, `v_proj`, `k_proj`, `o_proj`).
- Target masking: Cross-entropy loss computed strictly over the answer tokens by setting target labels of prompt tokens to `-100`.
