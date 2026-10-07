# Vietnamese LegalQA — UIT Data Science Challenge 2026 Portfolio

Technical portfolio and architecture dossier for the Vietnamese LegalQA system developed for the **UIT Data Science Challenge 2026 (DSC 2026)**.

## Verified Results & Competition Context

- **Official Competition:** [UIT Data Science Challenge 2026](https://www.citd.edu.vn/cuoc-thi-uit-data-science-challenge-2026/) (Organized by VNU-HCM University of Information Technology - CITD).
- **Private-Test Leaderboard:** 24th (Private Test Score: 0.5648).
- **Team Setting & Individual Ownership:** The competition submission was submitted under team name `UIT-TinDipLaPo`. Within this collaborative setting, team members explored and iterated separate architectural branches. The candidate independently designed, implemented, evaluated, and benchmarked the end-to-end retrieval, reranking, and generation pipeline documented in this repository.
- **Corpus Scale:** 8,532 statutory documents, segmented into 316,100 legal text retrieval chunks.

> **Data Release Note:** In accordance with academic integrity and competition governance, raw organizer-provided corpus dumps, proprietary training annotations, and private test evaluation sets are omitted from this public repository. This repository hosts candidate-owned architectural contracts, evaluation methodology, ablation summaries, and representative algorithm implementations.

---

## 1. Problem Formulation

Vietnamese LegalQA requires retrieving precise statutory provisions (articles, clauses, decrees, circulars) from a large corpus of unstructured legal texts in response to complex legal questions, followed by synthesizing legally grounded answers accompanied by verifiable legal citations.

Key technical challenges include:
1. **Lexical Mismatch vs. Citation Precision:** Real-world questions often use colloquial phrasing ("đi làm muộn bị trừ lương") while legal texts employ strict terminology ("khấu trừ tiền lương theo Bộ luật Lao động"). Conversely, questions referencing exact article or decree numbers require exact token matching.
2. **Context Window & Compute Bounds:** Legal statutes are lengthy. Passing entire legislative documents directly to an LLM exceeds context limits and causes needle-in-a-haystack degradation.
3. **Retrieval Failure vs. Generation Failure:** Distinguishing whether an incorrect answer stemmed from missing evidence in the top-K context or from LLM hallucination despite having correct evidence.

---

## 2. End-to-End Pipeline Architecture

The system is structured as an evidence-first retrieval-augmented generation (RAG) pipeline:

```
[User Query]
     │
     ├───► [BM25 Lexical Retriever]  ────► Top-K Lexical Candidates
     │
     └───► [Multilingual-E5 Dense]   ────► Top-K Semantic Candidates
                     │                              │
                     └──────────────┬───────────────┘
                                    ▼
                     [Reciprocal Rank Fusion (RRF)]
                                    ▼
                         [Top-M Merged Pool]
                                    ▼
                       [Cross-Encoder Reranker]
                                    ▼
                        [Top-N Relevant Chunks]
                                    ▼
                 [Evidence-Aware Context Construction]
                                    ▼
             [Generator: Qwen2.5-1.5B (4-bit QLoRA SFT)]
                                    ▼
              [Grounded Answer + Legal Citation Output]
```

### A. Preprocessing & Corpus Segmentation
- Cleaned and normalized 8,532 legal documents.
- Segmented statutes into 316,100 structured chunks preserving document provenance, decree numbering, and article headings.

### B. Hybrid Retrieval (BM25 + Dense E5 + RRF)
- **BM25 Lexical Retrieval:** Captures exact statutory terminology, article codes, and decree identifiers.
- **Dense Semantic Retrieval:** Utilizes `multilingual-e5-small` embeddings to capture paraphrased and conceptually aligned questions.
- **Reciprocal Rank Fusion (RRF):** Fuses ranking positions rather than raw uncalibrated scores:
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{k + r_m(d)}$$
  where $k = 60$ acts as a rank-smoothing constant preventing dominance by any single modality.

### C. Reranking & Context Assembly
- Cross-encoder reranking over the top candidates to rescore query-document pairs with joint cross-attention.
- Evidence-first context packing that fits within token limits while maintaining document-level citation references.

### D. Generation via QLoRA SFT
- Base model: Qwen2.5-1.5B (parameters strictly bounded under competition compute profiles).
- Training regime: 4-bit NormalFloat (NF4) quantization with LoRA adapters (QLoRA).
- Loss Masking: Answer-only loss masking with `-100` target labels on instruction and context tokens, ensuring gradient updates only optimize answer generation and citation syntax.

---

## 3. Diagnostic Evaluation & Methodology

To isolate and debug pipeline failure modes, offline evaluation used diagnostic qrels derived from validated citation targets:

| Evaluation Dimension | Metric / Diagnostic | Technical Purpose |
|---|---|---|
| **Retrieval Coverage** | Recall@1, Recall@5, Recall@10 | Measures whether ground-truth statutory evidence is captured in candidate pool. |
| **Ranking Quality** | MRR (Mean Reciprocal Rank), NDCG | Evaluates position quality of relevant statutory chunks before context truncation. |
| **Candidate-Depth Curve** | Recall vs. Top-K Depth | Quantifies diminishing returns of context window expansion versus token cost. |
| **Generation Fidelity** | Exact Citation Match, METEOR | Measures answer correctness conditioned strictly on retrieved evidence. |

### Diagnostic Finding: Retrieval vs. Generation Decoupling
By auditing validation failures against qrels:
- In cases where top-5 retrieval recall failed, generator retraining provided zero benefit; improving retrieval fusion was the necessary fix.
- In cases where retrieval achieved top-1 relevance, hallucinated answers were attributable to context dilution, resolved by answer-only loss masking and structured citation prompting.

---

## 4. Key Failure Modes & Insights

1. **Why Pure BM25 Failed on Conversational Queries:** Layperson terminology resulted in zero lexical overlap with formal legislative texts. Dense retrieval resolved this by projecting queries and statutes into semantic embedding space.
2. **Why Pure Dense Failed on Statutory Identifiers:** Dense embeddings frequently confused numerical decree citations (e.g., "Nghị định 45/2019" vs "Nghị định 45/2020"). BM25 resolved this by enforcing exact lexical match on decree identifiers.
3. **Why Score Summation Failed Without Calibration:** BM25 unbounded scores and cosine similarities operate on fundamentally incompatible distributions. RRF rank fusion eliminated distribution mismatch without requiring temperature calibration.
4. **Compute-Aware Engineering:** Restricting the parameter budget to 1.5B with 4-bit QLoRA enabled rapid, reproducible experimentation on single-GPU environments while maintaining competitive generation quality.

---

## 5. Repository Structure

```text
UIT-DSC-2026-LegalQA-Portfolio/
|-- docs/
|   |-- architecture.md          # Detailed system architecture specification
|   |-- experiment-summary.md     # Offline ablation logs and metric records
|   `-- reproducibility.md       # Deterministic configuration and seed controls
|-- src/
|   |-- core/
|   |   `-- contracts.py         # Data schemas and serialization contracts
|   |-- retrieval/
|   |   |-- fusion.py            # Reciprocal Rank Fusion implementation
|   |   `-- metrics.py           # Recall@K, MRR, NDCG evaluation routines
|   `-- generation/
|       `-- masking.py           # Answer-only cross-entropy loss masking
|-- tests/
|   |-- test_fusion.py           # Verification tests for RRF fusion logic
|   |-- test_metrics.py          # Verification tests for retrieval metrics
|   `-- test_masking.py          # Verification tests for loss masking logic
|-- LICENSE                      # MIT License
`-- README.md                    # Canonical project documentation
```

---

## 6. Reproducibility & Contracts

To run the verification test suite:

```bash
python -m pytest tests/ -v
```

All algorithms adhere to immutable data contracts:
- `Candidate`: dataclass tracking document ID, rank position, modality source, and fusion score.
- `Qrels`: gold standard mapping of query ID to relevant document IDs.
- Deterministic seed controls for all tokenization and data slicing routines.
