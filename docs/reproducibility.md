# Reproducibility Protocols & Evaluation Contracts

## 1. Principles of Reproducibility

To ensure every experimental claim can be defended and verified:

1. **Deterministic Seeds:**
   - Global seed `42` set for PyTorch, NumPy, and Python random modules.
   - Deterministic CuDNN algorithms enabled for embedding extraction.

2. **Artifact Lineage & Manifests:**
   - Every experimental run logs exact configurations in immutable YAML/JSON manifests.
   - Preprocessing hashes (SHA-256) recorded to ensure identical corpus slicing across splits.

3. **Evaluation Protocol:**
   - Clear separation between diagnostic qrels (used for offline development) and competition evaluation.
   - Strict leakage audits preventing test queries from contaminating BM25 vocabulary statistics or IDF weights.

4. **Hardware and Environment Constraints:**
   - Python 3.10+ / 3.12 compatible.
   - Designed to run within constrained compute environments (1x NVIDIA T4 or RTX GPU).
