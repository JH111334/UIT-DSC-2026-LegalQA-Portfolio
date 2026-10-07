from src.core.contracts import Candidate
from src.retrieval.fusion import reciprocal_rank_fusion

def test_rrf_scoring_order():
    run1 = [Candidate("docA", 10.0, 1, "bm25"), Candidate("docB", 5.0, 2, "bm25")]
    run2 = [Candidate("docB", 0.9, 1, "dense"), Candidate("docA", 0.8, 2, "dense")]

    fused = reciprocal_rank_fusion([run1, run2], k=60, top_n=2)
    assert len(fused) == 2
    # Both appear in rank 1 and 2, scores should be identical
    assert abs(fused[0].score - fused[1].score) < 1e-9

def test_rrf_single_modality_coverage():
    run1 = [Candidate("docA", 10.0, 1, "bm25")]
    run2 = [Candidate("docB", 0.9, 1, "dense")]

    fused = reciprocal_rank_fusion([run1, run2], k=60, top_n=5)
    doc_ids = {c.doc_id for c in fused}
    assert doc_ids == {"docA", "docB"}
