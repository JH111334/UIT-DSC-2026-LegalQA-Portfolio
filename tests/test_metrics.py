from src.retrieval.metrics import compute_recall_at_k, compute_mrr

def test_recall_at_k():
    qrels = {"q1": {"doc1", "doc2"}}
    rankings = {"q1": ["doc1", "doc3", "doc4"]}

    # Recall@1: 1 hit out of 2 = 0.5
    assert compute_recall_at_k(qrels, rankings, k=1) == 0.5
    # Recall@3: 1 hit out of 2 = 0.5
    assert compute_recall_at_k(qrels, rankings, k=3) == 0.5

def test_mrr_calculation():
    qrels = {"q1": {"doc2"}, "q2": {"doc1"}}
    rankings = {
        "q1": ["doc1", "doc2"], # rank 2 -> RR = 0.5
        "q2": ["doc1"]          # rank 1 -> RR = 1.0
    }
    # MRR = (0.5 + 1.0) / 2 = 0.75
    assert compute_mrr(qrels, rankings, k=5) == 0.75
