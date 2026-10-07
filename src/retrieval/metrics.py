from typing import Dict, List, Set

def compute_recall_at_k(qrels: Dict[str, Set[str]], rankings: Dict[str, List[str]], k: int) -> float:
    """
    Computes Recall@K averaged over all queries in qrels.
    """
    if not qrels:
        return 0.0

    recalls = []
    for qid, target_docs in qrels.items():
        if not target_docs:
            continue
        predicted_top_k = set(rankings.get(qid, [])[:k])
        hits = len(target_docs.intersection(predicted_top_k))
        recalls.append(hits / len(target_docs))

    return sum(recalls) / len(recalls) if recalls else 0.0

def compute_mrr(qrels: Dict[str, Set[str]], rankings: Dict[str, List[str]], k: int = 10) -> float:
    """
    Computes Mean Reciprocal Rank (MRR@K).
    """
    if not qrels:
        return 0.0

    reciprocal_ranks = []
    for qid, target_docs in qrels.items():
        if not target_docs:
            continue
        predicted = rankings.get(qid, [])[:k]
        rr = 0.0
        for rank, doc_id in enumerate(predicted, start=1):
            if doc_id in target_docs:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
