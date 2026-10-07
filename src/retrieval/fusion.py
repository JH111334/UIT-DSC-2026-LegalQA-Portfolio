from collections import defaultdict
from typing import List, Dict
from src.core.contracts import Candidate

def reciprocal_rank_fusion(runs: List[List[Candidate]], k: int = 60, top_n: int = 50) -> List[Candidate]:
    """
    Computes Reciprocal Rank Fusion (RRF) over multiple candidate rankings.
    Score(d) = sum_{run in runs} 1 / (k + rank(d))
    """
    scores: Dict[str, float] = defaultdict(float)
    sources: Dict[str, List[str]] = defaultdict(list)

    for run in runs:
        for idx, candidate in enumerate(run, start=1):
            rank = candidate.rank if candidate.rank > 0 else idx
            scores[candidate.doc_id] += 1.0 / (k + rank)
            sources[candidate.doc_id].append(candidate.source)

    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    fused_candidates: List[Candidate] = []
    for rank, (doc_id, score) in enumerate(sorted_docs[:top_n], start=1):
        fused_candidates.append(
            Candidate(
                doc_id=doc_id,
                score=score,
                rank=rank,
                source="+".join(sorted(set(sources[doc_id])))
            )
        )
    return fused_candidates
