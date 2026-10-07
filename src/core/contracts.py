from dataclasses import dataclass, field
from typing import List, Optional

@dataclass(frozen=True)
class Candidate:
    doc_id: str
    score: float
    rank: int
    source: str = "unknown"

@dataclass
class RetrievalRun:
    query_id: str
    candidates: List[Candidate] = field(default_factory=list)
