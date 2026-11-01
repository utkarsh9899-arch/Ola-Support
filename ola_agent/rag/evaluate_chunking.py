"""Document-level precision/recall for each chunking strategy (Task 5).

Retrieved chunks are mapped back to their parent doc_id and de-duplicated
before scoring:  P = |relevant ∩ retrieved| / |retrieved|,  R = |relevant ∩ retrieved| / |relevant|.
"""
from __future__ import annotations

from dataclasses import dataclass

import config
from ola_agent.rag.index import retrieve


@dataclass
class QueryScore:
    query: str
    chunk_doc_ids: list[str]
    retrieved_docs: list[str]
    relevant: list[str]
    hits: list[str]
    precision: float
    recall: float

    def arithmetic(self) -> str:
        nh, nr, nrel = len(self.hits), len(self.retrieved_docs), len(self.relevant)
        return f"P = {nh}/{nr} = {self.precision:.2f}   R = {nh}/{nrel} = {self.recall:.2f}"


def score_query(query: str, relevant: set[str], collection: str, k: int = config.TOP_K) -> QueryScore:
    chunk_doc_ids = [h.doc_id for h in retrieve(query, collection, k)]
    retrieved = list(dict.fromkeys(chunk_doc_ids))  # dedup, keep rank order
    hits = [d for d in retrieved if d in relevant]
    return QueryScore(
        query=query,
        chunk_doc_ids=chunk_doc_ids,
        retrieved_docs=retrieved,
        relevant=sorted(relevant),
        hits=hits,
        precision=len(hits) / len(retrieved) if retrieved else 0.0,
        recall=len(hits) / len(relevant) if relevant else 0.0,
    )


def evaluate(queries: list[dict], collection: str, k: int = config.TOP_K) -> list[QueryScore]:
    return [score_query(q["query"], q["relevant"], collection, k) for q in queries]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
