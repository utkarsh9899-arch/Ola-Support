"""Grounded generation with an empirically calibrated "I don't know" fallback (Task 4)."""
from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel

import config
from ola_agent.rag.index import Hit, embed, retrieve

IDK_ANSWER = (
    "I don't know. That question is not covered by Ola's support policies in my knowledge base, "
    "so I can't give a grounded answer."
)

GROUNDED_PROMPT = """You are an Ola customer-support assistant.
Answer the QUESTION using ONLY the CONTEXT below. Cite the [doc_id] of every sentence you use.
If the CONTEXT does not contain the answer, reply exactly: I don't know.

CONTEXT:
{context}
QUESTION: {question}
ANSWER:"""


class GroundedAnswer(BaseModel):
    query: str
    answer: str
    sources: list[str]
    context: list[dict]
    top_similarity: float
    fallback: bool


@lru_cache(maxsize=1)
def get_text_llm():
    if config.MOCK_LLM:
        from ola_agent.llm.mock_text_llm import MockTextLLM

        return MockTextLLM(embed)
    raise NotImplementedError("Real-LLM mode is optional and not configured; run with MOCK_LLM=1.")


def build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n".join(f"[{h.doc_id} | {h.title}] {h.text}" for h in hits)
    return GROUNDED_PROMPT.format(context=context, question=question)


def generate_answer(query: str, collection: str | None = None, k: int = config.TOP_K,
                    threshold: float = config.SIMILARITY_THRESHOLD, use_cache: bool = True) -> GroundedAnswer:
    """Cached front door to grounded generation (Task 16): a hit skips retrieval and the LLM call."""
    from ola_agent.rag.cache import GROUNDED_CACHE

    collection = collection or config.DEPLOYED_COLLECTION or config.COLLECTION_SENTENCE
    if not use_cache:
        return _generate(query, collection, k, threshold)
    key = GROUNDED_CACHE.key(query, collection, k, threshold)
    cached = GROUNDED_CACHE.get(key)
    if cached is not None:
        return cached.model_copy(update={"query": query})
    answer = _generate(query, collection, k, threshold)
    GROUNDED_CACHE.put(key, answer)
    return answer


def _generate(query: str, collection: str, k: int, threshold: float) -> GroundedAnswer:
    hits = retrieve(query, collection, k)
    top = hits[0].similarity if hits else 0.0
    if top < threshold:
        # Retrieval similarity is the only groundedness signal under MOCK_LLM.
        return GroundedAnswer(query=query, answer=IDK_ANSWER, sources=[], context=[], top_similarity=top, fallback=True)

    supporting = [h for h in hits if h.similarity >= threshold]
    answer = get_text_llm().complete(build_prompt(query, supporting))
    if answer.strip().startswith("I don't know"):
        return GroundedAnswer(query=query, answer=IDK_ANSWER, sources=[], context=[], top_similarity=top, fallback=True)
    return GroundedAnswer(
        query=query,
        answer=answer,
        sources=sorted({h.doc_id for h in supporting if f"[{h.doc_id}]" in answer}),
        context=[{"doc_id": h.doc_id, "text": h.text, "similarity": h.similarity} for h in supporting],
        top_similarity=top,
        fallback=False,
    )
