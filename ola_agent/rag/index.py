"""Embed chunks with SentenceTransformers and index them in ChromaDB (Task 3).

Each chunking strategy lives in its OWN collection. Collections use cosine
space, and embeddings are L2-normalised, so similarity = 1 - distance.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import config
from ola_agent.rag.chunking import STRATEGY_FIXED, STRATEGY_SENTENCE, Chunk, chunk_documents
from ola_agent.rag.kb import Document, load_documents

COLLECTIONS = {STRATEGY_FIXED: config.COLLECTION_FIXED, STRATEGY_SENTENCE: config.COLLECTION_SENTENCE}


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    similarity: float


@lru_cache(maxsize=1)
def get_embedder():
    config.enable_hf_offline()  # model was downloaded once during setup
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_client():
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(config.CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


def get_collection(name: str):
    return get_client().get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def embed(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


def upsert_chunks(collection_name: str, chunks: list[Chunk]) -> int:
    if not chunks:
        return 0
    get_collection(collection_name).upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embed([c.embedding_text for c in chunks]),
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )
    return len(chunks)


def build_index(docs: list[Document] | None = None) -> dict[str, int]:
    """Chunk every document both ways and upsert each set into its own collection."""
    docs = docs if docs is not None else load_documents()
    return {COLLECTIONS[s]: upsert_chunks(COLLECTIONS[s], chunk_documents(docs, s)) for s in (STRATEGY_FIXED, STRATEGY_SENTENCE)}


def retrieve(query: str, collection_name: str, k: int = config.TOP_K) -> list[Hit]:
    res = get_collection(collection_name).query(query_embeddings=embed([query]), n_results=k)
    return [
        Hit(chunk_id=cid, doc_id=meta["doc_id"], title=meta["title"], text=doc, similarity=round(1.0 - dist, 4))
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
