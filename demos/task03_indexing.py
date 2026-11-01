"""Task 3 demo: chunk both ways, embed, upsert into two collections, query both."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from demos._transcript import transcript  # noqa: E402
from ola_agent.rag.chunking import FIXED_OVERLAP_CHARS, FIXED_SIZE_CHARS, chunk_documents  # noqa: E402
from ola_agent.rag.index import build_index, get_collection, retrieve  # noqa: E402
from ola_agent.rag.kb import load_documents  # noqa: E402

SAMPLE_QUERY = "How long does a P1 ticket have to be resolved?"

if __name__ == "__main__":
    with transcript("task03_indexing"):
        docs = load_documents()
        print(f"Embedding model: {config.EMBEDDING_MODEL} (local, free)")
        print(f"Strategy A - fixed-size: {FIXED_SIZE_CHARS} chars, {FIXED_OVERLAP_CHARS} chars overlap (edges snapped to spaces)")
        print("Strategy B - sentence-based: one sentence per chunk\n")

        doc = docs[1]
        for strategy in ("fixed", "sentence"):
            print(f"Example chunks of {doc.doc_id} [{strategy}]:")
            for c in chunk_documents([doc], strategy):
                print(f"  {c.chunk_id:<36} ({len(c.text):>3} chars) {c.text}")
            print()

        counts = build_index(docs)
        for name, n in counts.items():
            print(f"collection.upsert() -> {name}: {n} chunks (collection.count() = {get_collection(name).count()})")

        print(f"\nSample query: {SAMPLE_QUERY!r}")
        for name in counts:
            print(f"\n  Top-3 from {name}:")
            for h in retrieve(SAMPLE_QUERY, name, k=3):
                print(f"    sim={h.similarity:.4f}  {h.doc_id:<26} {h.text[:90]}")
