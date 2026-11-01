"""Task 4 demo: grounded answers for 5 in-scope queries + 1 out-of-scope fallback, on both collections."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from demos._transcript import transcript  # noqa: E402
from ola_agent.rag.generate import generate_answer  # noqa: E402
from ola_agent.rag.index import build_index  # noqa: E402
from ola_agent.rag.queries import EVAL_QUERIES, OUT_OF_SCOPE_QUERY  # noqa: E402

if __name__ == "__main__":
    with transcript("task04_grounded_generation"):
        build_index()
        print(f"MOCK_LLM={config.MOCK_LLM} | top-k={config.TOP_K} | calibrated IDK threshold={config.SIMILARITY_THRESHOLD}\n")
        queries = [(q["query"], "in-scope") for q in EVAL_QUERIES] + [(OUT_OF_SCOPE_QUERY, "OUT-OF-SCOPE")]
        for collection in (config.COLLECTION_SENTENCE, config.COLLECTION_FIXED):
            print("=" * 100)
            print(f"Collection: {collection}")
            print("=" * 100)
            for i, (q, kind) in enumerate(queries, 1):
                r = generate_answer(q, collection=collection)
                verdict = "FALLBACK (I don't know)" if r.fallback else "answered from context"
                print(f"\nQ{i} [{kind}] {q}")
                print(f"  top-1 similarity = {r.top_similarity:.4f} ({'<' if r.fallback else '>='} {config.SIMILARITY_THRESHOLD}) -> {verdict}")
                print(f"  retrieved context: {[c['doc_id'] for c in r.context]}")
                print(f"  ANSWER: {r.answer}")
            print()
