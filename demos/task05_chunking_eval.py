"""Task 5 demo: per-query document-level precision/recall for BOTH collections."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from demos._transcript import transcript  # noqa: E402
from ola_agent.rag.evaluate_chunking import evaluate, mean  # noqa: E402
from ola_agent.rag.index import build_index  # noqa: E402
from ola_agent.rag.queries import EVAL_QUERIES  # noqa: E402

if __name__ == "__main__":
    with transcript("task05_chunking_eval"):
        build_index()
        print(f"Same {len(EVAL_QUERIES)} in-scope queries as Task 4, top-k={config.TOP_K} chunks, scored at document level.\n")
        results = {}
        for collection in (config.COLLECTION_FIXED, config.COLLECTION_SENTENCE):
            print("=" * 100)
            print(f"Collection: {collection}")
            print("=" * 100)
            scores = evaluate(EVAL_QUERIES, collection)
            results[collection] = scores
            for i, s in enumerate(scores, 1):
                print(f"Q{i}: {s.query}")
                print(f"    top-{config.TOP_K} chunks -> parent docs : {s.chunk_doc_ids}")
                print(f"    after dedup (retrieved)    : {s.retrieved_docs}")
                print(f"    relevant (gold)            : {s.relevant}")
                print(f"    relevant ∩ retrieved       : {s.hits}")
                print(f"    {s.arithmetic()}")
            p = [s.precision for s in scores]
            r = [s.recall for s in scores]
            print(f"\nMean precision = ({' + '.join(f'{x:.2f}' for x in p)}) / {len(p)} = {mean(p):.3f}")
            print(f"Mean recall    = ({' + '.join(f'{x:.2f}' for x in r)}) / {len(r)} = {mean(r):.3f}\n")

        print("=" * 100)
        print(f"{'Collection':<18} {'mean precision':>15} {'mean recall':>12}")
        means = {}
        for c, scores in results.items():
            means[c] = (round(mean([s.precision for s in scores]), 3), round(mean([s.recall for s in scores]), 3))
            print(f"{c:<18} {means[c][0]:>15.3f} {means[c][1]:>12.3f}")

        # The recommendation cites these exact numbers; fail loudly if the data ever changes.
        assert means[config.COLLECTION_SENTENCE] == (0.733, 0.900), means
        assert means[config.COLLECTION_FIXED] == (0.700, 0.800), means
        print("\nRECOMMENDATION")
        print(
            "I recommend deploying the sentence-based collection (ola_kb_sentence): on the same 5 queries it scored "
            "mean precision 0.733 vs 0.700 and mean recall 0.900 vs 0.800 for fixed-size chunks, and the gap comes "
            "from Q3, where only sentence chunks retrieved both the service-credit and the refund/compensation "
            "policy (R = 2/2 vs 1/2). Sentence chunks also gave the wider in-scope vs out-of-scope similarity gap "
            "during Task 4 calibration (+0.179 vs +0.146), which makes the 0.37 \"I don't know\" threshold safer, "
            "and because every chunk is one complete policy statement, grounded answers never start or stop mid-sentence."
        )
        print(f"\nDeployed collection for Parts 2-4: config.DEPLOYED_COLLECTION = {config.DEPLOYED_COLLECTION!r}")
