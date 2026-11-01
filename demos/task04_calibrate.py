"""Task 4 calibration: measure top-1 cosine similarity for in-scope vs out-of-scope
queries on BOTH collections, so the "I don't know" threshold is set from data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from demos._transcript import transcript  # noqa: E402
from ola_agent.rag.index import build_index, retrieve  # noqa: E402

IN_SCOPE = [
    "What is the resolution target for a P2 ticket?",
    "When does an L1 agent have to escalate a ticket?",
    "Can I get my refund in my Ola Money wallet?",
    "Which channels does Ola support use to contact customers?",
    "Is support available on Diwali?",
    "What happens if I complain about the same issue three times?",
    "How much service credit can I get in a month?",
    "How long are support tickets kept before deletion?",
]

OUT_OF_SCOPE = [
    "What will the weather be like in Mumbai tomorrow?",
    "Who won the cricket match yesterday?",
    "Give me a recipe for paneer butter masala.",
    "Write a Python function to reverse a linked list.",
    "What is the capital of Australia?",
    "What is Ola's share price today?",
]


def top1(query: str, collection: str) -> float:
    return retrieve(query, collection, k=1)[0].similarity


if __name__ == "__main__":
    with transcript("task04_threshold_calibration"):
        build_index()
        collections = [config.COLLECTION_FIXED, config.COLLECTION_SENTENCE]
        scores = {c: {"in": [], "out": []} for c in collections}

        print(f"{'kind':<5} {'query':<62} " + " ".join(f"{c:>16}" for c in collections))
        for kind, queries in (("in", IN_SCOPE), ("out", OUT_OF_SCOPE)):
            for q in queries:
                row = [top1(q, c) for c in collections]
                for c, s in zip(collections, row):
                    scores[c][kind].append(s)
                print(f"{kind:<5} {q:<62} " + " ".join(f"{s:>16.4f}" for s in row))

        print()
        for c in collections:
            lo_in, hi_out = min(scores[c]["in"]), max(scores[c]["out"])
            print(f"{c}: in-scope top-1 range {lo_in:.4f}-{max(scores[c]['in']):.4f} | "
                  f"out-of-scope range {min(scores[c]['out']):.4f}-{hi_out:.4f} | gap {lo_in - hi_out:+.4f}")

        all_in = min(min(scores[c]["in"]) for c in collections)
        all_out = max(max(scores[c]["out"]) for c in collections)
        print(f"\nAcross both collections: lowest in-scope = {all_in:.4f}, highest out-of-scope = {all_out:.4f}")
        if all_in > all_out:
            print(f"Clusters separate. Midpoint threshold = ({all_out:.4f} + {all_in:.4f}) / 2 = {(all_in + all_out) / 2:.4f}")
        else:
            print("Clusters OVERLAP - threshold cannot separate them; revisit queries/KB.")
