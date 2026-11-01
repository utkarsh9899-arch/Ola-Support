"""Task 2 demo: list every KB document, its topic, and its sentence count."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402,F401
from demos._transcript import transcript  # noqa: E402
from ola_agent.rag.kb import REQUIRED_TOPICS, load_documents, split_sentences  # noqa: E402

if __name__ == "__main__":
    with transcript("task02_knowledge_base"):
        docs = {d.doc_id: d for d in load_documents()}
        print(f"Knowledge base: {len(docs)} documents in kb/\n")
        print(f"{'Required topic':<40} {'doc_id':<28} sentences")
        for topic, doc_id in REQUIRED_TOPICS.items():
            n = len(split_sentences(docs[doc_id].text)) if doc_id in docs else 0
            print(f"{topic:<40} {doc_id:<28} {n}")
        covered = all(doc_id in docs for doc_id in REQUIRED_TOPICS.values())
        in_range = all(2 <= len(split_sentences(d.text)) <= 5 for d in docs.values())
        print(f"\nAll 12 required topics covered: {covered}")
        print(f"Every document has 2-5 sentences: {in_range}")
