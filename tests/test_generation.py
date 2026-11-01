import pytest

import config
from ola_agent.llm.mock_text_llm import _stitch
from ola_agent.rag.evaluate_chunking import score_query
from ola_agent.rag.generate import IDK_ANSWER, generate_answer
from ola_agent.rag.index import build_index
from ola_agent.rag.queries import EVAL_QUERIES, OUT_OF_SCOPE_QUERY


@pytest.fixture(scope="module", autouse=True)
def index():
    build_index()


def test_out_of_scope_query_triggers_fallback():
    r = generate_answer(OUT_OF_SCOPE_QUERY)
    assert r.fallback and r.answer == IDK_ANSWER and r.top_similarity < config.SIMILARITY_THRESHOLD


@pytest.mark.parametrize("q", EVAL_QUERIES, ids=lambda q: q["query"][:30])
def test_in_scope_answers_are_grounded_in_retrieved_context(q):
    r = generate_answer(q["query"])
    assert not r.fallback
    context = " ".join(c["text"] for c in r.context)
    # every answer sentence (citation stripped) is verbatim retrieved text
    for part in r.answer.split("] "):
        sentence = part.rsplit(" [", 1)[0].strip()
        assert sentence in context or all(w in context for w in sentence.split())


def test_precision_recall_arithmetic():
    s = score_query(EVAL_QUERIES[2]["query"], EVAL_QUERIES[2]["relevant"], config.COLLECTION_SENTENCE)
    assert len(s.retrieved_docs) == len(set(s.retrieved_docs))  # deduped
    assert s.precision == len(s.hits) / len(s.retrieved_docs)
    assert s.recall == len(s.hits) / len(s.relevant)


def test_stitch_joins_overlapping_fragments_only():
    assert _stitch("a b c d e f g", "c d e f g h i") == "a b c d e f g h i"
    assert _stitch("a b c", "x y z") is None
