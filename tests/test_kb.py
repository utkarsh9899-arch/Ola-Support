from ola_agent.rag.kb import REQUIRED_TOPICS, load_documents, split_sentences


def test_at_least_12_documents():
    assert len(load_documents()) >= 12


def test_every_required_topic_has_a_document():
    doc_ids = {d.doc_id for d in load_documents()}
    assert len(REQUIRED_TOPICS) == 12
    missing = {topic: doc for topic, doc in REQUIRED_TOPICS.items() if doc not in doc_ids}
    assert not missing


def test_every_document_has_2_to_5_sentences():
    for d in load_documents():
        n = len(split_sentences(d.text))
        assert 2 <= n <= 5, f"{d.doc_id} has {n} sentences"
