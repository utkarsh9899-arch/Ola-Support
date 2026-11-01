from ola_agent.rag.chunking import FIXED_SIZE_CHARS, chunk_documents, fixed_size_chunks, sentence_chunks
from ola_agent.rag.kb import load_documents, split_sentences


def test_fixed_chunks_respect_size_and_overlap():
    for doc in load_documents():
        chunks = fixed_size_chunks(doc)
        assert all(len(c.text) <= FIXED_SIZE_CHARS for c in chunks)
        for a, b in zip(chunks, chunks[1:]):
            # consecutive chunks share text (overlap)
            assert b.text.split()[0] in a.text


def test_sentence_chunks_are_one_sentence_each():
    for doc in load_documents():
        assert [c.text for c in sentence_chunks(doc)] == split_sentences(doc.text)


def test_every_chunk_maps_back_to_its_parent_document():
    docs = load_documents()
    ids = {d.doc_id for d in docs}
    for strategy in ("fixed", "sentence"):
        chunks = chunk_documents(docs, strategy)
        assert {c.doc_id for c in chunks} == ids
        assert len({c.chunk_id for c in chunks}) == len(chunks)
