"""Two chunking strategies for the knowledge base (Task 3)."""
from __future__ import annotations

from dataclasses import dataclass

from ola_agent.rag.kb import Document, split_sentences

FIXED_SIZE_CHARS = 200
FIXED_OVERLAP_CHARS = 50

STRATEGY_FIXED = "fixed"
STRATEGY_SENTENCE = "sentence"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    strategy: str
    index: int
    text: str

    @property
    def embedding_text(self) -> str:
        # The title is prepended for BOTH strategies so neither gets an unfair edge.
        return f"{self.title}: {self.text}"

    @property
    def metadata(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title, "strategy": self.strategy, "chunk_index": self.index}


def fixed_size_chunks(doc: Document, size: int = FIXED_SIZE_CHARS, overlap: int = FIXED_OVERLAP_CHARS) -> list[Chunk]:
    """Sliding character windows of ~`size` chars, consecutive windows sharing ~`overlap` chars.

    Window edges are moved to the nearest space so no word is cut in half; the
    split otherwise ignores sentence boundaries (that is the point of the contrast).
    """
    text = doc.text
    chunks, start, i = [], 0, 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunks.append(Chunk(f"{doc.doc_id}::{STRATEGY_FIXED}::{i}", doc.doc_id, doc.title, STRATEGY_FIXED, i, text[start:end].strip()))
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
        i += 1
    return chunks


def sentence_chunks(doc: Document) -> list[Chunk]:
    """One chunk per sentence."""
    return [
        Chunk(f"{doc.doc_id}::{STRATEGY_SENTENCE}::{i}", doc.doc_id, doc.title, STRATEGY_SENTENCE, i, sentence)
        for i, sentence in enumerate(split_sentences(doc.text))
    ]


def chunk_documents(docs: list[Document], strategy: str) -> list[Chunk]:
    fn = {STRATEGY_FIXED: fixed_size_chunks, STRATEGY_SENTENCE: sentence_chunks}[strategy]
    return [c for d in docs for c in fn(d)]
