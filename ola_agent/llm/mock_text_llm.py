"""Deterministic MOCK_LLM text model for the grounded-generation step.

It receives the real grounded prompt (CONTEXT + QUESTION blocks), and answers by
extracting the context sentences most similar to the question. It can only ever
output text that appears inside the CONTEXT block, so it is grounded by
construction. `call_count` lets the cache demo (Task 16) prove a call was saved.
"""
from __future__ import annotations

import re

import numpy as np

CONTEXT_RE = re.compile(r"CONTEXT:\n(.*?)\nQUESTION:", re.S)
QUESTION_RE = re.compile(r"QUESTION:\s*(.*?)\s*\nANSWER:", re.S)
CHUNK_RE = re.compile(r"^\[(?P<doc_id>[^\]|]+?)\s*\|\s*(?P<title>[^\]]+)\]\s*(?P<text>.+)$", re.M)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

MAX_ANSWER_SENTENCES = 2
COMPLETE_SENTENCE_BONUS = 0.05  # prefer full sentences over fixed-size chunk fragments
RELATIVE_CUTOFF = 0.10  # drop sentences much weaker than the best one
OVERLAP_NGRAM = 5  # two picks sharing a 5-word run are the same text (overlapping chunks)


def _ngrams(text: str, n: int = OVERLAP_NGRAM) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _stitch(a: str, b: str) -> str | None:
    """Join two overlapping fixed-size fragments (a's tail == b's head), using only their own words."""
    if a in b:
        return b
    if b in a:
        return a
    wa, wb = a.split(), b.split()
    for k in range(min(len(wa), len(wb)), OVERLAP_NGRAM - 1, -1):
        if wa[-k:] == wb[:k]:
            return " ".join(wa + wb[k:])
    return None


class MockTextLLM:
    model_name = "mock-extractive-v1"

    def __init__(self, embed_fn):
        self._embed = embed_fn
        self.call_count = 0

    def complete(self, prompt: str) -> str:
        self.call_count += 1
        context_m, question_m = CONTEXT_RE.search(prompt), QUESTION_RE.search(prompt)
        if not context_m or not question_m:
            return "I don't know."
        question = question_m.group(1).strip()

        candidates: list[tuple[str, str, str]] = []  # (doc_id, title, sentence)
        for m in CHUNK_RE.finditer(context_m.group(1)):
            for sent in SENTENCE_SPLIT.split(m.group("text").strip()):
                sent = sent.strip()
                if len(sent.split()) >= 4 and sent not in {s for _, _, s in candidates}:
                    candidates.append((m.group("doc_id"), m.group("title").strip(), sent))
        if not candidates:
            return "I don't know."

        # Score "title: sentence", mirroring how chunks were embedded for retrieval.
        vecs = np.array(self._embed([question] + [f"{t}: {s}" for _, t, s in candidates]))
        sims = vecs[1:] @ vecs[0]
        scored = []
        for (doc_id, _, sent), sim in zip(candidates, sims):
            complete = sent[0].isupper() and sent[-1] in ".!?"
            scored.append((float(sim) + (COMPLETE_SENTENCE_BONUS if complete else 0.0), doc_id, sent))
        scored.sort(key=lambda x: x[0], reverse=True)

        best = scored[0][0]
        chosen: list[list[str]] = []  # [doc_id, text]
        for score, doc_id, sent in scored:
            if score < best - RELATIVE_CUTOFF:
                break
            overlapping = next((c for c in chosen if c[0] == doc_id and _ngrams(c[1]) & _ngrams(sent)), None)
            if overlapping:
                # Overlapping fixed-size fragments of the same sentence: merge, don't repeat.
                merged = _stitch(overlapping[1], sent) or _stitch(sent, overlapping[1])
                if merged:
                    overlapping[1] = merged
                continue
            if len(chosen) < MAX_ANSWER_SENTENCES:
                chosen.append([doc_id, sent])
        return " ".join(f"{text} [{doc_id}]" for doc_id, text in chosen)
