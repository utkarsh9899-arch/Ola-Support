"""Guardrails (Task 10).

Input side : PII masking for the fixed-format phone-number field, prompt-injection detection.
Output side: groundedness check that refuses when the retrieved context does not support the answer.

Names, pickup/drop addresses and payment details are free text with no single fixed format, so
masking them is out of scope for this keyless MOCK_LLM masker (as the brief states).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import config

PHONE_MASK = "[PHONE_REDACTED]"
# Indian mobile numbers: optional +91 / 91 / 0 prefix, 10 digits starting 6-9, optional space/dash after 5 digits.
PHONE_RE = re.compile(r"(?<![\w+])(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")

INJECTION_PATTERNS = {
    "ignore_instructions": r"\b(ignore|disregard|forget|override)\b[\w\s,]{0,30}\b(instructions?|rules|guidelines|prompt|policies)\b",
    "role_override": r"\b(you are now|act as|pretend (to be|you are)|from now on you)\b",
    "prompt_exfiltration": r"\b(reveal|show|print|repeat|leak)\b[\w\s]{0,20}\b(system prompt|hidden prompt|instructions|your prompt)\b",
    "jailbreak_terms": r"\b(jailbreak|developer mode|do anything now|DAN mode|bypass (the )?(safety|guardrails?|filters?))\b",
}
_INJECTION_RES = {name: re.compile(p, re.I) for name, p in INJECTION_PATTERNS.items()}

BLOCKED_ANSWER = (
    "I can't help with that request. I can answer questions about Ola's support policies or check a "
    "support ticket's status."
)
UNGROUNDED_ANSWER = (
    "I don't know. The retrieved policy context does not support an answer to this question, so I won't guess."
)
UNGROUNDED_POLICY_NOTE = (
    "On your policy question: I don't know, because the retrieved policy context does not support an answer."
)
CITATION_RE = re.compile(r"\s*\[(\d\d_[a-z_]+)\]")


@dataclass
class InputGuardResult:
    original_length: int
    masked_text: str
    pii_masked: int = 0
    injection_matches: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return bool(self.injection_matches)

    @property
    def flags(self) -> list[str]:
        flags = [f"pii_masked:phone x{self.pii_masked}"] if self.pii_masked else []
        return flags + [f"prompt_injection:{m}" for m in self.injection_matches]


def mask_pii(text: str) -> tuple[str, int]:
    masked, n = PHONE_RE.subn(PHONE_MASK, text)
    return masked, n


def detect_injection(text: str) -> list[str]:
    return [name for name, rx in _INJECTION_RES.items() if rx.search(text)]


def guard_input(text: str) -> InputGuardResult:
    masked, n = mask_pii(text)
    return InputGuardResult(original_length=len(text), masked_text=masked, pii_masked=n,
                            injection_matches=detect_injection(masked))


@dataclass
class GroundednessResult:
    supported: bool
    reason: str
    unsupported_sentences: list[str] = field(default_factory=list)


FACT_TOKEN_RE = re.compile(r"OLA-T-\d{4}|\bP[1-4]\b|\d+(?:\.\d+)?")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def answer_sentences(answer: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(CITATION_RE.sub("", answer)) if s.strip()]


def sentence_supported(sentence: str, context_text: str, record_text: str, no_claim: set[str] = frozenset()) -> bool:
    """A sentence is supported when it is verbatim policy context, a fixed no-claim sentence, or a ticket
    sentence whose every id / priority / number appears in the ticket record."""
    if sentence in no_claim or sentence in context_text:
        return True
    tokens = FACT_TOKEN_RE.findall(sentence)
    return bool(record_text and tokens) and all(t in record_text for t in tokens)


def split_cited_sentences(answer: str) -> list[tuple[str, str]]:
    """'Sentence one. [doc_a] Sentence two. [doc_b]' -> [(sentence, doc_id), ...] for cited sentences."""
    out, pos = [], 0
    for m in CITATION_RE.finditer(answer):
        out.append((answer[pos:m.start()].strip(), m.group(1)))
        pos = m.end()
    return [(s, d) for s, d in out if s]


def check_groundedness(policy_answer: str, context: list[dict], top_similarity: float,
                       threshold: float = config.SIMILARITY_THRESHOLD) -> GroundednessResult:
    """Output-side check on the policy part of a draft.

    1. The question must be supported by retrieval at all (top-1 similarity >= calibrated threshold).
    2. Every cited sentence must appear in the retrieved context of the document it cites.
    """
    if top_similarity < threshold:
        return GroundednessResult(False, f"top-1 retrieval similarity {top_similarity:.4f} < threshold {threshold}")
    by_doc: dict[str, str] = {}
    for c in context:
        by_doc[c["doc_id"]] = by_doc.get(c["doc_id"], "") + " " + c["text"]
    unsupported = [s for s, doc in split_cited_sentences(policy_answer) if s not in by_doc.get(doc, "")]
    uncited = CITATION_RE.sub("", policy_answer).strip() if not split_cited_sentences(policy_answer) else ""
    if uncited:
        unsupported.append(uncited)
    if unsupported:
        return GroundednessResult(False, "answer contains sentences not found in the retrieved context", unsupported)
    return GroundednessResult(True, "every cited sentence is present in the retrieved context")
