"""Split a customer message into the two slots the crew's tools need."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ola_agent.tools.ticket_status import RECORD_ID_RE, normalize_record_id

# Words that only ask about a ticket's state; if nothing else remains, there is no policy question.
_STATUS_ONLY_WORDS = {
    "a", "about", "already", "an", "and", "be", "been", "can", "check", "closed", "current", "currently",
    "escalated", "for", "give", "has", "hi", "hello", "id", "in", "is", "it", "its", "me", "my", "not", "now",
    "number", "of", "on", "open", "pending", "please", "progress", "resolved", "show", "status", "still",
    "tell", "the", "this", "ticket", "update", "was", "what", "whats", "where", "yet", "you", "your",
}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _is_status_only(text: str) -> bool:
    return all(w in _STATUS_ONLY_WORDS for w in re.findall(r"[a-z]+", text.lower().replace("'", "")))


@dataclass(frozen=True)
class Slots:
    message: str
    record_id: str | None
    policy_question: str | None


def extract_slots(message: str) -> Slots:
    record_id = normalize_record_id(message)
    remainder = re.sub(r"\s+", " ", RECORD_ID_RE.sub(" ", message)).strip()
    # Keep only the sentences that ask something beyond the ticket's own state.
    policy_sentences = [s for s in _SENTENCE_SPLIT.split(remainder) if s.strip() and not _is_status_only(s)]
    policy_question = " ".join(policy_sentences).strip(" ,-") or None
    return Slots(message=message, record_id=record_id, policy_question=policy_question)
