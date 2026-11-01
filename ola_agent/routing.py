"""Split a customer message into the two slots the crew's tools need."""
from __future__ import annotations

import re
from dataclasses import dataclass

from ola_agent.tools.ticket_status import RECORD_ID_RE, normalize_record_id

# Words that only ask about a ticket's state; if nothing else remains, there is no policy question.
_STATUS_ONLY_WORDS = {
    "a", "about", "already", "an", "and", "be", "been", "can", "check", "closed", "current", "currently",
    "escalated", "for", "give", "has", "hi", "hello", "id", "in", "is", "it", "its", "me", "my", "not", "now",
    "number", "of", "on", "open", "pending", "please", "progress", "resolved", "should", "show", "status", "still",
    "tell", "the", "this", "ticket", "update", "was", "what", "whats", "where", "yet", "you", "your",
    # sentences that only hand over (masked) contact details are not policy questions either
    "call", "contact", "mobile", "other", "phone", "reach", "redacted",
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


# --- follow-up resolution from conversation history (Task 8) ------------------------
TICKET_REFERENCE_RE = re.compile(r"\b(it|its|this ticket|that ticket|my ticket|the ticket|that one)\b", re.I)
PRIORITY_REFERENCE_RE = re.compile(r"\b(that|this|its|the same|my ticket's) priority\b", re.I)
PRIORITY_IN_TEXT_RE = re.compile(r"\bpriority (P[1-4])\b")


@dataclass(frozen=True)
class Resolution:
    slots: Slots
    needs_clarification: bool
    notes: list[str]


def resolve_references(slots: Slots, history_texts: list[str]) -> Resolution:
    """Fill in 'it' / 'that priority' from earlier turns of THIS session only."""
    notes, record_id, policy = [], slots.record_id, slots.policy_question
    newest_first = list(reversed(history_texts))

    if record_id is None and TICKET_REFERENCE_RE.search(slots.message):
        record_id = next((normalize_record_id(t) for t in newest_first if normalize_record_id(t)), None)
        notes.append(f"'it' resolved from history -> {record_id}" if record_id else "'it' has no ticket in history")

    if PRIORITY_REFERENCE_RE.search(slots.message):
        priority = next((m.group(1) for t in newest_first if (m := PRIORITY_IN_TEXT_RE.search(t))), None)
        if priority:
            policy = PRIORITY_REFERENCE_RE.sub(f"priority {priority}", policy or slots.message)
            notes.append(f"'that priority' resolved from history -> {priority}")
        else:
            notes.append("'that priority' has no ticket priority in history")
            return Resolution(slots, True, notes)

    refers_to_ticket = TICKET_REFERENCE_RE.search(slots.message) is not None
    needs_clarification = record_id is None and refers_to_ticket and policy is None
    return Resolution(Slots(slots.message, record_id, policy), needs_clarification, notes)
