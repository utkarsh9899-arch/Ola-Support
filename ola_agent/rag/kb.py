"""Load the knowledge base from kb/*.md (Task 2).

Each file is one document: a '# Title' line followed by a 2-5 sentence body.
The file stem (e.g. '04_refund_compensation') is the document's doc_id.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import config

# Required topics from the brief -> the document that covers each one.
REQUIRED_TOPICS = {
    "ticket-priority classification rules": "01_ticket_priority",
    "SLA-by-severity policy": "02_sla_by_severity",
    "escalation matrix": "03_escalation_matrix",
    "refund/compensation policy": "04_refund_compensation",
    "customer-communication-channel policy": "05_communication_channels",
    "business-hours/holiday-support policy": "06_business_hours",
    "repeat-complaint-handling policy": "07_repeat_complaints",
    "service-credit policy": "08_service_credits",
    "feedback-collection process": "09_feedback_collection",
    "VIP-customer handling policy": "10_vip_handling",
    "outage-communication protocol": "11_outage_communication",
    "data-retention policy for tickets": "12_data_retention",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def parse_document(doc_id: str, raw: str) -> Document:
    lines = raw.strip().splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError(f"{doc_id}: first line must be '# Title'")
    title = lines[0][2:].strip()
    body = " ".join(line.strip() for line in lines[1:] if line.strip())
    return Document(doc_id=doc_id, title=title, text=body)


def load_documents(kb_dir=config.KB_DIR) -> list[Document]:
    return [parse_document(p.stem, p.read_text(encoding="utf-8")) for p in sorted(kb_dir.glob("*.md"))]
