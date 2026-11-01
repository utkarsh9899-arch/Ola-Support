"""check_support_ticket_status (Task 6): ticket lookup plus a designed escalation score.

escalation_score = W_ESCALATED * escalated + W_AGE * min(days_since_created / 30, 1)

* `escalated` dominates (0.6): a ticket someone already escalated is the strongest signal.
* The normalised age signal (0.4) says how long the ticket has been waiting, 0 = created today,
  1 = 30 days old. An older ticket still waiting is more urgent, so the score rises with age.
* The score is in [0, 1]: 1.0 only for an escalated ticket that is 30 days old.

Threshold: ESCALATION_THRESHOLD = W_AGE * P80(days_since_created) / 30 = 0.4 * 20 / 30 = 0.27, where
20 days is the 80th percentile of days_since_created in the generated dataset (SEED=42). So every
escalated ticket (score >= 0.6) is above it, and a non-escalated ticket is above it only when it is
older than 80% of all tickets (21+ days). Resolved/Closed tickets are never recommended for
escalation, whatever their score, because there is nothing left to escalate.
"""
from __future__ import annotations

import re

from dataset import SUPPORT_TICKETS

W_ESCALATED = 0.6
W_AGE = 0.4
MAX_AGE_DAYS = 30
AGE_P80_DAYS = 20  # 80th percentile of days_since_created (verified in tests/test_ticket_status.py)
ESCALATION_THRESHOLD = round(W_AGE * AGE_P80_DAYS / MAX_AGE_DAYS, 2)  # 0.27
TERMINAL_STATUSES = {"Resolved", "Closed"}

RECORD_ID_RE = re.compile(r"OLA-T-(\d{1,4})", re.I)
_BY_ID = {t["record_id"]: t for t in SUPPORT_TICKETS}


def normalize_record_id(text: str) -> str | None:
    m = RECORD_ID_RE.search(text or "")
    return f"OLA-T-{int(m.group(1)):04d}" if m else None


def escalation_score(escalated: bool, days_since_created: int) -> float:
    age = min(max(days_since_created, 0) / MAX_AGE_DAYS, 1.0)
    return round(W_ESCALATED * int(escalated) + W_AGE * age, 3)


def check_support_ticket_status(record_id: str) -> dict:
    rid = normalize_record_id(record_id)
    ticket = _BY_ID.get(rid) if rid else None
    if ticket is None:
        return {"record_id": record_id, "found": False, "error": f"No support ticket found with id {record_id!r}."}
    score = escalation_score(ticket["escalated"], ticket["days_since_created"])
    return {
        "record_id": rid,
        "found": True,
        "status": ticket["status"],
        "priority": ticket["priority"],
        "category": ticket["category"],
        "resolution_time_hours": ticket["resolution_time_hours"],
        "days_since_created": ticket["days_since_created"],
        "escalated": ticket["escalated"],
        "escalation_score": score,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "recommend_escalation": score >= ESCALATION_THRESHOLD and ticket["status"] not in TERMINAL_STATUSES,
    }
