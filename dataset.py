"""Seeded, deterministic generator for the Ola support-ticket dataset (Task 1).

Run `python dataset.py` to print the validation report. All values are
fabricated. The structure is checked by `validate()`; if a check fails, the
fix is to change SEED or the weights below and regenerate, never to edit
individual records.
"""
from __future__ import annotations

import random
from collections import Counter

# --- Design choices (also stated in README.md) -------------------------------
SEED = 42
N_RECORDS = 48

CATEGORY_WEIGHTS = {
    "Billing": 0.30,
    "Technical Issue": 0.22,
    "Account Access": 0.16,
    "Product Defect": 0.14,
    "General Inquiry": 0.18,
}

STATUS_WEIGHTS = {
    "Open": 0.25,
    "In Progress": 0.25,
    "Escalated": 0.12,
    "Resolved": 0.23,
    "Closed": 0.15,
}

# Priority depends on category so the data agrees with kb/01_ticket_priority.md
# (P1 = safety / trip blocked, P2 = payment failures and lockouts, P3 = disputes,
# bugs with a workaround and non-blocking defects, P4 = general inquiries).
PRIORITY_WEIGHTS_BY_CATEGORY = {
    "Billing": {"P2": 0.55, "P3": 0.40, "P4": 0.05},
    "Technical Issue": {"P1": 0.25, "P2": 0.30, "P3": 0.40, "P4": 0.05},
    "Account Access": {"P2": 0.70, "P3": 0.30},
    "Product Defect": {"P1": 0.15, "P2": 0.15, "P3": 0.70},
    "General Inquiry": {"P3": 0.20, "P4": 0.80},
}
PRIORITIES = ("P1", "P2", "P3", "P4")

# resolution_time_hours is drawn inside each priority's SLA resolution window from
# kb/02_sla_by_severity.md (P1 within 4 h ... P4 within 72 h), so the overall
# range is 0.5-72 hours: urgent safety/trip issues close within hours while
# low-priority inquiries can take up to three days.
RESOLUTION_HOURS_BY_PRIORITY = {
    "P1": (0.5, 4.0),
    "P2": (2.0, 24.0),
    "P3": (6.0, 48.0),
    "P4": (12.0, 72.0),
}

DAYS_SINCE_CREATED_RANGE = (0, 30)

# Tickets currently in "Escalated" status are always escalated=True; tickets in
# any other status were escalated earlier in their life with this probability.
PRIOR_ESCALATION_PROB = 0.08

# Fare attached to the ticket's trip, in INR (fabricated).
FARE_AMOUNT_INR_RANGE = (50, 1500)

# --- Validation thresholds from the brief ------------------------------------
MIN_RECORDS = 40
MIN_PER_CATEGORY = 3
MIN_PER_STATUS = 1
ESCALATED_PCT_BAND = (10.0, 30.0)


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()), k=1)[0]


def generate(seed: int = SEED, n: int = N_RECORDS) -> list[dict]:
    rng = random.Random(seed)
    tickets = []
    for i in range(1, n + 1):
        category = _weighted_choice(rng, CATEGORY_WEIGHTS)
        status = _weighted_choice(rng, STATUS_WEIGHTS)
        priority = _weighted_choice(rng, PRIORITY_WEIGHTS_BY_CATEGORY[category])
        low, high = RESOLUTION_HOURS_BY_PRIORITY[priority]
        resolution_time_hours = round(rng.uniform(low, high) * 2) / 2  # nearest 0.5 h
        days_since_created = rng.randint(*DAYS_SINCE_CREATED_RANGE)
        prior_escalation = rng.random() < PRIOR_ESCALATION_PROB
        escalated = status == "Escalated" or prior_escalation
        fare_amount_inr = rng.randint(*FARE_AMOUNT_INR_RANGE)
        tickets.append(
            {
                "record_id": f"OLA-T-{i:04d}",
                "category": category,
                "status": status,
                "priority": priority,
                "resolution_time_hours": resolution_time_hours,
                "days_since_created": days_since_created,
                "escalated": escalated,
                "fare_amount_inr": fare_amount_inr,
            }
        )
    return tickets


def summarize(tickets: list[dict]) -> dict:
    n = len(tickets)
    escalated = sum(t["escalated"] for t in tickets)
    return {
        "n": n,
        "per_category": Counter(t["category"] for t in tickets),
        "per_status": Counter(t["status"] for t in tickets),
        "per_priority": Counter(t["priority"] for t in tickets),
        "escalated_count": escalated,
        "escalated_pct": 100.0 * escalated / n,
    }


def validate(tickets: list[dict]) -> list[str]:
    """Return a list of failed checks (empty list = dataset is valid)."""
    s = summarize(tickets)
    failures = []
    if s["n"] < MIN_RECORDS:
        failures.append(f"only {s['n']} records (< {MIN_RECORDS})")
    for cat in CATEGORY_WEIGHTS:
        if s["per_category"][cat] < MIN_PER_CATEGORY:
            failures.append(f"category {cat!r} has {s['per_category'][cat]} records (< {MIN_PER_CATEGORY})")
    for status in STATUS_WEIGHTS:
        if s["per_status"][status] < MIN_PER_STATUS:
            failures.append(f"status {status!r} has {s['per_status'][status]} records (< {MIN_PER_STATUS})")
    lo, hi = ESCALATED_PCT_BAND
    if not lo <= s["escalated_pct"] <= hi:
        failures.append(f"escalated={s['escalated_pct']:.1f}% outside {lo}-{hi}%")
    dmin, dmax = DAYS_SINCE_CREATED_RANGE
    if any(not (dmin <= t["days_since_created"] <= dmax) for t in tickets):
        failures.append("days_since_created outside 0-30")
    if len({t["record_id"] for t in tickets}) != s["n"]:
        failures.append("duplicate record_id")
    return failures


def print_report(tickets: list[dict]) -> None:
    s = summarize(tickets)
    print(f"Ola support-ticket dataset  (SEED={SEED}, N={s['n']})")
    print("\nCount per category:")
    for cat in CATEGORY_WEIGHTS:
        print(f"  {cat:<16} {s['per_category'][cat]:>3}   (weight {CATEGORY_WEIGHTS[cat]:.2f})")
    print("\nCount per status:")
    for status in STATUS_WEIGHTS:
        print(f"  {status:<16} {s['per_status'][status]:>3}   (weight {STATUS_WEIGHTS[status]:.2f})")
    print("\nCount per priority:")
    for p in PRIORITIES:
        print(f"  {p:<16} {s['per_priority'][p]:>3}   (weights depend on category)")
    print(f"\nEscalated: {s['escalated_count']}/{s['n']} = {s['escalated_pct']:.1f}%  (required band 10-30%)")
    hours = [t["resolution_time_hours"] for t in tickets]
    fares = [t["fare_amount_inr"] for t in tickets]
    print(f"resolution_time_hours observed: {min(hours)}-{max(hours)}  (design range 0.5-72)")
    print(f"fare_amount_inr observed: {min(fares)}-{max(fares)}  (design range 50-1500)")
    failures = validate(tickets)
    print("\nValidation:", "PASS - all structural thresholds met" if not failures else "FAIL")
    for f in failures:
        print("  -", f)


SUPPORT_TICKETS: list[dict] = generate()

if __name__ == "__main__":
    print_report(SUPPORT_TICKETS)
