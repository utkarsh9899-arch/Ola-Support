import statistics

import dataset
from ola_agent.tools import ticket_status as ts


def test_threshold_is_derived_from_the_dataset_80th_percentile():
    days = [t["days_since_created"] for t in dataset.SUPPORT_TICKETS]
    p80 = statistics.quantiles(days, n=10, method="inclusive")[7]
    assert p80 == ts.AGE_P80_DAYS
    assert ts.ESCALATION_THRESHOLD == round(ts.W_AGE * p80 / 30, 2) == 0.27


def test_score_is_designed_not_a_boolean_or():
    assert ts.escalation_score(False, 0) == 0.0
    assert ts.escalation_score(False, 15) == 0.2
    assert ts.escalation_score(True, 0) == 0.6
    assert ts.escalation_score(True, 30) == 1.0
    assert 0.0 < ts.escalation_score(False, 29) < ts.escalation_score(True, 0)


def test_every_score_in_unit_interval():
    for t in dataset.SUPPORT_TICKETS:
        r = ts.check_support_ticket_status(t["record_id"])
        assert 0.0 <= r["escalation_score"] <= 1.0
        assert r["status"] == t["status"] and r["resolution_time_hours"] == t["resolution_time_hours"]


def test_lookup_accepts_loose_id_formats_and_handles_unknown():
    assert ts.check_support_ticket_status("ola-t-7")["record_id"] == "OLA-T-0007"
    assert ts.check_support_ticket_status("OLA-T-9999")["found"] is False
    assert ts.check_support_ticket_status("hello")["found"] is False


def test_closed_tickets_never_recommended_for_escalation():
    for t in dataset.SUPPORT_TICKETS:
        r = ts.check_support_ticket_status(t["record_id"])
        if t["status"] in ("Resolved", "Closed"):
            assert r["recommend_escalation"] is False
