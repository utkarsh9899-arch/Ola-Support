import dataset

REQUIRED_FIELDS = {"record_id", "category", "status", "resolution_time_hours", "days_since_created", "escalated"}


def test_dataset_meets_all_structural_thresholds():
    assert dataset.validate(dataset.SUPPORT_TICKETS) == []


def test_dataset_is_deterministic():
    assert dataset.generate() == dataset.generate() == dataset.SUPPORT_TICKETS


def test_every_record_has_required_fields_and_types():
    for t in dataset.SUPPORT_TICKETS:
        assert REQUIRED_FIELDS <= t.keys()
        assert isinstance(t["days_since_created"], int) and 0 <= t["days_since_created"] <= 30
        assert isinstance(t["escalated"], bool)
        assert 0.5 <= t["resolution_time_hours"] <= 72


def test_escalated_status_implies_escalated_flag():
    assert all(t["escalated"] for t in dataset.SUPPORT_TICKETS if t["status"] == "Escalated")


def test_priorities_agree_with_kb_priority_rules():
    for t in dataset.SUPPORT_TICKETS:
        assert t["priority"] in dataset.PRIORITY_WEIGHTS_BY_CATEGORY[t["category"]]
        if t["category"] in ("Billing", "Account Access", "General Inquiry"):
            assert t["priority"] != "P1"


def test_validate_catches_out_of_band_escalation():
    forced = [dict(t, escalated=True) for t in dataset.SUPPORT_TICKETS]
    assert any("escalated" in f for f in dataset.validate(forced))
