from ola_agent.routing import extract_slots


def test_ticket_only_message_has_no_policy_question():
    s = extract_slots("What is the status of ticket OLA-T-0044?")
    assert s.record_id == "OLA-T-0044" and s.policy_question is None


def test_policy_only_message_has_no_ticket():
    s = extract_slots("Who has to approve a refund above INR 2,000?")
    assert s.record_id is None and s.policy_question == "Who has to approve a refund above INR 2,000?"


def test_contact_detail_sentences_are_not_policy_questions():
    s = extract_slots("My phone number is [PHONE_REDACTED]. Can you check ticket OLA-T-0005?")
    assert s.record_id == "OLA-T-0005" and s.policy_question is None


def test_followup_resolution_uses_only_this_sessions_history():
    from ola_agent.routing import resolve_references

    hist = ["status of OLA-T-0016?", "Ticket OLA-T-0016 is currently In Progress (priority P2, Billing)"]
    r = resolve_references(extract_slots("Should it be escalated?"), hist)
    assert r.slots.record_id == "OLA-T-0016" and not r.needs_clarification
    fresh = resolve_references(extract_slots("Should it be escalated?"), [])
    assert fresh.slots.record_id is None and fresh.needs_clarification
    p = resolve_references(extract_slots("What is the resolution SLA for that priority?"), hist)
    assert p.slots.policy_question == "What is the resolution SLA for priority P2?"


def test_mixed_message_drops_status_only_sentences():
    s = extract_slots("Ticket OLA-T-0016 is still open. What happens if I complain about the same issue three times?")
    assert s.record_id == "OLA-T-0016"
    assert s.policy_question == "What happens if I complain about the same issue three times?"
