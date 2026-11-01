from ola_agent.routing import extract_slots


def test_ticket_only_message_has_no_policy_question():
    s = extract_slots("What is the status of ticket OLA-T-0044?")
    assert s.record_id == "OLA-T-0044" and s.policy_question is None


def test_policy_only_message_has_no_ticket():
    s = extract_slots("Who has to approve a refund above INR 2,000?")
    assert s.record_id is None and s.policy_question == "Who has to approve a refund above INR 2,000?"


def test_mixed_message_drops_status_only_sentences():
    s = extract_slots("Ticket OLA-T-0016 is still open. What happens if I complain about the same issue three times?")
    assert s.record_id == "OLA-T-0016"
    assert s.policy_question == "What happens if I complain about the same issue three times?"
