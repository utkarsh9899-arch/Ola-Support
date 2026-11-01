"""The shared query set for Tasks 4 and 5 (same >=5 in-scope queries for both).

`relevant` is the hand-labelled set of parent documents that answer each query.
"""

EVAL_QUERIES = [
    {
        "query": "How quickly must a P1 ticket get a first response and be resolved?",
        "relevant": {"02_sla_by_severity"},
    },
    {
        "query": "Who has to approve a refund above INR 2,000?",
        "relevant": {"04_refund_compensation", "03_escalation_matrix"},
    },
    {
        "query": "What compensation do I get if my driver arrives 20 minutes late?",
        "relevant": {"08_service_credits", "04_refund_compensation"},
    },
    {
        "query": "How does Ola keep customers informed during an outage?",
        "relevant": {"11_outage_communication"},
    },
    {
        "query": "How long are closed tickets stored, and can I ask for mine to be deleted?",
        "relevant": {"12_data_retention"},
    },
]

OUT_OF_SCOPE_QUERY = "Can you recommend a good movie to watch this weekend?"
