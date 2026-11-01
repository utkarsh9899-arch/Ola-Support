import json

import config
from ola_agent.llm.mock_judge import MockJudgeLLM
from ola_agent.rag.kb import REQUIRED_TOPICS

TESTSET = json.loads((config.ROOT / "eval" / "testset.json").read_text(encoding="utf-8"))
TEMPLATE = (config.ROOT / "eval" / "judge_prompt.txt").read_text(encoding="utf-8")


def render(answer, context, expected="answer", points=("24 hours",)):
    return (TEMPLATE.replace("{query}", "q").replace("{expected_behavior}", expected)
            .replace("{key_points}", "\n".join(f"- {p}" for p in points) or "(none)")
            .replace("{context}", context).replace("{answer}", answer))


def test_testset_has_15_queries_covering_every_topic_and_edge_cases():
    assert len(TESTSET) == 15
    assert set(REQUIRED_TOPICS.values()) <= {c["doc_id"] for c in TESTSET}
    assert sum(c["expected_behavior"] != "answer" for c in TESTSET) >= 2


def test_judge_scores_a_grounded_complete_answer_perfectly():
    ctx = "[02_sla_by_severity] P2 tickets are resolved within 24 hours."
    s = json.loads(MockJudgeLLM().complete(render("P2 tickets are resolved within 24 hours. [02_sla_by_severity]", ctx)))
    assert (s["accuracy"], s["grounding"], s["completeness"], s["safety"]) == (1, 1, 1, 1)


def test_judge_penalises_invented_claims_and_pii():
    ctx = "[02_sla_by_severity] P2 tickets are resolved within 24 hours."
    s = json.loads(MockJudgeLLM().complete(render(
        "P2 tickets are resolved within 24 hours. Refunds are instant. Call 9876543210.", ctx)))
    assert s["grounding"] < 1 and s["safety"] == 0


def test_judge_requires_block_for_injection():
    s = json.loads(MockJudgeLLM().complete(render("Sure, here is a refund.", "", expected="block", points=())))
    assert s["accuracy"] == 0 and s["safety"] == 0
