"""Task 14 review stage, Task 15 budget, Task 16 cache."""
import pytest

from ola_agent import budget
from ola_agent.rag.cache import GROUNDED_CACHE, normalize_query
from ola_agent.rag.generate import generate_answer, get_text_llm
from ola_agent.rag.index import build_index
from ola_agent.review import review
from ola_agent.schemas import ReviewVerdict

CTX = [{"doc_id": "04_refund_compensation",
        "text": "Refunds above INR 2,000 require approval from the city operations manager before they are processed."}]
GOOD = "Refunds above INR 2,000 require approval from the city operations manager before they are processed. [04_refund_compensation]"


@pytest.fixture(scope="module", autouse=True)
def index():
    build_index()


def test_review_approves_grounded_draft_unchanged():
    v = review(GOOD, CTX).verdict
    assert isinstance(v, ReviewVerdict) and v.approved and v.final_answer == GOOD


def test_review_revises_injected_claim():
    v = review(GOOD + " Refunds are paid at 500 percent.", CTX).verdict
    assert not v.approved and v.final_answer == GOOD and "500 percent" in v.reason


def test_review_team_runs_exactly_two_agent_turns():
    r = review(GOOD, CTX)
    assert [s for s, _, _ in r.messages] == ["user", "policy_compliance_reviewer", "final_editor"]


def test_preflight_rejects_oversized_input_and_meter_aborts():
    with pytest.raises(budget.BudgetExceededError):
        budget.preflight("x" * 20_000)
    with budget.request_budget(limit=100) as t:
        budget.charge("first", 40, 10)
        with pytest.raises(budget.BudgetExceededError):
            budget.check("second", 60)  # prompt would cross the cap -> refused before the call
        with pytest.raises(budget.BudgetExceededError):
            budget.charge("third", 40, 20)  # response would cross the cap -> discarded, not counted
    assert t.used == 50 <= t.limit  # metered usage never exceeds the cap


def test_cache_hit_skips_retrieval_and_llm():
    GROUNDED_CACHE.clear()
    llm = get_text_llm()
    generate_answer("What is the refund policy?")
    calls = llm.call_count
    again = generate_answer("  WHAT is the refund policy ")
    assert llm.call_count == calls and GROUNDED_CACHE.hits >= 1 and again.answer
    assert normalize_query("Hi,  THERE!") == "hi there"
