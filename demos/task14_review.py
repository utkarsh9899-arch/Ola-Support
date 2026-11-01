"""Task 14 demo: Autogen RoundRobinGroupChat review stage approving one draft and revising another."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402,F401
from autogen_agentchat.agents import AssistantAgent  # noqa: E402
from autogen_agentchat.teams import RoundRobinGroupChat  # noqa: E402

from demos._transcript import transcript  # noqa: E402
from ola_agent.llm.mock_autogen_client import MockReviewClient  # noqa: E402
from ola_agent.memory import ask  # noqa: E402
from ola_agent.rag.index import build_index  # noqa: E402
from ola_agent.review import build_review_team, build_task, review  # noqa: E402
from ola_agent.schemas import ReviewVerdict  # noqa: E402

INJECTED = "Ola refunds 500 percent of the fare within 5 minutes for any delay."


def show(title: str, draft: str, context: list, record: dict | None) -> None:
    print("=" * 100 + f"\n{title}\n" + "=" * 100)
    print(f"Composer draft: {draft}\n")
    r = review(draft, context, record)
    for source, mtype, content in r.messages:
        shown = content if source != "user" else content.replace("\n", "\n            ")
        print(f"[{source} | {mtype}]\n            {shown}\n")
    v = r.verdict
    print(f"ReviewVerdict (validated Pydantic): approved={v.approved}")
    print(f"  final_answer: {v.final_answer}")
    print(f"  reason      : {v.reason}")
    print(f"  unchanged   : {v.final_answer == draft}   | messages in run: {len(r.messages)} (task + 2 agent turns) "
          f"| model calls: {r.model_calls}\n")


if __name__ == "__main__":
    build_index()
    with transcript("task14_review"):
        team = build_review_team(MockReviewClient())
        print("Team: RoundRobinGroupChat([policy_compliance_reviewer, final_editor], max_turns=2,")
        print("      custom_message_types=[StructuredMessage[ReviewVerdict]])")
        print(f"      participants={[p.name for p in team._participants]}  max_turns={team._max_turns}")
        print("Final-Editor: AssistantAgent(..., output_content_type=ReviewVerdict)\n")

        # Real CrewAI drafts are the input. Query 1: approved unchanged.
        crew = ask("Who has to approve a refund above INR 2,000?", "review-demo-1")
        draft1 = crew.response.answer
        show("CASE 1 - real CrewAI Composer draft (grounded) -> expected: APPROVED UNCHANGED",
             draft1, crew.context, crew.lookup)

        # Query 2: a real ticket+policy draft with an ungrounded claim deliberately injected for the test.
        crew2 = ask("Ticket OLA-T-0044: what compensation do I get if my driver arrives 20 minutes late?", "review-demo-2")
        draft2 = f"{crew2.response.answer} {INJECTED}"
        show("CASE 2 - same kind of draft with an ungrounded claim injected -> expected: REVISED",
             draft2, crew2.context, crew2.lookup)

        print("=" * 100 + "\nWhy custom_message_types is required (same team without it):\n" + "=" * 100)
        c = MockReviewClient()
        bad = RoundRobinGroupChat([AssistantAgent("policy_compliance_reviewer", model_client=c),
                                   AssistantAgent("final_editor", model_client=c, output_content_type=ReviewVerdict)],
                                  max_turns=2)
        try:
            asyncio.run(bad.run(task=build_task(draft1, crew.context, crew.lookup)))
            print("no error")
        except Exception as e:  # noqa: BLE001
            print(f"{type(e).__name__}: {str(e)[:200]}")
