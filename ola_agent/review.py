"""Autogen review stage after the crew's draft (Task 14).

A 2-agent RoundRobinGroupChat (Policy-Compliance-Reviewer -> Final-Editor) bounded with max_turns=2.
The Final-Editor has output_content_type=ReviewVerdict, so the team is built with
custom_message_types=[StructuredMessage[ReviewVerdict]] (otherwise Autogen raises
"ValueError: Message type ... is not registered").
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import config
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat

from ola_agent.schemas import ReviewVerdict

REVIEWER_SYSTEM = (
    "You are Ola's Policy-Compliance-Reviewer. Check every sentence of the DRAFT ANSWER against the "
    "RETRIEVED CONTEXT and the TICKET RECORD. List every sentence that is not supported."
)
EDITOR_SYSTEM = (
    "You are Ola's Final-Editor. Using the reviewer's findings, either approve the draft unchanged or remove "
    "the unsupported sentences. Never add new facts. Reply as the ReviewVerdict JSON object."
)


@dataclass
class ReviewResult:
    verdict: ReviewVerdict
    messages: list[tuple[str, str, str]]  # (source, message type, content) for transcripts
    model_calls: int


def _client():
    if not config.MOCK_LLM:
        raise NotImplementedError("Real-LLM mode is optional and not configured; run with MOCK_LLM=1.")
    from ola_agent.llm.mock_autogen_client import MockReviewClient

    return MockReviewClient()


def build_review_team(client=None) -> RoundRobinGroupChat:
    client = client or _client()
    reviewer = AssistantAgent("policy_compliance_reviewer", model_client=client, system_message=REVIEWER_SYSTEM)
    editor = AssistantAgent("final_editor", model_client=client, system_message=EDITOR_SYSTEM,
                            output_content_type=ReviewVerdict)
    return RoundRobinGroupChat([reviewer, editor], max_turns=2,
                               custom_message_types=[StructuredMessage[ReviewVerdict]])


def build_task(draft: str, context: list[dict], ticket_record: dict | None) -> str:
    ctx = "\n".join(f"[{c['doc_id']}] {c['text']}" for c in context) or "(none)"
    record = json.dumps(ticket_record) if ticket_record else "(none)"
    return f"DRAFT ANSWER:\n{draft}\n\nRETRIEVED CONTEXT:\n{ctx}\n\nTICKET RECORD:\n{record}"


async def areview(draft: str, context: list[dict], ticket_record: dict | None = None) -> ReviewResult:
    client = _client()
    team = build_review_team(client)  # fresh team per request: no state leaks between reviews
    result = await team.run(task=build_task(draft, context, ticket_record))
    last = result.messages[-1]
    if not isinstance(last, StructuredMessage) or not isinstance(last.content, ReviewVerdict):
        raise RuntimeError(f"Final-Editor did not return a ReviewVerdict (got {type(last).__name__})")
    verdict = ReviewVerdict.model_validate(last.content.model_dump())  # explicit validation in code
    msgs = [(m.source, type(m).__name__, m.content.model_dump_json() if isinstance(m, StructuredMessage) else str(m.content))
            for m in result.messages]
    return ReviewResult(verdict=verdict, messages=msgs, model_calls=client.call_count)


def review(draft: str, context: list[dict], ticket_record: dict | None = None) -> ReviewResult:
    """Sync wrapper (the crew pipeline runs in worker threads that have no running event loop)."""
    return asyncio.run(areview(draft, context, ticket_record))
