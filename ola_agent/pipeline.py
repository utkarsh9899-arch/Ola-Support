"""End-to-end request pipeline.

input guardrails (PII mask, injection) -> session memory -> follow-up resolution -> CrewAI crew
-> SupportResponse validation -> output groundedness guardrail
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ola_agent import guardrails
from ola_agent.routing import extract_slots, resolve_references
from ola_agent.schemas import SupportResponse

CLARIFY_ANSWER = "Which ticket do you mean? Please share the ticket id (for example OLA-T-0012) and I'll check it."


@dataclass
class PipelineResult:
    response: SupportResponse
    masked_message: str
    retrieval: dict | None = None  # Retrieval Agent's tool output (context, similarity) when it ran
    lookup: dict | None = None  # Lookup Agent's tool output when it ran
    notes: list[str] = field(default_factory=list)
    crew_ran: bool = False

    @property
    def context(self) -> list[dict]:
        return (self.retrieval or {}).get("context", [])


def _task_json(raw: str) -> dict | None:
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) and "source" in obj else None
    except json.JSONDecodeError:
        return None


def blocked_result(guard: guardrails.InputGuardResult) -> PipelineResult:
    resp = SupportResponse(answer=guardrails.BLOCKED_ANSWER, intent="blocked", grounded=True, refused=True,
                           guardrail_flags=guard.flags)
    return PipelineResult(response=resp, masked_message=guard.masked_text, notes=["input blocked before the crew ran"])


def answer_masked(masked_message: str, history_texts: list[str], input_flags: list[str]) -> PipelineResult:
    """Run the crew on an already-guarded (masked, non-injection) message."""
    from ola_agent.crew import run_crew_slots

    resolution = resolve_references(extract_slots(masked_message), history_texts)
    if resolution.needs_clarification:
        resp = SupportResponse(answer=CLARIFY_ANSWER, intent="ticket_status", grounded=True,
                               guardrail_flags=list(input_flags))
        return PipelineResult(response=resp, masked_message=masked_message, notes=resolution.notes)

    response, output = run_crew_slots(resolution.slots)
    retrieval = _task_json(output.tasks_output[0].raw)
    lookup = _task_json(output.tasks_output[1].raw)
    response = SupportResponse.model_validate({**response.model_dump(),
                                               "guardrail_flags": list(input_flags) + response.guardrail_flags})
    result = PipelineResult(response=response, masked_message=masked_message, retrieval=retrieval, lookup=lookup,
                            notes=resolution.notes, crew_ran=True)
    return apply_output_guard(result)


def apply_output_guard(result: PipelineResult) -> PipelineResult:
    """Refuse the policy part when the retrieved context does not support it."""
    rag = result.retrieval
    if rag is None:
        return result  # no policy question was asked; ticket facts come straight from the record
    policy_answer = "" if rag.get("fallback") else rag["answer"]
    check = guardrails.check_groundedness(policy_answer, rag.get("context", []), rag.get("top_similarity", 0.0))
    if check.supported:
        return result
    resp = result.response
    ticket_found = bool(result.lookup and result.lookup.get("found"))
    ticket_part = (resp.answer.replace(policy_answer, "") if policy_answer else resp.answer).strip() if ticket_found else ""
    if ticket_part:
        answer = f"{ticket_part} {guardrails.UNGROUNDED_POLICY_NOTE}"
        intent, refused = "ticket_status", False
    else:
        answer, intent, refused = guardrails.UNGROUNDED_ANSWER, "out_of_scope", True
    result.response = SupportResponse.model_validate({
        **resp.model_dump(), "answer": answer, "intent": intent, "sources": [], "grounded": not refused,
        "refused": refused, "guardrail_flags": resp.guardrail_flags + [f"groundedness_refusal: {check.reason}"],
    })
    result.notes.append(f"output guard: {check.reason}" + (f" {check.unsupported_sentences}" if check.unsupported_sentences else ""))
    return result
