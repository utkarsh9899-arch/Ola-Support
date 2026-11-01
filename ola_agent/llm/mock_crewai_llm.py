"""Deterministic MOCK_LLM for CrewAI, built on CrewAI's own extension point `BaseLLM`.

It speaks CrewAI's text ReAct protocol (supports_function_calling() is False):

  1st call for an agent with tools -> "Thought / Action / Action Input"
  call after the tool ran           -> "Thought / Final Answer: <tool result>"
  agent without tools (Composer)    -> "Final Answer: <SupportResponse JSON>"

The two silent pitfalls named in the brief are handled explicitly:

1. CrewAI's system prompt literally contains "Observation: the result of the action".
   `_latest_observation` never reads system/user prompt text: it only looks inside
   assistant messages (text this mock generated, to which CrewAI appends the real
   "\nObservation: <result>") and only AFTER this mock's own last "Action Input:".
   On the first call there is no such message, so no placeholder can leak out.

2. Tools are never chosen by name. `_plan_tool_call` reads each tool's declared
   `args_schema` and fills every field from a slot in the task (`record_id` field <- the
   ticket id slot, `query` field <- the policy-question slot). A tool literally named
   `rag_lookup` is therefore dispatched by what it accepts, not by what it is called.
"""
from __future__ import annotations

import json
import re
from typing import Any

from crewai.llms.base_llm import BaseLLM
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ola_agent.rag.generate import IDK_ANSWER

SLOT_RE = {
    "record_id": re.compile(r"^TICKET_ID:\s*(.+)$", re.M),
    "query": re.compile(r"^POLICY_QUESTION:\s*(.+)$", re.M),
}
NOT_FOUND_HINT = "Please check the id and try again."  # fixed helper sentence; makes no factual claim
ACTION_INPUT_MARK = "Action Input:"
OBSERVATION_MARK = "Observation:"
NONE_VALUES = {"", "NONE", "N/A"}


def _content(msg: Any) -> str:
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
    if isinstance(content, list):  # multimodal-style content blocks
        content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def _role(msg: Any) -> str:
    return (msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")) or ""


def _json_objects(text: str) -> list[dict]:
    """Every top-level JSON object embedded in free text."""
    decoder, out, i = json.JSONDecoder(), [], 0
    while (i := text.find("{", i)) != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
            if isinstance(obj, dict):
                out.append(obj)
            i = end
        except json.JSONDecodeError:
            i += 1
    return out


class MockCrewLLM(BaseLLM):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "mock/ola-react-v1"
    llm_type: str = "mock"
    agent_tools: list[Any] = Field(default_factory=list, exclude=True)
    response_format: type[BaseModel] | None = Field(default=None, exclude=True)

    _calls: list[dict] = PrivateAttr(default_factory=list)

    # --- CrewAI capability flags ------------------------------------------------
    def supports_function_calling(self) -> bool:
        return False  # use the text ReAct protocol

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192

    def get_input_tokens_count(self, messages, tools=None, response_model=None) -> int:
        text = messages if isinstance(messages, str) else " ".join(_content(m) for m in messages)
        return len(text) // 4  # same 4-chars-per-token estimate CrewAI uses, without its warning

    @property
    def calls(self) -> list[dict]:
        return self._calls

    # --- main entry point ---------------------------------------------------------
    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None) -> str:
        msgs = [{"role": "user", "content": messages}] if isinstance(messages, str) else list(messages)
        task_text = "\n".join(_content(m) for m in msgs if _role(m) == "user")

        observation = self._latest_observation(msgs)
        if observation is not None:
            out = f"Thought: I now know the final answer\nFinal Answer: {observation}"
            kind = "final_from_observation"
        elif self.agent_tools:
            plan = self._plan_tool_call(task_text)
            if plan:
                tool, args = plan
                out = f"Thought: I need {tool.name} for this.\nAction: {tool.name}\n{ACTION_INPUT_MARK} {json.dumps(args)}"
                kind = "action"
            else:
                out = ("Thought: The task gives none of the inputs my tools need, so no tool call is needed.\n"
                       'Final Answer: {"skipped": true, "reason": "no input for this agent\'s tools"}')
                kind = "skip"
        else:
            out = "Thought: I can now compose the final answer.\nFinal Answer: " + self._compose(task_text, response_model)
            kind = "compose"
        self._calls.append({"kind": kind, "output": out})
        return out

    # --- pitfall 1: read only this mock's own text and what follows it -------------
    @staticmethod
    def _latest_observation(msgs: list) -> str | None:
        own = [i for i, m in enumerate(msgs) if _role(m) == "assistant" and ACTION_INPUT_MARK in _content(m)]
        if not own:
            return None  # first call: nothing has run yet, whatever the system prompt says
        last = own[-1]
        after_action = _content(msgs[last]).split(ACTION_INPUT_MARK)[-1]
        if OBSERVATION_MARK in after_action:
            return after_action.split(OBSERVATION_MARK, 1)[1].strip()
        for m in msgs[last + 1:]:  # some executors send the result as a separate tool message
            if _role(m) in ("tool", "function"):
                return _content(m).strip()
        return None

    # --- pitfall 2: dispatch from the declared argument schema, never the name ----
    def _plan_tool_call(self, task_text: str) -> tuple[Any, dict] | None:
        slots = {name: (rx.search(task_text).group(1).strip() if rx.search(task_text) else "")
                 for name, rx in SLOT_RE.items()}
        for tool in self.agent_tools:
            schema = getattr(tool, "args_schema", None)
            fields = getattr(schema, "model_fields", {}) if schema else {}
            args = {}
            for field_name in fields:
                value = slots.get(field_name, "")
                if value.upper() in NONE_VALUES:
                    break
                args[field_name] = value
            else:
                if args:
                    return tool, args
        return None

    # --- Response Composer --------------------------------------------------------
    def _compose(self, task_text: str, response_model: type[BaseModel] | None) -> str:
        results = {o["source"]: o for o in _json_objects(task_text) if o.get("source")}
        rag, ticket = results.get("kb_retrieval"), results.get("ticket_lookup")

        parts, sources, intent = [], [], "out_of_scope"
        payload: dict[str, Any] = {"record_id": None, "ticket_status": None, "escalation_score": None,
                                   "recommend_escalation": None}
        if ticket and ticket.get("found"):
            advice = "escalation is recommended" if ticket["recommend_escalation"] else "no escalation is needed"
            parts.append(
                f"Ticket {ticket['record_id']} is currently {ticket['status']} (priority {ticket['priority']}, "
                f"{ticket['category']}), with a resolution time of {ticket['resolution_time_hours']} hours. "
                f"Its escalation score is {ticket['escalation_score']} against a threshold of "
                f"{ticket['escalation_threshold']}, so {advice}."
            )
            payload.update(record_id=ticket["record_id"], ticket_status=ticket["status"],
                           escalation_score=ticket["escalation_score"],
                           recommend_escalation=ticket["recommend_escalation"])
            intent = "ticket_status"
        elif ticket:
            parts.append(f"I couldn't find a support ticket with id {ticket['record_id']}. {NOT_FOUND_HINT}")
            intent = "ticket_status"

        if rag and not rag.get("fallback"):
            parts.append(rag["answer"])
            sources = rag["sources"]
            intent = "mixed" if intent == "ticket_status" else "policy"
        elif rag and not parts:
            parts.append(IDK_ANSWER)

        refused = not parts or parts == [IDK_ANSWER]
        response = {
            "answer": " ".join(parts) if parts else IDK_ANSWER,
            "intent": "out_of_scope" if refused else intent,
            "sources": sources,
            "grounded": not refused,
            "refused": refused,
            "guardrail_flags": [],
            **payload,
        }
        model = response_model or self.response_format
        if model is not None:
            response = model.model_validate(response).model_dump()
        return json.dumps(response)
