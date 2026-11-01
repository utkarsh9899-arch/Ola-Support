"""Deterministic MOCK_LLM model client for the Autogen review team (Task 14).

Implements autogen_core's ChatCompletionClient. Behaviour depends on what Autogen asks for:

* plain text call (Policy-Compliance-Reviewer): checks every sentence of the DRAFT ANSWER against the
  RETRIEVED CONTEXT / TICKET RECORD in the task message and reports the unsupported ones on a line
  `UNSUPPORTED_SENTENCES=<json list>`.
* json_output=<Pydantic model> call (Final-Editor with output_content_type): reads the reviewer's findings
  and returns that model as JSON - approve the draft unchanged, or remove the unsupported sentences.

Findings are only read from non-system messages written by the reviewer, never from a system prompt.
"""
from __future__ import annotations

import json
import re
from typing import Any, AsyncGenerator, Mapping, Optional, Sequence

from autogen_core import CancellationToken
from autogen_core.models import (ChatCompletionClient, CreateResult, LLMMessage, ModelCapabilities, ModelFamily,
                                 ModelInfo, RequestUsage, SystemMessage)
from pydantic import BaseModel

from ola_agent import guardrails
from ola_agent.llm.mock_crewai_llm import NOT_FOUND_HINT

DRAFT_RE = re.compile(r"DRAFT ANSWER:\n(.*?)\n\nRETRIEVED CONTEXT:\n(.*?)\n\nTICKET RECORD:\n(.*?)\s*$", re.S)
FINDINGS_RE = re.compile(r"^UNSUPPORTED_SENTENCES=(\[.*\])$", re.M)
NO_CLAIM = {NOT_FOUND_HINT, *guardrails.answer_sentences(guardrails.UNGROUNDED_POLICY_NOTE)}
IDK_AFTER_REVIEW = "I don't know. None of the draft could be verified against the retrieved context."


def _text(m: LLMMessage) -> str:
    content = getattr(m, "content", "")
    return content if isinstance(content, str) else ""


class MockReviewClient(ChatCompletionClient):
    def __init__(self) -> None:
        self.call_count = 0
        self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    # --- the two agent behaviours ------------------------------------------------
    @staticmethod
    def _parse_task(messages: Sequence[LLMMessage]) -> tuple[str, str, str] | None:
        for m in messages:
            if not isinstance(m, SystemMessage) and (match := DRAFT_RE.search(_text(m))):
                return match.group(1).strip(), match.group(2), match.group(3)
        return None

    def _review(self, messages: Sequence[LLMMessage]) -> str:
        parsed = self._parse_task(messages)
        if parsed is None:
            return "COMPLIANCE REVIEW: no draft found.\nUNSUPPORTED_SENTENCES=[]"
        draft, context, record = parsed
        sentences = guardrails.answer_sentences(draft)
        unsupported = [s for s in sentences if not guardrails.sentence_supported(s, context, record, NO_CLAIM)]
        verdict = ("All sentences are supported; recommend approval." if not unsupported
                   else f"{len(unsupported)} sentence(s) are not supported by the retrieved context or ticket record.")
        return (f"COMPLIANCE REVIEW: checked {len(sentences)} sentence(s). {verdict}\n"
                f"UNSUPPORTED_SENTENCES={json.dumps(unsupported)}")

    def _edit(self, messages: Sequence[LLMMessage], model: type[BaseModel]) -> str:
        parsed = self._parse_task(messages)
        draft = parsed[0] if parsed else ""
        findings: list[str] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                continue  # never read findings from a system prompt
            for match in FINDINGS_RE.finditer(_text(m)):
                findings = json.loads(match.group(1))  # keep the latest reviewer finding
        if not findings:
            verdict = {"approved": True, "final_answer": draft,
                       "reason": "Policy-Compliance-Reviewer found every sentence supported; approved unchanged."}
        else:
            revised = draft
            for s in findings:
                revised = re.sub(re.escape(s) + r"(\s*\[\d\d_[a-z_]+\])?\s*", "", revised)
            revised = revised.strip() or IDK_AFTER_REVIEW
            verdict = {"approved": False, "final_answer": revised,
                       "reason": f"Removed {len(findings)} sentence(s) not supported by the retrieved context: "
                                 + "; ".join(f'"{s}"' for s in findings)}
        return model.model_validate(verdict).model_dump_json()

    # --- ChatCompletionClient API ------------------------------------------------
    async def create(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = [],
                     tool_choice: Any = "auto", json_output: Optional[bool | type[BaseModel]] = None,
                     extra_create_args: Mapping[str, Any] = {},
                     cancellation_token: Optional[CancellationToken] = None) -> CreateResult:
        from ola_agent.budget import check

        check("autogen", self.count_tokens(messages))
        self.call_count += 1
        if isinstance(json_output, type) and issubclass(json_output, BaseModel):
            content = self._edit(messages, json_output)
        else:
            content = self._review(messages)
        prompt_tokens = self.count_tokens(messages)
        usage = RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=len(content) // 4)
        self._usage = RequestUsage(self._usage.prompt_tokens + usage.prompt_tokens,
                                   self._usage.completion_tokens + usage.completion_tokens)
        from ola_agent.budget import charge

        charge("autogen:" + ("editor" if json_output else "reviewer"), usage.prompt_tokens, usage.completion_tokens)
        return CreateResult(finish_reason="stop", content=content, usage=usage, cached=False)

    async def create_stream(self, messages: Sequence[LLMMessage], **kwargs: Any) -> AsyncGenerator[str | CreateResult, None]:
        yield await self.create(messages, **kwargs)

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._usage

    def total_usage(self) -> RequestUsage:
        return self._usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = []) -> int:
        return sum(len(_text(m)) for m in messages) // 4

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Any] = []) -> int:
        return 8192 - self.count_tokens(messages)

    @property
    def capabilities(self) -> ModelCapabilities:  # deprecated in autogen, still abstract
        return ModelCapabilities(vision=False, function_calling=False, json_output=True)  # type: ignore[typeddict-item]

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(vision=False, function_calling=False, json_output=True, family=ModelFamily.UNKNOWN,
                         structured_output=True, multiple_system_messages=True)
