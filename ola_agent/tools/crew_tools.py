"""CrewAI tool wrappers for the RAG core and the ticket lookup.

Each tool returns JSON with a `source` key so downstream agents can tell results apart
without knowing which tool produced them.
"""
from __future__ import annotations

import json
from collections import Counter

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ola_agent.rag.generate import generate_answer
from ola_agent.tools.ticket_status import check_support_ticket_status

# Evidence for Task 7: how many times each tool actually ran.
TOOL_CALLS: Counter = Counter()


class RagQueryArgs(BaseModel):
    query: str = Field(description="The customer's support-policy question, in plain language.")


class TicketLookupArgs(BaseModel):
    record_id: str = Field(description="Support ticket id, e.g. OLA-T-0012.")


class RagSearchTool(BaseTool):
    name: str = "ola_policy_rag_search"
    description: str = (
        "Searches Ola's support-policy knowledge base and returns a grounded answer with the "
        "retrieved context and source document ids."
    )
    args_schema: type[BaseModel] = RagQueryArgs
    bound_agent_role: str | None = None  # set by governance.make_agent

    def _run(self, query: str) -> str:
        TOOL_CALLS[self.name] += 1
        r = generate_answer(query)
        return json.dumps({"source": "kb_retrieval", **r.model_dump()})


class TicketStatusTool(BaseTool):
    name: str = "check_support_ticket_status"
    description: str = (
        "Looks up one Ola support ticket by record_id and returns its status, resolution_time_hours "
        "and escalation_score."
    )
    args_schema: type[BaseModel] = TicketLookupArgs
    bound_agent_role: str | None = None  # set by governance.make_agent

    def _run(self, record_id: str) -> str:
        from ola_agent.governance import assert_tool_allowed

        assert_tool_allowed(self.name, self.bound_agent_role)  # runtime least-autonomy check
        TOOL_CALLS[self.name] += 1
        return json.dumps({"source": "ticket_lookup", **check_support_ticket_status(record_id)})
