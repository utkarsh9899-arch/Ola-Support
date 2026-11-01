"""Application-layer governance: least autonomy for tools (Task 15).

TOOL_ACL lists which agent roles may hold each tool. Agents are only built through
`make_agent`, which refuses to wire a tool to any role not on its list, and the ticket
tool re-checks its bound role every time it runs.
"""
from __future__ import annotations

TOOL_ACL: dict[str, set[str]] = {
    "check_support_ticket_status": {"Lookup Agent"},
    "ola_policy_rag_search": {"Retrieval Agent"},
}


class ToolPermissionError(PermissionError):
    pass


def assert_tool_allowed(tool_name: str, role: str | None) -> None:
    allowed = TOOL_ACL.get(tool_name)
    if allowed is None or role not in allowed:
        raise ToolPermissionError(
            f"Least-autonomy violation: agent role {role!r} may not use tool {tool_name!r} "
            f"(allowed: {sorted(allowed or [])})."
        )


def make_agent(role: str, tools: list | None = None, **agent_kwargs):
    """Build a CrewAI Agent, enforcing TOOL_ACL before any tool is attached."""
    from crewai import Agent

    tools = tools or []
    for tool in tools:
        assert_tool_allowed(tool.name, role)
        tool.bound_agent_role = role
    return Agent(role=role, tools=tools, **agent_kwargs)
