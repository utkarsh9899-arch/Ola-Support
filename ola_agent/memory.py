"""Session memory with LangChain (Task 8).

InMemoryChatMessageHistory per session_id + RunnableWithMessageHistory. In-process only:
history lives in `_STORE` for the life of the process and does not survive a restart.
Only MASKED text ever enters the history (input guardrails run before this layer).
"""
from __future__ import annotations

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from ola_agent import budget, guardrails
from ola_agent.pipeline import PipelineResult, answer_masked, blocked_result

_STORE: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _STORE:
        _STORE[session_id] = InMemoryChatMessageHistory()
    return _STORE[session_id]


def history_texts(session_id: str) -> list[str]:
    return [m.content for m in get_session_history(session_id).messages]


def _answer_with_history(inputs: dict) -> dict:
    history = [m.content for m in inputs.get("history", [])]
    result: PipelineResult = answer_masked(inputs["question"], history, inputs.get("input_flags", []))
    return {"answer": result.response.answer, "result": result}


# RunnableWithMessageHistory emits a LangChainDeprecationWarning pointing to LangGraph persistence.
# That is expected (see brief) and deliberately not silenced; the class works correctly here.
chat_chain = RunnableWithMessageHistory(
    RunnableLambda(_answer_with_history),
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
    output_messages_key="answer",
)


def ask(message: str, session_id: str, token_limit: int | None = None,
        skip_preflight: bool = False) -> PipelineResult:
    """Guard the raw input, check the token budget, then answer it inside the session's memory.

    Raises budget.BudgetExceededError when the request is projected to exceed, or does exceed, the cap.
    `skip_preflight` exists only to demonstrate that the runtime meter alone also stops a request.
    """
    guard = guardrails.guard_input(message)
    if guard.blocked:
        return blocked_result(guard)  # never reaches the crew or the session history
    projected = 0 if skip_preflight else budget.preflight(guard.masked_text, token_limit)
    with budget.request_budget(token_limit) as tracker:
        out = chat_chain.invoke(
            {"question": guard.masked_text, "input_flags": guard.flags},
            config={"configurable": {"session_id": session_id}},
        )
    result: PipelineResult = out["result"]
    result.tokens_projected, result.tokens_used = projected, tracker.used
    return result
