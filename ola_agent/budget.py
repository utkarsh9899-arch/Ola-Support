"""Runtime-layer governance: per-request token / cost budget (Task 15).

Two independent checks, so a request can never silently exceed its budget:

1. Pre-flight: before any LLM call, project the request's total tokens from the (masked) input size:
   projected = BASELINE_TOKENS + INPUT_COPIES * tokens(input). Over the cap -> rejected immediately.
2. Runtime metering: every model call (CrewAI agents, grounded generation, Autogen reviewers) charges its
   prompt + completion tokens to the request's tracker; crossing the cap raises and aborts the request.

Token counts use the same 4-characters-per-token estimate CrewAI applies to models without a tokenizer.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import config

# Measured on normal requests (demos/task15_governance.py prints the measurements): the heaviest
# (ticket + policy, 8 model calls) used 6,407 tokens, of which ~126 came from its 21-token message x 6.
BASELINE_TOKENS = 6500  # overhead of a full crew + review run excluding the user's text, rounded up
INPUT_COPIES = 6  # LLM calls that carry the user's text: 2 retrieval + 2 lookup + 1 composer + 1 generation


class BudgetExceededError(RuntimeError):
    def __init__(self, stage: str, tokens: int, limit: int):
        self.stage, self.tokens, self.limit = stage, tokens, limit
        super().__init__(f"{stage}: {tokens} tokens would exceed the per-request budget of {limit} tokens "
                         f"(~${cost_usd(tokens):.4f} vs cap ${cost_usd(limit):.4f}); request rejected.")


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def cost_usd(tokens: int) -> float:
    return tokens / 1000 * config.COST_PER_1K_TOKENS_USD


def projected_tokens(masked_text: str) -> int:
    return BASELINE_TOKENS + INPUT_COPIES * estimate_tokens(masked_text)


def preflight(masked_text: str, limit: int | None = None) -> int:
    limit = limit or config.MAX_TOKENS_PER_REQUEST
    projected = projected_tokens(masked_text)
    if projected > limit:
        raise BudgetExceededError("pre-flight projection", projected, limit)
    return projected


@dataclass
class BudgetTracker:
    limit: int
    used: int = 0
    calls: list[tuple[str, int]] = field(default_factory=list)

    def check(self, source: str, prompt_tokens: int) -> None:
        """Before a model call: refuse to send a prompt that would take the request over the cap."""
        if self.used + prompt_tokens > self.limit:
            raise BudgetExceededError(
                f"runtime metering refused model call #{len(self.calls) + 1} ({source}, {prompt_tokens} prompt "
                f"tokens; {self.used} already used)", self.used + prompt_tokens, self.limit)

    def charge(self, source: str, prompt_tokens: int, completion_tokens: int) -> None:
        """After a model call: record it; a response that would cross the cap is discarded, never counted."""
        tokens = prompt_tokens + completion_tokens
        if self.used + tokens > self.limit:
            raise BudgetExceededError(
                f"runtime metering discarded model call #{len(self.calls) + 1} ({source}; {self.used} already used)",
                self.used + tokens, self.limit)
        self.used += tokens
        self.calls.append((source, tokens))


_current: ContextVar[BudgetTracker | None] = ContextVar("ola_budget", default=None)


def check(source: str, prompt_tokens: int) -> None:
    tracker = _current.get()
    if tracker is not None:
        tracker.check(source, prompt_tokens)


def charge(source: str, prompt_tokens: int, completion_tokens: int) -> None:
    tracker = _current.get()
    if tracker is not None:
        tracker.charge(source, prompt_tokens, completion_tokens)


def current_tracker() -> BudgetTracker | None:
    return _current.get()


@contextmanager
def request_budget(limit: int | None = None):
    tracker = BudgetTracker(limit=limit or config.MAX_TOKENS_PER_REQUEST)
    token = _current.set(tracker)
    try:
        yield tracker
    finally:
        _current.reset(token)
