"""Task 16 demo: a repeated query hits the grounded-generation cache and skips retrieval + the LLM call."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402,F401
from demos._transcript import transcript  # noqa: E402
from ola_agent.memory import ask  # noqa: E402
from ola_agent.rag.cache import GROUNDED_CACHE, normalize_query  # noqa: E402
from ola_agent.rag.generate import generate_answer, get_text_llm  # noqa: E402
from ola_agent.rag.index import RETRIEVAL_CALLS, build_index  # noqa: E402
from ola_agent.tools.crew_tools import TOOL_CALLS  # noqa: E402


def counters() -> dict:
    return {"retrieval_calls": RETRIEVAL_CALLS["count"], "generation_llm_calls": get_text_llm().call_count,
            "cache_hits": GROUNDED_CACHE.hits, "cache_misses": GROUNDED_CACHE.misses}


def timed(fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    return out, (time.perf_counter() - start) * 1000


if __name__ == "__main__":
    build_index()
    GROUNDED_CACHE.clear()
    with transcript("task16_cache"):
        q1 = "What is the refund policy if I was charged twice?"
        q2 = "  what is the REFUND policy if I was charged twice  "
        print(f"normalize({q1!r}) = {normalize_query(q1)!r}")
        print(f"normalize({q2!r}) = {normalize_query(q2)!r}\n")

        print("=" * 100 + "\nA) Grounded-generation step called directly\n" + "=" * 100)
        c0 = counters()
        a1, t1 = timed(generate_answer, q1)
        c1 = counters()
        a2, t2 = timed(generate_answer, q2)
        c2 = counters()
        print(f"before          : {c0}")
        print(f"after 1st query : {c1}   ({t1:.1f} ms)  <- MISS: retrieval + LLM call")
        print(f"after 2nd query : {c2}   ({t2:.2f} ms)  <- HIT: no retrieval, no LLM call")
        print(f"generation-LLM calls made: 1st query = {c1['generation_llm_calls'] - c0['generation_llm_calls']}, "
              f"repeat = {c2['generation_llm_calls'] - c1['generation_llm_calls']} (the redundant call was avoided)")
        print(f"same answer     : {a1.answer == a2.answer}   speed-up: {t1 / max(t2, 1e-6):.0f}x\n")

        print("=" * 100 + "\nB) Through the full pipeline (guardrails -> crew -> RAG tool -> review)\n" + "=" * 100)
        q3 = "How long are closed tickets stored?"
        for label, q in (("1st", q3), ("repeat", q3.upper() + "  ")):
            before, tool_before = counters(), TOOL_CALLS["ola_policy_rag_search"]
            r, t = timed(ask, q, f"cache-{label}")
            after = counters()
            print(f"{label:<6} query {q!r}")
            print(f"       RAG tool invoked by Retrieval Agent: {TOOL_CALLS['ola_policy_rag_search'] - tool_before}x | "
                  f"retrieval calls +{after['retrieval_calls'] - before['retrieval_calls']} | "
                  f"generation LLM calls +{after['generation_llm_calls'] - before['generation_llm_calls']} | "
                  f"cache hits +{after['cache_hits'] - before['cache_hits']} | {t:.0f} ms total")
            print(f"       answer: {r.response.answer[:120]}...")
        print("\nThe crew still runs on the repeat (agents, review), but the grounded-generation step is served from")
        print("the cache: zero retrieval calls and zero generation-LLM calls for the repeated query.")
        print(f"\nCache size: {len(GROUNDED_CACHE)} entries | final counters: {counters()}")
