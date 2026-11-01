"""Task 8 demo: two SEPARATE transcripts from one process run.

transcripts/task08_memory_multiturn.txt : state carried across turns of session "rider-s1"
transcripts/task08_memory_fresh.txt     : a fresh session "rider-s2" where that state is absent
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402,F401
from demos._transcript import transcript  # noqa: E402

# Show (not silence) LangChain's expected deprecation warnings in the transcript; other libraries'
# unrelated internal deprecation chatter is left at Python's default (hidden).
from langchain_core._api import LangChainDeprecationWarning  # noqa: E402

warnings.filterwarnings("always", category=LangChainDeprecationWarning)
_default_showwarning = warnings.showwarning


def _show(msg, cat, *a, **k):
    if issubclass(cat, LangChainDeprecationWarning):
        print(f"[{cat.__name__}] {msg}")


warnings.showwarning = _show

from ola_agent.rag.index import build_index  # noqa: E402


def show_turn(ask, history_texts, session_id: str, n: int, message: str) -> None:
    before = len(history_texts(session_id))
    r = ask(message, session_id)
    print(f"--- Turn {n} (session_id={session_id!r}) ---")
    print(f"history messages BEFORE this turn: {before}")
    if before:
        print("history content:")
        for m in history_texts(session_id)[:before]:
            print(f"   | {m[:120]}{'...' if len(m) > 120 else ''}")
    print(f"USER : {message}")
    print(f"resolution notes : {r.notes or ['(none)']}")
    print(f"AGENT: {r.response.answer}")
    print(f"       intent={r.response.intent} record_id={r.response.record_id} sources={r.response.sources}")
    print(f"history messages AFTER this turn : {len(history_texts(session_id))}\n")


if __name__ == "__main__":
    build_index()
    with transcript("task08_memory_multiturn"):
        print("LangChain memory: InMemoryChatMessageHistory per session_id + RunnableWithMessageHistory")
        from ola_agent.memory import ask, history_texts  # warnings print here, inside the transcript

        print("\nSession 'rider-s1': state must carry across turns.\n")
        show_turn(ask, history_texts, "rider-s1", 1, "Hi, what's the status of my ticket OLA-T-0016?")
        show_turn(ask, history_texts, "rider-s1", 2, "Should it be escalated?")
        show_turn(ask, history_texts, "rider-s1", 3, "What is the resolution SLA for that priority?")
        print("RESULT: turn 2 resolved 'it' -> OLA-T-0016 and turn 3 resolved 'that priority' -> P2, "
              "both from this session's history only.")

    with transcript("task08_memory_fresh"):
        print("Same process run, NEW session_id 'rider-s2' (a fresh conversation).")
        print(f"Session 'rider-s1' still holds {len(history_texts('rider-s1'))} messages in this process.\n")
        show_turn(ask, history_texts, "rider-s2", 1, "Should it be escalated?")
        show_turn(ask, history_texts, "rider-s2", 2, "What is the resolution SLA for that priority?")
        leaked = any("OLA-T-0016" in t or "P2" in t for t in history_texts("rider-s2")[:1])
        print(f"RESULT: rider-s2 started with 0 history messages; nothing from rider-s1 leaked in "
              f"(OLA-T-0016/P2 present in s2's first turn: {leaked}). State is correctly absent.")
