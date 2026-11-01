"""Task 7 demo: 3-agent CrewAI crew run via .kickoff(); RAG tool and lookup tool invoked on different queries."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from demos._transcript import transcript  # noqa: E402
from ola_agent.crew import build_crew, kickoff_inputs  # noqa: E402
from ola_agent.rag.index import build_index  # noqa: E402
from ola_agent.routing import extract_slots  # noqa: E402
from ola_agent.schemas import SupportResponse  # noqa: E402
from ola_agent.tools.crew_tools import TOOL_CALLS  # noqa: E402

QUERIES = [
    "Who has to approve a refund above INR 2,000?",              # policy only -> RAG tool
    "What is the status of ticket OLA-T-0044?",                  # ticket only -> lookup tool
    "Ticket OLA-T-0016 is still open. What happens if I complain about the same issue three times?",  # both
]

if __name__ == "__main__":
    build_index()
    with transcript("task07_crew"):
        print(f"MOCK_LLM={config.MOCK_LLM}  CREWAI_DISABLE_TELEMETRY={__import__('os').environ['CREWAI_DISABLE_TELEMETRY']}\n")
        for i, q in enumerate(QUERIES, 1):
            crew = build_crew(verbose=False)
            if i == 1:
                print("Crew agents and their tools:")
                for a in crew.agents:
                    print(f"  - {a.role:<18} tools={[t.name for t in a.tools]}  llm={type(a.llm).__name__}")
                print(f"Process: {crew.process}\n")
            slots = extract_slots(q)
            before = dict(TOOL_CALLS)
            output = crew.kickoff(inputs=kickoff_inputs(slots))
            used = {k: TOOL_CALLS[k] - before.get(k, 0) for k in TOOL_CALLS if TOOL_CALLS[k] - before.get(k, 0)}
            print("=" * 100)
            print(f"Query {i}: {q}")
            print(f"  slots -> TICKET_ID={slots.record_id or 'NONE'} | POLICY_QUESTION={slots.policy_question or 'NONE'}")
            for task_out in output.tasks_output:
                print(f"  [{task_out.agent}] {task_out.raw[:160]}{'...' if len(task_out.raw) > 160 else ''}")
            for a in crew.agents:
                print(f"  {a.role} mock-LLM calls: {[c['kind'] for c in a.llm.calls]}")
            print(f"  TOOLS INVOKED during this kickoff: {used}")
            resp = SupportResponse.model_validate_json(output.raw)
            print(f"  SupportResponse schema valid: {isinstance(resp, SupportResponse)}  (crew pydantic output: {type(output.pydantic).__name__})")
            print(f"  FINAL ANSWER: {resp.answer}\n")
        print(f"Cumulative tool calls: {dict(TOOL_CALLS)}")
