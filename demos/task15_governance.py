"""Task 15 demo: four-layer governance - least autonomy, risk classification, runtime budget cap."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from demos._transcript import transcript  # noqa: E402
from ola_agent import budget  # noqa: E402
from ola_agent.crew import _llm, build_crew  # noqa: E402
from ola_agent.governance import TOOL_ACL, ToolPermissionError, make_agent  # noqa: E402
from ola_agent.memory import ask  # noqa: E402
from ola_agent.rag.index import build_index  # noqa: E402
from ola_agent.tools.crew_tools import TOOL_CALLS, TicketStatusTool  # noqa: E402

RISK_CLASSIFICATION = """RISK LEVEL: MEDIUM.
Under the Low / Medium / High scheme, this system is a customer-support-ticket assistant, which the scheme
places at Medium: its answers reach riders and drivers directly and can influence what they expect about
refunds, credits and escalations, so a wrong answer has real but recoverable customer impact. It is not Low,
because it does more than summarise or transcribe - it gives policy answers and reads individual ticket
records. It is not High: it handles no medical data and makes no hiring decisions, and although it talks
about refunds it never moves money or reads payment details (payment fields are out of scope and only
fabricated examples exist), and the ticket tool is read-only. Medium risk is why it carries input
guardrails, an output groundedness check, an independent Autogen review stage, least-autonomy tool access,
a per-request budget and masked audit logs. It would become High if it were allowed to issue refunds or
credits itself, or to process real card or bank details."""

LEAST_AUTONOMY = """LEAST-AUTONOMY GUARD (Application layer).
`governance.TOOL_ACL` maps each tool to the only agent roles allowed to hold it; check_support_ticket_status
is allowed for {'Lookup Agent'} only. Every agent in the crew is built through `governance.make_agent`, which
checks each tool against TOOL_ACL before it is attached and raises ToolPermissionError otherwise, so the
lookup tool cannot even be wired to the Retrieval Agent or the Response Composer. As a second, runtime layer,
the tool records the role it was bound to and re-checks TOOL_ACL every time it runs, so a tool instance that
was created or passed around outside make_agent refuses to execute. The Composer therefore can only use what
the Lookup Agent returned; it can never query ticket records itself."""

if __name__ == "__main__":
    build_index()
    with transcript("task15_governance"):
        print("FOUR-LAYER GOVERNANCE MODEL")
        print("  Data layer       : phone numbers masked before the model, memory, cache and logs; fabricated data only")
        print("  Model layer      : deterministic MOCK_LLM; output groundedness check; independent Autogen review")
        print("  Application layer: least autonomy for tools (below); risk classification (below)")
        print("  Runtime layer    : per-request token/cost budget cap (below); JSON-Lines audit log with trace ids\n")

        print("=" * 100 + "\nAPPLICATION LAYER - least autonomy\n" + "=" * 100)
        print(f"TOOL_ACL = {TOOL_ACL}\n")
        crew = build_crew()
        for a in crew.agents:
            print(f"  wired: {a.role:<18} tools={[t.name for t in a.tools]}")
        holders = [a.role for a in crew.agents if any(t.name == 'check_support_ticket_status' for t in a.tools)]
        print(f"  agents holding check_support_ticket_status: {holders}\n")

        for role in ("Response Composer", "Retrieval Agent"):
            try:
                make_agent(role, tools=[TicketStatusTool()], goal="g", backstory="b", llm=_llm())
                print(f"  wiring lookup tool to {role}: ALLOWED (unexpected)")
            except ToolPermissionError as e:
                print(f"  wiring lookup tool to {role}: BLOCKED -> {e}")
        before = TOOL_CALLS["check_support_ticket_status"]
        try:
            TicketStatusTool()._run(record_id="OLA-T-0044")  # stray instance, never bound to the Lookup Agent
        except ToolPermissionError as e:
            print(f"  calling an unbound lookup-tool instance directly: BLOCKED at runtime -> {e}")
        print(f"  lookup tool executions during the blocked attempts: {TOOL_CALLS['check_support_ticket_status'] - before}\n")
        print(LEAST_AUTONOMY + "\n")

        print("=" * 100 + "\nAPPLICATION LAYER - risk classification\n" + "=" * 100)
        print(RISK_CLASSIFICATION + "\n")

        print("=" * 100 + "\nRUNTIME LAYER - per-request token / cost budget\n" + "=" * 100)
        print(f"cap: {config.MAX_TOKENS_PER_REQUEST} tokens per request (~${budget.cost_usd(config.MAX_TOKENS_PER_REQUEST):.4f} "
              f"at a notional ${config.COST_PER_1K_TOKENS_USD}/1K tokens); tokens estimated at 4 chars/token")
        print(f"pre-flight projection = {budget.BASELINE_TOKENS} baseline + {budget.INPUT_COPIES} x tokens(input)\n")
        print("Measured usage of normal requests (metered across every model call):")
        for i, q in enumerate(["Can you book me a flight from Delhi to Goa next Friday?",
                               "What is the status of ticket OLA-T-0044?",
                               "Who has to approve a refund above INR 2,000?",
                               "Ticket OLA-T-0044: what compensation do I get if my driver arrives 20 minutes late?"]):
            r = ask(q, f"budget-normal-{i}")
            print(f"  {r.tokens_used:>5} used / {r.tokens_projected:>5} projected / {config.MAX_TOKENS_PER_REQUEST} cap  <- {q}")

        oversized = "Please read my full complaint history before answering. " * 360
        print(f"\n(1) Deliberately oversized request: {len(oversized):,} characters (~{budget.estimate_tokens(oversized):,} tokens)")
        llm_calls_before = sum(TOOL_CALLS.values())
        try:
            ask(oversized, "budget-oversized")
            print("    ACCEPTED (unexpected)")
        except budget.BudgetExceededError as e:
            print(f"    REJECTED -> {e}")
        print(f"    tools/LLM work done for it: {sum(TOOL_CALLS.values()) - llm_calls_before} tool calls (rejected before the crew ran)")

        print("\n(2) Defence in depth - runtime metering on its own: a normal mixed request under a deliberately")
        print("    small 2,000-token cap, with the pre-flight check bypassed ON PURPOSE for this demonstration")
        tool_before = dict(TOOL_CALLS)
        try:
            ask("Ticket OLA-T-0044: what compensation do I get if my driver arrives 20 minutes late?",
                "budget-runtime", token_limit=2000, skip_preflight=True)
            print("    COMPLETED (unexpected)")
        except budget.BudgetExceededError as e:
            print(f"    ABORTED mid-run -> {e}")
        print(f"    tool calls that had run before the abort: "
              f"{ {k: TOOL_CALLS[k] - tool_before.get(k, 0) for k in TOOL_CALLS} }")

        print("\n(3) Same oversized request through the API:")
        from api.main import app

        with TestClient(app) as client:
            resp = client.post("/ask", json={"query": oversized})
            print(f"    POST /ask -> HTTP {resp.status_code}: {resp.json()['detail'][:160]}...")
