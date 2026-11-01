"""Task 9 demo: the SupportResponse schema, validation of every crew response, and a rejected bad payload."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402,F401
from pydantic import ValidationError  # noqa: E402

from demos._transcript import transcript  # noqa: E402
from ola_agent.crew import run_crew  # noqa: E402
from ola_agent.rag.index import build_index  # noqa: E402
from ola_agent.schemas import SupportResponse  # noqa: E402

QUERIES = [
    "How long are closed tickets stored, and can I ask for mine to be deleted?",
    "What is the status of ticket OLA-T-0005?",
    "Ticket OLA-T-0044: who has to approve a refund above INR 2,000?",
    "What will the weather be like in Mumbai tomorrow?",
]

if __name__ == "__main__":
    build_index()
    with transcript("task09_structured_output"):
        print("SupportResponse JSON schema (attached as Task(output_pydantic=SupportResponse) and as the")
        print("Composer LLM's response_format; re-validated in code after every kickoff):")
        print(json.dumps(SupportResponse.model_json_schema(), indent=1)[:1800], "...\n")

        for q in QUERIES:
            resp, output = run_crew(q)
            validated = SupportResponse.model_validate_json(output.raw)
            print(f"Query: {q}")
            print(f"  crew output.pydantic type: {type(output.pydantic).__name__}")
            print(f"  SupportResponse.model_validate_json(output.raw): VALID ({validated.intent}, refused={validated.refused})")
            print(f"  {validated.model_dump_json()}\n")

        print("Negative check - a malformed response is rejected:")
        bad = {"answer": "", "intent": "chit-chat", "grounded": "maybe", "escalation_score": 1.7, "record_id": "T-12"}
        try:
            SupportResponse.model_validate(bad)
        except ValidationError as e:
            print(f"  ValidationError with {e.error_count()} errors:")
            for err in e.errors():
                print(f"    - {'.'.join(map(str, err['loc']))}: {err['msg']}")
