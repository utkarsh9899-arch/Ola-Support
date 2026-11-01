"""Regenerate every transcript (Tasks 1-16) and the evaluation, each in a fresh process, under MOCK_LLM=1.

    python demos/run_all_demos.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "demos/task01_dataset.py",
    "demos/task02_knowledge_base.py",
    "demos/task03_indexing.py",
    "demos/task04_calibrate.py",
    "demos/task04_grounded_generation.py",
    "demos/task05_chunking_eval.py",
    "demos/task06_ticket_tool.py",
    "demos/task07_crew.py",
    "demos/task08_memory.py",
    "demos/task09_structured_output.py",
    "demos/task10_guardrails.py",
    "demos/task11_12_api.py",
    "eval/run_eval.py",
    "demos/task14_review.py",
    "demos/task15_governance.py",
    "demos/task16_cache.py",
]

if __name__ == "__main__":
    env = {**os.environ, "MOCK_LLM": "1", "PYTHONIOENCODING": "utf-8"}
    failures = []
    for script in SCRIPTS:
        start = time.perf_counter()
        proc = subprocess.run([sys.executable, script], cwd=ROOT, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"{script:<40} {status:<18} {time.perf_counter() - start:6.1f}s")
        if proc.returncode != 0:
            failures.append(script)
            print(proc.stderr[-2000:])
    print(f"\n{len(SCRIPTS) - len(failures)}/{len(SCRIPTS)} scripts succeeded")
    sys.exit(1 if failures else 0)
