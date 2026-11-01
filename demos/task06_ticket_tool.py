"""Task 6 demo: escalation-score design, threshold justification, and sample lookups."""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402,F401
import dataset  # noqa: E402
from demos._transcript import transcript  # noqa: E402
from ola_agent.tools import ticket_status as ts  # noqa: E402

if __name__ == "__main__":
    with transcript("task06_ticket_tool"):
        print(f"escalation_score = {ts.W_ESCALATED} * escalated + {ts.W_AGE} * min(days_since_created / {ts.MAX_AGE_DAYS}, 1)\n")

        days = [t["days_since_created"] for t in dataset.SUPPORT_TICKETS]
        deciles = statistics.quantiles(days, n=10, method="inclusive")
        print("days_since_created distribution in the generated dataset (SEED=42, N=48):")
        print("  " + "  ".join(f"P{(i + 1) * 10}={d:g}" for i, d in enumerate(deciles)))
        print(f"\nThreshold = {ts.W_AGE} * P80 / 30 = {ts.W_AGE} * {deciles[7]:g} / 30 = {ts.W_AGE * deciles[7] / 30:.4f} -> {ts.ESCALATION_THRESHOLD}")
        print("  -> every escalated ticket (score >= 0.6) is above the threshold;")
        print("  -> a non-escalated ticket is above it only when older than the 80th percentile (21+ days);")
        print("  -> Resolved/Closed tickets are never recommended for escalation.\n")

        results = [ts.check_support_ticket_status(t["record_id"]) for t in dataset.SUPPORT_TICKETS]
        scores = sorted(r["escalation_score"] for r in results)
        above = [r for r in results if r["escalation_score"] >= ts.ESCALATION_THRESHOLD]
        recommended = [r for r in results if r["recommend_escalation"]]
        print(f"Score distribution: min={scores[0]}  median={statistics.median(scores)}  max={scores[-1]}")
        print(f"Tickets with score >= {ts.ESCALATION_THRESHOLD}: {len(above)}/{len(results)}; "
              f"recommended for escalation (excluding Resolved/Closed): {len(recommended)}/{len(results)}\n")

        for rid in ("OLA-T-0005", "OLA-T-0044", "OLA-T-0016", "ola-t-3", "OLA-T-9999"):
            print(f"check_support_ticket_status({rid!r})")
            print(f"  -> {ts.check_support_ticket_status(rid)}\n")
