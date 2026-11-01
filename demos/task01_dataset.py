"""Task 1 demo: generate the dataset, print counts, validate, show sample records."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402,F401  (sets offline env first)
import dataset  # noqa: E402
from demos._transcript import transcript  # noqa: E402

if __name__ == "__main__":
    with transcript("task01_dataset"):
        dataset.print_report(dataset.SUPPORT_TICKETS)
        regenerated = dataset.generate()
        print(f"\nDeterminism check (regenerate with same seed -> identical): {regenerated == dataset.SUPPORT_TICKETS}")
        print("\nFirst 5 records:")
        for t in dataset.SUPPORT_TICKETS[:5]:
            print(" ", t)
