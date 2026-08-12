"""Report version-4 constructed estimands and analytic covariance comparators."""

from __future__ import annotations

import sys
from pathlib import Path

BRANCH_DIR = Path(__file__).resolve().parent
if str(BRANCH_DIR) not in sys.path:
    sys.path.insert(0, str(BRANCH_DIR))

from mcw.pipeline import report_registry

if __name__ == "__main__":
    report_registry()
