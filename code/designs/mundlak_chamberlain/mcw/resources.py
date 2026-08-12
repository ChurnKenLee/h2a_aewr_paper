"""Declared dense-matrix resource guards for the version-4 estimator."""

from __future__ import annotations

import os
from dataclasses import dataclass

BYTES_PER_FLOAT64 = 8
BYTES_PER_GIB = 1024**3
DEFAULT_MAX_DENSE_GIB = 1.25
DEFAULT_MAX_ESTIMATED_PEAK_GIB = 6.0
DENSE_PEAK_COPIES = 4
GRAM_PEAK_COPIES = 3


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    dense_gib: float
    gram_gib: float
    estimated_peak_gib: float
    dense_limit_gib: float
    peak_limit_gib: float


def _positive_environment(name: str, default: float) -> float:
    text = os.getenv(name)
    if text is None:
        return default
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive finite number.") from error
    if not 0 < value < float("inf"):
        raise ValueError(f"{name} must be a positive finite number.")
    return value


def resource_budget(rows: int, columns: int) -> ResourceBudget:
    """Estimate the dense and fit working sets for an ``N x K`` design."""

    if rows < 1 or columns < 1:
        raise ValueError("Resource-budget dimensions must be positive.")
    dense_gib = rows * columns * BYTES_PER_FLOAT64 / BYTES_PER_GIB
    gram_gib = columns * columns * BYTES_PER_FLOAT64 / BYTES_PER_GIB
    return ResourceBudget(
        dense_gib=dense_gib,
        gram_gib=gram_gib,
        estimated_peak_gib=(
            DENSE_PEAK_COPIES * dense_gib + GRAM_PEAK_COPIES * gram_gib
        ),
        dense_limit_gib=_positive_environment(
            "MC_SPEC_MAX_DENSE_GIB", DEFAULT_MAX_DENSE_GIB
        ),
        peak_limit_gib=_positive_environment(
            "MC_SPEC_MAX_PEAK_GIB", DEFAULT_MAX_ESTIMATED_PEAK_GIB
        ),
    )


def guard_dense_allocation(rows: int, columns: int, *, label: str) -> ResourceBudget:
    """Reject a declared dense allocation before materializing it."""

    budget = resource_budget(rows, columns)
    if budget.dense_gib > budget.dense_limit_gib:
        raise MemoryError(
            f"{label} would allocate one {budget.dense_gib:.3f} GiB dense matrix, "
            f"above MC_SPEC_MAX_DENSE_GIB={budget.dense_limit_gib:.3f}."
        )
    return budget


def guard_fit_working_set(rows: int, columns: int) -> ResourceBudget:
    """Reject a selected fit whose declared dense peak exceeds the ceiling."""

    budget = guard_dense_allocation(rows, columns, label="Selected OLS design")
    if budget.estimated_peak_gib > budget.peak_limit_gib:
        raise MemoryError(
            "Selected OLS working set is estimated at "
            f"{budget.estimated_peak_gib:.3f} GiB, above "
            f"MC_SPEC_MAX_PEAK_GIB={budget.peak_limit_gib:.3f}."
        )
    return budget
