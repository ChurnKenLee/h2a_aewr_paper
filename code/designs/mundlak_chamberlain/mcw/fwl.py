"""Projection backends and full-rank OLS for MCW panel designs.

The nested fixed-effect backend handles county plus parent-by-year effects in
closed form.  The pooled backend applies no absorption: callers place every
intercept, calendar, group, and projection term directly in the nuisance
matrix.  Both paths retain causal-first rank selection and the *full-model*
leverage required by HC3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scipy.linalg

from .resources import guard_fit_working_set


def _as_2d(values: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float64)
    was_vector = array.ndim == 1
    if was_vector:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("Expected a vector or two-dimensional matrix.")
    return array, was_vector


def _factorize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    levels, codes = np.unique(values.astype(str), return_inverse=True)
    return codes.astype(np.int64, copy=False), levels


class OLSProjector(Protocol):
    """Projection contract consumed by :func:`fit_common_ols`."""

    @property
    def n_rows(self) -> int: ...

    @property
    def rank(self) -> int: ...

    def within(self, values: np.ndarray) -> np.ndarray: ...

    def leverage_diagonal(self) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class NoFixedEffectProjector:
    """Identity backend for pooled OLS with all structure explicit in nuisance.

    This backend deliberately contributes neither absorbed rank nor leverage.
    An intercept, year indicators, group indicators, and correlated-effects
    projection terms must therefore be supplied as columns of the nuisance
    matrix passed to :func:`fit_common_ols`.
    """

    n_rows: int

    def __post_init__(self) -> None:
        if isinstance(self.n_rows, bool) or int(self.n_rows) != self.n_rows:
            raise TypeError("NoFixedEffectProjector row count must be an integer.")
        if self.n_rows <= 0:
            raise ValueError("Cannot construct a projector for an empty design.")
        object.__setattr__(self, "n_rows", int(self.n_rows))

    @classmethod
    def from_row_count(cls, n_rows: int) -> NoFixedEffectProjector:
        """Construct an identity backend for ``n_rows`` pooled observations."""

        return cls(n_rows=n_rows)

    @property
    def rank(self) -> int:
        return 0

    def within(self, values: np.ndarray) -> np.ndarray:
        """Return values unchanged after enforcing the declared row count."""

        matrix, was_vector = _as_2d(values)
        if matrix.shape[0] != self.n_rows:
            raise ValueError("Matrix row count does not match pooled design.")
        return matrix[:, 0] if was_vector else matrix

    def leverage_diagonal(self) -> np.ndarray:
        """Return zero because this backend absorbs no columns."""

        return np.zeros(self.n_rows, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class NestedFixedEffectProjector:
    """Exact county plus parent-by-year projector for a balanced panel."""

    unit_codes: np.ndarray
    parent_codes: np.ndarray
    year_codes: np.ndarray
    unit_levels: np.ndarray
    parent_levels: np.ndarray
    year_levels: np.ndarray
    units_per_parent: np.ndarray

    @classmethod
    def from_arrays(
        cls,
        unit: np.ndarray,
        year: np.ndarray,
        parent: np.ndarray | None = None,
    ) -> NestedFixedEffectProjector:
        unit = np.asarray(unit).astype(str)
        year = np.asarray(year).astype(str)
        if parent is None:
            parent = np.repeat("national", unit.size)
        parent = np.asarray(parent).astype(str)
        if not (unit.size == year.size == parent.size):
            raise ValueError("Fixed-effect identifiers must have equal lengths.")
        if unit.size == 0:
            raise ValueError("Cannot construct fixed effects for an empty panel.")

        unit_codes, unit_levels = _factorize(unit)
        parent_codes, parent_levels = _factorize(parent)
        year_codes, year_levels = _factorize(year)
        n_units = unit_levels.size
        n_parents = parent_levels.size
        n_years = year_levels.size

        unit_counts = np.bincount(unit_codes, minlength=n_units)
        if not np.all(unit_counts == n_years):
            raise ValueError(
                "Version-4 exact FWL requires one observation per county-year "
                "on a balanced analysis panel."
            )

        combined = unit_codes * n_years + year_codes
        if np.unique(combined).size != unit.size:
            raise ValueError("Duplicate county-year rows in estimation sample.")

        unit_parent_min = np.full(n_units, n_parents, dtype=np.int64)
        unit_parent_max = np.full(n_units, -1, dtype=np.int64)
        np.minimum.at(unit_parent_min, unit_codes, parent_codes)
        np.maximum.at(unit_parent_max, unit_codes, parent_codes)
        if not np.array_equal(unit_parent_min, unit_parent_max):
            raise ValueError("Every county must be nested in one parent geography.")

        units_per_parent = np.bincount(unit_parent_min, minlength=n_parents)
        if np.any(units_per_parent == 0):
            raise ValueError("Empty parent fixed-effect component.")

        parent_year = parent_codes * n_years + year_codes
        expected_parent_year = np.repeat(units_per_parent, n_years)
        observed_parent_year = np.bincount(parent_year, minlength=n_parents * n_years)
        if not np.array_equal(observed_parent_year, expected_parent_year):
            raise ValueError(
                "Every parent component must contain the same counties in every year."
            )

        return cls(
            unit_codes=unit_codes,
            parent_codes=parent_codes,
            year_codes=year_codes,
            unit_levels=unit_levels,
            parent_levels=parent_levels,
            year_levels=year_levels,
            units_per_parent=units_per_parent,
        )

    @property
    def n_rows(self) -> int:
        return int(self.unit_codes.size)

    @property
    def n_units(self) -> int:
        return int(self.unit_levels.size)

    @property
    def n_parents(self) -> int:
        return int(self.parent_levels.size)

    @property
    def n_years(self) -> int:
        return int(self.year_levels.size)

    @property
    def rank(self) -> int:
        return self.n_units + self.n_parents * (self.n_years - 1)

    def within(self, values: np.ndarray) -> np.ndarray:
        """Apply the exact annihilator for county and parent-by-year FEs."""

        matrix, was_vector = _as_2d(values)
        if matrix.shape[0] != self.n_rows:
            raise ValueError("Matrix row count does not match fixed-effect panel.")
        width = matrix.shape[1]

        unit_sums = np.zeros((self.n_units, width), dtype=np.float64)
        np.add.at(unit_sums, self.unit_codes, matrix)
        unit_means = unit_sums / self.n_years

        parent_year_codes = self.parent_codes * self.n_years + self.year_codes
        parent_year_sums = np.zeros(
            (self.n_parents * self.n_years, width), dtype=np.float64
        )
        np.add.at(parent_year_sums, parent_year_codes, matrix)
        parent_year_counts = np.repeat(self.units_per_parent, self.n_years)
        parent_year_means = parent_year_sums / parent_year_counts[:, None]

        parent_sums = np.zeros((self.n_parents, width), dtype=np.float64)
        np.add.at(parent_sums, self.parent_codes, matrix)
        parent_means = parent_sums / (self.units_per_parent[:, None] * self.n_years)

        result = (
            matrix
            - unit_means[self.unit_codes]
            - parent_year_means[parent_year_codes]
            + parent_means[self.parent_codes]
        )
        return result[:, 0] if was_vector else result

    def leverage_diagonal(self) -> np.ndarray:
        """Return diag(P_FE), including both absorbed fixed-effect sets."""

        parent_size = self.units_per_parent[self.parent_codes].astype(np.float64)
        years = float(self.n_years)
        return 1.0 / years + 1.0 / parent_size - 1.0 / (years * parent_size)


@dataclass(frozen=True, slots=True)
class SelectedDesign:
    matrix: np.ndarray
    names: tuple[str, ...]
    causal_count: int
    selected_nuisance_indices: tuple[int, ...]
    dropped_nuisance_names: tuple[str, ...]


def _rank_from_qr(diagonal: np.ndarray, n_rows: int, n_cols: int) -> int:
    if diagonal.size == 0:
        return 0
    threshold = max(n_rows, n_cols) * np.finfo(np.float64).eps * diagonal.max()
    return int(np.count_nonzero(diagonal > threshold))


def causal_first_full_rank_design(
    causal: np.ndarray,
    nuisance: np.ndarray,
    causal_names: tuple[str, ...],
    nuisance_names: tuple[str, ...],
) -> SelectedDesign:
    """Keep all named causal columns, then a maximal nuisance complement."""

    causal, _ = _as_2d(causal)
    nuisance, _ = _as_2d(nuisance)
    if causal.shape[0] != nuisance.shape[0]:
        raise ValueError("Causal and nuisance matrices must share rows.")
    if causal.shape[1] != len(causal_names):
        raise ValueError("Causal name count does not match matrix width.")
    if nuisance.shape[1] != len(nuisance_names):
        raise ValueError("Nuisance name count does not match matrix width.")

    causal_norm = np.linalg.norm(causal, axis=0)
    zero_causal = [
        name for name, norm in zip(causal_names, causal_norm, strict=True) if norm == 0
    ]
    if zero_causal:
        raise ValueError(f"Causal coordinates are fully absorbed: {zero_causal}")
    causal_scaled = causal / causal_norm
    q_causal, r_causal, _ = scipy.linalg.qr(
        causal_scaled, mode="economic", pivoting=True, check_finite=False
    )
    causal_rank = _rank_from_qr(
        np.abs(np.diag(r_causal)), causal.shape[0], causal.shape[1]
    )
    if causal_rank != causal.shape[1]:
        raise ValueError(
            f"Named causal block has rank {causal_rank} for "
            f"{causal.shape[1]} coordinates; no causal term was dropped."
        )

    if nuisance.shape[1] == 0:
        return SelectedDesign(
            matrix=causal,
            names=causal_names,
            causal_count=causal.shape[1],
            selected_nuisance_indices=(),
            dropped_nuisance_names=(),
        )

    nuisance_orthogonal = nuisance - q_causal @ (q_causal.T @ nuisance)
    nuisance_norm = np.linalg.norm(nuisance_orthogonal, axis=0)
    nonzero = np.flatnonzero(nuisance_norm > 0)
    if nonzero.size:
        normalized = nuisance_orthogonal[:, nonzero] / nuisance_norm[nonzero]
        _, r_nuisance, pivot = scipy.linalg.qr(
            normalized, mode="economic", pivoting=True, check_finite=False
        )
        nuisance_rank = _rank_from_qr(
            np.abs(np.diag(r_nuisance)), normalized.shape[0], normalized.shape[1]
        )
        selected = tuple(int(nonzero[index]) for index in pivot[:nuisance_rank])
    else:
        selected = ()
    selected_set = set(selected)
    dropped = tuple(
        name for index, name in enumerate(nuisance_names) if index not in selected_set
    )
    matrix = np.column_stack((causal, nuisance[:, selected]))
    names = causal_names + tuple(nuisance_names[index] for index in selected)
    return SelectedDesign(
        matrix=matrix,
        names=names,
        causal_count=causal.shape[1],
        selected_nuisance_indices=selected,
        dropped_nuisance_names=dropped,
    )


@dataclass(frozen=True, slots=True)
class CommonOLSFit:
    coefficient: np.ndarray
    bread: np.ndarray
    residual: np.ndarray
    fitted: np.ndarray
    within_design: np.ndarray
    raw_design: np.ndarray
    design_names: tuple[str, ...]
    outcome_names: tuple[str, ...]
    causal_count: int
    selected_nuisance_indices: tuple[int, ...]
    dropped_nuisance_names: tuple[str, ...]
    leverage: np.ndarray
    fixed_effect_rank: int
    model_rank: int
    residual_df: int
    condition_number: float
    solve_relative_residual: float


def fit_common_ols(
    projector: OLSProjector,
    causal: np.ndarray,
    nuisance: np.ndarray,
    outcomes: np.ndarray,
    causal_names: tuple[str, ...],
    nuisance_names: tuple[str, ...],
    outcome_names: tuple[str, ...],
) -> CommonOLSFit:
    """Fit all primitive outcomes on one audited design and sample."""

    outcomes, _ = _as_2d(outcomes)
    if outcomes.shape[1] != len(outcome_names):
        raise ValueError("Outcome name count does not match matrix width.")
    if not np.all(np.isfinite(outcomes)):
        raise ValueError("Primitive outcomes must be finite on the common sample.")

    causal_within = projector.within(causal)
    nuisance_within = projector.within(nuisance)
    guard_fit_working_set(
        causal_within.shape[0], causal_within.shape[1] + nuisance_within.shape[1]
    )
    raw_causal_norm = np.linalg.norm(causal, axis=0)
    within_causal_norm = np.linalg.norm(causal_within, axis=0)
    absorbed = [
        name
        for name, raw_norm, within_norm in zip(
            causal_names,
            raw_causal_norm,
            within_causal_norm,
            strict=True,
        )
        if raw_norm == 0.0 or within_norm <= 1e-12 * max(raw_norm, np.finfo(float).tiny)
    ]
    if absorbed:
        raise ValueError(
            "Causal coordinates are numerically absorbed by the fixed effects: "
            f"{absorbed}"
        )
    selected = causal_first_full_rank_design(
        causal_within,
        nuisance_within,
        causal_names,
        nuisance_names,
    )
    raw_selected = np.column_stack(
        (causal, nuisance[:, selected.selected_nuisance_indices])
    )
    x = selected.matrix
    y = projector.within(outcomes)
    guard_fit_working_set(x.shape[0], x.shape[1])

    scales = np.sqrt(np.mean(np.square(x), axis=0))
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Non-finite or zero scale in selected design.")
    x_scaled = x / scales
    gram_scaled = x_scaled.T @ x_scaled
    eigenvalues = scipy.linalg.eigvalsh(gram_scaled, check_finite=False)
    if eigenvalues[0] <= 0:
        raise ValueError("Selected common design is not positive definite.")
    condition_number = float(eigenvalues[-1] / eigenvalues[0])
    factor = scipy.linalg.cho_factor(gram_scaled, lower=True, check_finite=False)
    beta_scaled = scipy.linalg.cho_solve(factor, x_scaled.T @ y, check_finite=False)
    coefficient = beta_scaled / scales[:, None]
    inverse_scaled = scipy.linalg.cho_solve(
        factor, np.eye(x.shape[1]), check_finite=False
    )
    inverse_scale = 1.0 / scales
    bread = inverse_scale[:, None] * inverse_scaled * inverse_scale[None, :]

    residual = y - x @ coefficient
    fitted = outcomes - residual
    normal_equation = x.T @ residual
    denominator = max(float(np.linalg.norm(x.T @ y)), np.finfo(float).tiny)
    solve_relative_residual = float(np.linalg.norm(normal_equation) / denominator)

    partial_leverage = np.einsum("ij,ij->i", x @ bread, x, optimize=True)
    leverage = projector.leverage_diagonal() + partial_leverage
    if np.any(leverage < -1e-10) or np.any(leverage >= 1.0 - 1e-10):
        raise ValueError(
            "Full-model leverage is outside [0, 1); HC3 would be undefined."
        )

    model_rank = projector.rank + x.shape[1]
    residual_df = x.shape[0] - model_rank
    if residual_df <= 0:
        raise ValueError("Non-positive residual degrees of freedom.")
    return CommonOLSFit(
        coefficient=coefficient,
        bread=bread,
        residual=residual,
        fitted=fitted,
        within_design=x,
        raw_design=raw_selected,
        design_names=selected.names,
        outcome_names=outcome_names,
        causal_count=selected.causal_count,
        selected_nuisance_indices=selected.selected_nuisance_indices,
        dropped_nuisance_names=selected.dropped_nuisance_names,
        leverage=leverage,
        fixed_effect_rank=projector.rank,
        model_rank=model_rank,
        residual_df=residual_df,
        condition_number=condition_number,
        solve_relative_residual=solve_relative_residual,
    )
