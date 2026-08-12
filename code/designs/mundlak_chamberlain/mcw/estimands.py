"""Named post-fit estimands for a common-design, multiple-outcome OLS fit.

The coefficient matrix convention is ``K x J``: rows are common design
coordinates and columns are primitive outcomes.  A :class:`NamedGradient`
uses the same convention.  Cross-outcome covariance blocks use
``(J, J, K, K)``, where ``blocks[a, b]`` is ``Cov(beta[:, a], beta[:, b])``.

All outcome denominators are row-aligned objects.  An estimand receives one
``TargetPopulation`` and applies its inclusion mask and weights to both its
effect and denominator.  This prevents, for example, an active-sample effect
from being divided by an all-county observed mean.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

Aggregation = Literal["weighted_mean", "weighted_sum"]


def _names(values: Sequence[str], label: str) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if not names:
        raise ValueError(f"{label} must not be empty.")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must be unique.")
    return names


def _row_ids(values: Sequence[object] | np.ndarray) -> tuple[str, ...]:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError("Row identifiers must be one-dimensional.")
    if any(value is None for value in array):
        raise ValueError("Row identifiers must not be null.")
    ids = tuple(str(value) for value in array)
    if len(set(ids)) != len(ids):
        raise ValueError("Row identifiers must be unique.")
    return ids


def _float_array(values: object, *, ndim: int, label: str) -> np.ndarray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{label} must be {ndim}-dimensional.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class CoefficientLayout:
    """Names and dimensions of a common ``K x J`` coefficient matrix."""

    coefficient_names: tuple[str, ...]
    outcome_names: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coefficient_names",
            _names(self.coefficient_names, "Coefficient names"),
        )
        object.__setattr__(
            self, "outcome_names", _names(self.outcome_names, "Outcome names")
        )

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.coefficient_names), len(self.outcome_names)

    def outcome_index(self, outcome: str) -> int:
        try:
            return self.outcome_names.index(outcome)
        except ValueError as error:
            raise ValueError(f"Unknown primitive outcome: {outcome}") from error


@dataclass(frozen=True, slots=True)
class CommonCoefficientMatrix:
    """A finite common-design coefficient matrix with named axes."""

    values: np.ndarray
    layout: CoefficientLayout

    def __post_init__(self) -> None:
        values = _float_array(self.values, ndim=2, label="Coefficient matrix")
        if values.shape != self.layout.shape:
            raise ValueError(
                "Coefficient matrix shape does not match its named layout: "
                f"{values.shape} != {self.layout.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Coefficient matrix must be finite.")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class RowVector:
    """A named row-level vector carrying an explicit row identity."""

    name: str
    row_ids: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        row_ids = _row_ids(self.row_ids)
        values = _float_array(self.values, ndim=1, label=self.name)
        if values.size != len(row_ids):
            raise ValueError(f"{self.name} and its row identifiers differ in length.")
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_polars(
        cls,
        frame: pl.DataFrame,
        *,
        row_id_column: str,
        value_column: str,
        name: str | None = None,
    ) -> RowVector:
        """Build a row-aligned vector without converting through pandas."""

        return cls(
            name=name or value_column,
            row_ids=tuple(frame.get_column(row_id_column).cast(pl.String).to_list()),
            values=frame.get_column(value_column).to_numpy(),
        )


@dataclass(frozen=True, slots=True)
class RowGradient:
    """One ``K``-vector of primitive-effect loadings for each target row."""

    name: str
    row_ids: tuple[str, ...]
    values: np.ndarray
    coefficient_names: tuple[str, ...]

    def __post_init__(self) -> None:
        row_ids = _row_ids(self.row_ids)
        coefficient_names = _names(self.coefficient_names, "Coefficient names")
        values = _float_array(self.values, ndim=2, label=self.name)
        expected = (len(row_ids), len(coefficient_names))
        if values.shape != expected:
            raise ValueError(
                f"{self.name} row-gradient shape is {values.shape}, expected {expected}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{self.name} row gradients must be finite.")
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "coefficient_names", coefficient_names)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_polars(
        cls,
        frame: pl.DataFrame,
        *,
        row_id_column: str,
        coefficient_columns: Sequence[str],
        name: str,
    ) -> RowGradient:
        """Build row gradients from named Polars columns in coefficient order."""

        columns = tuple(coefficient_columns)
        return cls(
            name=name,
            row_ids=tuple(frame.get_column(row_id_column).cast(pl.String).to_list()),
            values=frame.select(columns).to_numpy(),
            coefficient_names=columns,
        )


@dataclass(frozen=True, slots=True)
class TargetPopulation:
    """One declared row population and one set of nonnegative target weights."""

    name: str
    row_ids: tuple[str, ...]
    include: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        row_ids = _row_ids(self.row_ids)
        include = np.array(self.include, dtype=np.bool_, copy=True)
        weights = np.array(self.weights, dtype=np.float64, copy=True)
        if include.ndim != 1 or weights.ndim != 1:
            raise ValueError("Target inclusion and weights must be one-dimensional.")
        if include.size != len(row_ids) or weights.size != len(row_ids):
            raise ValueError("Target rows, inclusion, and weights differ in length.")
        if not np.any(include):
            raise ValueError("Target population is empty.")
        selected_weights = weights[include]
        if (
            np.any(~np.isfinite(selected_weights))
            or np.any(selected_weights < 0)
            or float(selected_weights.sum()) <= 0
        ):
            raise ValueError(
                "Selected target weights must be finite, nonnegative, and nonzero."
            )
        weights[~include] = 0.0
        include.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "include", include)
        object.__setattr__(self, "weights", weights)

    @classmethod
    def all_rows(
        cls,
        row_ids: Sequence[object] | np.ndarray,
        *,
        weights: Sequence[float] | np.ndarray | None = None,
        name: str = "all_rows",
    ) -> TargetPopulation:
        ids = _row_ids(row_ids)
        supplied_weights = np.ones(len(ids)) if weights is None else weights
        return cls(
            name=name,
            row_ids=ids,
            include=np.ones(len(ids), dtype=np.bool_),
            weights=np.asarray(supplied_weights),
        )

    @classmethod
    def from_polars(
        cls,
        frame: pl.DataFrame,
        *,
        row_id_column: str,
        include_column: str | None = None,
        weight_column: str | None = None,
        name: str = "target",
    ) -> TargetPopulation:
        """Build a target directly from a Polars frame."""

        rows = tuple(frame.get_column(row_id_column).cast(pl.String).to_list())
        if include_column is None:
            include = np.ones(len(rows), dtype=np.bool_)
        else:
            series = frame.get_column(include_column)
            if series.null_count():
                raise ValueError("Target inclusion column must not contain nulls.")
            include = series.cast(pl.Boolean).to_numpy()
        if weight_column is None:
            weights = np.ones(len(rows), dtype=np.float64)
        else:
            weights = frame.get_column(weight_column).to_numpy()
        return cls(name=name, row_ids=rows, include=include, weights=weights)

    @property
    def observations(self) -> int:
        return int(np.count_nonzero(self.include))

    @property
    def weight_sum(self) -> float:
        return float(self.weights[self.include].sum())

    def _check_rows(self, row_ids: tuple[str, ...], label: str) -> None:
        if self.row_ids != row_ids:
            raise ValueError(
                f"{label} does not have the target's exact rows in the same order."
            )

    def weighted_sum(self, vector: RowVector) -> float:
        self._check_rows(vector.row_ids, vector.name)
        selected = vector.values[self.include]
        if not np.all(np.isfinite(selected)):
            raise ValueError(f"{vector.name} is non-finite on the target population.")
        return float(self.weights[self.include] @ selected)

    def weighted_mean(self, vector: RowVector) -> float:
        return self.weighted_sum(vector) / self.weight_sum

    def aggregate_gradient(
        self, gradient: RowGradient, aggregation: Aggregation
    ) -> np.ndarray:
        self._check_rows(gradient.row_ids, gradient.name)
        selected = gradient.values[self.include]
        result = self.weights[self.include] @ selected
        if aggregation == "weighted_mean":
            result = result / self.weight_sum
        elif aggregation != "weighted_sum":
            raise ValueError(f"Unknown aggregation: {aggregation}")
        return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class NamedGradient:
    """A named analytic gradient on the common ``K x J`` coefficient basis."""

    name: str
    kind: str
    target_name: str
    values: np.ndarray
    layout: CoefficientLayout

    def __post_init__(self) -> None:
        values = _float_array(self.values, ndim=2, label=self.name)
        if values.shape != self.layout.shape:
            raise ValueError(
                f"Gradient shape {values.shape} does not match {self.layout.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Gradient must be finite.")
        object.__setattr__(self, "values", values)

    def evaluate(self, coefficients: CommonCoefficientMatrix) -> float:
        if coefficients.layout != self.layout:
            raise ValueError("Gradient and coefficient layouts do not match exactly.")
        return float(np.sum(self.values * coefficients.values))


@dataclass(frozen=True, slots=True)
class CombinedOutcomeTerm:
    """One term for an independent, row-level combined-outcome oracle."""

    outcome: str
    row_gradient: RowGradient
    multiplier: float = 1.0
    aggregation: Aggregation = "weighted_mean"

    def __post_init__(self) -> None:
        if not np.isfinite(self.multiplier):
            raise ValueError("Combined-outcome multiplier must be finite.")
        if self.aggregation not in {"weighted_mean", "weighted_sum"}:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


@dataclass(frozen=True, slots=True)
class DeltaResult:
    """Analytic delta-method result without imposing a reference distribution."""

    name: str
    kind: str
    target_name: str
    estimate: float
    variance: float
    standard_error: float


def _check_row_gradient(
    row_gradient: RowGradient,
    layout: CoefficientLayout,
) -> None:
    if row_gradient.coefficient_names != layout.coefficient_names:
        raise ValueError(
            f"{row_gradient.name} is not on the declared coefficient basis."
        )


def _named_gradient(
    *,
    name: str,
    kind: str,
    target: TargetPopulation,
    layout: CoefficientLayout,
    contributions: Sequence[tuple[str, np.ndarray]],
) -> NamedGradient:
    values = np.zeros(layout.shape, dtype=np.float64)
    for outcome, contribution in contributions:
        vector = np.asarray(contribution, dtype=np.float64)
        if vector.shape != (layout.shape[0],):
            raise ValueError("Gradient contribution has the wrong coefficient width.")
        values[:, layout.outcome_index(outcome)] += vector
    return NamedGradient(
        name=name,
        kind=kind,
        target_name=target.name,
        values=values,
        layout=layout,
    )


def linear_primitive_effect(
    *,
    name: str,
    outcome: str,
    row_gradient: RowGradient,
    target: TargetPopulation,
    layout: CoefficientLayout,
    aggregation: Aggregation = "weighted_mean",
) -> NamedGradient:
    """Construct a weighted primitive-outcome effect."""

    _check_row_gradient(row_gradient, layout)
    contribution = target.aggregate_gradient(row_gradient, aggregation)
    return _named_gradient(
        name=name,
        kind=f"linear_primitive_{aggregation}",
        target=target,
        layout=layout,
        contributions=((outcome, contribution),),
    )


def per_baseline_worker_effect(
    *,
    name: str,
    outcome: str,
    row_gradient: RowGradient,
    baseline_workers: RowVector,
    target: TargetPopulation,
    layout: CoefficientLayout,
    scale: float = 1.0,
) -> NamedGradient:
    """Effect total per weighted baseline worker on exactly the target rows."""

    if not np.isfinite(scale):
        raise ValueError("Per-worker scale must be finite.")
    _check_row_gradient(row_gradient, layout)
    target._check_rows(baseline_workers.row_ids, baseline_workers.name)
    selected_baseline = baseline_workers.values[target.include]
    if np.any(~np.isfinite(selected_baseline)) or np.any(selected_baseline < 0):
        raise ValueError(
            "Baseline workers must be finite and nonnegative on target rows."
        )
    denominator = target.weighted_sum(baseline_workers)
    if denominator <= 0:
        raise ValueError("Weighted baseline-worker denominator must be positive.")
    contribution = (
        scale * target.aggregate_gradient(row_gradient, "weighted_sum") / denominator
    )
    return _named_gradient(
        name=name,
        kind="per_baseline_worker_weighted_effect",
        target=target,
        layout=layout,
        contributions=((outcome, contribution),),
    )


def fixed_observed_mean_elasticity(
    *,
    name: str,
    outcome: str,
    row_gradient: RowGradient,
    observed_outcome: RowVector,
    target: TargetPopulation,
    layout: CoefficientLayout,
    log_point_scale: float = 100.0,
) -> NamedGradient:
    """Elasticity using a fixed observed mean from the identical target rows."""

    if not np.isfinite(log_point_scale):
        raise ValueError("Elasticity scale must be finite.")
    _check_row_gradient(row_gradient, layout)
    observed_mean = target.weighted_mean(observed_outcome)
    if not np.isfinite(observed_mean) or abs(observed_mean) <= np.finfo(float).tiny:
        raise ValueError("Observed target-population mean must be nonzero and finite.")
    contribution = (
        log_point_scale
        * target.aggregate_gradient(row_gradient, "weighted_mean")
        / observed_mean
    )
    return _named_gradient(
        name=name,
        kind="fixed_observed_mean_elasticity",
        target=target,
        layout=layout,
        contributions=((outcome, contribution),),
    )


def fixed_observed_mean_percent_per_treatment_unit(
    *,
    name: str,
    outcome: str,
    row_gradient: RowGradient,
    observed_outcome: RowVector,
    target: TargetPopulation,
    layout: CoefficientLayout,
    percent_scale: float = 100.0,
) -> NamedGradient:
    """Derivative as percent of the fixed mean per declared treatment unit.

    This normalization is not an elasticity unless a separately justified
    chain rule maps one treatment unit to a proportional change in the
    underlying policy variable.
    """

    if not np.isfinite(percent_scale):
        raise ValueError("Percent scale must be finite.")
    _check_row_gradient(row_gradient, layout)
    observed_mean = target.weighted_mean(observed_outcome)
    if not np.isfinite(observed_mean) or abs(observed_mean) <= np.finfo(float).tiny:
        raise ValueError("Observed target-population mean must be nonzero and finite.")
    contribution = (
        percent_scale
        * target.aggregate_gradient(row_gradient, "weighted_mean")
        / observed_mean
    )
    return _named_gradient(
        name=name,
        kind="fixed_observed_mean_percent_per_treatment_unit",
        target=target,
        layout=layout,
        contributions=((outcome, contribution),),
    )


def ratio_of_aggregate_derivative(
    *,
    name: str,
    numerator_outcome: str,
    denominator_outcome: str,
    numerator_row_gradient: RowGradient,
    denominator_row_gradient: RowGradient,
    observed_numerator: RowVector,
    observed_denominator: RowVector,
    target: TargetPopulation,
    layout: CoefficientLayout,
) -> NamedGradient:
    """Derivative of a ratio of weighted outcome aggregates.

    This is ``d(N / D) = dN / D - N dD / D^2``.  It is deliberately
    distinct from :func:`average_unit_ratio_derivative`.
    """

    _check_row_gradient(numerator_row_gradient, layout)
    _check_row_gradient(denominator_row_gradient, layout)
    numerator = target.weighted_sum(observed_numerator)
    denominator = target.weighted_sum(observed_denominator)
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Observed aggregate denominator must be positive and finite.")
    numerator_effect = target.aggregate_gradient(numerator_row_gradient, "weighted_sum")
    denominator_effect = target.aggregate_gradient(
        denominator_row_gradient, "weighted_sum"
    )
    return _named_gradient(
        name=name,
        kind="ratio_of_aggregates_derivative",
        target=target,
        layout=layout,
        contributions=(
            (numerator_outcome, numerator_effect / denominator),
            (
                denominator_outcome,
                -numerator * denominator_effect / denominator**2,
            ),
        ),
    )


def positions_per_application_derivative(
    *,
    positions_row_gradient: RowGradient,
    applications_row_gradient: RowGradient,
    observed_positions: RowVector,
    observed_applications: RowVector,
    target: TargetPopulation,
    layout: CoefficientLayout,
    positions_outcome: str = "certified_positions",
    applications_outcome: str = "applications",
    name: str = "positions_per_application_ratio_of_aggregates_derivative",
) -> NamedGradient:
    return ratio_of_aggregate_derivative(
        name=name,
        numerator_outcome=positions_outcome,
        denominator_outcome=applications_outcome,
        numerator_row_gradient=positions_row_gradient,
        denominator_row_gradient=applications_row_gradient,
        observed_numerator=observed_positions,
        observed_denominator=observed_applications,
        target=target,
        layout=layout,
    )


def hours_per_position_derivative(
    *,
    hours_row_gradient: RowGradient,
    positions_row_gradient: RowGradient,
    observed_hours: RowVector,
    observed_positions: RowVector,
    target: TargetPopulation,
    layout: CoefficientLayout,
    hours_outcome: str = "certified_hours",
    positions_outcome: str = "certified_positions",
    name: str = "hours_per_position_ratio_of_aggregates_derivative",
) -> NamedGradient:
    return ratio_of_aggregate_derivative(
        name=name,
        numerator_outcome=hours_outcome,
        denominator_outcome=positions_outcome,
        numerator_row_gradient=hours_row_gradient,
        denominator_row_gradient=positions_row_gradient,
        observed_numerator=observed_hours,
        observed_denominator=observed_positions,
        target=target,
        layout=layout,
    )


def average_unit_ratio_derivative(
    *,
    name: str,
    numerator_outcome: str,
    denominator_outcome: str,
    numerator_row_gradient: RowGradient,
    denominator_row_gradient: RowGradient,
    observed_numerator: RowVector,
    observed_denominator: RowVector,
    target: TargetPopulation,
    layout: CoefficientLayout,
) -> NamedGradient:
    """Derivative of a weighted average of row-level outcome ratios."""

    _check_row_gradient(numerator_row_gradient, layout)
    _check_row_gradient(denominator_row_gradient, layout)
    target._check_rows(observed_numerator.row_ids, observed_numerator.name)
    target._check_rows(observed_denominator.row_ids, observed_denominator.name)
    target._check_rows(numerator_row_gradient.row_ids, numerator_row_gradient.name)
    target._check_rows(denominator_row_gradient.row_ids, denominator_row_gradient.name)

    keep = target.include
    numerator = observed_numerator.values[keep]
    denominator = observed_denominator.values[keep]
    if np.any(~np.isfinite(numerator)) or np.any(~np.isfinite(denominator)):
        raise ValueError("Observed row ratios must be finite on target rows.")
    if np.any(denominator <= 0):
        raise ValueError(
            "Average-unit-ratio denominators must be positive on every target row."
        )
    normalized_weights = target.weights[keep] / target.weight_sum
    numerator_multiplier = normalized_weights / denominator
    denominator_multiplier = -normalized_weights * numerator / denominator**2
    numerator_effect = numerator_multiplier @ numerator_row_gradient.values[keep]
    denominator_effect = denominator_multiplier @ denominator_row_gradient.values[keep]
    return _named_gradient(
        name=name,
        kind="average_unit_ratio_derivative",
        target=target,
        layout=layout,
        contributions=(
            (numerator_outcome, numerator_effect),
            (denominator_outcome, denominator_effect),
        ),
    )


def direct_combined_outcome_oracle(
    coefficients: CommonCoefficientMatrix,
    *,
    target: TargetPopulation,
    terms: Sequence[CombinedOutcomeTerm],
) -> float:
    """Evaluate combined row-level outcome effects before forming a gradient.

    This intentionally computes ``row_gradient @ beta`` term by term.  It is
    therefore a useful direct oracle for analytic named-gradient tests.
    """

    if not terms:
        raise ValueError("Combined-outcome oracle requires at least one term.")
    result = 0.0
    for term in terms:
        _check_row_gradient(term.row_gradient, coefficients.layout)
        target._check_rows(term.row_gradient.row_ids, term.row_gradient.name)
        outcome_index = coefficients.layout.outcome_index(term.outcome)
        row_effect = RowVector(
            name=f"direct_effect::{term.outcome}",
            row_ids=term.row_gradient.row_ids,
            values=(term.row_gradient.values @ coefficients.values[:, outcome_index]),
        )
        if term.aggregation == "weighted_mean":
            aggregate = target.weighted_mean(row_effect)
        else:
            aggregate = target.weighted_sum(row_effect)
        result += term.multiplier * aggregate
    return float(result)


def _covariance_array(
    covariance_blocks: np.ndarray | Mapping[tuple[str, str], np.ndarray],
    layout: CoefficientLayout,
) -> np.ndarray:
    outcomes = layout.outcome_names
    k, j = layout.shape
    if isinstance(covariance_blocks, Mapping):
        blocks = np.empty((j, j, k, k), dtype=np.float64)
        for left_index, left in enumerate(outcomes):
            for right_index, right in enumerate(outcomes):
                direct = covariance_blocks.get((left, right))
                if direct is not None:
                    block = direct
                else:
                    reverse = covariance_blocks.get((right, left))
                    if reverse is None:
                        raise ValueError(
                            f"Missing covariance block for ({left}, {right})."
                        )
                    block = np.asarray(reverse).T
                block = np.asarray(block, dtype=np.float64)
                if block.shape != (k, k):
                    raise ValueError(
                        f"Covariance block ({left}, {right}) has shape {block.shape}."
                    )
                blocks[left_index, right_index] = block
    else:
        blocks = np.array(covariance_blocks, dtype=np.float64, copy=True)
        expected = (j, j, k, k)
        if blocks.shape != expected:
            raise ValueError(
                f"Cross-outcome covariance shape is {blocks.shape}, expected {expected}."
            )
    if not np.all(np.isfinite(blocks)):
        raise ValueError("Cross-outcome covariance blocks must be finite.")
    if not np.allclose(blocks, blocks.transpose(1, 0, 3, 2), rtol=1e-9, atol=1e-11):
        raise ValueError("Cross-outcome covariance blocks are not symmetric.")
    return blocks


def apply_delta_method(
    gradient: NamedGradient,
    coefficients: CommonCoefficientMatrix,
    covariance_blocks: np.ndarray | Mapping[tuple[str, str], np.ndarray],
) -> DeltaResult:
    """Apply an analytic named gradient to caller-supplied covariance blocks."""

    if coefficients.layout != gradient.layout:
        raise ValueError("Gradient and coefficient layouts do not match exactly.")
    blocks = _covariance_array(covariance_blocks, gradient.layout)
    variance = float(
        np.einsum(
            "ka,abkl,lb->",
            gradient.values,
            blocks,
            gradient.values,
            optimize=True,
        )
    )
    tolerance = 1e-10 * max(float(np.max(np.abs(blocks))), 1.0)
    if variance < -tolerance:
        raise ValueError("Delta variance is negative; covariance is not valid here.")
    variance = max(variance, 0.0)
    return DeltaResult(
        name=gradient.name,
        kind=gradient.kind,
        target_name=gradient.target_name,
        estimate=gradient.evaluate(coefficients),
        variance=variance,
        standard_error=float(np.sqrt(variance)),
    )
