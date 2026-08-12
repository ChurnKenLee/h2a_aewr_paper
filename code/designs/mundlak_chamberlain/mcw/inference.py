"""Analytic covariance tools for the version-4 MCW specification program.

The functions in this module operate on fitted arrays.  They do not construct
treatments, choose clusters, fit models, or implement resampling.  Outcome
coefficients are stacked outcome-major: all coefficients for outcome zero,
then all coefficients for outcome one, and so on.

The continuous-dose CCV mixture is experimental.  The Lean development proves
the covariance-kernel identities and the binary bridge, but it does not prove
that a scalar second-moment weight is sufficient for arbitrary continuous
assignment processes.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy import linalg

FloatArray = npt.NDArray[np.float64]


class InferenceContractError(ValueError):
    """Raised when an inference input or numerical invariant is invalid."""


@dataclass(frozen=True, slots=True)
class CovarianceCheck:
    """A covariance after symmetry and positive-semidefinite validation."""

    covariance: FloatArray
    maximum_asymmetry: float
    minimum_eigenvalue: float
    numerical_rank: int
    tolerance: float


@dataclass(frozen=True, slots=True)
class ClusterSandwich:
    """Cross-outcome CR0 and CR1 covariance matrices."""

    cr0: FloatArray
    cr1: FloatArray
    cr1_factor: float
    n_clusters: int


@dataclass(frozen=True, slots=True)
class LinearGradientSandwich:
    """Scalar HC3/CR0/CR1 variances for one cross-outcome linear gradient."""

    hc3_variance: float
    cr0_variance: float
    cr1_variance: float
    cr1_factor: float
    n_clusters: int
    raw_row_scores: FloatArray = field(repr=False, compare=False)
    common_contrast_direction: FloatArray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    outcome_loadings: FloatArray | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def hc3_standard_error(self) -> float:
        """Return the HC3 standard error."""

        return float(np.sqrt(self.hc3_variance))

    @property
    def cr0_standard_error(self) -> float:
        """Return the CR0 standard error."""

        return float(np.sqrt(self.cr0_variance))

    @property
    def cr1_standard_error(self) -> float:
        """Return the CR1 standard error."""

        return float(np.sqrt(self.cr1_variance))


@dataclass(frozen=True, slots=True)
class ExperimentalScalarCCV:
    """Experimental scalar-contrast continuous-dose CCV components."""

    hc3: FloatArray
    cr0: FloatArray
    cr1: FloatArray
    ccv_hc3: FloatArray
    ccv_hc3_cr1: FloatArray
    omega: FloatArray
    kappa: FloatArray
    lambda_weight: float
    omega_cv: float
    kappa_cv: float
    omega_zero_share: float
    cr1_factor: float
    n_clusters: int
    q: float = 1.0


def _float_array(value: npt.ArrayLike, *, name: str) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InferenceContractError(f"{name} must be numeric.") from error
    if not np.all(np.isfinite(array)):
        raise InferenceContractError(f"{name} must contain only finite values.")
    return array


def _matrix(value: npt.ArrayLike, *, name: str) -> FloatArray:
    array = _float_array(value, name=name)
    if array.ndim != 2 or min(array.shape) < 1:
        raise InferenceContractError(f"{name} must be a nonempty matrix.")
    return array


def _outcome_matrix(
    residuals: npt.ArrayLike,
    *,
    n_observations: int,
) -> FloatArray:
    array = _float_array(residuals, name="residuals")
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] != n_observations:
        raise InferenceContractError(
            "residuals must have one row per design observation."
        )
    if array.shape[1] < 1:
        raise InferenceContractError("residuals must contain at least one outcome.")
    return array


def validate_covariance(
    covariance: npt.ArrayLike,
    *,
    name: str = "covariance",
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 1e-10,
) -> CovarianceCheck:
    """Validate symmetry and PSD without repairing material violations.

    Numerical asymmetry below the declared tolerance is symmetrized.  Negative
    eigenvalues are never truncated; a materially negative eigenvalue raises.
    """

    matrix = _matrix(covariance, name=name)
    if matrix.shape[0] != matrix.shape[1]:
        raise InferenceContractError(f"{name} must be square.")
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise InferenceContractError("Covariance tolerances must be nonnegative.")

    scale = max(float(np.max(np.abs(matrix))), 1.0)
    tolerance = absolute_tolerance + relative_tolerance * scale
    maximum_asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    if maximum_asymmetry > tolerance:
        raise InferenceContractError(
            f"{name} is materially asymmetric: {maximum_asymmetry:.3e} > "
            f"{tolerance:.3e}."
        )

    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues = linalg.eigvalsh(symmetric, check_finite=False)
    eigen_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    psd_tolerance = absolute_tolerance + relative_tolerance * eigen_scale
    minimum_eigenvalue = float(eigenvalues[0])
    if minimum_eigenvalue < -psd_tolerance:
        raise InferenceContractError(
            f"{name} is not positive semidefinite: minimum eigenvalue "
            f"{minimum_eigenvalue:.3e} < {-psd_tolerance:.3e}."
        )

    return CovarianceCheck(
        covariance=symmetric,
        maximum_asymmetry=maximum_asymmetry,
        minimum_eigenvalue=minimum_eigenvalue,
        numerical_rank=int(np.sum(eigenvalues > psd_tolerance)),
        tolerance=psd_tolerance,
    )


def _validated_bread(design: FloatArray, bread: npt.ArrayLike) -> FloatArray:
    candidate = _matrix(bread, name="bread")
    n_coefficients = design.shape[1]
    if candidate.shape != (n_coefficients, n_coefficients):
        raise InferenceContractError(
            "bread must have one row and column per design coefficient."
        )
    checked = validate_covariance(candidate, name="bread")
    gram = design.T @ design
    inverse_error = linalg.norm(gram @ checked.covariance - np.eye(n_coefficients))
    inverse_scale = max(linalg.norm(gram) * linalg.norm(checked.covariance), 1.0)
    if inverse_error > 1e-8 * inverse_scale:
        raise InferenceContractError(
            "bread is not the inverse of the supplied common-design Gram matrix."
        )
    return checked.covariance


def ols_bread(design: npt.ArrayLike) -> FloatArray:
    """Return ``(X'X)^-1`` after an explicit full-rank check."""

    matrix = _matrix(design, name="design")
    gram = matrix.T @ matrix
    if np.linalg.matrix_rank(gram) != gram.shape[0]:
        raise InferenceContractError("The common design is rank deficient.")
    return linalg.inv(gram, check_finite=False)


def _common_inputs(
    design: npt.ArrayLike,
    residuals: npt.ArrayLike,
    bread: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    matrix = _matrix(design, name="design")
    outcomes = _outcome_matrix(residuals, n_observations=matrix.shape[0])
    inverse = _validated_bread(matrix, bread)
    return matrix, outcomes, inverse


def _leverage(
    leverage: npt.ArrayLike,
    *,
    n_observations: int,
    denominator_tolerance: float = 1e-10,
) -> FloatArray:
    values = _float_array(leverage, name="full_model_leverage")
    if values.ndim != 1 or values.size != n_observations:
        raise InferenceContractError(
            "full_model_leverage must contain one value per observation."
        )
    scale_tolerance = 1e-10
    if float(np.min(values)) < -scale_tolerance:
        raise InferenceContractError("Full-model leverage cannot be negative.")
    if float(np.max(values)) > 1.0 + scale_tolerance:
        raise InferenceContractError("Full-model leverage cannot exceed one.")
    if float(np.min(1.0 - values)) <= denominator_tolerance:
        raise InferenceContractError(
            "HC3 is undefined because at least one full-model leverage is too "
            "close to one."
        )
    return values


def _cluster_codes(
    clusters: npt.ArrayLike,
    *,
    n_observations: int,
) -> tuple[npt.NDArray[np.int64], int]:
    labels = np.asarray(clusters)
    if labels.ndim != 1 or labels.size != n_observations:
        raise InferenceContractError(
            "clusters must contain one one-way cluster label per observation."
        )
    if labels.dtype.kind in {"f", "c"} and np.any(~np.isfinite(labels)):
        raise InferenceContractError("clusters cannot contain missing values.")
    if labels.dtype.kind == "O":
        for value in labels:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                raise InferenceContractError("clusters cannot contain missing values.")
    try:
        _, codes = np.unique(labels, return_inverse=True)
    except TypeError as error:
        raise InferenceContractError(
            "clusters must use mutually comparable scalar labels."
        ) from error
    n_clusters = int(codes.max()) + 1 if codes.size else 0
    if n_clusters < 2:
        raise InferenceContractError(
            "Cluster inference requires at least two clusters."
        )
    return codes.astype(np.int64, copy=False), n_clusters


def _coefficient_loadings(design: FloatArray, bread: FloatArray) -> FloatArray:
    return design @ bread.T


def _stacked_influence(
    loadings: FloatArray,
    residuals: FloatArray,
) -> FloatArray:
    influence = residuals[:, :, None] * loadings[:, None, :]
    return influence.reshape(influence.shape[0], -1)


def hc3_cross_outcome_covariance(
    design: npt.ArrayLike,
    residuals: npt.ArrayLike,
    bread: npt.ArrayLike,
    full_model_leverage: npt.ArrayLike,
) -> FloatArray:
    """Return the common-design, cross-outcome HC3 covariance.

    The caller must supply the exact diagonal of the full model hat matrix,
    including fixed effects and nuisance regressors.  A partial-regression
    leverage such as ``Dtilde**2 / sum(Dtilde**2)`` is not a substitute.
    """

    matrix, outcomes, inverse = _common_inputs(design, residuals, bread)
    leverage = _leverage(full_model_leverage, n_observations=matrix.shape[0])
    adjusted = outcomes / (1.0 - leverage[:, None])
    influence = _stacked_influence(
        _coefficient_loadings(matrix, inverse),
        adjusted,
    )
    covariance = influence.T @ influence
    return validate_covariance(covariance, name="HC3 covariance").covariance


def cr1_small_sample_factor(
    n_observations: int,
    n_clusters: int,
    n_parameters: int,
) -> float:
    """Return ``G/(G-1) * (N-1)/(N-K)`` with explicit guards."""

    if n_clusters < 2:
        raise InferenceContractError("CR1 requires at least two clusters.")
    if n_parameters < 1 or n_observations <= n_parameters:
        raise InferenceContractError("CR1 requires 1 <= K < N.")
    return (n_clusters / (n_clusters - 1.0)) * (
        (n_observations - 1.0) / (n_observations - n_parameters)
    )


def cluster_cross_outcome_covariance(
    design: npt.ArrayLike,
    residuals: npt.ArrayLike,
    bread: npt.ArrayLike,
    clusters: npt.ArrayLike,
    *,
    n_parameters: int,
) -> ClusterSandwich:
    """Return common-design CR0 and conventional-factor CR1 covariances."""

    matrix, outcomes, inverse = _common_inputs(design, residuals, bread)
    codes, n_clusters = _cluster_codes(
        clusters,
        n_observations=matrix.shape[0],
    )
    influence = _stacked_influence(
        _coefficient_loadings(matrix, inverse),
        outcomes,
    )
    cluster_influence = np.zeros((n_clusters, influence.shape[1]))
    np.add.at(cluster_influence, codes, influence)
    cr0 = validate_covariance(
        cluster_influence.T @ cluster_influence,
        name="CR0 covariance",
    ).covariance
    factor = cr1_small_sample_factor(
        matrix.shape[0],
        n_clusters,
        n_parameters,
    )
    cr1 = validate_covariance(factor * cr0, name="CR1 covariance").covariance
    return ClusterSandwich(
        cr0=cr0,
        cr1=cr1,
        cr1_factor=factor,
        n_clusters=n_clusters,
    )


def _residualized_contrast_directions(
    design: FloatArray,
    bread: FloatArray,
    contrasts: FloatArray,
) -> FloatArray:
    """Compute and guard directions after the common inputs are validated."""

    directions = design @ (bread @ contrasts)
    normal_equation_errors = linalg.norm(
        design.T @ directions - contrasts,
        axis=0,
    )
    contrast_scales = np.maximum(linalg.norm(contrasts, axis=0), 1.0)
    # The rich pooled projection is intentionally close to the declared
    # condition-number warning boundary.  Require seven reliable decimal
    # digits in the contrast normal equation; this is still stricter than the
    # production fit's reported solve-residual guard.  Apply the guard to each
    # direction so one large contrast cannot mask failure of another column.
    failed = np.flatnonzero(normal_equation_errors > 1e-7 * contrast_scales)
    if failed.size:
        raise InferenceContractError(
            "Contrast direction columns failed their normal-equation check: "
            + ", ".join(str(int(index)) for index in failed)
        )
    return directions


def residualized_contrast_directions(
    design: npt.ArrayLike,
    bread: npt.ArrayLike,
    contrasts: npt.ArrayLike,
) -> FloatArray:
    """Return normalized FWL directions for ``K x C`` coefficient contrasts.

    The common ``N x K`` design and ``K x K`` bread are validated once.  Each
    contrast column must be nonzero, and the returned array has shape ``N x C``.
    For a selected coefficient, its column is
    ``Dtilde / (Dtilde'Dtilde)``.  Global normalization changes reported omega
    levels but not lambda, omega CV, or kappa.
    """

    matrix = _matrix(design, name="design")
    inverse = _validated_bread(matrix, bread)
    candidates = _float_array(contrasts, name="contrasts")
    if (
        candidates.ndim != 2
        or candidates.shape[0] != matrix.shape[1]
        or candidates.shape[1] < 1
    ):
        raise InferenceContractError(
            "contrasts must be a nonempty K x C matrix with one row per "
            "common-design coefficient."
        )
    zero_columns = np.flatnonzero(~np.any(candidates != 0.0, axis=0))
    if zero_columns.size:
        raise InferenceContractError(
            "contrast columns cannot be identically zero: "
            + ", ".join(str(int(index)) for index in zero_columns)
        )
    return _residualized_contrast_directions(matrix, inverse, candidates)


def residualized_contrast_direction(
    design: npt.ArrayLike,
    bread: npt.ArrayLike,
    contrast: npt.ArrayLike,
) -> FloatArray:
    """Return the normalized FWL direction for a scalar coefficient contrast.

    For a selected scalar coefficient this is ``Dtilde / (Dtilde'Dtilde)``.
    Its global normalization changes reported omega levels but not lambda,
    omega CV, or kappa.
    """

    matrix = _matrix(design, name="design")
    inverse = _validated_bread(matrix, bread)
    vector = _float_array(contrast, name="contrast")
    if vector.ndim != 1 or vector.size != matrix.shape[1]:
        raise InferenceContractError(
            "contrast must contain one value per common-design coefficient."
        )
    if not np.any(vector != 0.0):
        raise InferenceContractError("contrast cannot be identically zero.")
    return _residualized_contrast_directions(
        matrix,
        inverse,
        vector[:, None],
    )[:, 0]


def linear_gradient_cross_outcome_inference(
    design: npt.ArrayLike,
    residuals: npt.ArrayLike,
    bread: npt.ArrayLike,
    full_model_leverage: npt.ArrayLike,
    clusters: npt.ArrayLike,
    gradient: npt.ArrayLike,
    *,
    n_parameters: int,
) -> LinearGradientSandwich:
    """Contract exact HC3/CR0/CR1 inference for one ``K x J`` gradient.

    Coefficients are conceptually stacked outcome-major, but the full
    ``(K J) x (K J)`` covariance is never materialized.  Instead, the batched
    residualized directions turn each observation into one scalar score.  A
    gradient column may be zero when an outcome does not enter the estimand;
    the complete gradient must contain at least one nonzero value.
    """

    derivative = _float_array(gradient, name="gradient")
    if derivative.ndim != 2 or min(derivative.shape) < 1:
        raise InferenceContractError("gradient must be a nonempty K x J matrix.")
    if not np.any(derivative != 0.0):
        raise InferenceContractError("gradient cannot be identically zero.")
    return batch_linear_gradient_cross_outcome_inference(
        design,
        residuals,
        bread,
        full_model_leverage,
        clusters,
        derivative[None, :, :],
        n_parameters=n_parameters,
    )[0]


def _proportional_outcome_loadings(
    gradient: FloatArray,
) -> tuple[int, FloatArray] | None:
    """Factor ``gradient`` as one coefficient contrast times outcome loadings."""

    column_norms = linalg.norm(gradient, axis=0)
    reference = int(np.argmax(column_norms))
    base = gradient[:, reference]
    denominator = float(base @ base)
    if denominator == 0.0:
        return None
    loadings = (base @ gradient) / denominator
    reconstruction = base[:, None] * loadings[None, :]
    error = linalg.norm(gradient - reconstruction)
    scale = max(linalg.norm(gradient), np.finfo(np.float64).tiny)
    if error > 1e-10 * scale:
        return None
    return reference, np.asarray(loadings, dtype=np.float64)


def batch_linear_gradient_cross_outcome_inference(
    design: npt.ArrayLike,
    residuals: npt.ArrayLike,
    bread: npt.ArrayLike,
    full_model_leverage: npt.ArrayLike,
    clusters: npt.ArrayLike,
    gradients: npt.ArrayLike,
    *,
    n_parameters: int,
) -> tuple[LinearGradientSandwich, ...]:
    """Contract HC3/CR0/CR1 for multiple ``G x K x J`` gradients.

    The common design, bread, residuals, leverage, and cluster partition are
    validated exactly once.  All active coefficient contrasts are transformed
    to residualized directions in one matrix multiplication.  Each result
    retains its raw scalar row scores.  When the outcome columns of a gradient
    are proportional, it also retains a common coefficient direction and the
    corresponding outcome loadings for scalar-direction diagnostics.
    """

    matrix, outcomes, inverse = _common_inputs(design, residuals, bread)
    leverage = _leverage(
        full_model_leverage,
        n_observations=matrix.shape[0],
    )
    codes, n_clusters = _cluster_codes(
        clusters,
        n_observations=matrix.shape[0],
    )
    derivatives = _float_array(gradients, name="gradients")
    if derivatives.ndim != 3 or min(derivatives.shape) < 1:
        raise InferenceContractError("gradients must be a nonempty G x K x J array.")
    n_gradients, n_coefficients, n_outcomes = derivatives.shape
    if n_coefficients != matrix.shape[1]:
        raise InferenceContractError(
            "gradients must have one coefficient row per design column."
        )
    if n_outcomes != outcomes.shape[1]:
        raise InferenceContractError(
            "gradients must have one outcome column per residual outcome."
        )

    flattened = derivatives.transpose(1, 0, 2).reshape(n_coefficients, -1)
    active = np.any(flattened != 0.0, axis=0)
    active_by_gradient = active.reshape(n_gradients, n_outcomes).any(axis=1)
    zero_gradients = np.flatnonzero(~active_by_gradient)
    if zero_gradients.size:
        raise InferenceContractError(
            "gradient matrices cannot be identically zero: "
            + ", ".join(str(int(index)) for index in zero_gradients)
        )

    active_flat_indices = np.flatnonzero(active)
    directions = _residualized_contrast_directions(
        matrix,
        inverse,
        flattened[:, active],
    )
    gradient_indices = active_flat_indices // n_outcomes
    outcome_indices = active_flat_indices % n_outcomes
    raw_scores = np.zeros((matrix.shape[0], n_gradients), dtype=np.float64)
    for direction_index, (gradient_index, outcome_index) in enumerate(
        zip(gradient_indices, outcome_indices, strict=True)
    ):
        raw_scores[:, gradient_index] += (
            directions[:, direction_index] * outcomes[:, outcome_index]
        )

    hc3_scores = raw_scores / (1.0 - leverage[:, None])
    hc3_variances = np.einsum(
        "ij,ij->j",
        hc3_scores,
        hc3_scores,
        optimize=True,
    )
    cluster_scores = np.zeros((n_clusters, n_gradients), dtype=np.float64)
    np.add.at(cluster_scores, codes, raw_scores)
    cr0_variances = np.einsum(
        "ij,ij->j",
        cluster_scores,
        cluster_scores,
        optimize=True,
    )
    factor = cr1_small_sample_factor(
        matrix.shape[0],
        n_clusters,
        n_parameters,
    )
    direction_positions = np.full(flattened.shape[1], -1, dtype=np.int64)
    direction_positions[active] = np.arange(active_flat_indices.size)

    results: list[LinearGradientSandwich] = []
    for gradient_index in range(n_gradients):
        factorization = _proportional_outcome_loadings(derivatives[gradient_index])
        common_direction = None
        outcome_loadings = None
        if factorization is not None:
            reference, outcome_loadings = factorization
            flat_index = gradient_index * n_outcomes + reference
            common_direction = directions[:, direction_positions[flat_index]]
        results.append(
            LinearGradientSandwich(
                hc3_variance=float(hc3_variances[gradient_index]),
                cr0_variance=float(cr0_variances[gradient_index]),
                cr1_variance=float(factor * cr0_variances[gradient_index]),
                cr1_factor=factor,
                n_clusters=n_clusters,
                raw_row_scores=raw_scores[:, gradient_index],
                common_contrast_direction=common_direction,
                outcome_loadings=outcome_loadings,
            )
        )
    return tuple(results)


def _population_cv(values: FloatArray, *, name: str) -> float:
    mean = float(np.mean(values))
    if mean <= 0:
        raise InferenceContractError(f"{name} has no positive identifying variation.")
    variance = float(np.mean((values - mean) ** 2))
    return float(np.sqrt(variance) / mean)


def experimental_scalar_ccv_hc3(
    contrast_direction: npt.ArrayLike,
    residuals: npt.ArrayLike,
    full_model_leverage: npt.ArrayLike,
    clusters: npt.ArrayLike,
    *,
    n_parameters: int,
    zero_tolerance: float = 1e-14,
) -> ExperimentalScalarCCV:
    """Compute the experimental q=1 scalar continuous-dose CCV-HC3.

    ``contrast_direction`` is the normalized residualized scalar direction,
    usually returned by :func:`residualized_contrast_direction`.  The same
    direction and cluster partition are used for every outcome.
    """

    direction = _float_array(contrast_direction, name="contrast_direction")
    if direction.ndim != 1 or direction.size < 1:
        raise InferenceContractError("contrast_direction must be a nonempty vector.")
    outcomes = _outcome_matrix(residuals, n_observations=direction.size)
    leverage = _leverage(full_model_leverage, n_observations=direction.size)
    codes, n_clusters = _cluster_codes(
        clusters,
        n_observations=direction.size,
    )
    if zero_tolerance < 0:
        raise InferenceContractError("zero_tolerance must be nonnegative.")

    counts = np.bincount(codes, minlength=n_clusters).astype(np.float64)
    second = (
        np.bincount(
            codes,
            weights=direction**2,
            minlength=n_clusters,
        )
        / counts
    )
    fourth = (
        np.bincount(
            codes,
            weights=direction**4,
            minlength=n_clusters,
        )
        / counts
    )
    omega_scale = float(np.max(second))
    if omega_scale <= 0:
        raise InferenceContractError(
            "Residualized contrast direction has no cluster-level variation."
        )

    # Scale before forming the second moment.  The contrast direction is
    # normalized and can be numerically small in a large panel, while lambda
    # is exactly invariant to any global rescaling.
    scaled_omega = second / omega_scale
    mean_omega = float(np.mean(scaled_omega))
    second_moment = float(np.mean(scaled_omega**2))
    moment_ratio = mean_omega**2 / second_moment
    numerical_tolerance = 1e-10
    if moment_ratio < -numerical_tolerance or moment_ratio > 1 + numerical_tolerance:
        raise InferenceContractError(
            "The omega moment ratio lies materially outside [0, 1]."
        )
    moment_ratio = min(max(moment_ratio, 0.0), 1.0)
    lambda_weight = 1.0 - moment_ratio

    kappa = np.full(n_clusters, np.nan)
    positive = second > zero_tolerance * omega_scale
    kappa[positive] = fourth[positive] / second[positive] ** 2
    finite_kappa = kappa[np.isfinite(kappa)]
    kappa_cv = (
        _population_cv(finite_kappa, name="kappa") if finite_kappa.size else np.nan
    )
    omega_cv = _population_cv(second, name="omega")

    robust_influence = direction[:, None] * outcomes / (1.0 - leverage[:, None])
    hc3 = validate_covariance(
        robust_influence.T @ robust_influence,
        name="scalar-contrast HC3 covariance",
    ).covariance

    score = direction[:, None] * outcomes
    cluster_score = np.zeros((n_clusters, outcomes.shape[1]))
    np.add.at(cluster_score, codes, score)
    cr0 = validate_covariance(
        cluster_score.T @ cluster_score,
        name="scalar-contrast CR0 covariance",
    ).covariance
    factor = cr1_small_sample_factor(
        direction.size,
        n_clusters,
        n_parameters,
    )
    cr1 = validate_covariance(factor * cr0, name="scalar-contrast CR1").covariance
    ccv_hc3 = validate_covariance(
        lambda_weight * cr0 + (1.0 - lambda_weight) * hc3,
        name="experimental CCV-HC3 covariance",
    ).covariance
    ccv_hc3_cr1 = validate_covariance(
        lambda_weight * cr1 + (1.0 - lambda_weight) * hc3,
        name="experimental CCV-HC3-CR1 covariance",
    ).covariance

    return ExperimentalScalarCCV(
        hc3=hc3,
        cr0=cr0,
        cr1=cr1,
        ccv_hc3=ccv_hc3,
        ccv_hc3_cr1=ccv_hc3_cr1,
        omega=second,
        kappa=kappa,
        lambda_weight=lambda_weight,
        omega_cv=omega_cv,
        kappa_cv=kappa_cv,
        omega_zero_share=float(np.mean(~positive)),
        cr1_factor=factor,
        n_clusters=n_clusters,
    )


def direct_cv3_covariance(
    full_sample_estimate: npt.ArrayLike,
    leave_cluster_out_estimates: npt.ArrayLike,
) -> FloatArray:
    """Return CV3 from literal leave-one-cluster-out estimates.

    The caller is responsible for actually deleting each cluster and refitting
    the full model, including fixed effects.  No FWL-conditional shortcut is
    used or claimed to be exact.
    """

    full = _float_array(full_sample_estimate, name="full_sample_estimate")
    if full.ndim == 0:
        full = full.reshape(1)
    if full.ndim != 1 or full.size < 1:
        raise InferenceContractError(
            "full_sample_estimate must be a nonempty coefficient vector."
        )
    leave_out = _float_array(
        leave_cluster_out_estimates,
        name="leave_cluster_out_estimates",
    )
    if leave_out.ndim == 1 and full.size == 1:
        leave_out = leave_out[:, None]
    if leave_out.ndim != 2 or leave_out.shape[1] != full.size:
        raise InferenceContractError(
            "leave_cluster_out_estimates must have one row per deleted cluster "
            "and one column per full-sample estimate."
        )
    n_clusters = leave_out.shape[0]
    if n_clusters < 2:
        raise InferenceContractError("CV3 requires at least two deleted clusters.")
    deviations = leave_out - full[None, :]
    covariance = ((n_clusters - 1.0) / n_clusters) * (deviations.T @ deviations)
    return validate_covariance(covariance, name="direct-refit CV3").covariance


def cr2_cross_outcome_covariance_dense(
    design: npt.ArrayLike,
    residuals: npt.ArrayLike,
    bread: npt.ArrayLike,
    clusters: npt.ArrayLike,
    *,
    max_cluster_size: int = 250,
    max_dense_elements: int = 1_000_000,
    eigenvalue_tolerance: float = 1e-10,
) -> FloatArray:
    """Return dense full-model CR2 for small verification fixtures.

    This constructs each ``H_gg`` block and therefore deliberately refuses
    large clusters.  It is a test/reference implementation, not the production
    path for the county-year panel.
    """

    matrix, outcomes, inverse = _common_inputs(design, residuals, bread)
    codes, n_clusters = _cluster_codes(
        clusters,
        n_observations=matrix.shape[0],
    )
    if max_cluster_size < 1 or max_dense_elements < 1:
        raise InferenceContractError("CR2 dense guards must be positive.")
    counts = np.bincount(codes, minlength=n_clusters)
    if int(np.max(counts)) > max_cluster_size:
        raise InferenceContractError(
            "Dense CR2 cluster-size guard exceeded; use a production block "
            "algorithm instead."
        )
    if int(np.sum(counts.astype(np.int64) ** 2)) > max_dense_elements:
        raise InferenceContractError(
            "Dense CR2 element guard exceeded; use a production block algorithm."
        )
    if eigenvalue_tolerance <= 0:
        raise InferenceContractError("eigenvalue_tolerance must be positive.")

    dimension = outcomes.shape[1] * matrix.shape[1]
    covariance = np.zeros((dimension, dimension))
    for cluster in range(n_clusters):
        selected = codes == cluster
        cluster_design = matrix[selected]
        h_block = cluster_design @ inverse @ cluster_design.T
        annihilator = np.eye(int(np.sum(selected))) - h_block
        annihilator = (annihilator + annihilator.T) / 2.0
        eigenvalues, eigenvectors = linalg.eigh(
            annihilator,
            check_finite=False,
        )
        scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        if float(eigenvalues[0]) <= eigenvalue_tolerance * scale:
            raise InferenceContractError(
                "Dense CR2 adjustment is singular or numerically unstable."
            )
        adjustment = (
            eigenvectors * (1.0 / np.sqrt(eigenvalues))[None, :]
        ) @ eigenvectors.T
        adjusted_residuals = adjustment @ outcomes[selected]
        cluster_contribution = (
            inverse @ cluster_design.T @ adjusted_residuals
        ).T.reshape(-1)
        covariance += np.outer(cluster_contribution, cluster_contribution)

    return validate_covariance(covariance, name="dense CR2 covariance").covariance


def assemble_block_covariance(
    blocks: Sequence[Sequence[npt.ArrayLike]],
) -> FloatArray:
    """Assemble reciprocal covariance blocks and validate the full matrix."""

    n_rows = len(blocks)
    if n_rows < 1 or any(len(row) != n_rows for row in blocks):
        raise InferenceContractError("blocks must form a nonempty square grid.")
    diagonal_sizes: list[int] = []
    converted: list[list[FloatArray]] = []
    for row_index, row in enumerate(blocks):
        converted_row: list[FloatArray] = []
        for column_index, block in enumerate(row):
            converted_row.append(
                _matrix(block, name=f"blocks[{row_index}][{column_index}]")
            )
        converted.append(converted_row)
        diagonal = converted_row[row_index]
        if diagonal.shape[0] != diagonal.shape[1]:
            raise InferenceContractError("Diagonal covariance blocks must be square.")
        diagonal_sizes.append(diagonal.shape[0])

    for row_index in range(n_rows):
        for column_index in range(n_rows):
            expected = (diagonal_sizes[row_index], diagonal_sizes[column_index])
            block = converted[row_index][column_index]
            if block.shape != expected:
                raise InferenceContractError(
                    f"blocks[{row_index}][{column_index}] has shape "
                    f"{block.shape}, expected {expected}."
                )
            reciprocal = converted[column_index][row_index].T
            scale = max(float(np.max(np.abs(block))), 1.0)
            if float(np.max(np.abs(block - reciprocal))) > 1e-10 * scale:
                raise InferenceContractError(
                    "Off-diagonal covariance blocks are not reciprocal transposes."
                )

    covariance = np.block(converted)
    return validate_covariance(covariance, name="assembled block covariance").covariance


def apply_gradient(
    covariance: npt.ArrayLike,
    gradient: npt.ArrayLike,
) -> FloatArray:
    """Apply an arbitrary scalar or vector delta-method gradient ``G V G'``."""

    checked = validate_covariance(covariance, name="coefficient covariance")
    derivative = _float_array(gradient, name="gradient")
    if derivative.ndim == 1:
        derivative = derivative[None, :]
    if derivative.ndim != 2 or derivative.shape[1] != checked.covariance.shape[0]:
        raise InferenceContractError(
            "gradient must have one column per stacked coefficient."
        )
    transformed = derivative @ checked.covariance @ derivative.T
    return validate_covariance(
        transformed,
        name="delta-method covariance",
    ).covariance


__all__ = [
    "ClusterSandwich",
    "CovarianceCheck",
    "ExperimentalScalarCCV",
    "InferenceContractError",
    "LinearGradientSandwich",
    "apply_gradient",
    "assemble_block_covariance",
    "batch_linear_gradient_cross_outcome_inference",
    "cluster_cross_outcome_covariance",
    "cr1_small_sample_factor",
    "cr2_cross_outcome_covariance_dense",
    "direct_cv3_covariance",
    "experimental_scalar_ccv_hc3",
    "hc3_cross_outcome_covariance",
    "linear_gradient_cross_outcome_inference",
    "ols_bread",
    "residualized_contrast_direction",
    "residualized_contrast_directions",
    "validate_covariance",
]
