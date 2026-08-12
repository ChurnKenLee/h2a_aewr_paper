"""Synthetic oracles for the MCW analytic covariance layer."""

import sys
import unittest
from pathlib import Path

import numpy as np

BRANCH_ROOT = Path(__file__).resolve().parents[1]
if str(BRANCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BRANCH_ROOT))

from mcw.inference import (
    InferenceContractError,
    apply_gradient,
    assemble_block_covariance,
    batch_linear_gradient_cross_outcome_inference,
    cluster_cross_outcome_covariance,
    cr2_cross_outcome_covariance_dense,
    direct_cv3_covariance,
    experimental_scalar_ccv_hc3,
    hc3_cross_outcome_covariance,
    linear_gradient_cross_outcome_inference,
    ols_bread,
    residualized_contrast_direction,
    residualized_contrast_directions,
    validate_covariance,
)
from scipy import linalg


class InferenceFixtures(unittest.TestCase):
    """Use one deterministic full-rank common design throughout."""

    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(20260812)
        n_observations = 48
        regressors = rng.normal(size=(n_observations, 3))
        cls.design = np.column_stack((np.ones(n_observations), regressors))
        cls.bread = ols_bread(cls.design)
        cls.hat = np.einsum(
            "ij,jk,ik->i",
            cls.design,
            cls.bread,
            cls.design,
        )
        raw_errors = rng.normal(size=(n_observations, 2))
        projection = cls.design @ cls.bread @ cls.design.T
        cls.residuals = (np.eye(n_observations) - projection) @ raw_errors
        cls.clusters = np.repeat(np.arange(8), 6)

    def test_hc3_matches_dense_manual_cross_outcome_formula(self) -> None:
        actual = hc3_cross_outcome_covariance(
            self.design,
            self.residuals,
            self.bread,
            self.hat,
        )
        n_coefficients = self.design.shape[1]
        expected = np.empty_like(actual)
        for outcome_a in range(2):
            for outcome_b in range(2):
                weights = (
                    self.residuals[:, outcome_a]
                    * self.residuals[:, outcome_b]
                    / (1.0 - self.hat) ** 2
                )
                meat = self.design.T @ (weights[:, None] * self.design)
                block = self.bread @ meat @ self.bread
                row = slice(
                    outcome_a * n_coefficients,
                    (outcome_a + 1) * n_coefficients,
                )
                column = slice(
                    outcome_b * n_coefficients,
                    (outcome_b + 1) * n_coefficients,
                )
                expected[row, column] = block
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_cross_outcome_blocks_equal_direct_combined_outcome(self) -> None:
        outcome_weights = np.array([0.7, -1.25])
        combined_residual = self.residuals @ outcome_weights
        n_coefficients = self.design.shape[1]
        combination = np.hstack(
            [weight * np.eye(n_coefficients) for weight in outcome_weights]
        )

        block_hc3 = hc3_cross_outcome_covariance(
            self.design,
            self.residuals,
            self.bread,
            self.hat,
        )
        direct_hc3 = hc3_cross_outcome_covariance(
            self.design,
            combined_residual,
            self.bread,
            self.hat,
        )
        np.testing.assert_allclose(
            combination @ block_hc3 @ combination.T,
            direct_hc3,
            rtol=1e-12,
            atol=1e-12,
        )

        block_cluster = cluster_cross_outcome_covariance(
            self.design,
            self.residuals,
            self.bread,
            self.clusters,
            n_parameters=n_coefficients,
        )
        direct_cluster = cluster_cross_outcome_covariance(
            self.design,
            combined_residual,
            self.bread,
            self.clusters,
            n_parameters=n_coefficients,
        )
        np.testing.assert_allclose(
            combination @ block_cluster.cr0 @ combination.T,
            direct_cluster.cr0,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            combination @ block_cluster.cr1 @ combination.T,
            direct_cluster.cr1,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_scalar_ccv_lambda_identities_and_endpoints(self) -> None:
        clusters = np.repeat(np.arange(4), 4)
        scales = np.repeat(np.array([1.0, 2.0, 0.5, 3.0]), 4)
        signs = np.tile(np.array([1.0, -1.0, 1.0, -1.0]), 4)
        direction = scales * signs
        residuals = np.column_stack(
            (
                np.linspace(-1.0, 1.0, direction.size),
                np.cos(np.arange(direction.size)),
            )
        )
        leverage = np.full(direction.size, 0.1)
        result = experimental_scalar_ccv_hc3(
            direction,
            residuals,
            leverage,
            clusters,
            n_parameters=2,
        )

        expected_omega = np.array([1.0, 4.0, 0.25, 9.0])
        expected_lambda = 1.0 - expected_omega.mean() ** 2 / np.mean(expected_omega**2)
        np.testing.assert_allclose(result.omega, expected_omega)
        self.assertAlmostEqual(result.lambda_weight, expected_lambda)
        self.assertAlmostEqual(
            result.lambda_weight,
            result.omega_cv**2 / (1.0 + result.omega_cv**2),
        )
        np.testing.assert_allclose(result.kappa, np.ones(4))
        np.testing.assert_allclose(
            result.ccv_hc3,
            expected_lambda * result.cr0 + (1.0 - expected_lambda) * result.hc3,
        )
        np.testing.assert_allclose(
            result.ccv_hc3_cr1,
            expected_lambda * result.cr1 + (1.0 - expected_lambda) * result.hc3,
        )

        small_scale = experimental_scalar_ccv_hc3(
            direction * 1e-8,
            residuals,
            leverage,
            clusters,
            n_parameters=2,
        )
        self.assertAlmostEqual(small_scale.lambda_weight, result.lambda_weight)
        self.assertAlmostEqual(small_scale.omega_cv, result.omega_cv)
        np.testing.assert_allclose(small_scale.kappa, result.kappa)

        constant = experimental_scalar_ccv_hc3(
            signs,
            residuals,
            leverage,
            clusters,
            n_parameters=2,
        )
        self.assertAlmostEqual(constant.lambda_weight, 0.0)
        np.testing.assert_allclose(constant.ccv_hc3, constant.hc3)

        with self.assertRaises(InferenceContractError):
            experimental_scalar_ccv_hc3(
                np.zeros_like(direction),
                residuals,
                leverage,
                clusters,
                n_parameters=2,
            )

    def test_direct_cv3_uses_literal_leave_cluster_estimates(self) -> None:
        full = np.array([1.0, -0.5])
        leave_out = np.array(
            [
                [1.1, -0.4],
                [0.8, -0.7],
                [1.05, -0.45],
                [1.2, -0.6],
            ]
        )
        deviations = leave_out - full
        expected = (3.0 / 4.0) * deviations.T @ deviations
        actual = direct_cv3_covariance(full, leave_out)
        np.testing.assert_allclose(actual, expected)

    def test_dense_cr2_matches_manual_full_hat_blocks(self) -> None:
        actual = cr2_cross_outcome_covariance_dense(
            self.design,
            self.residuals,
            self.bread,
            self.clusters,
            max_cluster_size=10,
        )
        n_coefficients = self.design.shape[1]
        expected = np.zeros_like(actual)
        for cluster in np.unique(self.clusters):
            selected = self.clusters == cluster
            cluster_design = self.design[selected]
            h_block = cluster_design @ self.bread @ cluster_design.T
            eigenvalues, eigenvectors = linalg.eigh(np.eye(6) - h_block)
            adjustment = (
                eigenvectors * (1.0 / np.sqrt(eigenvalues))[None, :]
            ) @ eigenvectors.T
            adjusted = adjustment @ self.residuals[selected]
            contribution = (self.bread @ cluster_design.T @ adjusted).T.reshape(
                2 * n_coefficients
            )
            expected += np.outer(contribution, contribution)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

        with self.assertRaises(InferenceContractError):
            cr2_cross_outcome_covariance_dense(
                self.design,
                self.residuals,
                self.bread,
                self.clusters,
                max_cluster_size=5,
            )

    def test_contrast_direction_blocks_gradient_and_psd_guards(self) -> None:
        contrast = np.array([0.0, 1.0, -0.5, 0.25])
        direction = residualized_contrast_direction(
            self.design,
            self.bread,
            contrast,
        )
        np.testing.assert_allclose(self.design.T @ direction, contrast, atol=1e-12)

        block = hc3_cross_outcome_covariance(
            self.design,
            self.residuals,
            self.bread,
            self.hat,
        )
        n_coefficients = self.design.shape[1]
        blocks = [
            [
                block[
                    row * n_coefficients : (row + 1) * n_coefficients,
                    column * n_coefficients : (column + 1) * n_coefficients,
                ]
                for column in range(2)
            ]
            for row in range(2)
        ]
        np.testing.assert_allclose(assemble_block_covariance(blocks), block)

        gradient = np.vstack(
            (
                np.r_[contrast, np.zeros(n_coefficients)],
                np.r_[np.zeros(n_coefficients), contrast],
            )
        )
        expected = gradient @ block @ gradient.T
        np.testing.assert_allclose(apply_gradient(block, gradient), expected)

        with self.assertRaises(InferenceContractError):
            validate_covariance(np.array([[1.0, 2.0], [2.0, 1.0]]))

    def test_batch_contrast_directions_match_scalar_calls(self) -> None:
        contrasts = np.column_stack(
            (
                np.array([0.0, 1.0, -0.5, 0.25]),
                np.array([1.0, -0.25, 0.0, 0.75]),
                np.array([-0.5, 0.0, 1.25, 0.0]),
            )
        )
        actual = residualized_contrast_directions(
            self.design,
            self.bread,
            contrasts,
        )
        expected = np.column_stack(
            [
                residualized_contrast_direction(
                    self.design,
                    self.bread,
                    contrasts[:, column],
                )
                for column in range(contrasts.shape[1])
            ]
        )

        self.assertEqual(actual.shape, (self.design.shape[0], contrasts.shape[1]))
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            self.design.T @ actual,
            contrasts,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_batch_contrast_directions_reject_zero_columns(self) -> None:
        contrasts = np.column_stack(
            (
                np.array([0.0, 1.0, -0.5, 0.25]),
                np.zeros(self.design.shape[1]),
            )
        )
        with self.assertRaisesRegex(
            InferenceContractError,
            "identically zero: 1",
        ):
            residualized_contrast_directions(
                self.design,
                self.bread,
                contrasts,
            )

        with self.assertRaisesRegex(InferenceContractError, "K x C matrix"):
            residualized_contrast_directions(
                self.design,
                self.bread,
                contrasts.T,
            )

    def test_linear_gradient_inference_matches_full_covariance_contraction(
        self,
    ) -> None:
        gradient = np.array(
            [
                [0.0, 0.5],
                [1.0, -0.25],
                [-0.5, 0.75],
                [0.25, 0.0],
            ]
        )
        actual = linear_gradient_cross_outcome_inference(
            self.design,
            self.residuals,
            self.bread,
            self.hat,
            self.clusters,
            gradient,
            n_parameters=self.design.shape[1],
        )
        stacked_gradient = gradient.T.reshape(-1)
        full_hc3 = hc3_cross_outcome_covariance(
            self.design,
            self.residuals,
            self.bread,
            self.hat,
        )
        full_cluster = cluster_cross_outcome_covariance(
            self.design,
            self.residuals,
            self.bread,
            self.clusters,
            n_parameters=self.design.shape[1],
        )
        expected_hc3 = float(stacked_gradient @ full_hc3 @ stacked_gradient)
        expected_cr0 = float(stacked_gradient @ full_cluster.cr0 @ stacked_gradient)
        expected_cr1 = float(stacked_gradient @ full_cluster.cr1 @ stacked_gradient)

        self.assertAlmostEqual(actual.hc3_variance, expected_hc3)
        self.assertAlmostEqual(actual.cr0_variance, expected_cr0)
        self.assertAlmostEqual(actual.cr1_variance, expected_cr1)
        self.assertAlmostEqual(actual.cr1_factor, full_cluster.cr1_factor)
        self.assertEqual(actual.n_clusters, full_cluster.n_clusters)
        self.assertAlmostEqual(
            actual.hc3_standard_error**2,
            actual.hc3_variance,
        )
        self.assertAlmostEqual(
            actual.cr0_standard_error**2,
            actual.cr0_variance,
        )
        self.assertAlmostEqual(
            actual.cr1_standard_error**2,
            actual.cr1_variance,
        )

    def test_linear_gradient_inference_allows_inactive_outcomes(self) -> None:
        gradient = np.column_stack(
            (
                np.array([0.0, 1.0, -0.5, 0.25]),
                np.zeros(self.design.shape[1]),
            )
        )
        actual = linear_gradient_cross_outcome_inference(
            self.design,
            self.residuals,
            self.bread,
            self.hat,
            self.clusters,
            gradient,
            n_parameters=self.design.shape[1],
        )
        expected = linear_gradient_cross_outcome_inference(
            self.design,
            self.residuals[:, 0],
            self.bread,
            self.hat,
            self.clusters,
            gradient[:, :1],
            n_parameters=self.design.shape[1],
        )
        self.assertEqual(actual, expected)

        with self.assertRaisesRegex(
            InferenceContractError,
            "gradient cannot be identically zero",
        ):
            linear_gradient_cross_outcome_inference(
                self.design,
                self.residuals,
                self.bread,
                self.hat,
                self.clusters,
                np.zeros_like(gradient),
                n_parameters=self.design.shape[1],
            )

    def test_batch_linear_gradients_match_repeated_and_stacked_inference(
        self,
    ) -> None:
        base = np.array([0.0, 1.0, -0.5, 0.25])
        gradients = np.stack(
            (
                np.array(
                    [
                        [0.0, 0.5],
                        [1.0, -0.25],
                        [-0.5, 0.75],
                        [0.25, 0.0],
                    ]
                ),
                np.column_stack((base, np.zeros_like(base))),
                np.column_stack((base, -2.0 * base)),
            )
        )
        actual = batch_linear_gradient_cross_outcome_inference(
            self.design,
            self.residuals,
            self.bread,
            self.hat,
            self.clusters,
            gradients,
            n_parameters=self.design.shape[1],
        )
        repeated = tuple(
            linear_gradient_cross_outcome_inference(
                self.design,
                self.residuals,
                self.bread,
                self.hat,
                self.clusters,
                gradient,
                n_parameters=self.design.shape[1],
            )
            for gradient in gradients
        )
        full_hc3 = hc3_cross_outcome_covariance(
            self.design,
            self.residuals,
            self.bread,
            self.hat,
        )
        full_cluster = cluster_cross_outcome_covariance(
            self.design,
            self.residuals,
            self.bread,
            self.clusters,
            n_parameters=self.design.shape[1],
        )

        self.assertEqual(len(actual), len(gradients))
        for gradient, batch_result, scalar_result in zip(
            gradients,
            actual,
            repeated,
            strict=True,
        ):
            np.testing.assert_allclose(
                [
                    batch_result.hc3_variance,
                    batch_result.cr0_variance,
                    batch_result.cr1_variance,
                    batch_result.cr1_factor,
                ],
                [
                    scalar_result.hc3_variance,
                    scalar_result.cr0_variance,
                    scalar_result.cr1_variance,
                    scalar_result.cr1_factor,
                ],
                rtol=1e-13,
                atol=1e-13,
            )
            self.assertEqual(batch_result.n_clusters, scalar_result.n_clusters)
            np.testing.assert_allclose(
                batch_result.raw_row_scores,
                scalar_result.raw_row_scores,
                rtol=1e-13,
                atol=1e-13,
            )
            stacked = gradient.T.reshape(-1)
            self.assertAlmostEqual(
                batch_result.hc3_variance,
                float(stacked @ full_hc3 @ stacked),
            )
            self.assertAlmostEqual(
                batch_result.cr0_variance,
                float(stacked @ full_cluster.cr0 @ stacked),
            )
            self.assertAlmostEqual(
                batch_result.cr1_variance,
                float(stacked @ full_cluster.cr1 @ stacked),
            )

        self.assertIsNone(actual[0].common_contrast_direction)
        self.assertIsNone(actual[0].outcome_loadings)
        proportional = actual[2]
        self.assertIsNotNone(proportional.common_contrast_direction)
        self.assertIsNotNone(proportional.outcome_loadings)
        common_direction = proportional.common_contrast_direction
        loadings = proportional.outcome_loadings
        assert common_direction is not None
        assert loadings is not None
        np.testing.assert_allclose(
            proportional.raw_row_scores,
            common_direction * (self.residuals @ loadings),
            rtol=1e-13,
            atol=1e-13,
        )
        reconstructed_gradient = (self.design.T @ common_direction)[:, None] * loadings[
            None, :
        ]
        np.testing.assert_allclose(
            reconstructed_gradient,
            gradients[2],
            rtol=1e-12,
            atol=1e-12,
        )

    def test_batch_linear_gradients_reject_zero_gradient(self) -> None:
        valid = np.column_stack(
            (
                np.array([0.0, 1.0, -0.5, 0.25]),
                np.zeros(self.design.shape[1]),
            )
        )
        gradients = np.stack((valid, np.zeros_like(valid)))
        with self.assertRaisesRegex(
            InferenceContractError,
            "identically zero: 1",
        ):
            batch_linear_gradient_cross_outcome_inference(
                self.design,
                self.residuals,
                self.bread,
                self.hat,
                self.clusters,
                gradients,
                n_parameters=self.design.shape[1],
            )


if __name__ == "__main__":
    unittest.main()
