import Mathlib

/-!
# Continuous-dose Causal Cluster Variance, in Lean 4

This file formalizes the algebraic core of the LaTeX note:

* FWL/OLS score expansion after residualization, stated for an arbitrary finite
  observation index.
* Robust and clustered finite-sample variance formulas.
* The identity saying that the clustered estimator is the robust estimator plus
  within-cluster cross-products.
* The binary fixed-effect reduction: for a 0/1 treatment, the residualized
  within-cluster second moment is `Wbar * (1 - Wbar)`.
* The location-scale fourth-moment scaling calculation.
* A design-covariance sandwich estimator for arbitrary real-valued treatments.
* Mathlib-based asymptotic studentization and interval coverage for the feasible
  design-covariance standard error, conditional on a design-studentized CLT and
  consistency of the estimated residual/covariance inputs.
-/

noncomputable section

open scoped BigOperators
open Finset

namespace ContinuousCCV

/-! ## 1. Observation-level FWL algebra -/

namespace ObservationLevel

-- LEAN 4 FIX: Swapped Type* for Type _ to avoid parser unexpected token errors
variable {α : Type _} [Fintype α]

/-- S = sum of Dtilde_i^2. -/
def denom (Dtilde : α → ℝ) : ℝ :=
  ∑ a : α, Dtilde a ^ 2

/-- FWL scalar OLS coefficient: (Dtilde' Ytilde) / (Dtilde' Dtilde). -/
def betaHat (Dtilde Ytilde : α → ℝ) : ℝ :=
  (∑ a : α, Dtilde a * Ytilde a) / denom Dtilde

/-- The summed score: sum of Dtilde_i * utilde_i. -/
def scoreSum (Dtilde utilde : α → ℝ) : ℝ :=
  ∑ a : α, Dtilde a * utilde a

/--
Finite-sample FWL score expansion:
betaHat - beta = score / denom, whenever the denominator is nonzero.
-/
theorem betaHat_error
    (Dtilde Ytilde utilde : α → ℝ) (β : ℝ)
    (hY : ∀ a : α, Ytilde a = Dtilde a * β + utilde a)
    (hS : denom Dtilde ≠ 0) :
    betaHat Dtilde Ytilde - β = scoreSum Dtilde utilde / denom Dtilde := by
  classical
  unfold betaHat scoreSum
  unfold denom at hS ⊢
  have hnum :
      (∑ a : α, Dtilde a * Ytilde a) =
        β * (∑ a : α, Dtilde a ^ 2) + ∑ a : α, Dtilde a * utilde a := by
    calc
      (∑ a : α, Dtilde a * Ytilde a)
          = ∑ a : α, Dtilde a * (Dtilde a * β + utilde a) := by
              apply Finset.sum_congr rfl
              intro a _ha
              rw [hY a]
      _ = ∑ a : α, (β * (Dtilde a ^ 2) + Dtilde a * utilde a) := by
              apply Finset.sum_congr rfl
              intro a _ha
              ring
      _ = (∑ a : α, β * (Dtilde a ^ 2)) + ∑ a : α, Dtilde a * utilde a := by
              rw [Finset.sum_add_distrib]
      _ = β * (∑ a : α, Dtilde a ^ 2) + ∑ a : α, Dtilde a * utilde a := by
              rw [← Finset.mul_sum]
  rw [hnum]
  field_simp [hS]
  ring

end ObservationLevel

/-! ## 2. Cluster-level variance algebra -/

namespace ClusterLevel

-- LEAN 4 FIX: Separated dependent variable typeclass binders to ensure sequential elaboration
variable {G : Type _} [Fintype G] [DecidableEq G]
variable {I : G → Type _}
variable [∀ g : G, Fintype (I g)]
variable [∀ g : G, DecidableEq (I g)]

/-- Cluster score: sum of psi_gi. -/
def scoreCluster (ψ : ∀ g : G, I g → ℝ) (g : G) : ℝ :=
  ∑ i : I g, ψ g i

/-- Common scale factor: S^(-2). -/
def scale (S : ℝ) : ℝ :=
  (S ^ 2)⁻¹

/-- Eicker-Huber-White/robust scalar variance. -/
def robustVar (S : ℝ) (ψ : ∀ g : G, I g → ℝ) : ℝ :=
  scale S * ∑ g : G, ∑ i : I g, ψ g i ^ 2

/-- Liang-Zeger clustered scalar variance. -/
def clusterVar (S : ℝ) (ψ : ∀ g : G, I g → ℝ) : ℝ :=
  scale S * ∑ g : G, scoreCluster ψ g ^ 2

/-- Same clustered variance, written as all within-cluster pair products. -/
def pairwiseClusterVar (S : ℝ) (ψ : ∀ g : G, I g → ℝ) : ℝ :=
  scale S * ∑ g : G, ∑ i : I g, ∑ j : I g, ψ g i * ψ g j

/-- The within-cluster off-diagonal cross-product component. -/
def crossProductVar (S : ℝ) (ψ : ∀ g : G, I g → ℝ) : ℝ :=
  scale S * ∑ g : G, ∑ i : I g, ∑ j ∈ (Finset.univ.erase i), ψ g i * ψ g j

/-- A finite-sum identity: the square of a sum is diagonal terms plus off-diagonal terms. -/
lemma sum_sq_eq_diag_add_offdiag
    {ι : Type _} [Fintype ι] [DecidableEq ι] (a : ι → ℝ) :
    (∑ i : ι, a i) ^ 2 =
      (∑ i : ι, a i ^ 2) + ∑ i : ι, ∑ j ∈ (Finset.univ.erase i), a i * a j := by
  classical
  calc
    (∑ i : ι, a i) ^ 2 = (∑ i : ι, a i) * ∑ j : ι, a j := by
      ring
    _ = ∑ i : ι, ∑ j : ι, a i * a j := by
      simpa using (Fintype.sum_mul_sum (fun i : ι => a i) (fun j : ι => a j))
    _ = ∑ i : ι, (a i * a i + ∑ j ∈ (Finset.univ.erase i), a i * a j) := by
      apply Finset.sum_congr rfl
      intro i _hi
      have h :=
        Finset.add_sum_erase
          (s := (Finset.univ : Finset ι))
          (f := fun j : ι => a i * a j)
          (a := i)
          (by simp)
      simpa using h.symm
    _ = (∑ i : ι, a i * a i) + ∑ i : ι, ∑ j ∈ (Finset.univ.erase i), a i * a j := by
      rw [Finset.sum_add_distrib]
    _ = (∑ i : ι, a i ^ 2) + ∑ i : ι, ∑ j ∈ (Finset.univ.erase i), a i * a j := by
      congr 1
      apply Finset.sum_congr rfl
      intro i _hi
      ring

/-- Clustered variance is the scaled sum of all within-cluster pairwise products. -/
theorem clusterVar_eq_pairwiseClusterVar
    (S : ℝ) (ψ : ∀ g : G, I g → ℝ) :
    clusterVar S ψ = pairwiseClusterVar S ψ := by
  classical
  unfold clusterVar pairwiseClusterVar scoreCluster
  congr 1
  apply Finset.sum_congr rfl
  intro g _hg
  calc
    (∑ i : I g, ψ g i) ^ 2 = (∑ i : I g, ψ g i) * ∑ j : I g, ψ g j := by
      ring
    _ = ∑ i : I g, ∑ j : I g, ψ g i * ψ g j := by
      simpa using (Fintype.sum_mul_sum (fun i : I g => ψ g i) (fun j : I g => ψ g j))

/--
The Liang-Zeger clustered variance equals the robust variance plus within-cluster
score cross-products.
-/
theorem clusterVar_eq_robustVar_add_crossProductVar
    (S : ℝ) (ψ : ∀ g : G, I g → ℝ) :
    clusterVar S ψ = robustVar S ψ + crossProductVar S ψ := by
  classical
  unfold clusterVar robustVar crossProductVar scoreCluster
  calc
    scale S * ∑ g : G, (∑ i : I g, ψ g i) ^ 2
        = scale S * ∑ g : G,
            ((∑ i : I g, ψ g i ^ 2) +
              ∑ i : I g, ∑ j ∈ (Finset.univ.erase i), ψ g i * ψ g j) := by
            congr 1
            apply Finset.sum_congr rfl
            intro g _hg
            exact sum_sq_eq_diag_add_offdiag (fun i : I g => ψ g i)
    _ = scale S *
          ((∑ g : G, ∑ i : I g, ψ g i ^ 2) +
            ∑ g : G, ∑ i : I g, ∑ j ∈ (Finset.univ.erase i), ψ g i * ψ g j) := by
            rw [Finset.sum_add_distrib]
    _ = scale S * (∑ g : G, ∑ i : I g, ψ g i ^ 2) +
          scale S * (∑ g : G, ∑ i : I g, ∑ j ∈ (Finset.univ.erase i), ψ g i * ψ g j) := by
            ring

end ClusterLevel

/-! ## 3. Binary fixed-effect reduction -/

namespace BinaryReduction

/-- Mean of a finite vector indexed by Fin n. -/
def meanFin {n : ℕ} (w : Fin n → ℝ) : ℝ :=
  (∑ i : Fin n, w i) / (n : ℝ)

/-- Residualized within-cluster second moment after demeaning. -/
def omegaResidualFin {n : ℕ} (w : Fin n → ℝ) : ℝ :=
  (∑ i : Fin n, (w i - meanFin w) ^ 2) / (n : ℝ)

/-- A helper expansion for centered finite sums. -/
lemma centered_sum_sq_expansion
    {n : ℕ} (w : Fin n → ℝ) :
    ∑ i : Fin n, (w i - meanFin w) ^ 2 =
      (∑ i : Fin n, w i ^ 2)
        - 2 * meanFin w * (∑ i : Fin n, w i)
        + (n : ℝ) * meanFin w ^ 2 := by
  classical
  unfold meanFin
  simp_rw [sub_sq]
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib]
  have hmul :
      (∑ i : Fin n, 2 * w i * ((∑ i : Fin n, w i) / (n : ℝ))) =
        2 * ((∑ i : Fin n, w i) / (n : ℝ)) * (∑ i : Fin n, w i) := by
    rw [← Finset.sum_mul]
    rw [← Finset.mul_sum]
    ring
  have hconst :
      (∑ _i : Fin n, ((∑ i : Fin n, w i) / (n : ℝ)) ^ 2) =
        (n : ℝ) * (((∑ i : Fin n, w i) / (n : ℝ)) ^ 2) := by
    simp [Fintype.card_fin]
  rw [hmul, hconst]

/--
Binary reduction: if each w_i is 0 or 1, then the within-cluster second moment
of the demeaned treatment is Wbar * (1 - Wbar).
-/
theorem omegaResidualFin_eq_mean_mul_one_sub_mean
    {n : ℕ} [NeZero n] (w : Fin n → ℝ)
    (hw : ∀ i : Fin n, w i ^ 2 = w i) :
    omegaResidualFin w = meanFin w * (1 - meanFin w) := by
  classical
  have hn_nat : n ≠ 0 := NeZero.ne n
  have hn : (n : ℝ) ≠ 0 := by
    exact_mod_cast hn_nat
  have hsq : (∑ i : Fin n, w i ^ 2) = ∑ i : Fin n, w i := by
    apply Finset.sum_congr rfl
    intro i _hi
    exact hw i
  unfold omegaResidualFin
  rw [centered_sum_sq_expansion, hsq]
  unfold meanFin
  field_simp [hn]
  ring

/-- Boolean-valued treatment version, where the binary identity is automatic. -/
theorem omegaResidualFin_bool_eq_mean_mul_one_sub_mean
    {n : ℕ} [NeZero n] (W : Fin n → Bool) :
    omegaResidualFin (fun i : Fin n => if W i then (1 : ℝ) else 0) =
      meanFin (fun i : Fin n => if W i then (1 : ℝ) else 0) *
        (1 - meanFin (fun i : Fin n => if W i then (1 : ℝ) else 0)) := by
  classical
  apply omegaResidualFin_eq_mean_mul_one_sub_mean
  intro i
  cases W i <;> simp

end BinaryReduction

/-! ## 4. Location-scale higher-moment calculation -/

namespace LocationScale

/-- Finite average over an index type. -/
def avg {ι : Type _} [Fintype ι] (x : ι → ℝ) : ℝ :=
  (∑ i : ι, x i) / (Fintype.card ι : ℝ)

/-- Second moment scales by sigma^2 under D_i = sigma * epsilon_i. -/
theorem secondMoment_scale
    {ι : Type _} [Fintype ι] (σ : ℝ) (ε : ι → ℝ)
    (hvar : avg (fun i : ι => ε i ^ 2) = 1) :
    avg (fun i : ι => (σ * ε i) ^ 2) = σ ^ 2 := by
  classical
  unfold avg at hvar ⊢
  calc
    (∑ i : ι, (σ * ε i) ^ 2) / (Fintype.card ι : ℝ)
        = (∑ i : ι, σ ^ 2 * ε i ^ 2) / (Fintype.card ι : ℝ) := by
            congr 1
            apply Finset.sum_congr rfl
            intro i _hi
            ring
    _ = (σ ^ 2 * ∑ i : ι, ε i ^ 2) / (Fintype.card ι : ℝ) := by
            rw [Finset.mul_sum]
    _ = σ ^ 2 * ((∑ i : ι, ε i ^ 2) / (Fintype.card ι : ℝ)) := by
            ring
    _ = σ ^ 2 := by
            rw [hvar]
            ring

/-- Fourth moment scales by sigma^4 under D_i = sigma * epsilon_i. -/
theorem fourthMoment_scale
    {ι : Type _} [Fintype ι] (σ κ : ℝ) (ε : ι → ℝ)
    (hkurt : avg (fun i : ι => ε i ^ 4) = κ) :
    avg (fun i : ι => (σ * ε i) ^ 4) = σ ^ 4 * κ := by
  classical
  unfold avg at hkurt ⊢
  calc
    (∑ i : ι, (σ * ε i) ^ 4) / (Fintype.card ι : ℝ)
        = (∑ i : ι, σ ^ 4 * ε i ^ 4) / (Fintype.card ι : ℝ) := by
            congr 1
            apply Finset.sum_congr rfl
            intro i _hi
            ring
    _ = (σ ^ 4 * ∑ i : ι, ε i ^ 4) / (Fintype.card ι : ℝ) := by
            rw [Finset.mul_sum]
    _ = σ ^ 4 * ((∑ i : ι, ε i ^ 4) / (Fintype.card ι : ℝ)) := by
            ring
    _ = σ ^ 4 * κ := by
            rw [hkurt]

/-- The paper's location-scale conclusion: fourth moments scale proportional to the squared variance. -/
theorem fourthMoment_eq_kappa_times_secondMoment_sq
    {ι : Type _} [Fintype ι] (σ κ : ℝ) (ε : ι → ℝ)
    (hvar : avg (fun i : ι => ε i ^ 2) = 1)
    (hkurt : avg (fun i : ι => ε i ^ 4) = κ) :
    avg (fun i : ι => (σ * ε i) ^ 4) =
      (avg (fun i : ι => (σ * ε i) ^ 2)) ^ 2 * κ := by
  rw [fourthMoment_scale σ κ ε hkurt, secondMoment_scale σ ε hvar]
  ring

end LocationScale

/-! ## 5. Design-covariance sandwich for continuous treatments -/

namespace DesignCovariance

variable {α : Type _} [Fintype α]

/--
The quadratic form u' Sigma u, written as finite sums over an arbitrary
observation index.  `Sigma` is the residualized treatment assignment covariance
kernel.
-/
def quad (u : α → ℝ) (Sigma : α → α → ℝ) : ℝ :=
  ∑ i : α, ∑ j : α, u i * Sigma i j * u j

/-- The design variance of the linearized estimator S^{-1} Dtilde'u. -/
def designVariance (S : ℝ) (u : α → ℝ) (Sigma : α → α → ℝ) : ℝ :=
  ClusterLevel.scale S * quad u Sigma

/-- The feasible design-covariance CCV estimator using residual and covariance estimates. -/
def dcCCV (S : ℝ) (uhat : α → ℝ) (SigmaHat : α → α → ℝ) : ℝ :=
  designVariance S uhat SigmaHat

/-- The population/design standard error associated with a covariance-kernel variance. -/
def designSE (S : ℝ) (u : α → ℝ) (Sigma : α → α → ℝ) : ℝ :=
  Real.sqrt (designVariance S u Sigma)

/-- The feasible standard error from the design-covariance CCV estimator. -/
def dcSE (S : ℝ) (uhat : α → ℝ) (SigmaHat : α → α → ℝ) : ℝ :=
  Real.sqrt (dcCCV S uhat SigmaHat)

/-- Positive semidefiniteness of a finite covariance kernel. -/
def IsPSD (Sigma : α → α → ℝ) : Prop :=
  ∀ u : α → ℝ, 0 ≤ quad u Sigma

/-- A covariance-kernel design variance is nonnegative when the kernel is PSD. -/
theorem designVariance_nonneg
    (S : ℝ) (u : α → ℝ) (Sigma : α → α → ℝ)
    (hPSD : IsPSD Sigma) :
    0 ≤ designVariance S u Sigma := by
  unfold designVariance ClusterLevel.scale
  exact mul_nonneg (inv_nonneg.mpr (sq_nonneg S)) (hPSD u)

/-- The feasible design-covariance estimator is nonnegative for a PSD covariance estimate. -/
theorem dcCCV_nonneg
    (S : ℝ) (uhat : α → ℝ) (SigmaHat : α → α → ℝ)
    (hPSD : IsPSD SigmaHat) :
    0 ≤ dcCCV S uhat SigmaHat := by
  exact designVariance_nonneg S uhat SigmaHat hPSD

/--
Exact finite-sample validity: if the residual estimate and covariance-kernel
estimate are exact, the design-covariance CCV estimator equals the design
variance by construction.
-/
theorem dcCCV_eq_designVariance_of_exact
    (S : ℝ) (u uhat : α → ℝ) (Sigma SigmaHat : α → α → ℝ)
    (hu : uhat = u) (hSigma : SigmaHat = Sigma) :
    dcCCV S uhat SigmaHat = designVariance S u Sigma := by
  rw [dcCCV, hu, hSigma]

/-- Exact standard-error validity follows from exact variance validity. -/
theorem dcSE_eq_designSE_of_exact
    (S : ℝ) (u uhat : α → ℝ) (Sigma SigmaHat : α → α → ℝ)
    (hu : uhat = u) (hSigma : SigmaHat = Sigma) :
    dcSE S uhat SigmaHat = designSE S u Sigma := by
  unfold dcSE designSE
  rw [dcCCV_eq_designVariance_of_exact S u uhat Sigma SigmaHat hu hSigma]

/--
The theorem is stated for an arbitrary real-valued residualized treatment.  No
binary restriction is present; all continuous-treatment design information enters
through the covariance kernel.
-/
theorem arbitrary_continuous_treatment_exact_valid
    (Dtilde u uhat : α → ℝ) (Sigma SigmaHat : α → α → ℝ)
    (hu : uhat = u) (hSigma : SigmaHat = Sigma) :
    dcCCV (ObservationLevel.denom Dtilde) uhat SigmaHat =
      designVariance (ObservationLevel.denom Dtilde) u Sigma := by
  exact dcCCV_eq_designVariance_of_exact
    (S := ObservationLevel.denom Dtilde)
    (u := u) (uhat := uhat) (Sigma := Sigma) (SigmaHat := SigmaHat)
    hu hSigma

/-- The rank-one unrestricted covariance kernel generated by a score regressor. -/
def outerKernel (x : α → ℝ) : α → α → ℝ :=
  fun i j : α => x i * x j

/-- The unrestricted rank-one sandwich equals the squared summed score. -/
theorem quad_outerKernel (x u : α → ℝ) :
    quad u (outerKernel x) = (∑ i : α, x i * u i) ^ 2 := by
  classical
  unfold quad outerKernel
  calc
    (∑ i : α, ∑ j : α, u i * (x i * x j) * u j)
        = ∑ i : α, ∑ j : α, (x i * u i) * (x j * u j) := by
            apply Finset.sum_congr rfl
            intro i _hi
            apply Finset.sum_congr rfl
            intro j _hj
            ring
    _ = (∑ i : α, x i * u i) * ∑ j : α, x j * u j := by
            simpa using
              (Fintype.sum_mul_sum
                (fun i : α => x i * u i)
                (fun j : α => x j * u j)).symm
    _ = (∑ i : α, x i * u i) ^ 2 := by
            ring

/--
If the covariance kernel is the unrestricted rank-one outer product, the
design-covariance estimator reduces to the fully clustered/all-pairs score
variance for the finite observation index.
-/
theorem designVariance_outerKernel_eq_score_sq
    (S : ℝ) (Dtilde u : α → ℝ) :
    designVariance S u (outerKernel Dtilde) =
      ClusterLevel.scale S * (∑ i : α, Dtilde i * u i) ^ 2 := by
  unfold designVariance
  rw [quad_outerKernel]

/-- Squared-error loss for comparing variance estimators to a design target. -/
def sqLoss (candidate target : ℝ) : ℝ :=
  (candidate - target) ^ 2

/--
Oracle optimality: an exact design-variance estimator has zero squared loss and
therefore weakly dominates every other real-valued candidate for that target.
-/
theorem exact_designVariance_sqLoss_optimal
    (candidate target other : ℝ) (h : candidate = target) :
    sqLoss candidate target ≤ sqLoss other target := by
  unfold sqLoss
  rw [h]
  nlinarith [sq_nonneg (other - target)]

/--
Direct optimality statement for the design-covariance CCV estimator: under exact
residual and covariance inputs, its squared loss against the design variance is
zero, so no other real-valued variance candidate can have smaller squared loss.
-/
theorem dcCCV_sqLoss_optimal_of_exact
    (S : ℝ) (u uhat : α → ℝ) (Sigma SigmaHat : α → α → ℝ)
    (other : ℝ) (hu : uhat = u) (hSigma : SigmaHat = Sigma) :
    sqLoss (dcCCV S uhat SigmaHat) (designVariance S u Sigma) ≤
      sqLoss other (designVariance S u Sigma) := by
  exact exact_designVariance_sqLoss_optimal
    (candidate := dcCCV S uhat SigmaHat)
    (target := designVariance S u Sigma)
    (other := other)
    (h := dcCCV_eq_designVariance_of_exact S u uhat Sigma SigmaHat hu hSigma)

end DesignCovariance

/-! ## 6. Finite design probability model -/

namespace FiniteDesign

variable {Ω α : Type _} [Fintype Ω] [Fintype α]

/-- Finite-state expectation under probability masses `p`. -/
def expectation (p : Ω → ℝ) (X : Ω → ℝ) : ℝ :=
  ∑ ω : Ω, p ω * X ω

/-- Finite-state variance. -/
def variance (p : Ω → ℝ) (X : Ω → ℝ) : ℝ :=
  expectation p (fun ω : Ω => (X ω - expectation p X) ^ 2)

/-- A finite probability mass function. -/
structure IsProbabilityMass (p : Ω → ℝ) : Prop where
  nonneg : ∀ ω : Ω, 0 ≤ p ω
  total_mass : (∑ ω : Ω, p ω) = 1

/-- Linearized treatment score before scaling by the OLS denominator. -/
def linearScore (D : Ω → α → ℝ) (u : α → ℝ) (ω : Ω) : ℝ :=
  ∑ i : α, D ω i * u i

/-- Linearized estimator error with fixed denominator `S`. -/
def betaErrorLinear (S : ℝ) (D : Ω → α → ℝ) (u : α → ℝ) (ω : Ω) : ℝ :=
  S⁻¹ * linearScore D u ω

/-- The actual OLS coefficient error in each finite design state. -/
def betaHatError (D Y : Ω → α → ℝ) (β : ℝ) (ω : Ω) : ℝ :=
  ObservationLevel.betaHat (D ω) (Y ω) - β

/--
The score regressor that exactly represents the random-denominator OLS error:
`D_i / (D'D)` in each design state.
-/
def normalizedTreatment (D : Ω → α → ℝ) (ω : Ω) (i : α) : ℝ :=
  (ObservationLevel.denom (D ω))⁻¹ * D ω i

/-- Centered treatment coordinate under the finite design law. -/
def centeredD (p : Ω → ℝ) (D : Ω → α → ℝ) (i : α) (ω : Ω) : ℝ :=
  D ω i - expectation p (fun ω' : Ω => D ω' i)

/-- The covariance kernel of the residualized treatment assignment. -/
def covKernel (p : Ω → ℝ) (D : Ω → α → ℝ) (i j : α) : ℝ :=
  expectation p (fun ω : Ω => centeredD p D i ω * centeredD p D j ω)

lemma expectation_linearScore
    (p : Ω → ℝ) (D : Ω → α → ℝ) (u : α → ℝ) :
    expectation p (linearScore D u) =
      ∑ i : α, expectation p (fun ω : Ω => D ω i) * u i := by
  classical
  unfold expectation linearScore
  calc
    (∑ ω : Ω, p ω * ∑ i : α, D ω i * u i)
        = ∑ ω : Ω, ∑ i : α, p ω * (D ω i * u i) := by
            apply Finset.sum_congr rfl
            intro ω _hω
            rw [Finset.mul_sum]
    _ = ∑ i : α, ∑ ω : Ω, p ω * (D ω i * u i) := by
            rw [Finset.sum_comm]
    _ = ∑ i : α, (∑ ω : Ω, p ω * D ω i) * u i := by
            apply Finset.sum_congr rfl
            intro i _hi
            simp_rw [← mul_assoc]
            rw [Finset.sum_mul]

lemma linearScore_sub_expectation
    (p : Ω → ℝ) (D : Ω → α → ℝ) (u : α → ℝ) (ω : Ω) :
    linearScore D u ω - expectation p (linearScore D u) =
      ∑ i : α, centeredD p D i ω * u i := by
  classical
  rw [expectation_linearScore]
  unfold linearScore centeredD
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _hi
  ring

lemma sum_sq_as_double_sum
    (a : α → ℝ) :
    (∑ i : α, a i) ^ 2 = ∑ i : α, ∑ j : α, a i * a j := by
  classical
  calc
    (∑ i : α, a i) ^ 2 = (∑ i : α, a i) * ∑ j : α, a j := by
      ring
    _ = ∑ i : α, ∑ j : α, a i * a j := by
      simpa using (Fintype.sum_mul_sum (fun i : α => a i) (fun j : α => a j))

/--
Statistical variance identity for a finite design probability model: the
variance of the linear score equals the quadratic form using the treatment
assignment covariance kernel.
-/
theorem variance_linearScore_eq_quad_covKernel
    (p : Ω → ℝ) (D : Ω → α → ℝ) (u : α → ℝ) :
    variance p (linearScore D u) =
      DesignCovariance.quad u (covKernel p D) := by
  classical
  unfold variance
  calc
    expectation p (fun ω : Ω => (linearScore D u ω - expectation p (linearScore D u)) ^ 2)
        = ∑ ω : Ω, p ω *
            (∑ i : α, centeredD p D i ω * u i) ^ 2 := by
            unfold expectation
            apply Finset.sum_congr rfl
            intro ω _hω
            change p ω * ((linearScore D u ω - expectation p (linearScore D u)) ^ 2) =
              p ω * (∑ i : α, centeredD p D i ω * u i) ^ 2
            rw [linearScore_sub_expectation]
    _ = ∑ ω : Ω, p ω *
            (∑ i : α, ∑ j : α,
              (centeredD p D i ω * u i) * (centeredD p D j ω * u j)) := by
            apply Finset.sum_congr rfl
            intro ω _hω
            rw [sum_sq_as_double_sum]
    _ = ∑ ω : Ω, ∑ i : α, ∑ j : α,
            p ω * ((centeredD p D i ω * u i) * (centeredD p D j ω * u j)) := by
            apply Finset.sum_congr rfl
            intro ω _hω
            rw [Finset.mul_sum]
            apply Finset.sum_congr rfl
            intro i _hi
            rw [Finset.mul_sum]
    _ = ∑ i : α, ∑ j : α, ∑ ω : Ω,
            p ω * ((centeredD p D i ω * u i) * (centeredD p D j ω * u j)) := by
            rw [Finset.sum_comm]
            apply Finset.sum_congr rfl
            intro i _hi
            rw [Finset.sum_comm]
    _ = ∑ i : α, ∑ j : α,
            u i * (∑ ω : Ω,
              p ω * (centeredD p D i ω * centeredD p D j ω)) * u j := by
            apply Finset.sum_congr rfl
            intro i _hi
            apply Finset.sum_congr rfl
            intro j _hj
            rw [Finset.mul_sum]
            rw [Finset.sum_mul]
            apply Finset.sum_congr rfl
            intro ω _hω
            ring
    _ = DesignCovariance.quad u (covKernel p D) := by
            unfold DesignCovariance.quad covKernel expectation
            simp

/-- The finite covariance kernel is symmetric. -/
theorem covKernel_symm
    (p : Ω → ℝ) (D : Ω → α → ℝ) (i j : α) :
    covKernel p D i j = covKernel p D j i := by
  classical
  unfold covKernel expectation
  apply Finset.sum_congr rfl
  intro ω _hω
  ring

/-- Finite-state variance is nonnegative under nonnegative probability masses. -/
theorem variance_nonneg
    (p : Ω → ℝ) (X : Ω → ℝ)
    (hp_nonneg : ∀ ω : Ω, 0 ≤ p ω) :
    0 ≤ variance p X := by
  classical
  unfold variance expectation
  exact Finset.sum_nonneg
    (fun ω _hω => mul_nonneg (hp_nonneg ω) (sq_nonneg (X ω - ∑ ω' : Ω, p ω' * X ω')))

/-- The true finite-design covariance kernel is positive semidefinite. -/
theorem covKernel_isPSD
    (p : Ω → ℝ) (D : Ω → α → ℝ)
    (hp : IsProbabilityMass p) :
    DesignCovariance.IsPSD (covKernel p D) := by
  intro u
  rw [← variance_linearScore_eq_quad_covKernel]
  exact variance_nonneg p (linearScore D u) hp.nonneg

/-- The design variance induced by the true finite-design covariance kernel is nonnegative. -/
theorem designVariance_covKernel_nonneg
    (S : ℝ) (p : Ω → ℝ) (D : Ω → α → ℝ) (u : α → ℝ)
    (hp : IsProbabilityMass p) :
    0 ≤ DesignCovariance.designVariance S u (covKernel p D) :=
  DesignCovariance.designVariance_nonneg S u (covKernel p D) (covKernel_isPSD p D hp)

lemma expectation_const_mul
    (p : Ω → ℝ) (c : ℝ) (X : Ω → ℝ) :
    expectation p (fun ω : Ω => c * X ω) = c * expectation p X := by
  classical
  unfold expectation
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro ω _hω
  ring

lemma variance_const_mul
    (p : Ω → ℝ) (c : ℝ) (X : Ω → ℝ) :
    variance p (fun ω : Ω => c * X ω) = c ^ 2 * variance p X := by
  classical
  unfold variance
  rw [expectation_const_mul]
  unfold expectation
  calc
    (∑ ω : Ω, p ω * (c * X ω - c * (∑ ω : Ω, p ω * X ω)) ^ 2)
        = ∑ ω : Ω, p ω * (c ^ 2 * (X ω - ∑ ω : Ω, p ω * X ω) ^ 2) := by
            apply Finset.sum_congr rfl
            intro ω _hω
            ring
    _ = c ^ 2 * ∑ ω : Ω, p ω * (X ω - ∑ ω : Ω, p ω * X ω) ^ 2 := by
            rw [Finset.mul_sum]
            apply Finset.sum_congr rfl
            intro ω _hω
            ring

lemma inv_sq_eq_scale (S : ℝ) :
    S⁻¹ ^ 2 = ClusterLevel.scale S := by
  unfold ClusterLevel.scale
  ring

/--
Correct design variance for the linearized estimator under a finite design law.
This is an actual finite probability-model theorem, not an asymptotic
placeholder.
-/
theorem variance_betaErrorLinear_eq_designVariance
    (S : ℝ) (p : Ω → ℝ) (D : Ω → α → ℝ) (u : α → ℝ) :
    variance p (betaErrorLinear S D u) =
      DesignCovariance.designVariance S u (covKernel p D) := by
  classical
  unfold betaErrorLinear
  rw [variance_const_mul]
  rw [variance_linearScore_eq_quad_covKernel]
  unfold DesignCovariance.designVariance
  rw [inv_sq_eq_scale]

/--
If the OLS denominator is fixed across design states, the actual OLS error is the
linearized score with denominator `S`.
-/
theorem betaHatError_eq_betaErrorLinear_of_fixed_denom
    (S : ℝ) (D Y : Ω → α → ℝ) (β : ℝ) (u : α → ℝ)
    (hY : ∀ (ω : Ω) (i : α), Y ω i = D ω i * β + u i)
    (hS : ∀ ω : Ω, ObservationLevel.denom (D ω) = S)
    (hS_ne : S ≠ 0) :
    betaHatError D Y β = betaErrorLinear S D u := by
  funext ω
  unfold betaHatError betaErrorLinear linearScore
  have hden_ne : ObservationLevel.denom (D ω) ≠ 0 := by
    rw [hS ω]
    exact hS_ne
  have hpoint :
      ObservationLevel.betaHat (D ω) (Y ω) - β =
        ObservationLevel.scoreSum (D ω) u / ObservationLevel.denom (D ω) :=
    ObservationLevel.betaHat_error
      (Dtilde := D ω) (Ytilde := Y ω) (utilde := u) (β := β)
      (hY := hY ω) (hS := hden_ne)
  rw [hpoint, hS ω]
  unfold ObservationLevel.scoreSum
  rw [div_eq_inv_mul]

/--
Finite-sample variance of the actual OLS error under a fixed denominator.
This is the raw-treatment covariance-kernel version of the design sandwich.
-/
theorem variance_betaHatError_eq_designVariance_of_fixed_denom
    (S : ℝ) (p : Ω → ℝ) (D Y : Ω → α → ℝ) (β : ℝ) (u : α → ℝ)
    (hY : ∀ (ω : Ω) (i : α), Y ω i = D ω i * β + u i)
    (hS : ∀ ω : Ω, ObservationLevel.denom (D ω) = S)
    (hS_ne : S ≠ 0) :
    variance p (betaHatError D Y β) =
      DesignCovariance.designVariance S u (covKernel p D) := by
  rw [betaHatError_eq_betaErrorLinear_of_fixed_denom
    (S := S) (D := D) (Y := Y) (β := β) (u := u) hY hS hS_ne]
  exact variance_betaErrorLinear_eq_designVariance S p D u

/--
Without a fixed denominator, the actual OLS error is still exactly linear in `u`
after absorbing the random denominator into the treatment regressor.
-/
theorem betaHatError_eq_linearScore_normalizedTreatment
    (D Y : Ω → α → ℝ) (β : ℝ) (u : α → ℝ)
    (hY : ∀ (ω : Ω) (i : α), Y ω i = D ω i * β + u i)
    (hDenom : ∀ ω : Ω, ObservationLevel.denom (D ω) ≠ 0) :
    betaHatError D Y β = linearScore (normalizedTreatment D) u := by
  funext ω
  unfold betaHatError
  have hpoint :
      ObservationLevel.betaHat (D ω) (Y ω) - β =
        ObservationLevel.scoreSum (D ω) u / ObservationLevel.denom (D ω) :=
    ObservationLevel.betaHat_error
      (Dtilde := D ω) (Ytilde := Y ω) (utilde := u) (β := β)
      (hY := hY ω) (hS := hDenom ω)
  rw [hpoint]
  unfold ObservationLevel.scoreSum linearScore normalizedTreatment
  calc
    (∑ a : α, D ω a * u a) / ObservationLevel.denom (D ω)
        = (ObservationLevel.denom (D ω))⁻¹ * (∑ a : α, D ω a * u a) := by
            rw [div_eq_inv_mul]
    _ = ∑ a : α, (ObservationLevel.denom (D ω))⁻¹ * (D ω a * u a) := by
            rw [Finset.mul_sum]
    _ = ∑ a : α, ((ObservationLevel.denom (D ω))⁻¹ * D ω a) * u a := by
            apply Finset.sum_congr rfl
            intro a _ha
            ring

/--
Exact finite-sample variance of the actual OLS error with a random denominator.
The covariance kernel must be taken over `D / (D'D)`, so the outer sandwich scale
is `1`.
-/
theorem variance_betaHatError_eq_designVariance_normalizedTreatment
    (p : Ω → ℝ) (D Y : Ω → α → ℝ) (β : ℝ) (u : α → ℝ)
    (hY : ∀ (ω : Ω) (i : α), Y ω i = D ω i * β + u i)
    (hDenom : ∀ ω : Ω, ObservationLevel.denom (D ω) ≠ 0) :
    variance p (betaHatError D Y β) =
      DesignCovariance.designVariance 1 u (covKernel p (normalizedTreatment D)) := by
  rw [betaHatError_eq_linearScore_normalizedTreatment
    (D := D) (Y := Y) (β := β) (u := u) hY hDenom]
  rw [variance_linearScore_eq_quad_covKernel
    (p := p) (D := normalizedTreatment D) (u := u)]
  unfold DesignCovariance.designVariance ClusterLevel.scale
  ring

/--
With exact residuals and exact covariance kernel from the finite design law, the
design-covariance CCV estimator equals the true variance of the linearized
estimator.
-/
theorem dcCCV_eq_finite_design_variance_of_exact
    (S : ℝ) (p : Ω → ℝ) (D : Ω → α → ℝ)
    (u uhat : α → ℝ) (SigmaHat : α → α → ℝ)
    (hu : uhat = u) (hSigma : SigmaHat = covKernel p D) :
    DesignCovariance.dcCCV S uhat SigmaHat =
      variance p (betaErrorLinear S D u) := by
  rw [variance_betaErrorLinear_eq_designVariance]
  exact (DesignCovariance.dcCCV_eq_designVariance_of_exact
    (S := S) (u := u) (uhat := uhat)
    (Sigma := covKernel p D) (SigmaHat := SigmaHat) hu hSigma)

/--
Finite-sample exactness for the actual OLS estimator when the denominator is
fixed across assignment states and the covariance estimate is exact.
-/
theorem dcCCV_eq_finite_design_variance_of_ols_fixed_denom
    (S : ℝ) (p : Ω → ℝ) (D Y : Ω → α → ℝ) (β : ℝ)
    (u uhat : α → ℝ) (SigmaHat : α → α → ℝ)
    (hY : ∀ (ω : Ω) (i : α), Y ω i = D ω i * β + u i)
    (hS : ∀ ω : Ω, ObservationLevel.denom (D ω) = S)
    (hS_ne : S ≠ 0)
    (hu : uhat = u) (hSigma : SigmaHat = covKernel p D) :
    DesignCovariance.dcCCV S uhat SigmaHat =
      variance p (betaHatError D Y β) := by
  rw [variance_betaHatError_eq_designVariance_of_fixed_denom
    (S := S) (p := p) (D := D) (Y := Y) (β := β) (u := u) hY hS hS_ne]
  exact (DesignCovariance.dcCCV_eq_designVariance_of_exact
    (S := S) (u := u) (uhat := uhat)
    (Sigma := covKernel p D) (SigmaHat := SigmaHat) hu hSigma)

/--
Finite-sample exactness for the actual OLS estimator with a random denominator.
Here `SigmaHat` estimates the covariance kernel of `D / (D'D)`, not raw `D`.
-/
theorem dcCCV_eq_finite_design_variance_of_ols
    (p : Ω → ℝ) (D Y : Ω → α → ℝ) (β : ℝ)
    (u uhat : α → ℝ) (SigmaHat : α → α → ℝ)
    (hY : ∀ (ω : Ω) (i : α), Y ω i = D ω i * β + u i)
    (hDenom : ∀ ω : Ω, ObservationLevel.denom (D ω) ≠ 0)
    (hu : uhat = u) (hSigma : SigmaHat = covKernel p (normalizedTreatment D)) :
    DesignCovariance.dcCCV 1 uhat SigmaHat =
      variance p (betaHatError D Y β) := by
  rw [variance_betaHatError_eq_designVariance_normalizedTreatment
    (p := p) (D := D) (Y := Y) (β := β) (u := u) hY hDenom]
  exact (DesignCovariance.dcCCV_eq_designVariance_of_exact
    (S := 1) (u := u) (uhat := uhat)
    (Sigma := covKernel p (normalizedTreatment D)) (SigmaHat := SigmaHat) hu hSigma)

/-- Exact standard-error equality for the fixed-denominator finite-sample theorem. -/
theorem dcSE_eq_sqrt_finite_design_variance_of_ols_fixed_denom
    (S : ℝ) (p : Ω → ℝ) (D Y : Ω → α → ℝ) (β : ℝ)
    (u uhat : α → ℝ) (SigmaHat : α → α → ℝ)
    (hY : ∀ (ω : Ω) (i : α), Y ω i = D ω i * β + u i)
    (hS : ∀ ω : Ω, ObservationLevel.denom (D ω) = S)
    (hS_ne : S ≠ 0)
    (hu : uhat = u) (hSigma : SigmaHat = covKernel p D) :
    DesignCovariance.dcSE S uhat SigmaHat =
      Real.sqrt (variance p (betaHatError D Y β)) := by
  unfold DesignCovariance.dcSE
  rw [dcCCV_eq_finite_design_variance_of_ols_fixed_denom
    (S := S) (p := p) (D := D) (Y := Y) (β := β)
    (u := u) (uhat := uhat) (SigmaHat := SigmaHat)
    hY hS hS_ne hu hSigma]

/-- Exact standard-error equality for the random-denominator finite-sample theorem. -/
theorem dcSE_eq_sqrt_finite_design_variance_of_ols
    (p : Ω → ℝ) (D Y : Ω → α → ℝ) (β : ℝ)
    (u uhat : α → ℝ) (SigmaHat : α → α → ℝ)
    (hY : ∀ (ω : Ω) (i : α), Y ω i = D ω i * β + u i)
    (hDenom : ∀ ω : Ω, ObservationLevel.denom (D ω) ≠ 0)
    (hu : uhat = u) (hSigma : SigmaHat = covKernel p (normalizedTreatment D)) :
    DesignCovariance.dcSE 1 uhat SigmaHat =
      Real.sqrt (variance p (betaHatError D Y β)) := by
  unfold DesignCovariance.dcSE
  rw [dcCCV_eq_finite_design_variance_of_ols
    (p := p) (D := D) (Y := Y) (β := β)
    (u := u) (uhat := uhat) (SigmaHat := SigmaHat)
    hY hDenom hu hSigma]

end FiniteDesign

/-! ## 7. Algebraic extensions -/

namespace HeterogeneousTreatment

variable {α : Type _} [Fintype α]

/--
Variance-weighted target coefficient for heterogeneous treatment effects under
the scalar residualized-treatment regression.
-/
def weightedBeta (Dtilde beta : α → ℝ) : ℝ :=
  (∑ a : α, Dtilde a ^ 2 * beta a) / ObservationLevel.denom Dtilde

/--
The extra finite-sample term generated by treatment-effect heterogeneity when
expanding around an arbitrary scalar target `betaBar`.
-/
def heterogeneityScore (Dtilde beta : α → ℝ) (betaBar : ℝ) : ℝ :=
  ∑ a : α, Dtilde a ^ 2 * (beta a - betaBar)

/--
Finite-sample FWL expansion with heterogeneous treatment effects.  When
`Y_i = D_i * beta_i + u_i`, the scalar OLS coefficient centered at an arbitrary
target `betaBar` equals the usual residual score plus a heterogeneity-weighting
term.
-/
theorem betaHat_error_heterogeneous
    (Dtilde Ytilde utilde beta : α → ℝ) (betaBar : ℝ)
    (hY : ∀ a : α, Ytilde a = Dtilde a * beta a + utilde a)
    (hS : ObservationLevel.denom Dtilde ≠ 0) :
    ObservationLevel.betaHat Dtilde Ytilde - betaBar =
      ObservationLevel.scoreSum Dtilde utilde / ObservationLevel.denom Dtilde +
        heterogeneityScore Dtilde beta betaBar / ObservationLevel.denom Dtilde := by
  classical
  unfold ObservationLevel.betaHat ObservationLevel.scoreSum heterogeneityScore
  unfold ObservationLevel.denom at hS ⊢
  have hnum :
      (∑ a : α, Dtilde a * Ytilde a) =
        betaBar * (∑ a : α, Dtilde a ^ 2) +
          ∑ a : α, Dtilde a ^ 2 * (beta a - betaBar) +
          ∑ a : α, Dtilde a * utilde a := by
    calc
      (∑ a : α, Dtilde a * Ytilde a)
          = ∑ a : α, Dtilde a * (Dtilde a * beta a + utilde a) := by
              apply Finset.sum_congr rfl
              intro a _ha
              rw [hY a]
      _ = ∑ a : α, (Dtilde a ^ 2 * beta a + Dtilde a * utilde a) := by
              apply Finset.sum_congr rfl
              intro a _ha
              ring
      _ = (∑ a : α, Dtilde a ^ 2 * beta a) +
            ∑ a : α, Dtilde a * utilde a := by
              rw [Finset.sum_add_distrib]
      _ = (∑ a : α,
            (betaBar * Dtilde a ^ 2 + Dtilde a ^ 2 * (beta a - betaBar))) +
            ∑ a : α, Dtilde a * utilde a := by
              congr 1
              apply Finset.sum_congr rfl
              intro a _ha
              ring
      _ = ((∑ a : α, betaBar * Dtilde a ^ 2) +
            ∑ a : α, Dtilde a ^ 2 * (beta a - betaBar)) +
            ∑ a : α, Dtilde a * utilde a := by
              rw [Finset.sum_add_distrib]
      _ = betaBar * (∑ a : α, Dtilde a ^ 2) +
            ∑ a : α, Dtilde a ^ 2 * (beta a - betaBar) +
            ∑ a : α, Dtilde a * utilde a := by
              rw [← Finset.mul_sum]
  rw [hnum]
  field_simp [hS]
  ring

/--
If the scalar target is the variance-weighted treatment effect, the
heterogeneity score is zero, so heterogeneous-treatment FWL reduces to the usual
residual score expansion.
-/
theorem heterogeneityScore_weightedBeta_eq_zero
    (Dtilde beta : α → ℝ)
    (hS : ObservationLevel.denom Dtilde ≠ 0) :
    heterogeneityScore Dtilde beta (weightedBeta Dtilde beta) = 0 := by
  classical
  unfold ObservationLevel.denom at hS
  unfold heterogeneityScore weightedBeta ObservationLevel.denom
  let A : ℝ := ∑ a : α, Dtilde a ^ 2 * beta a
  let S : ℝ := ∑ a : α, Dtilde a ^ 2
  have hS' : S ≠ 0 := by
    simpa [S] using hS
  change (∑ a : α, Dtilde a ^ 2 * (beta a - A / S)) = 0
  calc
    (∑ a : α, Dtilde a ^ 2 * (beta a - A / S))
        = (∑ a : α, Dtilde a ^ 2 * beta a) -
            ∑ a : α, Dtilde a ^ 2 * (A / S) := by
            simp_rw [mul_sub]
            rw [Finset.sum_sub_distrib]
    _ = A - S * (A / S) := by
            rw [Finset.sum_mul]
    _ = 0 := by
            field_simp [hS']
            ring

theorem betaHat_error_weightedBeta
    (Dtilde Ytilde utilde beta : α → ℝ)
    (hY : ∀ a : α, Ytilde a = Dtilde a * beta a + utilde a)
    (hS : ObservationLevel.denom Dtilde ≠ 0) :
    ObservationLevel.betaHat Dtilde Ytilde - weightedBeta Dtilde beta =
      ObservationLevel.scoreSum Dtilde utilde / ObservationLevel.denom Dtilde := by
  rw [betaHat_error_heterogeneous
    (Dtilde := Dtilde) (Ytilde := Ytilde) (utilde := utilde)
    (beta := beta) (betaBar := weightedBeta Dtilde beta) hY hS]
  rw [heterogeneityScore_weightedBeta_eq_zero Dtilde beta hS]
  ring

end HeterogeneousTreatment

namespace MultiWayCluster

variable {α G T : Type _}
variable [Fintype α] [Fintype G] [Fintype T]
variable [DecidableEq α] [DecidableEq G] [DecidableEq T]

/-- Sum of observation scores inside one cluster label. -/
def scoreBy (cluster : α → G) (ψ : α → ℝ) (g : G) : ℝ :=
  ∑ i ∈ (Finset.univ.filter fun i : α => cluster i = g), ψ i

/-- One-way clustered scalar variance for a flat observation index. -/
def clusterVarBy (S : ℝ) (cluster : α → G) (ψ : α → ℝ) : ℝ :=
  ClusterLevel.scale S * ∑ g : G, scoreBy cluster ψ g ^ 2

/-- Product/intersection cluster label for two clustering dimensions. -/
def intersectionCluster (clusterG : α → G) (clusterT : α → T) : α → G × T :=
  fun i : α => (clusterG i, clusterT i)

/--
Two-way clustered variance estimator as finite inclusion-exclusion over two
cluster dimensions and their intersection cells.
-/
def twoWayClusterVar
    (S : ℝ) (clusterG : α → G) (clusterT : α → T) (ψ : α → ℝ) : ℝ :=
  clusterVarBy S clusterG ψ +
    clusterVarBy S clusterT ψ -
    clusterVarBy S (intersectionCluster clusterG clusterT) ψ

/-- The two-way clustered estimator is exactly the inclusion-exclusion formula. -/
theorem twoWayClusterVar_eq_inclusion_exclusion
    (S : ℝ) (clusterG : α → G) (clusterT : α → T) (ψ : α → ℝ) :
    twoWayClusterVar S clusterG clusterT ψ =
      clusterVarBy S clusterG ψ +
        clusterVarBy S clusterT ψ -
        clusterVarBy S (intersectionCluster clusterG clusterT) ψ := by
  rfl

/-- Intersection-cell scores are just scores by the product cluster label. -/
theorem scoreBy_intersection_eq_product_label
    (clusterG : α → G) (clusterT : α → T) (ψ : α → ℝ) (gt : G × T) :
    scoreBy (intersectionCluster clusterG clusterT) ψ gt =
      ∑ i ∈ (Finset.univ.filter fun i : α => clusterG i = gt.1 ∧ clusterT i = gt.2),
        ψ i := by
  classical
  rcases gt with ⟨g, t⟩
  unfold scoreBy intersectionCluster
  apply Finset.sum_congr
  · ext i
    simp
  · intro i _hi
    rfl

end MultiWayCluster

namespace DyadicClustering

variable {V : Type _} [DecidableEq V]

/-- Two directed dyads share a vertex if any endpoint is common. -/
def sharesVertex (d e : V × V) : Prop :=
  d.1 = e.1 ∨ d.1 = e.2 ∨ d.2 = e.1 ∨ d.2 = e.2

/-- A dyadic covariance kernel is supported only on dyads sharing a vertex. -/
def SupportedOnSharedVertex (Sigma : V × V → V × V → ℝ) : Prop :=
  ∀ d e : V × V, ¬ sharesVertex d e → Sigma d e = 0

/-- Restrict any dyadic covariance kernel to the shared-vertex support. -/
noncomputable def restrictedKernel (Sigma : V × V → V × V → ℝ) : V × V → V × V → ℝ :=
  fun d e => by
    classical
    exact if sharesVertex d e then Sigma d e else 0

theorem restrictedKernel_supported
    (Sigma : V × V → V × V → ℝ) :
    SupportedOnSharedVertex (restrictedKernel Sigma) := by
  intro d e hnot
  by_cases h : sharesVertex d e
  · exact False.elim (hnot h)
  · simp [restrictedKernel, h]

theorem restrictedKernel_eq_of_sharesVertex
    (Sigma : V × V → V × V → ℝ) {d e : V × V}
    (h : sharesVertex d e) :
    restrictedKernel Sigma d e = Sigma d e := by
  simp [restrictedKernel, h]

theorem restrictedKernel_eq_zero_of_not_sharesVertex
    (Sigma : V × V → V × V → ℝ) {d e : V × V}
    (h : ¬ sharesVertex d e) :
    restrictedKernel Sigma d e = 0 := by
  simp [restrictedKernel, h]

end DyadicClustering

namespace GroupLevelTreatment

variable {G : Type _} [Fintype G]
variable {I : G → Type _} [∀ g : G, Fintype (I g)]

/--
Structural group-level treatment condition: within each cluster, the continuous
treatment equals a cluster-level value.
-/
def IsGroupLevelTreatment (D : ∀ g : G, I g → ℝ) (Dbar : G → ℝ) : Prop :=
  ∀ (g : G) (i : I g), D g i = Dbar g

/-- Cluster score when treatment is constant within cluster. -/
theorem scoreCluster_groupLevel_treatment
    (D : ∀ g : G, I g → ℝ) (Dbar : G → ℝ) (u : ∀ g : G, I g → ℝ)
    (hD : IsGroupLevelTreatment D Dbar) (g : G) :
    ClusterLevel.scoreCluster (fun g i => D g i * u g i) g =
      Dbar g * ∑ i : I g, u g i := by
  classical
  unfold ClusterLevel.scoreCluster
  calc
    (∑ i : I g, D g i * u g i)
        = ∑ i : I g, Dbar g * u g i := by
            apply Finset.sum_congr rfl
            intro i _hi
            rw [hD g i]
    _ = Dbar g * ∑ i : I g, u g i := by
            rw [Finset.mul_sum]

/--
Under group-level treatment, the cluster-robust score component is the
cluster-level treatment squared times the squared within-cluster residual sum.
-/
theorem scoreCluster_groupLevel_treatment_sq
    (D : ∀ g : G, I g → ℝ) (Dbar : G → ℝ) (u : ∀ g : G, I g → ℝ)
    (hD : IsGroupLevelTreatment D Dbar) (g : G) :
    ClusterLevel.scoreCluster (fun g i => D g i * u g i) g ^ 2 =
      Dbar g ^ 2 * (∑ i : I g, u g i) ^ 2 := by
  rw [scoreCluster_groupLevel_treatment D Dbar u hD g]
  ring

end GroupLevelTreatment

namespace TWFE

variable {I T : Type _} [Fintype I] [Fintype T]

/-- Unit mean in a two-way panel. -/
def rowMean (X : I → T → ℝ) (i : I) : ℝ :=
  (∑ t : T, X i t) / (Fintype.card T : ℝ)

/-- Time mean in a two-way panel. -/
def colMean (X : I → T → ℝ) (t : T) : ℝ :=
  (∑ i : I, X i t) / (Fintype.card I : ℝ)

/-- Grand mean in a two-way panel. -/
def grandMean (X : I → T → ℝ) : ℝ :=
  (∑ i : I, ∑ t : T, X i t) /
    ((Fintype.card I : ℝ) * (Fintype.card T : ℝ))

/-- Two-way fixed-effect residualization by double demeaning. -/
def doubleDemean (X : I → T → ℝ) (i : I) (t : T) : ℝ :=
  X i t - rowMean X i - colMean X t + grandMean X

/--
Double demeaning removes additive unit effects, time effects, and constants.
This is the algebraic core behind TWFE residualization.
-/
theorem doubleDemean_additive_unit_time_eq_zero
    (a : I → ℝ) (b : T → ℝ) (c : ℝ)
    (hI : (Fintype.card I : ℝ) ≠ 0)
    (hT : (Fintype.card T : ℝ) ≠ 0) :
    doubleDemean (fun i t => a i + b t + c) = fun _i _t => 0 := by
  classical
  funext i t
  unfold doubleDemean rowMean colMean grandMean
  simp_rw [Finset.sum_add_distrib]
  simp only [Finset.sum_const, nsmul_eq_mul, Finset.card_univ]
  field_simp [hI, hT, mul_ne_zero hI hT]
  rw [← Finset.mul_sum]
  ring_nf

end TWFE

namespace VectorTreatment

variable {α K : Type _} [Fintype α] [Fintype K]

/-- Matrix-vector multiplication, represented as finite functions. -/
def matVec (A : K → K → ℝ) (v : K → ℝ) (k : K) : ℝ :=
  ∑ l : K, A k l * v l

/-- Residualized treatment Gram matrix `D' D`. -/
def gram (D : α → K → ℝ) (k l : K) : ℝ :=
  ∑ a : α, D a k * D a l

/-- Cross-product vector `D'Y`. -/
def cross (D : α → K → ℝ) (Y : α → ℝ) (k : K) : ℝ :=
  ∑ a : α, D a k * Y a

/-- Score vector `D'u`. -/
def score (D : α → K → ℝ) (u : α → ℝ) (k : K) : ℝ :=
  ∑ a : α, D a k * u a

/-- Vector OLS coefficient using a supplied inverse/solver for the Gram matrix. -/
def betaHat (Ainv : K → K → ℝ) (D : α → K → ℝ) (Y : α → ℝ) : K → ℝ :=
  matVec Ainv (cross D Y)

/-- The normal-equation cross product decomposes into `D'D beta + D'u`. -/
theorem cross_eq_gram_beta_add_score
    (D : α → K → ℝ) (Y u : α → ℝ) (beta : K → ℝ)
    (hY : ∀ a : α, Y a = (∑ k : K, D a k * beta k) + u a) :
    cross D Y = fun k : K => matVec (gram D) beta k + score D u k := by
  classical
  funext k
  unfold cross matVec gram score
  calc
    (∑ a : α, D a k * Y a)
        = ∑ a : α, D a k * ((∑ l : K, D a l * beta l) + u a) := by
            apply Finset.sum_congr rfl
            intro a _ha
            rw [hY a]
    _ = ∑ a : α, ((∑ l : K, (D a k * D a l) * beta l) + D a k * u a) := by
            apply Finset.sum_congr rfl
            intro a _ha
            rw [mul_add]
            congr 1
            rw [Finset.mul_sum]
            apply Finset.sum_congr rfl
            intro l _hl
            ring
    _ = (∑ a : α, ∑ l : K, (D a k * D a l) * beta l) +
          ∑ a : α, D a k * u a := by
            rw [Finset.sum_add_distrib]
    _ = (∑ l : K, ∑ a : α, (D a k * D a l) * beta l) +
          ∑ a : α, D a k * u a := by
            rw [Finset.sum_comm]
    _ = (∑ l : K, (∑ a : α, D a k * D a l) * beta l) +
          ∑ a : α, D a k * u a := by
            congr 1
            apply Finset.sum_congr rfl
            intro l _hl
            rw [Finset.sum_mul]
    _ = (∑ l : K, (∑ a : α, D a k * D a l) * beta l) +
          ∑ a : α, D a k * u a := by
            rfl

lemma matVec_add
    (A : K → K → ℝ) (v w : K → ℝ) :
    matVec A (fun k => v k + w k) =
      fun k => matVec A v k + matVec A w k := by
  classical
  funext k
  unfold matVec
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro l _hl
  ring

/--
Vector FWL score expansion.  `Ainv` is any left inverse/solver for the Gram
matrix on the coefficient space.
-/
theorem betaHat_error
    (Ainv : K → K → ℝ) (D : α → K → ℝ) (Y u : α → ℝ) (beta : K → ℝ)
    (hY : ∀ a : α, Y a = (∑ k : K, D a k * beta k) + u a)
    (hLeftInv : matVec Ainv (matVec (gram D) beta) = beta) :
    betaHat Ainv D Y - beta = matVec Ainv (score D u) := by
  classical
  funext k
  have hcross := cross_eq_gram_beta_add_score D Y u beta hY
  unfold betaHat
  rw [hcross]
  rw [matVec_add]
  have hleft_k : matVec Ainv (matVec (gram D) beta) k = beta k := by
    exact congrFun hLeftInv k
  change matVec Ainv (matVec (gram D) beta) k +
      matVec Ainv (score D u) k - beta k =
    matVec Ainv (score D u) k
  rw [hleft_k]
  ring

end VectorTreatment

namespace MatrixSandwich

variable {K : Type _} [Fintype K]

/-- Finite-vector dot product. -/
def dot (x y : K → ℝ) : ℝ :=
  ∑ k : K, x k * y k

/-- Quadratic form for a finite matrix represented as a function. -/
def quad (A : K → K → ℝ) (v : K → ℝ) : ℝ :=
  ∑ k : K, ∑ l : K, v k * A k l * v l

/-- Positive semidefiniteness for finite matrix kernels. -/
def IsPSD (A : K → K → ℝ) : Prop :=
  ∀ v : K → ℝ, 0 ≤ quad A v

/--
Sandwich variance of a transformed score vector.  This is the core matrix
infrastructure for vector treatments, 2SLS, CRVE, GMM, and GLM sandwiches.
-/
def sandwichVariance (H meat : K → K → ℝ) (v : K → ℝ) : ℝ :=
  quad meat (VectorTreatment.matVec H v)

/-- A sandwich quadratic form is nonnegative when the middle meat matrix is PSD. -/
theorem sandwichVariance_nonneg
    (H meat : K → K → ℝ) (v : K → ℝ)
    (hPSD : IsPSD meat) :
    0 ≤ sandwichVariance H meat v := by
  exact hPSD (VectorTreatment.matVec H v)

/-- Scalar GMM/GLM sandwich target, using a supplied inverse Hessian/Jacobian. -/
def gmmSandwich (Hinv meat : K → K → ℝ) (scoreDirection : K → ℝ) : ℝ :=
  sandwichVariance Hinv meat scoreDirection

theorem gmmSandwich_nonneg
    (Hinv meat : K → K → ℝ) (scoreDirection : K → ℝ)
    (hPSD : IsPSD meat) :
    0 ≤ gmmSandwich Hinv meat scoreDirection :=
  sandwichVariance_nonneg Hinv meat scoreDirection hPSD

end MatrixSandwich

namespace HACKernel

variable {α : Type _}

/--
Distance-weighted covariance kernel.  This covers Conley/spatial HAC,
network-HAC, and panel-HAC once `dist` is specialized to geographic, graph, or
temporal distance.
-/
def weightedKernel
    (dist : α → α → ℝ) (bandwidth : ℝ) (Kfun : ℝ → ℝ)
    (base : α → α → ℝ) : α → α → ℝ :=
  fun i j => Kfun (dist i j / bandwidth) * base i j

/-- Compact-support condition for a scalar HAC kernel. -/
def CompactSupportAt (Kfun : ℝ → ℝ) (radius : ℝ) : Prop :=
  ∀ x : ℝ, radius < |x| → Kfun x = 0

/-- A compactly supported HAC kernel is zero outside its normalized radius. -/
theorem weightedKernel_eq_zero_of_outside_support
    (dist : α → α → ℝ) (bandwidth radius : ℝ) (Kfun : ℝ → ℝ)
    (base : α → α → ℝ)
    (hK : CompactSupportAt Kfun radius)
    {i j : α}
    (hout : radius < |dist i j / bandwidth|) :
    weightedKernel dist bandwidth Kfun base i j = 0 := by
  unfold weightedKernel
  rw [hK (dist i j / bandwidth) hout]
  ring

end HACKernel

namespace TriangularArray

variable {α : ℕ → Type _} [∀ n : ℕ, Fintype (α n)]

/-- Triangular-array quadratic form with dimension/index type varying in `n`. -/
def quad
    (u : ∀ n : ℕ, α n → ℝ)
    (Sigma : ∀ n : ℕ, α n → α n → ℝ)
    (n : ℕ) : ℝ :=
  ∑ i : α n, ∑ j : α n, u n i * Sigma n i j * u n j

/-- Triangular-array design variance target. -/
def designVariance
    (S : ℕ → ℝ)
    (u : ∀ n : ℕ, α n → ℝ)
    (Sigma : ∀ n : ℕ, α n → α n → ℝ)
    (n : ℕ) : ℝ :=
  ClusterLevel.scale (S n) * quad u Sigma n

/-- PSD condition for every finite covariance kernel in a triangular array. -/
def IsPSD
    (Sigma : ∀ n : ℕ, α n → α n → ℝ) : Prop :=
  ∀ n : ℕ, ∀ v : α n → ℝ,
    0 ≤ ∑ i : α n, ∑ j : α n, v i * Sigma n i j * v j

theorem designVariance_nonneg
    (S : ℕ → ℝ)
    (u : ∀ n : ℕ, α n → ℝ)
    (Sigma : ∀ n : ℕ, α n → α n → ℝ)
    (hPSD : IsPSD Sigma)
    (n : ℕ) :
    0 ≤ designVariance S u Sigma n := by
  unfold designVariance quad ClusterLevel.scale
  exact mul_nonneg (inv_nonneg.mpr (sq_nonneg (S n))) (hPSD n (u n))

/--
Ratio consistency target for growing covariance dimensions.  Concrete
triangular-array LLNs or weak-dependence arguments should prove this predicate.
-/
def RatioConsistent
    (vHat vTarget : ℕ → ℝ) : Prop :=
  Filter.Tendsto (fun n : ℕ => vHat n / vTarget n) Filter.atTop (nhds 1)

end TriangularArray

/-! ## 8. Asymptotic consistency and inference -/

namespace Asymptotics

open MeasureTheory ProbabilityTheory Filter
open scoped Topology

/-! ### Deterministic asymptotic algebra validated by Lean -/

/--
Finite-dimensional continuous mapping for the covariance sandwich numerator:
pointwise convergence of residuals and covariance-kernel entries implies
convergence of the quadratic form.
-/
theorem quad_tendsto_of_pointwise
    {α : Type _} [Fintype α]
    (uhat : ℕ → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → α → α → ℝ) (Sigma : α → α → ℝ)
    (hu : ∀ i : α, Filter.Tendsto (fun n : ℕ => uhat n i) Filter.atTop (nhds (u i)))
    (hSigma :
      ∀ i j : α, Filter.Tendsto (fun n : ℕ => SigmaHat n i j) Filter.atTop (nhds (Sigma i j))) :
    Filter.Tendsto
      (fun n : ℕ => DesignCovariance.quad (uhat n) (SigmaHat n))
      Filter.atTop
      (nhds (DesignCovariance.quad u Sigma)) := by
  classical
  unfold DesignCovariance.quad
  simpa using
    (tendsto_finsetSum
      (s := (Finset.univ : Finset α))
      (x := Filter.atTop)
      (f := fun i n => ∑ j : α, uhat n i * SigmaHat n i j * uhat n j)
      (a := fun i => ∑ j : α, u i * Sigma i j * u j)
      (by
        intro i _hi
        simpa using
          (tendsto_finsetSum
            (s := (Finset.univ : Finset α))
            (x := Filter.atTop)
            (f := fun j n => uhat n i * SigmaHat n i j * uhat n j)
            (a := fun j => u i * Sigma i j * u j)
            (by
              intro j _hj
              exact ((hu i).mul (hSigma i j)).mul (hu j)))))

/--
Consistency of the design-covariance estimator under pointwise residual and
covariance-kernel consistency, with the outer denominator scale fixed.
-/
theorem dcCCV_tendsto_of_pointwise
    {α : Type _} [Fintype α]
    (S : ℝ) (uhat : ℕ → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → α → α → ℝ) (Sigma : α → α → ℝ)
    (hu : ∀ i : α, Filter.Tendsto (fun n : ℕ => uhat n i) Filter.atTop (nhds (u i)))
    (hSigma :
      ∀ i j : α, Filter.Tendsto (fun n : ℕ => SigmaHat n i j) Filter.atTop (nhds (Sigma i j))) :
    Filter.Tendsto
      (fun n : ℕ => DesignCovariance.dcCCV S (uhat n) (SigmaHat n))
      Filter.atTop
      (nhds (DesignCovariance.designVariance S u Sigma)) := by
  have hquad := quad_tendsto_of_pointwise uhat u SigmaHat Sigma hu hSigma
  unfold DesignCovariance.dcCCV DesignCovariance.designVariance
  exact tendsto_const_nhds.mul hquad

/--
Ratio consistency of `dcCCV` follows from pointwise residual/covariance
consistency whenever the limiting design variance is nonzero.
-/
theorem dcCCV_ratio_tendsto_one_of_pointwise
    {α : Type _} [Fintype α]
    (S : ℝ) (uhat : ℕ → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → α → α → ℝ) (Sigma : α → α → ℝ)
    (hu : ∀ i : α, Filter.Tendsto (fun n : ℕ => uhat n i) Filter.atTop (nhds (u i)))
    (hSigma :
      ∀ i j : α, Filter.Tendsto (fun n : ℕ => SigmaHat n i j) Filter.atTop (nhds (Sigma i j)))
    (hV : DesignCovariance.designVariance S u Sigma ≠ 0) :
    Filter.Tendsto
      (fun n : ℕ =>
        DesignCovariance.dcCCV S (uhat n) (SigmaHat n) /
          DesignCovariance.designVariance S u Sigma)
      Filter.atTop
      (nhds 1) := by
  have hdc := dcCCV_tendsto_of_pointwise S uhat u SigmaHat Sigma hu hSigma
  have hratio := hdc.div tendsto_const_nhds hV
  simpa [hV] using hratio

/--
Almost-sure pointwise consistency of the residual vector and covariance-kernel
entries implies convergence in measure of the feasible design-covariance
variance estimator.
-/
theorem dcCCV_tendstoInMeasure_of_ae_pointwise
    {Ω α : Type _} [MeasurableSpace Ω] [Fintype α]
    {P : Measure Ω} [IsFiniteMeasure P]
    (S : ℝ) (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (hmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω => DesignCovariance.dcCCV S (uhat n ω) (SigmaHat n ω)) P)
    (hu :
      ∀ i : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)))
    (hSigma :
      ∀ i j : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => SigmaHat n ω i j) Filter.atTop (nhds (Sigma i j))) :
    TendstoInMeasure P
      (fun n ω => DesignCovariance.dcCCV S (uhat n ω) (SigmaHat n ω))
      Filter.atTop
      (fun _ : Ω => DesignCovariance.designVariance S u Sigma) := by
  classical
  refine tendstoInMeasure_of_tendsto_ae hmeas ?_
  have hu_all :
      ∀ᵐ ω ∂P, ∀ i : α,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)) :=
    ae_all_iff.mpr hu
  have hSigma_all :
      ∀ᵐ ω ∂P, ∀ i j : α,
        Filter.Tendsto (fun n : ℕ => SigmaHat n ω i j) Filter.atTop (nhds (Sigma i j)) :=
    ae_all_iff.mpr (fun i => ae_all_iff.mpr (hSigma i))
  filter_upwards [hu_all, hSigma_all] with ω hωu hωSigma
  exact dcCCV_tendsto_of_pointwise
    S
    (fun n i => uhat n ω i)
    u
    (fun n i j => SigmaHat n ω i j)
    Sigma
    (fun i => hωu i)
    (fun i j => hωSigma i j)

/--
Almost-sure pointwise residual/covariance consistency also yields convergence in
measure of the feasible-to-design variance ratio, provided the design variance is
nonzero.
-/
theorem dcCCV_ratio_tendstoInMeasure_one_of_ae_pointwise
    {Ω α : Type _} [MeasurableSpace Ω] [Fintype α]
    {P : Measure Ω} [IsFiniteMeasure P]
    (S : ℝ) (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (hmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω =>
            DesignCovariance.dcCCV S (uhat n ω) (SigmaHat n ω) /
              DesignCovariance.designVariance S u Sigma) P)
    (hu :
      ∀ i : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)))
    (hSigma :
      ∀ i j : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => SigmaHat n ω i j) Filter.atTop (nhds (Sigma i j)))
    (hV : DesignCovariance.designVariance S u Sigma ≠ 0) :
    TendstoInMeasure P
      (fun n ω =>
        DesignCovariance.dcCCV S (uhat n ω) (SigmaHat n ω) /
          DesignCovariance.designVariance S u Sigma)
      Filter.atTop
      (fun _ : Ω => (1 : ℝ)) := by
  classical
  refine tendstoInMeasure_of_tendsto_ae hmeas ?_
  have hu_all :
      ∀ᵐ ω ∂P, ∀ i : α,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)) :=
    ae_all_iff.mpr hu
  have hSigma_all :
      ∀ᵐ ω ∂P, ∀ i j : α,
        Filter.Tendsto (fun n : ℕ => SigmaHat n ω i j) Filter.atTop (nhds (Sigma i j)) :=
    ae_all_iff.mpr (fun i => ae_all_iff.mpr (hSigma i))
  filter_upwards [hu_all, hSigma_all] with ω hωu hωSigma
  exact dcCCV_ratio_tendsto_one_of_pointwise
    S
    (fun n i => uhat n ω i)
    u
    (fun n i j => SigmaHat n ω i j)
    Sigma
    (fun i => hωu i)
    (fun i j => hωSigma i j)
    hV

/--
Almost-sure pointwise residual/covariance consistency yields consistency of the
inverse standard-error ratio `seDesign / dcSE`, which is the Slutsky multiplier
used for feasible studentization.
-/
theorem dcSE_inv_ratio_tendstoInMeasure_one_of_ae_pointwise
    {Ω α : Type _} [MeasurableSpace Ω] [Fintype α]
    {P : Measure Ω} [IsFiniteMeasure P]
    (S : ℝ) (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (hmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω =>
            DesignCovariance.designSE S u Sigma /
              DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω)) P)
    (hu :
      ∀ i : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)))
    (hSigma :
      ∀ i j : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => SigmaHat n ω i j) Filter.atTop (nhds (Sigma i j)))
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma) :
    TendstoInMeasure P
      (fun n ω =>
        DesignCovariance.designSE S u Sigma /
          DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
      Filter.atTop
      (fun _ : Ω => (1 : ℝ)) := by
  classical
  refine tendstoInMeasure_of_tendsto_ae hmeas ?_
  have hu_all :
      ∀ᵐ ω ∂P, ∀ i : α,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)) :=
    ae_all_iff.mpr hu
  have hSigma_all :
      ∀ᵐ ω ∂P, ∀ i j : α,
        Filter.Tendsto (fun n : ℕ => SigmaHat n ω i j) Filter.atTop (nhds (Sigma i j)) :=
    ae_all_iff.mpr (fun i => ae_all_iff.mpr (hSigma i))
  filter_upwards [hu_all, hSigma_all] with ω hωu hωSigma
  have hdc :
      Filter.Tendsto
        (fun n : ℕ => DesignCovariance.dcCCV S (uhat n ω) (SigmaHat n ω))
        Filter.atTop
        (nhds (DesignCovariance.designVariance S u Sigma)) :=
    dcCCV_tendsto_of_pointwise
      S
      (fun n i => uhat n ω i)
      u
      (fun n i j => SigmaHat n ω i j)
      Sigma
      (fun i => hωu i)
      (fun i j => hωSigma i j)
  have hsqrt :
      Filter.Tendsto
        (fun n : ℕ => Real.sqrt (DesignCovariance.dcCCV S (uhat n ω) (SigmaHat n ω)))
        Filter.atTop
        (nhds (Real.sqrt (DesignCovariance.designVariance S u Sigma))) :=
    hdc.sqrt
  have hsqrt_ne : Real.sqrt (DesignCovariance.designVariance S u Sigma) ≠ 0 :=
    Real.sqrt_ne_zero'.mpr hVpos
  have hratio :
      Filter.Tendsto
        (fun n : ℕ =>
          Real.sqrt (DesignCovariance.designVariance S u Sigma) /
            Real.sqrt (DesignCovariance.dcCCV S (uhat n ω) (SigmaHat n ω)))
        Filter.atTop
        (nhds
          (Real.sqrt (DesignCovariance.designVariance S u Sigma) /
            Real.sqrt (DesignCovariance.designVariance S u Sigma))) :=
    tendsto_const_nhds.div hsqrt hsqrt_ne
  simpa [DesignCovariance.designSE, DesignCovariance.dcSE, hsqrt_ne] using hratio

/--
Weak-dependence SLLN interface for covariance-kernel consistency.  A mixing,
HAC, panel-HAC, or network-dependence proof should discharge `hSigma`; this
theorem then converts that entrywise kernel consistency into `dcCCV`
consistency.
-/
theorem dcCCV_tendstoInMeasure_of_weakDependenceKernel
    {Ω α : Type _} [MeasurableSpace Ω] [Fintype α]
    {P : Measure Ω} [IsFiniteMeasure P]
    (S : ℝ) (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (hmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω => DesignCovariance.dcCCV S (uhat n ω) (SigmaHat n ω)) P)
    (hu :
      ∀ i : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)))
    (hSigma :
      ∀ i j : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => SigmaHat n ω i j) Filter.atTop (nhds (Sigma i j))) :
    TendstoInMeasure P
      (fun n ω => DesignCovariance.dcCCV S (uhat n ω) (SigmaHat n ω))
      Filter.atTop
      (fun _ : Ω => DesignCovariance.designVariance S u Sigma) :=
  dcCCV_tendstoInMeasure_of_ae_pointwise S uhat u SigmaHat Sigma hmeas hu hSigma

/--
Weak-dependence interface for feasible standard-error consistency.  The hard
probability input is again the entrywise SLLN/consistency statement `hSigma`.
-/
theorem dcSE_inv_ratio_tendstoInMeasure_one_of_weakDependenceKernel
    {Ω α : Type _} [MeasurableSpace Ω] [Fintype α]
    {P : Measure Ω} [IsFiniteMeasure P]
    (S : ℝ) (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (hmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω =>
            DesignCovariance.designSE S u Sigma /
              DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω)) P)
    (hu :
      ∀ i : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)))
    (hSigma :
      ∀ i j : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => SigmaHat n ω i j) Filter.atTop (nhds (Sigma i j)))
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma) :
    TendstoInMeasure P
      (fun n ω =>
        DesignCovariance.designSE S u Sigma /
          DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
      Filter.atTop
      (fun _ : Ω => (1 : ℝ)) :=
  dcSE_inv_ratio_tendstoInMeasure_one_of_ae_pointwise
    S uhat u SigmaHat Sigma hmeas hu hSigma hVpos

/--
If the residual estimator is eventually exact almost surely, then it satisfies
the pointwise residual-consistency premise used by the feasible `dcCCV`
asymptotic theorem.
-/
theorem residual_tendsto_ae_of_ae_forall_eq
    {Ω α : Type _} [MeasurableSpace Ω]
    {P : Measure Ω}
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (hexact : ∀ᵐ ω ∂P, ∀ n : ℕ, ∀ i : α, uhat n ω i = u i) :
    ∀ i : α, ∀ᵐ ω ∂P,
      Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)) := by
  intro i
  filter_upwards [hexact] with ω hω
  have hfun : (fun n : ℕ => uhat n ω i) = fun _n : ℕ => u i := by
    funext n
    exact hω n i
  simpa [hfun] using tendsto_const_nhds

/--
If the covariance-kernel estimator is eventually exact almost surely, then it
satisfies the pointwise kernel-consistency premise used by the feasible `dcCCV`
asymptotic theorem.
-/
theorem kernel_tendsto_ae_of_ae_forall_eq
    {Ω α : Type _} [MeasurableSpace Ω]
    {P : Measure Ω}
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (hexact :
      ∀ᵐ ω ∂P, ∀ n : ℕ, ∀ i j : α, SigmaHat n ω i j = Sigma i j) :
    ∀ i j : α, ∀ᵐ ω ∂P,
      Filter.Tendsto (fun n : ℕ => SigmaHat n ω i j) Filter.atTop
        (nhds (Sigma i j)) := by
  intro i j
  filter_upwards [hexact] with ω hω
  have hfun : (fun n : ℕ => SigmaHat n ω i j) = fun _n : ℕ => Sigma i j := by
    funext n
    exact hω n i j
  simpa [hfun] using tendsto_const_nhds

/--
Exact residuals and an exact covariance kernel imply feasible `dcCCV`
converges in measure to the design variance.  This is the asymptotic counterpart
of the exact finite-sample design-variance identity.
-/
theorem dcCCV_tendstoInMeasure_of_ae_exact
    {Ω α : Type _} [MeasurableSpace Ω] [Fintype α]
    {P : Measure Ω} [IsFiniteMeasure P]
    (S : ℝ) (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (huexact : ∀ᵐ ω ∂P, ∀ n : ℕ, ∀ i : α, uhat n ω i = u i)
    (hSigmaexact :
      ∀ᵐ ω ∂P, ∀ n : ℕ, ∀ i j : α, SigmaHat n ω i j = Sigma i j) :
    TendstoInMeasure P
      (fun n ω => DesignCovariance.dcCCV S (uhat n ω) (SigmaHat n ω))
      Filter.atTop
      (fun _ : Ω => DesignCovariance.designVariance S u Sigma) := by
  have hmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω => DesignCovariance.dcCCV S (uhat n ω) (SigmaHat n ω)) P := by
    intro n
    have hEq :
        (fun ω : Ω => DesignCovariance.dcCCV S (uhat n ω) (SigmaHat n ω)) =ᵐ[P]
          fun _ω : Ω => DesignCovariance.designVariance S u Sigma := by
      filter_upwards [huexact, hSigmaexact] with ω huω hSigmaω
      have hu : uhat n ω = u := by
        funext i
        exact huω n i
      have hSigma : SigmaHat n ω = Sigma := by
        funext i j
        exact hSigmaω n i j
      exact DesignCovariance.dcCCV_eq_designVariance_of_exact
        S u (uhat n ω) Sigma (SigmaHat n ω) hu hSigma
    exact aestronglyMeasurable_const.congr hEq.symm
  exact dcCCV_tendstoInMeasure_of_ae_pointwise
    S uhat u SigmaHat Sigma hmeas
    (residual_tendsto_ae_of_ae_forall_eq uhat u huexact)
    (kernel_tendsto_ae_of_ae_forall_eq SigmaHat Sigma hSigmaexact)

/--
Exact residuals and an exact covariance kernel imply feasible standard-error
ratio consistency.  In this case the ratio is almost surely equal to one for
every sample size.
-/
theorem dcSE_inv_ratio_tendstoInMeasure_one_of_ae_exact
    {Ω α : Type _} [MeasurableSpace Ω] [Fintype α]
    {P : Measure Ω} [IsFiniteMeasure P]
    (S : ℝ) (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (huexact : ∀ᵐ ω ∂P, ∀ n : ℕ, ∀ i : α, uhat n ω i = u i)
    (hSigmaexact :
      ∀ᵐ ω ∂P, ∀ n : ℕ, ∀ i j : α, SigmaHat n ω i j = Sigma i j)
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma) :
    TendstoInMeasure P
      (fun n ω =>
        DesignCovariance.designSE S u Sigma /
          DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
      Filter.atTop
      (fun _ : Ω => (1 : ℝ)) := by
  have hDesignSE_ne : DesignCovariance.designSE S u Sigma ≠ 0 := by
    unfold DesignCovariance.designSE
    exact Real.sqrt_ne_zero'.mpr hVpos
  have hmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω =>
            DesignCovariance.designSE S u Sigma /
              DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω)) P := by
    intro n
    have hEq :
        (fun ω : Ω =>
          DesignCovariance.designSE S u Sigma /
            DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω)) =ᵐ[P]
          fun _ω : Ω => (1 : ℝ) := by
      filter_upwards [huexact, hSigmaexact] with ω huω hSigmaω
      have hu : uhat n ω = u := by
        funext i
        exact huω n i
      have hSigma : SigmaHat n ω = Sigma := by
        funext i j
        exact hSigmaω n i j
      rw [DesignCovariance.dcSE_eq_designSE_of_exact
        S u (uhat n ω) Sigma (SigmaHat n ω) hu hSigma]
      exact div_self hDesignSE_ne
    exact aestronglyMeasurable_const.congr hEq.symm
  exact dcSE_inv_ratio_tendstoInMeasure_one_of_ae_pointwise
    S uhat u SigmaHat Sigma hmeas
    (residual_tendsto_ae_of_ae_forall_eq uhat u huexact)
    (kernel_tendsto_ae_of_ae_forall_eq SigmaHat Sigma hSigmaexact)
    hVpos

/-!
### Scope notes for covariance-kernel consistency

The formalized sample-kernel route below proves consistency from Mathlib's iid
strong law, entry by entry.  This covers iid observations and iid clusters by
letting the sample index `k` range over independent clusters.

Two natural extensions are intentionally left as assumptions/proof obligations
rather than proved here:

* Weak dependence or mixing across administrative units.  To use that route,
  replace `sampleKernelMean_tendsto_ae` with a weak-dependence SLLN proving
  `∀ i j, SigmaHat n ω i j → Sigma i j` almost surely or in probability, then
  apply the existing `dcCCV_tendstoInMeasure_of_ae_pointwise` theorem.
* Growing covariance dimension.  The current theorem fixes a finite index type
  `α`, so all finite sums are over a fixed set.  If the number of units,
  clusters, or cells grows with `n`, the proof needs a triangular-array version
  with `α n`, uniform kernel convergence, and nondegenerate limiting variance.
-/

/-- Empirical average of a real-valued stochastic sequence. -/
def sampleMean {Ω : Type _} (X : ℕ → Ω → ℝ) (n : ℕ) (ω : Ω) : ℝ :=
  (n : ℝ)⁻¹ * ∑ k ∈ Finset.range n, X k ω

/--
Strong-law consistency of a sample mean.  This is the basic ingredient used for
entrywise consistency of a sample covariance-kernel estimator.
-/
theorem sampleMean_tendsto_ae
    {Ω : Type _} [MeasurableSpace Ω] {P : Measure Ω}
    (X : ℕ → Ω → ℝ)
    (hint : Integrable (X 0) P)
    (hindep : Pairwise fun i j => IndepFun (X i) (X j) P)
    (hident : ∀ i : ℕ, IdentDistrib (X i) (X 0) P P) :
    ∀ᵐ ω ∂P,
      Filter.Tendsto (fun n : ℕ => sampleMean X n ω)
        Filter.atTop (nhds (∫ ω, X 0 ω ∂P)) := by
  simpa [sampleMean, smul_eq_mul] using
    (ProbabilityTheory.strong_law_ae X hint hindep hident)

/-- Entrywise sample-average estimator for a covariance kernel. -/
def sampleKernelMean {Ω α : Type _}
    (Z : ℕ → Ω → α → α → ℝ) (n : ℕ) (ω : Ω) (i j : α) : ℝ :=
  sampleMean (fun k ω => Z k ω i j) n ω

/--
Entrywise strong-law consistency of a sample covariance-kernel estimator.
`Z k ω i j` is the k-th sample contribution to covariance entry `(i,j)`.
-/
theorem sampleKernelMean_tendsto_ae
    {Ω α : Type _} [MeasurableSpace Ω] [Fintype α] {P : Measure Ω}
    (Z : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (hint : ∀ i j : α, Integrable (fun ω : Ω => Z 0 ω i j) P)
    (hindep :
      ∀ i j : α, Pairwise fun k l => IndepFun (fun ω : Ω => Z k ω i j)
        (fun ω : Ω => Z l ω i j) P)
    (hident :
      ∀ i j : α, ∀ k : ℕ,
        IdentDistrib (fun ω : Ω => Z k ω i j) (fun ω : Ω => Z 0 ω i j) P P)
    (hmean : ∀ i j : α, (∫ ω, Z 0 ω i j ∂P) = Sigma i j) :
    ∀ i j : α, ∀ᵐ ω ∂P,
      Filter.Tendsto (fun n : ℕ => sampleKernelMean Z n ω i j)
        Filter.atTop (nhds (Sigma i j)) := by
  intro i j
  have h :=
    sampleMean_tendsto_ae
      (fun k ω => Z k ω i j)
      (hint i j)
      (hindep i j)
      (hident i j)
  simpa [sampleKernelMean, hmean i j] using h

/--
If the covariance-kernel estimator is an entrywise sample mean satisfying the
strong law, then `dcCCV` converges in measure to the design variance.
-/
theorem dcCCV_tendstoInMeasure_of_sampleKernelMean
    {Ω α : Type _} [MeasurableSpace Ω] [Fintype α]
    {P : Measure Ω} [IsFiniteMeasure P]
    (S : ℝ) (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (Z : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (hmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω =>
            DesignCovariance.dcCCV S (uhat n ω) (sampleKernelMean Z n ω)) P)
    (hu :
      ∀ i : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)))
    (hint : ∀ i j : α, Integrable (fun ω : Ω => Z 0 ω i j) P)
    (hindep :
      ∀ i j : α, Pairwise fun k l => IndepFun (fun ω : Ω => Z k ω i j)
        (fun ω : Ω => Z l ω i j) P)
    (hident :
      ∀ i j : α, ∀ k : ℕ,
        IdentDistrib (fun ω : Ω => Z k ω i j) (fun ω : Ω => Z 0 ω i j) P P)
    (hmean : ∀ i j : α, (∫ ω, Z 0 ω i j ∂P) = Sigma i j) :
    TendstoInMeasure P
      (fun n ω => DesignCovariance.dcCCV S (uhat n ω) (sampleKernelMean Z n ω))
      Filter.atTop
      (fun _ : Ω => DesignCovariance.designVariance S u Sigma) := by
  exact dcCCV_tendstoInMeasure_of_ae_pointwise
    S uhat u (sampleKernelMean Z) Sigma hmeas hu
    (sampleKernelMean_tendsto_ae Z Sigma hint hindep hident hmean)

/--
Sample-kernel version of inverse standard-error ratio consistency.  This is the
direct input needed to studentize by `dcSE`.
-/
theorem dcSE_inv_ratio_tendstoInMeasure_one_of_sampleKernelMean
    {Ω α : Type _} [MeasurableSpace Ω] [Fintype α]
    {P : Measure Ω} [IsFiniteMeasure P]
    (S : ℝ) (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (Z : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (hmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω =>
            DesignCovariance.designSE S u Sigma /
              DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Z n ω)) P)
    (hu :
      ∀ i : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)))
    (hint : ∀ i j : α, Integrable (fun ω : Ω => Z 0 ω i j) P)
    (hindep :
      ∀ i j : α, Pairwise fun k l => IndepFun (fun ω : Ω => Z k ω i j)
        (fun ω : Ω => Z l ω i j) P)
    (hident :
      ∀ i j : α, ∀ k : ℕ,
        IdentDistrib (fun ω : Ω => Z k ω i j) (fun ω : Ω => Z 0 ω i j) P P)
    (hmean : ∀ i j : α, (∫ ω, Z 0 ω i j ∂P) = Sigma i j)
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma) :
    TendstoInMeasure P
      (fun n ω =>
        DesignCovariance.designSE S u Sigma /
          DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Z n ω))
      Filter.atTop
      (fun _ : Ω => (1 : ℝ)) := by
  exact dcSE_inv_ratio_tendstoInMeasure_one_of_ae_pointwise
    S uhat u (sampleKernelMean Z) Sigma hmeas hu
    (sampleKernelMean_tendsto_ae Z Sigma hint hindep hident hmean)
    hVpos

/-- First-order variance target `A_k / s_k^2`. -/
def varianceTarget (A s : ℕ → ℝ) (n : ℕ) : ℝ :=
  A n / s n ^ 2

/-- Feasible sandwich variance `Ahat_k / S_k^2`. -/
def varianceHat (Ahat S : ℕ → ℝ) (n : ℕ) : ℝ :=
  Ahat n / S n ^ 2

/--
The ratio of feasible to target variances is the product of the numerator ratio
and squared denominator ratio, whenever denominators are nonzero.
-/
theorem varianceHat_div_target_eq
    (Ahat A S s : ℕ → ℝ) (n : ℕ)
    (hA : A n ≠ 0) (hS : S n ≠ 0) (hs : s n ≠ 0) :
    varianceHat Ahat S n / varianceTarget A s n =
      (Ahat n / A n) * (s n / S n) ^ 2 := by
  have hS2 : S n ^ 2 ≠ 0 := pow_ne_zero 2 hS
  have hs2 : s n ^ 2 ≠ 0 := pow_ne_zero 2 hs
  have htarget : A n / s n ^ 2 ≠ 0 := div_ne_zero hA hs2
  unfold varianceHat varianceTarget
  field_simp [hA, hS, hs, hS2, hs2, htarget]

/--
If the numerator ratio converges to one and the denominator ratio converges to
one, then the feasible variance is ratio-consistent for the target variance.
This is the Slutsky-free algebraic core of the SE consistency proof.
-/
theorem variance_ratio_consistent
    (Ahat A S s : ℕ → ℝ)
    (hA : ∀ n : ℕ, A n ≠ 0)
    (hS : ∀ n : ℕ, S n ≠ 0)
    (hs : ∀ n : ℕ, s n ≠ 0)
    (hnum : Filter.Tendsto (fun n : ℕ => Ahat n / A n) Filter.atTop (nhds 1))
    (hden : Filter.Tendsto (fun n : ℕ => s n / S n) Filter.atTop (nhds 1)) :
    Filter.Tendsto
      (fun n : ℕ => varianceHat Ahat S n / varianceTarget A s n)
      Filter.atTop
      (nhds 1) := by
  have hsq : Filter.Tendsto (fun n : ℕ => (s n / S n) ^ 2) Filter.atTop (nhds (1 ^ 2)) :=
    hden.pow 2
  have hprod :
      Filter.Tendsto
        (fun n : ℕ => (Ahat n / A n) * (s n / S n) ^ 2)
        Filter.atTop
        (nhds (1 * 1 ^ 2)) :=
    hnum.mul hsq
  have hprod_one :
      Filter.Tendsto
        (fun n : ℕ => (Ahat n / A n) * (s n / S n) ^ 2)
        Filter.atTop
        (nhds 1) := by
    simpa using hprod
  refine hprod_one.congr' ?_
  exact Filter.Eventually.of_forall
    (fun n : ℕ => (varianceHat_div_target_eq Ahat A S s n (hA n) (hS n) (hs n)).symm)

/-- Feasible and target standard errors. -/
def seHat (Ahat S : ℕ → ℝ) (n : ℕ) : ℝ :=
  Real.sqrt (varianceHat Ahat S n)

def seTarget (A s : ℕ → ℝ) (n : ℕ) : ℝ :=
  Real.sqrt (varianceTarget A s n)

/-- The SE ratio equals the square root of the variance ratio when variances are nonnegative. -/
theorem seHat_div_target_eq_sqrt_variance_ratio
    (Ahat A S s : ℕ → ℝ) (n : ℕ)
    (hV : 0 ≤ varianceHat Ahat S n)
    (_hT : 0 ≤ varianceTarget A s n) :
    seHat Ahat S n / seTarget A s n =
      Real.sqrt (varianceHat Ahat S n / varianceTarget A s n) := by
  unfold seHat seTarget
  by_cases hzero : varianceTarget A s n = 0
  · rw [hzero]
    simp
  · rw [← Real.sqrt_div hV]

/-!
### Concrete iid asymptotic inference

The next results use Mathlib's measure-theoretic convergence notions to prove a
genuine iid-score asymptotic-normality and coverage pipeline:

* Mathlib's iid CLT gives convergence in distribution for normalized scores.
* Mathlib's Slutsky theorem studentizes by an inverse standard-error ratio that
  converges in probability to one.
* The portmanteau theorem gives convergence of interval coverage probabilities.

This is still not the full continuous-treatment DiD CCV theorem.  It covers the
iid normalized-score model stated in the theorem assumptions below.  A clustered
or treatment-design theorem still has to prove that its linearized score array
satisfies an appropriate CLT and that its CCV standard-error ratio converges.
-/

/-- The probability law of a random variable, packaged as a `ProbabilityMeasure`. -/
def lawPM {Ω E : Type _} [MeasurableSpace Ω] [MeasurableSpace E]
    (P : Measure Ω) [IsProbabilityMeasure P] (X : Ω → E) (hX : AEMeasurable X P) :
    ProbabilityMeasure E :=
  ⟨P.map X, Measure.isProbabilityMeasure_map hX⟩

/--
Slutsky studentization.  If a normalized statistic converges in distribution to
`Z` and the multiplicative inverse standard-error ratio converges in probability
to one, then the studentized statistic has the same limiting distribution.

The multiplicative ratio is stated as `seInv`, typically
`seTarget / seHat`.  This avoids division by a random denominator in the
continuous mapping step; multiplication is globally continuous.
-/
theorem studentized_tendstoInDistribution_of_clt_and_se_inv_consistency
    {Ω Ωlim : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (normalized seInv studentized : ℕ → Ω → ℝ) (Z : Ωlim → ℝ)
    (hCLT : TendstoInDistribution normalized Filter.atTop Z (fun _ : ℕ => P) Plim)
    (hSE : TendstoInMeasure P seInv Filter.atTop (fun _ : Ω => (1 : ℝ)))
    (hSEmeas : ∀ n : ℕ, AEMeasurable (seInv n) P)
    (hstudentized :
      ∀ n : ℕ, studentized n =ᵐ[P] fun ω : Ω => normalized n ω * seInv n ω) :
    TendstoInDistribution studentized Filter.atTop Z (fun _ : ℕ => P) Plim := by
  have hmul :
      TendstoInDistribution
        (fun n ω => normalized n ω * seInv n ω) Filter.atTop
        (fun ω => Z ω * (1 : ℝ)) (fun _ : ℕ => P) Plim := by
    exact hCLT.continuous_comp_prodMk_of_tendstoInMeasure_const
      (g := fun p : ℝ × ℝ => p.1 * p.2) (by fun_prop) hSE hSEmeas
  exact hmul.congr (fun n => (hstudentized n).symm)
    (Filter.Eventually.of_forall (fun _ => by simp))

/--
Additive Slutsky linearization.  If a linearized statistic has a distributional
limit and the remainder is `o_p(1)`, then any statistic that is almost surely
the sum of the linearized statistic and the remainder has the same limit.
-/
theorem tendstoInDistribution_of_clt_and_op_remainder
    {Ω Ωlim : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (linearized remainder statistic : ℕ → Ω → ℝ) (Z : Ωlim → ℝ)
    (hCLT : TendstoInDistribution linearized Filter.atTop Z (fun _ : ℕ => P) Plim)
    (hrem : TendstoInMeasure P remainder Filter.atTop (fun _ : Ω => (0 : ℝ)))
    (hremmeas : ∀ n : ℕ, AEMeasurable (remainder n) P)
    (hstat :
      ∀ n : ℕ, statistic n =ᵐ[P]
        fun ω : Ω => linearized n ω + remainder n ω) :
    TendstoInDistribution statistic Filter.atTop Z (fun _ : ℕ => P) Plim := by
  have hadd :
      TendstoInDistribution
        (fun n ω => linearized n ω + remainder n ω) Filter.atTop
        (fun ω => Z ω + (0 : ℝ)) (fun _ : ℕ => P) Plim := by
    exact hCLT.continuous_comp_prodMk_of_tendstoInMeasure_const
      (g := fun p : ℝ × ℝ => p.1 + p.2) (by fun_prop) hrem hremmeas
  exact hadd.congr (fun n => (hstat n).symm)
    (Filter.Eventually.of_forall (fun _ => by simp))

/-- A zero remainder is `o_p(1)`. -/
theorem zero_remainder_tendstoInMeasure
    {Ω : Type _} [MeasurableSpace Ω] {P : Measure Ω} [IsFiniteMeasure P] :
    TendstoInMeasure P (fun _n : ℕ => fun _ω : Ω => (0 : ℝ))
      Filter.atTop (fun _ : Ω => (0 : ℝ)) := by
  refine tendstoInMeasure_of_tendsto_ae (fun _ => aestronglyMeasurable_const) ?_
  exact Filter.Eventually.of_forall
    (fun _ω : Ω => tendsto_const_nhds)

/--
Almost-sure convergence of a measurable remainder to zero implies the
`o_p(1)` remainder condition used in the asymptotic-linearization theorem.
-/
theorem op_remainder_of_tendsto_ae
    {Ω : Type _} [MeasurableSpace Ω] {P : Measure Ω} [IsFiniteMeasure P]
    (remainder : ℕ → Ω → ℝ)
    (hmeas : ∀ n : ℕ, AEStronglyMeasurable (remainder n) P)
    (hrem_ae :
      ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => remainder n ω) Filter.atTop (nhds (0 : ℝ))) :
    TendstoInMeasure P remainder Filter.atTop (fun _ : Ω => (0 : ℝ)) := by
  exact tendstoInMeasure_of_tendsto_ae hmeas hrem_ae

/--
Lp convergence of a measurable remainder to zero implies the `o_p(1)`
remainder condition.  For an L2-style assumption, instantiate `p = 2`.
-/
theorem op_remainder_of_tendsto_eLpNorm
    {Ω : Type _} [MeasurableSpace Ω] {P : Measure Ω}
    (remainder : ℕ → Ω → ℝ) (p : ENNReal)
    (hp_ne_zero : p ≠ 0)
    (hmeas : ∀ n : ℕ, AEStronglyMeasurable (remainder n) P)
    (hLp :
      Filter.Tendsto
        (fun n : ℕ =>
          eLpNorm ((remainder n) - fun _ : Ω => (0 : ℝ)) p P)
        Filter.atTop (nhds 0)) :
    TendstoInMeasure P remainder Filter.atTop (fun _ : Ω => (0 : ℝ)) := by
  exact tendstoInMeasure_of_tendsto_eLpNorm
    hp_ne_zero hmeas aestronglyMeasurable_const hLp

/--
Mathlib's iid central limit theorem for normalized scalar scores with mean zero
and second moment one.
-/
theorem iid_score_clt
    {Ω Ωlim : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (X : ℕ → Ω → ℝ) (Z : Ωlim → ℝ)
    (hZ : HasLaw Z (gaussianReal 0 1) Plim)
    (h0 : ∫ x, X 0 x ∂P = 0)
    (h1 : ∫ x, (X 0 ^ 2) x ∂P = 1)
    (hindep : iIndepFun X P)
    (hident : ∀ i : ℕ, IdentDistrib (X i) (X 0) P P) :
    TendstoInDistribution
      (fun (n : ℕ) ω =>
        ((Real.sqrt (n : ℝ))⁻¹) * (∑ k ∈ Finset.range n, X k ω))
      Filter.atTop Z (fun _ : ℕ => P) Plim := by
  exact tendstoInDistribution_inv_sqrt_mul_sum hZ h0 h1 hindep hident

/--
Design-studentized CLT from an iid normalized score plus an `o_p(1)`
linearization remainder.
-/
theorem designStudentized_tendstoInDistribution_of_iid_score_and_op_remainder
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (X : ℕ → Ω → ℝ) (remainder : ℕ → Ω → ℝ)
    (u : α → ℝ) (Sigma : α → α → ℝ) (Z : Ωlim → ℝ)
    (hZ : HasLaw Z (gaussianReal 0 1) Plim)
    (h0 : ∫ x, X 0 x ∂P = 0)
    (h1 : ∫ x, (X 0 ^ 2) x ∂P = 1)
    (hindep : iIndepFun X P)
    (hident : ∀ i : ℕ, IdentDistrib (X i) (X 0) P P)
    (hrem :
      TendstoInMeasure P remainder Filter.atTop (fun _ : Ω => (0 : ℝ)))
    (hremmeas : ∀ n : ℕ, AEMeasurable (remainder n) P)
    (hlinearization :
      ∀ n : ℕ,
        (fun ω : Ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma) =ᵐ[P]
          fun ω : Ω =>
            ((Real.sqrt (n : ℝ))⁻¹) * (∑ k ∈ Finset.range n, X k ω) +
              remainder n ω) :
    TendstoInDistribution
      (fun n ω =>
        (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma)
      Filter.atTop Z (fun _ : ℕ => P) Plim := by
  let normalizedScore : ℕ → Ω → ℝ :=
    fun n ω => ((Real.sqrt (n : ℝ))⁻¹) * (∑ k ∈ Finset.range n, X k ω)
  have hCLT :
      TendstoInDistribution normalizedScore Filter.atTop Z (fun _ : ℕ => P) Plim := by
    exact iid_score_clt X Z hZ h0 h1 hindep hident
  exact tendstoInDistribution_of_clt_and_op_remainder
    (linearized := normalizedScore)
    (remainder := remainder)
    (statistic := fun n ω =>
      (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma)
    (Z := Z)
    hCLT hrem hremmeas hlinearization

/--
Concrete iid-score studentized asymptotic normality.  The theorem assumes the
linearized studentized statistic is almost surely the normalized iid score times
an inverse standard-error ratio, and that this ratio converges in probability to
one.
-/
theorem iid_studentized_score_tendstoInDistribution
    {Ω Ωlim : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (X : ℕ → Ω → ℝ) (Z : Ωlim → ℝ)
    (seInv studentized : ℕ → Ω → ℝ)
    (hZ : HasLaw Z (gaussianReal 0 1) Plim)
    (h0 : ∫ x, X 0 x ∂P = 0)
    (h1 : ∫ x, (X 0 ^ 2) x ∂P = 1)
    (hindep : iIndepFun X P)
    (hident : ∀ i : ℕ, IdentDistrib (X i) (X 0) P P)
    (hSE : TendstoInMeasure P seInv Filter.atTop (fun _ : Ω => (1 : ℝ)))
    (hSEmeas : ∀ n : ℕ, AEMeasurable (seInv n) P)
    (hstudentized :
      ∀ n : ℕ, studentized n =ᵐ[P]
        fun ω : Ω =>
          (((Real.sqrt (n : ℝ))⁻¹) * (∑ k ∈ Finset.range n, X k ω)) * seInv n ω) :
    TendstoInDistribution studentized Filter.atTop Z (fun _ : ℕ => P) Plim := by
  let normalized : ℕ → Ω → ℝ :=
    fun (n : ℕ) ω => ((Real.sqrt (n : ℝ))⁻¹) * (∑ k ∈ Finset.range n, X k ω)
  have hCLT : TendstoInDistribution normalized Filter.atTop Z (fun _ : ℕ => P) Plim := by
    exact iid_score_clt X Z hZ h0 h1 hindep hident
  exact studentized_tendstoInDistribution_of_clt_and_se_inv_consistency
    normalized seInv studentized Z hCLT hSE hSEmeas hstudentized

/--
Portmanteau interval-coverage step.  If `T_n` converges in distribution to `Z`
and the limiting law gives zero mass to the boundary of `[-z,z]`, then the
coverage probabilities of `T_n ∈ [-z,z]` converge to the limiting coverage.

The probabilities are expressed as `NNReal` values via `ProbabilityMeasure`;
equivalently they are the `toNNReal` coercions of the underlying measure
probabilities.
-/
theorem interval_coverage_tendsto_of_tendstoInDistribution
    {Ω Ωlim : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (T : ℕ → Ω → ℝ) (Z : Ωlim → ℝ) (z : ℝ)
    (hT : TendstoInDistribution T Filter.atTop Z (fun _ : ℕ => P) Plim)
    (hfrontier : (Plim.map Z) (frontier (Set.Icc (-z) z)) = 0) :
    Filter.Tendsto
      (fun n => ((P.map (T n)) (Set.Icc (-z) z)).toNNReal)
      Filter.atTop
      (nhds (((Plim.map Z) (Set.Icc (-z) z)).toNNReal)) := by
  have hnull : (lawPM Plim Z hT.aemeasurable_limit) (frontier (Set.Icc (-z) z)) = 0 := by
    simp [lawPM, hfrontier]
  simpa [lawPM] using
    (ProbabilityMeasure.tendsto_measure_of_null_frontier_of_tendsto
      (μs_lim := hT.tendsto) (E := Set.Icc (-z) z) hnull)

/--
Interval coverage convergence to a named nominal target.  The target is supplied
as an assumption about the limiting law, e.g. by choosing `z` as the appropriate
standard-normal critical value outside this theorem.
-/
theorem interval_coverage_tendsto_to_nominal_of_tendstoInDistribution
    {Ω Ωlim : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (T : ℕ → Ω → ℝ) (Z : Ωlim → ℝ) (z : ℝ) (target : NNReal)
    (hT : TendstoInDistribution T Filter.atTop Z (fun _ : ℕ => P) Plim)
    (hfrontier : (Plim.map Z) (frontier (Set.Icc (-z) z)) = 0)
    (hlimit : ((Plim.map Z) (Set.Icc (-z) z)).toNNReal = target) :
    Filter.Tendsto
      (fun n => ((P.map (T n)) (Set.Icc (-z) z)).toNNReal)
      Filter.atTop
      (nhds target) := by
  simpa [hlimit] using
    interval_coverage_tendsto_of_tendstoInDistribution T Z z hT hfrontier

/--
Concrete iid-score studentized interval coverage.  This combines the iid CLT,
Slutsky studentization, and portmanteau coverage steps.
-/
theorem iid_studentized_score_interval_coverage_tendsto_to_nominal
    {Ω Ωlim : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (X : ℕ → Ω → ℝ) (Z : Ωlim → ℝ)
    (seInv studentized : ℕ → Ω → ℝ)
    (z : ℝ) (target : NNReal)
    (hZ : HasLaw Z (gaussianReal 0 1) Plim)
    (h0 : ∫ x, X 0 x ∂P = 0)
    (h1 : ∫ x, (X 0 ^ 2) x ∂P = 1)
    (hindep : iIndepFun X P)
    (hident : ∀ i : ℕ, IdentDistrib (X i) (X 0) P P)
    (hSE : TendstoInMeasure P seInv Filter.atTop (fun _ : Ω => (1 : ℝ)))
    (hSEmeas : ∀ n : ℕ, AEMeasurable (seInv n) P)
    (hstudentized :
      ∀ n : ℕ, studentized n =ᵐ[P]
        fun ω : Ω =>
          (((Real.sqrt (n : ℝ))⁻¹) * (∑ k ∈ Finset.range n, X k ω)) * seInv n ω)
    (hfrontier : (Plim.map Z) (frontier (Set.Icc (-z) z)) = 0)
    (hlimit : ((Plim.map Z) (Set.Icc (-z) z)).toNNReal = target) :
    Filter.Tendsto
      (fun n => ((P.map (studentized n)) (Set.Icc (-z) z)).toNNReal)
      Filter.atTop
      (nhds target) := by
  have hT : TendstoInDistribution studentized Filter.atTop Z (fun _ : ℕ => P) Plim :=
    iid_studentized_score_tendstoInDistribution
      X Z seInv studentized hZ h0 h1 hindep hident hSE hSEmeas hstudentized
  exact interval_coverage_tendsto_to_nominal_of_tendstoInDistribution
    studentized Z z target hT hfrontier hlimit

/--
Pointwise algebra behind studentization: if both standard errors are nonzero,
studentizing by the feasible standard error is the same as design-studentizing
and multiplying by the inverse standard-error ratio `seDesign / seHat`.
-/
theorem studentized_eq_designStudentized_mul_se_ratio
    (betaHat beta seDesign seHat : ℝ)
    (hDesign : seDesign ≠ 0) (hHat : seHat ≠ 0) :
    (betaHat - beta) / seHat =
      ((betaHat - beta) / seDesign) * (seDesign / seHat) := by
  field_simp [hDesign, hHat]

/--
General fixed-effect/DiD CCV asymptotic validity theorem.

This theorem is intentionally stated after FE residualization and FWL
linearization.  `betaHat` may be the coefficient from any continuous-treatment
DiD/fixed-effect regression; `seDesign` is the infeasible design standard error
from the true residualized assignment covariance; and `seHat` is the feasible
CCV standard error, such as the design-covariance sandwich estimator.

No iid assumption and no binary-treatment restriction appears here.  The required
model-specific inputs are:

* a CLT for the design-studentized FE coefficient;
* consistency of the inverse standard-error ratio `seDesign / seHat`;
* nonzero standard errors eventually almost surely.

For a concrete continuous-treatment design, the hard work is proving those input
assumptions from the assignment model and the FE residualization/covariance
estimator.  Once they are available, Lean proves valid studentized inference.
-/
theorem continuous_did_fe_ccv_studentized_tendstoInDistribution
    {Ω Ωlim : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat seDesign seHat : ℕ → Ω → ℝ) (beta : ℝ) (Z : Ωlim → ℝ)
    (hCLT :
      TendstoInDistribution
        (fun n ω => (betaHat n ω - beta) / seDesign n ω)
        Filter.atTop Z (fun _ : ℕ => P) Plim)
    (hSE :
      TendstoInMeasure P
        (fun n ω => seDesign n ω / seHat n ω)
        Filter.atTop (fun _ : Ω => (1 : ℝ)))
    (hSEmeas :
      ∀ n : ℕ, AEMeasurable (fun ω : Ω => seDesign n ω / seHat n ω) P)
    (hNonzero :
      ∀ n : ℕ, ∀ᵐ ω ∂P, seDesign n ω ≠ 0 ∧ seHat n ω ≠ 0) :
    TendstoInDistribution
      (fun n ω => (betaHat n ω - beta) / seHat n ω)
      Filter.atTop Z (fun _ : ℕ => P) Plim := by
  exact studentized_tendstoInDistribution_of_clt_and_se_inv_consistency
    (normalized := fun n ω => (betaHat n ω - beta) / seDesign n ω)
    (seInv := fun n ω => seDesign n ω / seHat n ω)
    (studentized := fun n ω => (betaHat n ω - beta) / seHat n ω)
    (Z := Z)
    hCLT hSE hSEmeas
    (fun n => by
      filter_upwards [hNonzero n] with ω hω
      exact studentized_eq_designStudentized_mul_se_ratio
        (betaHat n ω) beta (seDesign n ω) (seHat n ω) hω.1 hω.2)

/-!
### Modular asymptotic assumption bundles

The structures below collect the minimal assumptions that remain
model-specific.  They are deliberately small: each one corresponds to one
standard proof obligation in an asymptotic CCV argument.
-/

/-- Distributional limit for a chosen linearized score process. -/
structure ScoreCLT
    {Ω Ωlim : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim]
    (P : Measure Ω) (Plim : Measure Ωlim)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (score : ℕ → Ω → ℝ) (Z : Ωlim → ℝ) : Prop where
  clt : TendstoInDistribution score Filter.atTop Z (fun _ : ℕ => P) Plim

/--
Asymptotic linearization of a statistic around a linearized score, with an
`o_p(1)` remainder.
-/
structure AsymptoticLinearization
    {Ω : Type _} [MeasurableSpace Ω]
    (P : Measure Ω)
    (statistic linearized remainder : ℕ → Ω → ℝ) : Prop where
  remainder_op :
    TendstoInMeasure P remainder Filter.atTop (fun _ : Ω => (0 : ℝ))
  remainder_meas : ∀ n : ℕ, AEMeasurable (remainder n) P
  linearization :
    ∀ n : ℕ, statistic n =ᵐ[P]
      fun ω : Ω => linearized n ω + remainder n ω

/-- Pointwise consistency of the feasible residual vector. -/
structure ResidualConsistency
    {Ω α : Type _} [MeasurableSpace Ω]
    (P : Measure Ω) (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ) : Prop where
  pointwise :
    ∀ i : α, ∀ᵐ ω ∂P,
      Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i))

/-- Pointwise consistency of the feasible covariance-kernel estimator. -/
structure KernelConsistency
    {Ω α : Type _} [MeasurableSpace Ω]
    (P : Measure Ω) (SigmaHat : ℕ → Ω → α → α → ℝ)
    (Sigma : α → α → ℝ) : Prop where
  pointwise :
    ∀ i j : α, ∀ᵐ ω ∂P,
      Filter.Tendsto (fun n : ℕ => SigmaHat n ω i j) Filter.atTop
        (nhds (Sigma i j))

/-- Generic standard-error ratio consistency assumptions. -/
structure SEConsistency
    {Ω : Type _} [MeasurableSpace Ω]
    (P : Measure Ω) (seDesign seHat : ℕ → Ω → ℝ) : Prop where
  ratio :
    TendstoInMeasure P
      (fun n ω => seDesign n ω / seHat n ω)
      Filter.atTop (fun _ : Ω => (1 : ℝ))
  ratio_meas :
    ∀ n : ℕ, AEMeasurable (fun ω : Ω => seDesign n ω / seHat n ω) P
  nonzero :
    ∀ n : ℕ, ∀ᵐ ω ∂P, seDesign n ω ≠ 0 ∧ seHat n ω ≠ 0

/--
The extra regularity needed to turn residual/kernel consistency into
standard-error consistency for the concrete `dcCCV` standard error.
-/
structure DCCVRegularity
    {Ω α : Type _} [MeasurableSpace Ω] [Fintype α]
    (P : Measure Ω) (S : ℝ)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ) : Prop where
  ratio_meas :
    ∀ n : ℕ,
      AEStronglyMeasurable
        (fun ω : Ω =>
          DesignCovariance.designSE S u Sigma /
            DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω)) P
  variance_pos : 0 < DesignCovariance.designVariance S u Sigma
  hat_nonzero :
    ∀ n : ℕ, ∀ᵐ ω ∂P,
      DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω) ≠ 0

/-- Exact feasible residuals and covariance kernel, as a stricter special case. -/
structure ExactDCCVFeasible
    {Ω α : Type _} [MeasurableSpace Ω]
    (P : Measure Ω) (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ) : Prop where
  residual_exact : ∀ᵐ ω ∂P, ∀ n : ℕ, ∀ i : α, uhat n ω i = u i
  kernel_exact :
    ∀ᵐ ω ∂P, ∀ n : ℕ, ∀ i j : α, SigmaHat n ω i j = Sigma i j

/--
A score CLT plus an asymptotic linearization gives the distributional limit of
the original statistic.
-/
theorem tendstoInDistribution_of_scoreCLT_and_linearization
    {Ω Ωlim : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (statistic linearized remainder : ℕ → Ω → ℝ) (Z : Ωlim → ℝ)
    (hScore : ScoreCLT P Plim linearized Z)
    (hLin : AsymptoticLinearization P statistic linearized remainder) :
    TendstoInDistribution statistic Filter.atTop Z (fun _ : ℕ => P) Plim := by
  exact tendstoInDistribution_of_clt_and_op_remainder
    linearized remainder statistic Z
    hScore.clt hLin.remainder_op hLin.remainder_meas hLin.linearization

/--
Generic feasible-studentized inference from the three modular inputs:
linearization, score CLT, and standard-error consistency.
-/
theorem continuous_did_fe_ccv_studentized_tendstoInDistribution_of_structures
    {Ω Ωlim : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat seDesign seHat : ℕ → Ω → ℝ) (beta : ℝ)
    (linearized remainder : ℕ → Ω → ℝ) (Z : Ωlim → ℝ)
    (hScore : ScoreCLT P Plim linearized Z)
    (hLin :
      AsymptoticLinearization P
        (fun n ω => (betaHat n ω - beta) / seDesign n ω)
        linearized remainder)
    (hSE : SEConsistency P seDesign seHat) :
    TendstoInDistribution
      (fun n ω => (betaHat n ω - beta) / seHat n ω)
      Filter.atTop Z (fun _ : ℕ => P) Plim := by
  have hCLT :
      TendstoInDistribution
        (fun n ω => (betaHat n ω - beta) / seDesign n ω)
        Filter.atTop Z (fun _ : ℕ => P) Plim :=
    tendstoInDistribution_of_scoreCLT_and_linearization
      (fun n ω => (betaHat n ω - beta) / seDesign n ω)
      linearized remainder Z hScore hLin
  exact continuous_did_fe_ccv_studentized_tendstoInDistribution
    betaHat seDesign seHat beta Z
    hCLT hSE.ratio hSE.ratio_meas hSE.nonzero

/-- Exact residuals imply the residual-consistency structure. -/
theorem residualConsistency_of_ae_exact
    {Ω α : Type _} [MeasurableSpace Ω]
    {P : Measure Ω}
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (hexact : ∀ᵐ ω ∂P, ∀ n : ℕ, ∀ i : α, uhat n ω i = u i) :
    ResidualConsistency P uhat u :=
  ⟨residual_tendsto_ae_of_ae_forall_eq uhat u hexact⟩

/-- Exact covariance kernels imply the kernel-consistency structure. -/
theorem kernelConsistency_of_ae_exact
    {Ω α : Type _} [MeasurableSpace Ω]
    {P : Measure Ω}
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (hexact :
      ∀ᵐ ω ∂P, ∀ n : ℕ, ∀ i j : α, SigmaHat n ω i j = Sigma i j) :
    KernelConsistency P SigmaHat Sigma :=
  ⟨kernel_tendsto_ae_of_ae_forall_eq SigmaHat Sigma hexact⟩

/-- The sample-kernel SLLN supplies the kernel-consistency structure. -/
theorem kernelConsistency_of_sampleKernelMean
    {Ω α : Type _} [MeasurableSpace Ω] [Fintype α] {P : Measure Ω}
    (Zsample : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (hint : ∀ i j : α, Integrable (fun ω : Ω => Zsample 0 ω i j) P)
    (hindep :
      ∀ i j : α, Pairwise fun k l => IndepFun (fun ω : Ω => Zsample k ω i j)
        (fun ω : Ω => Zsample l ω i j) P)
    (hident :
      ∀ i j : α, ∀ k : ℕ,
        IdentDistrib (fun ω : Ω => Zsample k ω i j)
          (fun ω : Ω => Zsample 0 ω i j) P P)
    (hmean : ∀ i j : α, (∫ ω, Zsample 0 ω i j ∂P) = Sigma i j) :
    KernelConsistency P (sampleKernelMean Zsample) Sigma :=
  ⟨sampleKernelMean_tendsto_ae Zsample Sigma hint hindep hident hmean⟩

/--
Residual/kernel consistency plus concrete `dcCCV` regularity gives the generic
standard-error consistency structure.
-/
theorem seConsistency_dcCCV_of_residual_kernel
    {Ω α : Type _} [MeasurableSpace Ω] [Fintype α]
    {P : Measure Ω} [IsFiniteMeasure P]
    (S : ℝ) (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (hRes : ResidualConsistency P uhat u)
    (hKer : KernelConsistency P SigmaHat Sigma)
    (hReg : DCCVRegularity P S uhat u SigmaHat Sigma) :
    SEConsistency P
      (fun _n _ω => DesignCovariance.designSE S u Sigma)
      (fun n ω => DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω)) := by
  have hDesignSE_ne : DesignCovariance.designSE S u Sigma ≠ 0 := by
    unfold DesignCovariance.designSE
    exact Real.sqrt_ne_zero'.mpr hReg.variance_pos
  refine ⟨?ratio, ?ratio_meas, ?nonzero⟩
  · exact dcSE_inv_ratio_tendstoInMeasure_one_of_ae_pointwise
      S uhat u SigmaHat Sigma hReg.ratio_meas
      hRes.pointwise hKer.pointwise hReg.variance_pos
  · intro n
    exact (hReg.ratio_meas n).aemeasurable
  · intro n
    filter_upwards [hReg.hat_nonzero n] with ω hhat
    exact ⟨hDesignSE_ne, hhat⟩

/-- Exact feasible objects supply `DCCVRegularity`. -/
theorem dcCCVRegularity_of_ae_exact
    {Ω α : Type _} [MeasurableSpace Ω] [Fintype α]
    {P : Measure Ω}
    (S : ℝ) (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (hExact : ExactDCCVFeasible P uhat u SigmaHat Sigma)
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma) :
    DCCVRegularity P S uhat u SigmaHat Sigma := by
  have hDesignSE_ne : DesignCovariance.designSE S u Sigma ≠ 0 := by
    unfold DesignCovariance.designSE
    exact Real.sqrt_ne_zero'.mpr hVpos
  refine ⟨?ratio_meas, hVpos, ?hat_nonzero⟩
  · intro n
    have hEq :
        (fun ω : Ω =>
          DesignCovariance.designSE S u Sigma /
            DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω)) =ᵐ[P]
          fun _ω : Ω => (1 : ℝ) := by
      filter_upwards [hExact.residual_exact, hExact.kernel_exact] with ω huω hSigmaω
      have hu : uhat n ω = u := by
        funext i
        exact huω n i
      have hSigma : SigmaHat n ω = Sigma := by
        funext i j
        exact hSigmaω n i j
      rw [DesignCovariance.dcSE_eq_designSE_of_exact
        S u (uhat n ω) Sigma (SigmaHat n ω) hu hSigma]
      exact div_self hDesignSE_ne
    exact aestronglyMeasurable_const.congr hEq.symm
  · intro n
    filter_upwards [hExact.residual_exact, hExact.kernel_exact] with ω huω hSigmaω
    have hu : uhat n ω = u := by
      funext i
      exact huω n i
    have hSigma : SigmaHat n ω = Sigma := by
      funext i j
      exact hSigmaω n i j
    rw [DesignCovariance.dcSE_eq_designSE_of_exact
      S u (uhat n ω) Sigma (SigmaHat n ω) hu hSigma]
    exact hDesignSE_ne

/--
Concrete `dcCCV` feasible asymptotic normality from the modular assumptions.
This is the main structure-based endpoint for the remaining asymptotic proof.
-/
theorem continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_structures
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (linearized remainder : ℕ → Ω → ℝ)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Z : Ωlim → ℝ)
    (hScore : ScoreCLT P Plim linearized Z)
    (hLin :
      AsymptoticLinearization P
        (fun n ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma)
        linearized remainder)
    (hRes : ResidualConsistency P uhat u)
    (hKer : KernelConsistency P SigmaHat Sigma)
    (hReg : DCCVRegularity P S uhat u SigmaHat Sigma) :
    TendstoInDistribution
      (fun n ω =>
        (betaHat n ω - beta) / DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
      Filter.atTop Z (fun _ : ℕ => P) Plim := by
  have hSE :
      SEConsistency P
        (fun _n _ω => DesignCovariance.designSE S u Sigma)
        (fun n ω => DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω)) :=
    seConsistency_dcCCV_of_residual_kernel
      S uhat u SigmaHat Sigma hRes hKer hReg
  exact continuous_did_fe_ccv_studentized_tendstoInDistribution_of_structures
    (betaHat := betaHat)
    (seDesign := fun _n _ω => DesignCovariance.designSE S u Sigma)
    (seHat := fun n ω => DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
    (beta := beta)
    (linearized := linearized)
    (remainder := remainder)
    (Z := Z)
    hScore hLin hSE

/--
Structure-based exact-feasible specialization.  Exact residuals and exact
covariance kernels are represented by `ExactDCCVFeasible`; this theorem keeps
that stricter route separate from sample-kernel or weak-dependence routes.
-/
theorem continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_structures_ae_exact
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (linearized remainder : ℕ → Ω → ℝ)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Z : Ωlim → ℝ)
    (hScore : ScoreCLT P Plim linearized Z)
    (hLin :
      AsymptoticLinearization P
        (fun n ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma)
        linearized remainder)
    (hExact : ExactDCCVFeasible P uhat u SigmaHat Sigma)
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma) :
    TendstoInDistribution
      (fun n ω =>
        (betaHat n ω - beta) / DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
      Filter.atTop Z (fun _ : ℕ => P) Plim := by
  exact continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_structures
    betaHat beta S linearized remainder uhat u SigmaHat Sigma Z
    hScore hLin
    (residualConsistency_of_ae_exact uhat u hExact.residual_exact)
    (kernelConsistency_of_ae_exact SigmaHat Sigma hExact.kernel_exact)
    (dcCCVRegularity_of_ae_exact S uhat u SigmaHat Sigma hExact hVpos)

/--
Coverage version of the structure-based `dcCCV` asymptotic theorem.
-/
theorem continuous_did_fe_dcCCV_interval_coverage_tendsto_to_nominal_of_structures
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (linearized remainder : ℕ → Ω → ℝ)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Z : Ωlim → ℝ) (z : ℝ) (target : NNReal)
    (hScore : ScoreCLT P Plim linearized Z)
    (hLin :
      AsymptoticLinearization P
        (fun n ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma)
        linearized remainder)
    (hRes : ResidualConsistency P uhat u)
    (hKer : KernelConsistency P SigmaHat Sigma)
    (hReg : DCCVRegularity P S uhat u SigmaHat Sigma)
    (hfrontier :
      (Plim.map Z) (frontier (Set.Icc (-z) z)) = 0)
    (hlimit :
      ((Plim.map Z) (Set.Icc (-z) z)).toNNReal = target) :
    Filter.Tendsto
      (fun n =>
        ((P.map
          (fun ω : Ω =>
            (betaHat n ω - beta) / DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω)))
          (Set.Icc (-z) z)).toNNReal)
      Filter.atTop
      (nhds target) := by
  have hT :
      TendstoInDistribution
        (fun n ω =>
          (betaHat n ω - beta) / DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
        Filter.atTop Z (fun _ : ℕ => P) Plim :=
    continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_structures
      betaHat beta S linearized remainder uhat u SigmaHat Sigma Z
      hScore hLin hRes hKer hReg
  exact interval_coverage_tendsto_to_nominal_of_tendstoInDistribution
    (fun n ω =>
      (betaHat n ω - beta) / DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
    Z z target hT hfrontier hlimit

/--
Coverage version of the structure-based exact-feasible specialization.
-/
theorem continuous_did_fe_dcCCV_interval_coverage_tendsto_to_nominal_of_structures_ae_exact
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (linearized remainder : ℕ → Ω → ℝ)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Z : Ωlim → ℝ) (z : ℝ) (target : NNReal)
    (hScore : ScoreCLT P Plim linearized Z)
    (hLin :
      AsymptoticLinearization P
        (fun n ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma)
        linearized remainder)
    (hExact : ExactDCCVFeasible P uhat u SigmaHat Sigma)
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma)
    (hfrontier :
      (Plim.map Z) (frontier (Set.Icc (-z) z)) = 0)
    (hlimit :
      ((Plim.map Z) (Set.Icc (-z) z)).toNNReal = target) :
    Filter.Tendsto
      (fun n =>
        ((P.map
          (fun ω : Ω =>
            (betaHat n ω - beta) / DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω)))
          (Set.Icc (-z) z)).toNNReal)
      Filter.atTop
      (nhds target) := by
  have hT :
      TendstoInDistribution
        (fun n ω =>
          (betaHat n ω - beta) / DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
        Filter.atTop Z (fun _ : ℕ => P) Plim :=
    continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_structures_ae_exact
      betaHat beta S linearized remainder uhat u SigmaHat Sigma Z
      hScore hLin hExact hVpos
  exact interval_coverage_tendsto_to_nominal_of_tendstoInDistribution
    (fun n ω =>
      (betaHat n ω - beta) / DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
    Z z target hT hfrontier hlimit

/--
Concrete fixed-effect/DiD asymptotic normality for the design-covariance CCV
standard error.  The remaining model-specific inputs are:

* a CLT for the infeasible design-studentized coefficient;
* almost-sure pointwise consistency of the residual vector and covariance-kernel
  estimator;
* positive limiting design variance and nonzero feasible standard errors.
-/
theorem continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_ae_pointwise
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Z : Ωlim → ℝ)
    (hCLT :
      TendstoInDistribution
        (fun n ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma)
        Filter.atTop Z (fun _ : ℕ => P) Plim)
    (hSEmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω =>
            DesignCovariance.designSE S u Sigma /
              DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω)) P)
    (hu :
      ∀ i : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)))
    (hSigma :
      ∀ i j : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => SigmaHat n ω i j) Filter.atTop (nhds (Sigma i j)))
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma)
    (hHatNonzero :
      ∀ n : ℕ, ∀ᵐ ω ∂P, DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω) ≠ 0) :
    TendstoInDistribution
      (fun n ω =>
        (betaHat n ω - beta) / DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
      Filter.atTop Z (fun _ : ℕ => P) Plim := by
  have hSE :
      TendstoInMeasure P
        (fun n ω =>
          DesignCovariance.designSE S u Sigma /
            DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
        Filter.atTop (fun _ : Ω => (1 : ℝ)) :=
    dcSE_inv_ratio_tendstoInMeasure_one_of_ae_pointwise
      S uhat u SigmaHat Sigma hSEmeas hu hSigma hVpos
  have hDesignSE_ne : DesignCovariance.designSE S u Sigma ≠ 0 := by
    unfold DesignCovariance.designSE
    exact Real.sqrt_ne_zero'.mpr hVpos
  exact continuous_did_fe_ccv_studentized_tendstoInDistribution
    (betaHat := betaHat)
    (seDesign := fun _ _ => DesignCovariance.designSE S u Sigma)
    (seHat := fun n ω => DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
    (beta := beta)
    (Z := Z)
    hCLT
    hSE
    (fun n => (hSEmeas n).aemeasurable)
    (fun n => by
      filter_upwards [hHatNonzero n] with ω hhat
      exact ⟨hDesignSE_ne, hhat⟩)

/--
Concrete fixed-effect/DiD asymptotic normality when the feasible residuals and
covariance kernel are exact almost surely.  Compared with the pointwise theorem,
this discharges residual consistency, kernel consistency, feasible-SE
measurability, and feasible-SE nonzero premises.
-/
theorem continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_ae_exact
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Z : Ωlim → ℝ)
    (hCLT :
      TendstoInDistribution
        (fun n ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma)
        Filter.atTop Z (fun _ : ℕ => P) Plim)
    (huexact : ∀ᵐ ω ∂P, ∀ n : ℕ, ∀ i : α, uhat n ω i = u i)
    (hSigmaexact :
      ∀ᵐ ω ∂P, ∀ n : ℕ, ∀ i j : α, SigmaHat n ω i j = Sigma i j)
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma) :
    TendstoInDistribution
      (fun n ω =>
        (betaHat n ω - beta) / DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
      Filter.atTop Z (fun _ : ℕ => P) Plim := by
  have hDesignSE_ne : DesignCovariance.designSE S u Sigma ≠ 0 := by
    unfold DesignCovariance.designSE
    exact Real.sqrt_ne_zero'.mpr hVpos
  have hSEmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω =>
            DesignCovariance.designSE S u Sigma /
              DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω)) P := by
    intro n
    have hEq :
        (fun ω : Ω =>
          DesignCovariance.designSE S u Sigma /
            DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω)) =ᵐ[P]
          fun _ω : Ω => (1 : ℝ) := by
      filter_upwards [huexact, hSigmaexact] with ω huω hSigmaω
      have hu : uhat n ω = u := by
        funext i
        exact huω n i
      have hSigma : SigmaHat n ω = Sigma := by
        funext i j
        exact hSigmaω n i j
      rw [DesignCovariance.dcSE_eq_designSE_of_exact
        S u (uhat n ω) Sigma (SigmaHat n ω) hu hSigma]
      exact div_self hDesignSE_ne
    exact aestronglyMeasurable_const.congr hEq.symm
  have hHatNonzero :
      ∀ n : ℕ, ∀ᵐ ω ∂P,
        DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω) ≠ 0 := by
    intro n
    filter_upwards [huexact, hSigmaexact] with ω huω hSigmaω
    have hu : uhat n ω = u := by
      funext i
      exact huω n i
    have hSigma : SigmaHat n ω = Sigma := by
      funext i j
      exact hSigmaω n i j
    rw [DesignCovariance.dcSE_eq_designSE_of_exact
      S u (uhat n ω) Sigma (SigmaHat n ω) hu hSigma]
    exact hDesignSE_ne
  exact continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_ae_pointwise
    betaHat beta S uhat u SigmaHat Sigma Z
    hCLT hSEmeas
    (residual_tendsto_ae_of_ae_forall_eq uhat u huexact)
    (kernel_tendsto_ae_of_ae_forall_eq SigmaHat Sigma hSigmaexact)
    hVpos hHatNonzero

/--
Iid-score specialization with exact feasible residuals and covariance kernel.
Mathlib supplies the scalar iid CLT; exact feasible objects discharge the
standard-error consistency side.
-/
theorem continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_iid_score_ae_exact
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (X : ℕ → Ω → ℝ)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Z : Ωlim → ℝ)
    (hZ : HasLaw Z (gaussianReal 0 1) Plim)
    (h0 : ∫ x, X 0 x ∂P = 0)
    (h1 : ∫ x, (X 0 ^ 2) x ∂P = 1)
    (hindepX : iIndepFun X P)
    (hidentX : ∀ i : ℕ, IdentDistrib (X i) (X 0) P P)
    (hDesignStudentized :
      ∀ n : ℕ,
        (fun ω : Ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma) =ᵐ[P]
          fun ω : Ω =>
            ((Real.sqrt (n : ℝ))⁻¹) * (∑ k ∈ Finset.range n, X k ω))
    (huexact : ∀ᵐ ω ∂P, ∀ n : ℕ, ∀ i : α, uhat n ω i = u i)
    (hSigmaexact :
      ∀ᵐ ω ∂P, ∀ n : ℕ, ∀ i j : α, SigmaHat n ω i j = Sigma i j)
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma) :
    TendstoInDistribution
      (fun n ω =>
        (betaHat n ω - beta) / DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
      Filter.atTop Z (fun _ : ℕ => P) Plim := by
  have hScoreCLT :
      TendstoInDistribution
        (fun (n : ℕ) ω =>
          ((Real.sqrt (n : ℝ))⁻¹) * (∑ k ∈ Finset.range n, X k ω))
        Filter.atTop Z (fun _ : ℕ => P) Plim :=
    iid_score_clt X Z hZ h0 h1 hindepX hidentX
  have hCLT :
      TendstoInDistribution
        (fun n ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma)
        Filter.atTop Z (fun _ : ℕ => P) Plim :=
    hScoreCLT.congr
      (fun n => (hDesignStudentized n).symm)
      (Filter.Eventually.of_forall (fun _ => by simp))
  exact continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_ae_exact
    betaHat beta S uhat u SigmaHat Sigma Z hCLT huexact hSigmaexact hVpos

/--
Iid-score/exact-feasible `dcCCV` asymptotic normality with the standard
`o_p(1)` linearization remainder.
-/
theorem continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_iid_score_ae_exact_and_op_remainder
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (X : ℕ → Ω → ℝ) (remainder : ℕ → Ω → ℝ)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Z : Ωlim → ℝ)
    (hZ : HasLaw Z (gaussianReal 0 1) Plim)
    (h0 : ∫ x, X 0 x ∂P = 0)
    (h1 : ∫ x, (X 0 ^ 2) x ∂P = 1)
    (hindepX : iIndepFun X P)
    (hidentX : ∀ i : ℕ, IdentDistrib (X i) (X 0) P P)
    (hrem :
      TendstoInMeasure P remainder Filter.atTop (fun _ : Ω => (0 : ℝ)))
    (hremmeas : ∀ n : ℕ, AEMeasurable (remainder n) P)
    (hlinearization :
      ∀ n : ℕ,
        (fun ω : Ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma) =ᵐ[P]
          fun ω : Ω =>
            ((Real.sqrt (n : ℝ))⁻¹) * (∑ k ∈ Finset.range n, X k ω) +
              remainder n ω)
    (huexact : ∀ᵐ ω ∂P, ∀ n : ℕ, ∀ i : α, uhat n ω i = u i)
    (hSigmaexact :
      ∀ᵐ ω ∂P, ∀ n : ℕ, ∀ i j : α, SigmaHat n ω i j = Sigma i j)
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma) :
    TendstoInDistribution
      (fun n ω =>
        (betaHat n ω - beta) / DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
      Filter.atTop Z (fun _ : ℕ => P) Plim := by
  have hCLT :
      TendstoInDistribution
        (fun n ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma)
        Filter.atTop Z (fun _ : ℕ => P) Plim :=
    designStudentized_tendstoInDistribution_of_iid_score_and_op_remainder
      betaHat beta S X remainder u Sigma Z
      hZ h0 h1 hindepX hidentX hrem hremmeas hlinearization
  exact continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_ae_exact
    betaHat beta S uhat u SigmaHat Sigma Z hCLT huexact hSigmaexact hVpos

/--
Interval-coverage version of the concrete design-covariance CCV asymptotic
theorem.
-/
theorem continuous_did_fe_dcCCV_interval_coverage_tendsto_to_nominal_of_ae_pointwise
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Z : Ωlim → ℝ) (z : ℝ) (target : NNReal)
    (hCLT :
      TendstoInDistribution
        (fun n ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma)
        Filter.atTop Z (fun _ : ℕ => P) Plim)
    (hSEmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω =>
            DesignCovariance.designSE S u Sigma /
              DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω)) P)
    (hu :
      ∀ i : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)))
    (hSigma :
      ∀ i j : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => SigmaHat n ω i j) Filter.atTop (nhds (Sigma i j)))
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma)
    (hHatNonzero :
      ∀ n : ℕ, ∀ᵐ ω ∂P, DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω) ≠ 0)
    (hfrontier :
      (Plim.map Z) (frontier (Set.Icc (-z) z)) = 0)
    (hlimit :
      ((Plim.map Z) (Set.Icc (-z) z)).toNNReal = target) :
    Filter.Tendsto
      (fun n =>
        ((P.map
          (fun ω : Ω =>
            (betaHat n ω - beta) / DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω)))
          (Set.Icc (-z) z)).toNNReal)
      Filter.atTop
      (nhds target) := by
  have hT :
      TendstoInDistribution
        (fun n ω =>
          (betaHat n ω - beta) / DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
        Filter.atTop Z (fun _ : ℕ => P) Plim :=
    continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_ae_pointwise
      betaHat beta S uhat u SigmaHat Sigma Z
      hCLT hSEmeas hu hSigma hVpos hHatNonzero
  exact interval_coverage_tendsto_to_nominal_of_tendstoInDistribution
    (fun n ω =>
      (betaHat n ω - beta) / DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
    Z z target hT hfrontier hlimit

/--
Interval coverage when the feasible residuals and covariance kernel are exact
almost surely.  This is the direct coverage-level corollary of the exact
studentized `dcCCV` asymptotic theorem.
-/
theorem continuous_did_fe_dcCCV_interval_coverage_tendsto_to_nominal_of_ae_exact
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (SigmaHat : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Z : Ωlim → ℝ) (z : ℝ) (target : NNReal)
    (hCLT :
      TendstoInDistribution
        (fun n ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma)
        Filter.atTop Z (fun _ : ℕ => P) Plim)
    (huexact : ∀ᵐ ω ∂P, ∀ n : ℕ, ∀ i : α, uhat n ω i = u i)
    (hSigmaexact :
      ∀ᵐ ω ∂P, ∀ n : ℕ, ∀ i j : α, SigmaHat n ω i j = Sigma i j)
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma)
    (hfrontier :
      (Plim.map Z) (frontier (Set.Icc (-z) z)) = 0)
    (hlimit :
      ((Plim.map Z) (Set.Icc (-z) z)).toNNReal = target) :
    Filter.Tendsto
      (fun n =>
        ((P.map
          (fun ω : Ω =>
            (betaHat n ω - beta) / DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω)))
          (Set.Icc (-z) z)).toNNReal)
      Filter.atTop
      (nhds target) := by
  have hT :
      TendstoInDistribution
        (fun n ω =>
          (betaHat n ω - beta) / DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
        Filter.atTop Z (fun _ : ℕ => P) Plim :=
    continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_ae_exact
      betaHat beta S uhat u SigmaHat Sigma Z
      hCLT huexact hSigmaexact hVpos
  exact interval_coverage_tendsto_to_nominal_of_tendstoInDistribution
    (fun n ω =>
      (betaHat n ω - beta) / DesignCovariance.dcSE S (uhat n ω) (SigmaHat n ω))
    Z z target hT hfrontier hlimit

/--
Studentized asymptotic normality when the feasible covariance kernel is an
entrywise sample mean whose entries satisfy the strong law.
-/
theorem continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_sampleKernelMean
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (Zsample : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Zlim : Ωlim → ℝ)
    (hCLT :
      TendstoInDistribution
        (fun n ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma)
        Filter.atTop Zlim (fun _ : ℕ => P) Plim)
    (hSEmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω =>
            DesignCovariance.designSE S u Sigma /
              DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω)) P)
    (hu :
      ∀ i : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)))
    (hint : ∀ i j : α, Integrable (fun ω : Ω => Zsample 0 ω i j) P)
    (hindep :
      ∀ i j : α, Pairwise fun k l => IndepFun (fun ω : Ω => Zsample k ω i j)
        (fun ω : Ω => Zsample l ω i j) P)
    (hident :
      ∀ i j : α, ∀ k : ℕ,
        IdentDistrib (fun ω : Ω => Zsample k ω i j)
          (fun ω : Ω => Zsample 0 ω i j) P P)
    (hmean : ∀ i j : α, (∫ ω, Zsample 0 ω i j ∂P) = Sigma i j)
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma)
    (hHatNonzero :
      ∀ n : ℕ, ∀ᵐ ω ∂P,
        DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω) ≠ 0) :
    TendstoInDistribution
      (fun n ω =>
        (betaHat n ω - beta) /
          DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω))
      Filter.atTop Zlim (fun _ : ℕ => P) Plim := by
  have hSE :
      TendstoInMeasure P
        (fun n ω =>
          DesignCovariance.designSE S u Sigma /
            DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω))
        Filter.atTop (fun _ : Ω => (1 : ℝ)) :=
    dcSE_inv_ratio_tendstoInMeasure_one_of_sampleKernelMean
      S uhat u Zsample Sigma hSEmeas hu hint hindep hident hmean hVpos
  have hDesignSE_ne : DesignCovariance.designSE S u Sigma ≠ 0 := by
    unfold DesignCovariance.designSE
    exact Real.sqrt_ne_zero'.mpr hVpos
  exact continuous_did_fe_ccv_studentized_tendstoInDistribution
    (betaHat := betaHat)
    (seDesign := fun _ _ => DesignCovariance.designSE S u Sigma)
    (seHat := fun n ω =>
      DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω))
    (beta := beta)
    (Z := Zlim)
    hCLT
    hSE
    (fun n => (hSEmeas n).aemeasurable)
    (fun n => by
      filter_upwards [hHatNonzero n] with ω hhat
      exact ⟨hDesignSE_ne, hhat⟩)

/--
Iid-score specialization of the design-covariance CCV asymptotic-normality
theorem.  Mathlib supplies the iid CLT; the covariance kernel is handled by the
sample-kernel strong law.
-/
theorem continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_iid_score_sampleKernelMean
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (X : ℕ → Ω → ℝ)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (Zsample : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Zlim : Ωlim → ℝ)
    (hZ : HasLaw Zlim (gaussianReal 0 1) Plim)
    (h0 : ∫ x, X 0 x ∂P = 0)
    (h1 : ∫ x, (X 0 ^ 2) x ∂P = 1)
    (hindepX : iIndepFun X P)
    (hidentX : ∀ i : ℕ, IdentDistrib (X i) (X 0) P P)
    (hDesignStudentized :
      ∀ n : ℕ,
        (fun ω : Ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma) =ᵐ[P]
          fun ω : Ω =>
            ((Real.sqrt (n : ℝ))⁻¹) * (∑ k ∈ Finset.range n, X k ω))
    (hSEmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω =>
            DesignCovariance.designSE S u Sigma /
              DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω)) P)
    (hu :
      ∀ i : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)))
    (hint : ∀ i j : α, Integrable (fun ω : Ω => Zsample 0 ω i j) P)
    (hindep :
      ∀ i j : α, Pairwise fun k l => IndepFun (fun ω : Ω => Zsample k ω i j)
        (fun ω : Ω => Zsample l ω i j) P)
    (hident :
      ∀ i j : α, ∀ k : ℕ,
        IdentDistrib (fun ω : Ω => Zsample k ω i j)
          (fun ω : Ω => Zsample 0 ω i j) P P)
    (hmean : ∀ i j : α, (∫ ω, Zsample 0 ω i j ∂P) = Sigma i j)
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma)
    (hHatNonzero :
      ∀ n : ℕ, ∀ᵐ ω ∂P,
        DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω) ≠ 0) :
    TendstoInDistribution
      (fun n ω =>
        (betaHat n ω - beta) /
          DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω))
      Filter.atTop Zlim (fun _ : ℕ => P) Plim := by
  have hScoreCLT :
      TendstoInDistribution
        (fun (n : ℕ) ω =>
          ((Real.sqrt (n : ℝ))⁻¹) * (∑ k ∈ Finset.range n, X k ω))
        Filter.atTop Zlim (fun _ : ℕ => P) Plim :=
    iid_score_clt X Zlim hZ h0 h1 hindepX hidentX
  have hCLT :
      TendstoInDistribution
        (fun n ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma)
        Filter.atTop Zlim (fun _ : ℕ => P) Plim :=
    hScoreCLT.congr
      (fun n => (hDesignStudentized n).symm)
      (Filter.Eventually.of_forall (fun _ => by simp))
  exact continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_sampleKernelMean
    betaHat beta S uhat u Zsample Sigma Zlim
    hCLT hSEmeas hu hint hindep hident hmean hVpos hHatNonzero

/--
Iid-score/sample-kernel design-covariance CCV asymptotic normality with the
standard `o_p(1)` linearization remainder.  This is the practical theorem for
the usual proof shape:

`(betaHat_n - beta) / seDesign = normalizedScore_n + remainder_n`,
with `remainder_n = o_p(1)`.
-/
theorem continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_iid_score_sampleKernelMean_and_op_remainder
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (X : ℕ → Ω → ℝ) (remainder : ℕ → Ω → ℝ)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (Zsample : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Zlim : Ωlim → ℝ)
    (hZ : HasLaw Zlim (gaussianReal 0 1) Plim)
    (h0 : ∫ x, X 0 x ∂P = 0)
    (h1 : ∫ x, (X 0 ^ 2) x ∂P = 1)
    (hindepX : iIndepFun X P)
    (hidentX : ∀ i : ℕ, IdentDistrib (X i) (X 0) P P)
    (hrem :
      TendstoInMeasure P remainder Filter.atTop (fun _ : Ω => (0 : ℝ)))
    (hremmeas : ∀ n : ℕ, AEMeasurable (remainder n) P)
    (hlinearization :
      ∀ n : ℕ,
        (fun ω : Ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma) =ᵐ[P]
          fun ω : Ω =>
            ((Real.sqrt (n : ℝ))⁻¹) * (∑ k ∈ Finset.range n, X k ω) +
              remainder n ω)
    (hSEmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω =>
            DesignCovariance.designSE S u Sigma /
              DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω)) P)
    (hu :
      ∀ i : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)))
    (hint : ∀ i j : α, Integrable (fun ω : Ω => Zsample 0 ω i j) P)
    (hindep :
      ∀ i j : α, Pairwise fun k l => IndepFun (fun ω : Ω => Zsample k ω i j)
        (fun ω : Ω => Zsample l ω i j) P)
    (hident :
      ∀ i j : α, ∀ k : ℕ,
        IdentDistrib (fun ω : Ω => Zsample k ω i j)
          (fun ω : Ω => Zsample 0 ω i j) P P)
    (hmean : ∀ i j : α, (∫ ω, Zsample 0 ω i j ∂P) = Sigma i j)
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma)
    (hHatNonzero :
      ∀ n : ℕ, ∀ᵐ ω ∂P,
        DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω) ≠ 0) :
    TendstoInDistribution
      (fun n ω =>
        (betaHat n ω - beta) /
          DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω))
      Filter.atTop Zlim (fun _ : ℕ => P) Plim := by
  have hCLT :
      TendstoInDistribution
        (fun n ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma)
        Filter.atTop Zlim (fun _ : ℕ => P) Plim :=
    designStudentized_tendstoInDistribution_of_iid_score_and_op_remainder
      betaHat beta S X remainder u Sigma Zlim
      hZ h0 h1 hindepX hidentX hrem hremmeas hlinearization
  exact continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_sampleKernelMean
    betaHat beta S uhat u Zsample Sigma Zlim
    hCLT hSEmeas hu hint hindep hident hmean hVpos hHatNonzero

/--
Same feasible `dcCCV` asymptotic normality theorem, but with the `o_p(1)`
remainder condition discharged by almost-sure convergence of the linearization
remainder.
-/
theorem continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_iid_score_sampleKernelMean_and_ae_remainder
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (X : ℕ → Ω → ℝ) (remainder : ℕ → Ω → ℝ)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (Zsample : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Zlim : Ωlim → ℝ)
    (hZ : HasLaw Zlim (gaussianReal 0 1) Plim)
    (h0 : ∫ x, X 0 x ∂P = 0)
    (h1 : ∫ x, (X 0 ^ 2) x ∂P = 1)
    (hindepX : iIndepFun X P)
    (hidentX : ∀ i : ℕ, IdentDistrib (X i) (X 0) P P)
    (hremmeas : ∀ n : ℕ, AEStronglyMeasurable (remainder n) P)
    (hrem_ae :
      ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => remainder n ω) Filter.atTop (nhds (0 : ℝ)))
    (hlinearization :
      ∀ n : ℕ,
        (fun ω : Ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma) =ᵐ[P]
          fun ω : Ω =>
            ((Real.sqrt (n : ℝ))⁻¹) * (∑ k ∈ Finset.range n, X k ω) +
              remainder n ω)
    (hSEmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω =>
            DesignCovariance.designSE S u Sigma /
              DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω)) P)
    (hu :
      ∀ i : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)))
    (hint : ∀ i j : α, Integrable (fun ω : Ω => Zsample 0 ω i j) P)
    (hindep :
      ∀ i j : α, Pairwise fun k l => IndepFun (fun ω : Ω => Zsample k ω i j)
        (fun ω : Ω => Zsample l ω i j) P)
    (hident :
      ∀ i j : α, ∀ k : ℕ,
        IdentDistrib (fun ω : Ω => Zsample k ω i j)
          (fun ω : Ω => Zsample 0 ω i j) P P)
    (hmean : ∀ i j : α, (∫ ω, Zsample 0 ω i j ∂P) = Sigma i j)
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma)
    (hHatNonzero :
      ∀ n : ℕ, ∀ᵐ ω ∂P,
        DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω) ≠ 0) :
    TendstoInDistribution
      (fun n ω =>
        (betaHat n ω - beta) /
          DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω))
      Filter.atTop Zlim (fun _ : ℕ => P) Plim := by
  exact
    continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_iid_score_sampleKernelMean_and_op_remainder
      betaHat beta S X remainder uhat u Zsample Sigma Zlim
      hZ h0 h1 hindepX hidentX
      (op_remainder_of_tendsto_ae remainder hremmeas hrem_ae)
      (fun n => (hremmeas n).aemeasurable)
      hlinearization hSEmeas hu hint hindep hident hmean hVpos hHatNonzero

/--
Same feasible `dcCCV` asymptotic normality theorem, but with the `o_p(1)`
remainder condition discharged by Lp convergence of the linearization remainder
to zero.  The L2 case is obtained by setting `p = 2`.
-/
theorem continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_iid_score_sampleKernelMean_and_eLpNorm_remainder
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (X : ℕ → Ω → ℝ) (remainder : ℕ → Ω → ℝ) (p : ENNReal)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (Zsample : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Zlim : Ωlim → ℝ)
    (hZ : HasLaw Zlim (gaussianReal 0 1) Plim)
    (h0 : ∫ x, X 0 x ∂P = 0)
    (h1 : ∫ x, (X 0 ^ 2) x ∂P = 1)
    (hindepX : iIndepFun X P)
    (hidentX : ∀ i : ℕ, IdentDistrib (X i) (X 0) P P)
    (hp_ne_zero : p ≠ 0)
    (hremmeas : ∀ n : ℕ, AEStronglyMeasurable (remainder n) P)
    (hLp :
      Filter.Tendsto
        (fun n : ℕ =>
          eLpNorm ((remainder n) - fun _ : Ω => (0 : ℝ)) p P)
        Filter.atTop (nhds 0))
    (hlinearization :
      ∀ n : ℕ,
        (fun ω : Ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma) =ᵐ[P]
          fun ω : Ω =>
            ((Real.sqrt (n : ℝ))⁻¹) * (∑ k ∈ Finset.range n, X k ω) +
              remainder n ω)
    (hSEmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω =>
            DesignCovariance.designSE S u Sigma /
              DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω)) P)
    (hu :
      ∀ i : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)))
    (hint : ∀ i j : α, Integrable (fun ω : Ω => Zsample 0 ω i j) P)
    (hindep :
      ∀ i j : α, Pairwise fun k l => IndepFun (fun ω : Ω => Zsample k ω i j)
        (fun ω : Ω => Zsample l ω i j) P)
    (hident :
      ∀ i j : α, ∀ k : ℕ,
        IdentDistrib (fun ω : Ω => Zsample k ω i j)
          (fun ω : Ω => Zsample 0 ω i j) P P)
    (hmean : ∀ i j : α, (∫ ω, Zsample 0 ω i j ∂P) = Sigma i j)
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma)
    (hHatNonzero :
      ∀ n : ℕ, ∀ᵐ ω ∂P,
        DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω) ≠ 0) :
    TendstoInDistribution
      (fun n ω =>
        (betaHat n ω - beta) /
          DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω))
      Filter.atTop Zlim (fun _ : ℕ => P) Plim := by
  exact
    continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_iid_score_sampleKernelMean_and_op_remainder
      betaHat beta S X remainder uhat u Zsample Sigma Zlim
      hZ h0 h1 hindepX hidentX
      (op_remainder_of_tendsto_eLpNorm remainder p hp_ne_zero hremmeas hLp)
      (fun n => (hremmeas n).aemeasurable)
      hlinearization hSEmeas hu hint hindep hident hmean hVpos hHatNonzero

/--
Interval coverage when the feasible covariance kernel is an entrywise sample
mean satisfying the strong law.
-/
theorem continuous_did_fe_dcCCV_interval_coverage_tendsto_to_nominal_of_sampleKernelMean
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (Zsample : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Zlim : Ωlim → ℝ) (z : ℝ) (target : NNReal)
    (hCLT :
      TendstoInDistribution
        (fun n ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma)
        Filter.atTop Zlim (fun _ : ℕ => P) Plim)
    (hSEmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω =>
            DesignCovariance.designSE S u Sigma /
              DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω)) P)
    (hu :
      ∀ i : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)))
    (hint : ∀ i j : α, Integrable (fun ω : Ω => Zsample 0 ω i j) P)
    (hindep :
      ∀ i j : α, Pairwise fun k l => IndepFun (fun ω : Ω => Zsample k ω i j)
        (fun ω : Ω => Zsample l ω i j) P)
    (hident :
      ∀ i j : α, ∀ k : ℕ,
        IdentDistrib (fun ω : Ω => Zsample k ω i j)
          (fun ω : Ω => Zsample 0 ω i j) P P)
    (hmean : ∀ i j : α, (∫ ω, Zsample 0 ω i j ∂P) = Sigma i j)
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma)
    (hHatNonzero :
      ∀ n : ℕ, ∀ᵐ ω ∂P,
        DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω) ≠ 0)
    (hfrontier :
      (Plim.map Zlim) (frontier (Set.Icc (-z) z)) = 0)
    (hlimit :
      ((Plim.map Zlim) (Set.Icc (-z) z)).toNNReal = target) :
    Filter.Tendsto
      (fun n =>
        ((P.map
          (fun ω : Ω =>
            (betaHat n ω - beta) /
              DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω)))
          (Set.Icc (-z) z)).toNNReal)
      Filter.atTop
      (nhds target) := by
  have hT :
      TendstoInDistribution
        (fun n ω =>
          (betaHat n ω - beta) /
            DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω))
        Filter.atTop Zlim (fun _ : ℕ => P) Plim :=
    continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_sampleKernelMean
      betaHat beta S uhat u Zsample Sigma Zlim
      hCLT hSEmeas hu hint hindep hident hmean hVpos hHatNonzero
  exact interval_coverage_tendsto_to_nominal_of_tendstoInDistribution
    (fun n ω =>
      (betaHat n ω - beta) /
        DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω))
    Zlim z target hT hfrontier hlimit

/--
Iid-score specialization of the design-covariance CCV interval-coverage
theorem.  Mathlib supplies the iid CLT, while the covariance-kernel estimator is
handled by the sample-kernel strong law.
-/
theorem continuous_did_fe_dcCCV_interval_coverage_tendsto_to_nominal_of_iid_score_sampleKernelMean
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (X : ℕ → Ω → ℝ)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (Zsample : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Zlim : Ωlim → ℝ) (z : ℝ) (target : NNReal)
    (hZ : HasLaw Zlim (gaussianReal 0 1) Plim)
    (h0 : ∫ x, X 0 x ∂P = 0)
    (h1 : ∫ x, (X 0 ^ 2) x ∂P = 1)
    (hindepX : iIndepFun X P)
    (hidentX : ∀ i : ℕ, IdentDistrib (X i) (X 0) P P)
    (hDesignStudentized :
      ∀ n : ℕ,
        (fun ω : Ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma) =ᵐ[P]
          fun ω : Ω =>
            ((Real.sqrt (n : ℝ))⁻¹) * (∑ k ∈ Finset.range n, X k ω))
    (hSEmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω =>
            DesignCovariance.designSE S u Sigma /
              DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω)) P)
    (hu :
      ∀ i : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)))
    (hint : ∀ i j : α, Integrable (fun ω : Ω => Zsample 0 ω i j) P)
    (hindep :
      ∀ i j : α, Pairwise fun k l => IndepFun (fun ω : Ω => Zsample k ω i j)
        (fun ω : Ω => Zsample l ω i j) P)
    (hident :
      ∀ i j : α, ∀ k : ℕ,
        IdentDistrib (fun ω : Ω => Zsample k ω i j)
          (fun ω : Ω => Zsample 0 ω i j) P P)
    (hmean : ∀ i j : α, (∫ ω, Zsample 0 ω i j ∂P) = Sigma i j)
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma)
    (hHatNonzero :
      ∀ n : ℕ, ∀ᵐ ω ∂P,
        DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω) ≠ 0)
    (hfrontier :
      (Plim.map Zlim) (frontier (Set.Icc (-z) z)) = 0)
    (hlimit :
      ((Plim.map Zlim) (Set.Icc (-z) z)).toNNReal = target) :
    Filter.Tendsto
      (fun n =>
        ((P.map
          (fun ω : Ω =>
            (betaHat n ω - beta) /
              DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω)))
          (Set.Icc (-z) z)).toNNReal)
      Filter.atTop
      (nhds target) := by
  have hT :
      TendstoInDistribution
        (fun n ω =>
          (betaHat n ω - beta) /
            DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω))
        Filter.atTop Zlim (fun _ : ℕ => P) Plim :=
    continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_iid_score_sampleKernelMean
      betaHat beta S X uhat u Zsample Sigma Zlim
      hZ h0 h1 hindepX hidentX hDesignStudentized
      hSEmeas hu hint hindep hident hmean hVpos hHatNonzero
  exact interval_coverage_tendsto_to_nominal_of_tendstoInDistribution
    (fun n ω =>
      (betaHat n ω - beta) /
        DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω))
    Zlim z target hT hfrontier hlimit

/--
Iid-score/sample-kernel design-covariance CCV interval coverage with an
`o_p(1)` linearization remainder.
-/
theorem continuous_did_fe_dcCCV_interval_coverage_tendsto_to_nominal_of_iid_score_sampleKernelMean_and_op_remainder
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (X : ℕ → Ω → ℝ) (remainder : ℕ → Ω → ℝ)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (Zsample : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Zlim : Ωlim → ℝ) (z : ℝ) (target : NNReal)
    (hZ : HasLaw Zlim (gaussianReal 0 1) Plim)
    (h0 : ∫ x, X 0 x ∂P = 0)
    (h1 : ∫ x, (X 0 ^ 2) x ∂P = 1)
    (hindepX : iIndepFun X P)
    (hidentX : ∀ i : ℕ, IdentDistrib (X i) (X 0) P P)
    (hrem :
      TendstoInMeasure P remainder Filter.atTop (fun _ : Ω => (0 : ℝ)))
    (hremmeas : ∀ n : ℕ, AEMeasurable (remainder n) P)
    (hlinearization :
      ∀ n : ℕ,
        (fun ω : Ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma) =ᵐ[P]
          fun ω : Ω =>
            ((Real.sqrt (n : ℝ))⁻¹) * (∑ k ∈ Finset.range n, X k ω) +
              remainder n ω)
    (hSEmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω =>
            DesignCovariance.designSE S u Sigma /
              DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω)) P)
    (hu :
      ∀ i : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)))
    (hint : ∀ i j : α, Integrable (fun ω : Ω => Zsample 0 ω i j) P)
    (hindep :
      ∀ i j : α, Pairwise fun k l => IndepFun (fun ω : Ω => Zsample k ω i j)
        (fun ω : Ω => Zsample l ω i j) P)
    (hident :
      ∀ i j : α, ∀ k : ℕ,
        IdentDistrib (fun ω : Ω => Zsample k ω i j)
          (fun ω : Ω => Zsample 0 ω i j) P P)
    (hmean : ∀ i j : α, (∫ ω, Zsample 0 ω i j ∂P) = Sigma i j)
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma)
    (hHatNonzero :
      ∀ n : ℕ, ∀ᵐ ω ∂P,
        DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω) ≠ 0)
    (hfrontier :
      (Plim.map Zlim) (frontier (Set.Icc (-z) z)) = 0)
    (hlimit :
      ((Plim.map Zlim) (Set.Icc (-z) z)).toNNReal = target) :
    Filter.Tendsto
      (fun n =>
        ((P.map
          (fun ω : Ω =>
            (betaHat n ω - beta) /
              DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω)))
          (Set.Icc (-z) z)).toNNReal)
      Filter.atTop
      (nhds target) := by
  have hT :
      TendstoInDistribution
        (fun n ω =>
          (betaHat n ω - beta) /
            DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω))
        Filter.atTop Zlim (fun _ : ℕ => P) Plim :=
    continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_iid_score_sampleKernelMean_and_op_remainder
      betaHat beta S X remainder uhat u Zsample Sigma Zlim
      hZ h0 h1 hindepX hidentX hrem hremmeas hlinearization
      hSEmeas hu hint hindep hident hmean hVpos hHatNonzero
  exact interval_coverage_tendsto_to_nominal_of_tendstoInDistribution
    (fun n ω =>
      (betaHat n ω - beta) /
        DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω))
    Zlim z target hT hfrontier hlimit

/--
Interval coverage under iid-score/sample-kernel assumptions when the
linearization remainder converges to zero almost surely.
-/
theorem continuous_did_fe_dcCCV_interval_coverage_tendsto_to_nominal_of_iid_score_sampleKernelMean_and_ae_remainder
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (X : ℕ → Ω → ℝ) (remainder : ℕ → Ω → ℝ)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (Zsample : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Zlim : Ωlim → ℝ) (z : ℝ) (target : NNReal)
    (hZ : HasLaw Zlim (gaussianReal 0 1) Plim)
    (h0 : ∫ x, X 0 x ∂P = 0)
    (h1 : ∫ x, (X 0 ^ 2) x ∂P = 1)
    (hindepX : iIndepFun X P)
    (hidentX : ∀ i : ℕ, IdentDistrib (X i) (X 0) P P)
    (hremmeas : ∀ n : ℕ, AEStronglyMeasurable (remainder n) P)
    (hrem_ae :
      ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => remainder n ω) Filter.atTop (nhds (0 : ℝ)))
    (hlinearization :
      ∀ n : ℕ,
        (fun ω : Ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma) =ᵐ[P]
          fun ω : Ω =>
            ((Real.sqrt (n : ℝ))⁻¹) * (∑ k ∈ Finset.range n, X k ω) +
              remainder n ω)
    (hSEmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω =>
            DesignCovariance.designSE S u Sigma /
              DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω)) P)
    (hu :
      ∀ i : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)))
    (hint : ∀ i j : α, Integrable (fun ω : Ω => Zsample 0 ω i j) P)
    (hindep :
      ∀ i j : α, Pairwise fun k l => IndepFun (fun ω : Ω => Zsample k ω i j)
        (fun ω : Ω => Zsample l ω i j) P)
    (hident :
      ∀ i j : α, ∀ k : ℕ,
        IdentDistrib (fun ω : Ω => Zsample k ω i j)
          (fun ω : Ω => Zsample 0 ω i j) P P)
    (hmean : ∀ i j : α, (∫ ω, Zsample 0 ω i j ∂P) = Sigma i j)
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma)
    (hHatNonzero :
      ∀ n : ℕ, ∀ᵐ ω ∂P,
        DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω) ≠ 0)
    (hfrontier :
      (Plim.map Zlim) (frontier (Set.Icc (-z) z)) = 0)
    (hlimit :
      ((Plim.map Zlim) (Set.Icc (-z) z)).toNNReal = target) :
    Filter.Tendsto
      (fun n =>
        ((P.map
          (fun ω : Ω =>
            (betaHat n ω - beta) /
              DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω)))
          (Set.Icc (-z) z)).toNNReal)
      Filter.atTop
      (nhds target) := by
  have hT :
      TendstoInDistribution
        (fun n ω =>
          (betaHat n ω - beta) /
            DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω))
        Filter.atTop Zlim (fun _ : ℕ => P) Plim :=
    continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_iid_score_sampleKernelMean_and_ae_remainder
      betaHat beta S X remainder uhat u Zsample Sigma Zlim
      hZ h0 h1 hindepX hidentX hremmeas hrem_ae hlinearization
      hSEmeas hu hint hindep hident hmean hVpos hHatNonzero
  exact interval_coverage_tendsto_to_nominal_of_tendstoInDistribution
    (fun n ω =>
      (betaHat n ω - beta) /
        DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω))
    Zlim z target hT hfrontier hlimit

/--
Interval coverage under iid-score/sample-kernel assumptions when the
linearization remainder converges to zero in Lp.  The L2 case is obtained by
setting `p = 2`.
-/
theorem continuous_did_fe_dcCCV_interval_coverage_tendsto_to_nominal_of_iid_score_sampleKernelMean_and_eLpNorm_remainder
    {Ω Ωlim α : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim] [Fintype α]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat : ℕ → Ω → ℝ) (beta : ℝ) (S : ℝ)
    (X : ℕ → Ω → ℝ) (remainder : ℕ → Ω → ℝ) (p : ENNReal)
    (uhat : ℕ → Ω → α → ℝ) (u : α → ℝ)
    (Zsample : ℕ → Ω → α → α → ℝ) (Sigma : α → α → ℝ)
    (Zlim : Ωlim → ℝ) (z : ℝ) (target : NNReal)
    (hZ : HasLaw Zlim (gaussianReal 0 1) Plim)
    (h0 : ∫ x, X 0 x ∂P = 0)
    (h1 : ∫ x, (X 0 ^ 2) x ∂P = 1)
    (hindepX : iIndepFun X P)
    (hidentX : ∀ i : ℕ, IdentDistrib (X i) (X 0) P P)
    (hp_ne_zero : p ≠ 0)
    (hremmeas : ∀ n : ℕ, AEStronglyMeasurable (remainder n) P)
    (hLp :
      Filter.Tendsto
        (fun n : ℕ =>
          eLpNorm ((remainder n) - fun _ : Ω => (0 : ℝ)) p P)
        Filter.atTop (nhds 0))
    (hlinearization :
      ∀ n : ℕ,
        (fun ω : Ω =>
          (betaHat n ω - beta) / DesignCovariance.designSE S u Sigma) =ᵐ[P]
          fun ω : Ω =>
            ((Real.sqrt (n : ℝ))⁻¹) * (∑ k ∈ Finset.range n, X k ω) +
              remainder n ω)
    (hSEmeas :
      ∀ n : ℕ,
        AEStronglyMeasurable
          (fun ω : Ω =>
            DesignCovariance.designSE S u Sigma /
              DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω)) P)
    (hu :
      ∀ i : α, ∀ᵐ ω ∂P,
        Filter.Tendsto (fun n : ℕ => uhat n ω i) Filter.atTop (nhds (u i)))
    (hint : ∀ i j : α, Integrable (fun ω : Ω => Zsample 0 ω i j) P)
    (hindep :
      ∀ i j : α, Pairwise fun k l => IndepFun (fun ω : Ω => Zsample k ω i j)
        (fun ω : Ω => Zsample l ω i j) P)
    (hident :
      ∀ i j : α, ∀ k : ℕ,
        IdentDistrib (fun ω : Ω => Zsample k ω i j)
          (fun ω : Ω => Zsample 0 ω i j) P P)
    (hmean : ∀ i j : α, (∫ ω, Zsample 0 ω i j ∂P) = Sigma i j)
    (hVpos : 0 < DesignCovariance.designVariance S u Sigma)
    (hHatNonzero :
      ∀ n : ℕ, ∀ᵐ ω ∂P,
        DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω) ≠ 0)
    (hfrontier :
      (Plim.map Zlim) (frontier (Set.Icc (-z) z)) = 0)
    (hlimit :
      ((Plim.map Zlim) (Set.Icc (-z) z)).toNNReal = target) :
    Filter.Tendsto
      (fun n =>
        ((P.map
          (fun ω : Ω =>
            (betaHat n ω - beta) /
              DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω)))
          (Set.Icc (-z) z)).toNNReal)
      Filter.atTop
      (nhds target) := by
  have hT :
      TendstoInDistribution
        (fun n ω =>
          (betaHat n ω - beta) /
            DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω))
        Filter.atTop Zlim (fun _ : ℕ => P) Plim :=
    continuous_did_fe_dcCCV_studentized_tendstoInDistribution_of_iid_score_sampleKernelMean_and_eLpNorm_remainder
      betaHat beta S X remainder p uhat u Zsample Sigma Zlim
      hZ h0 h1 hindepX hidentX hp_ne_zero hremmeas hLp hlinearization
      hSEmeas hu hint hindep hident hmean hVpos hHatNonzero
  exact interval_coverage_tendsto_to_nominal_of_tendstoInDistribution
    (fun n ω =>
      (betaHat n ω - beta) /
        DesignCovariance.dcSE S (uhat n ω) (sampleKernelMean Zsample n ω))
    Zlim z target hT hfrontier hlimit

/--
General fixed-effect/DiD CCV interval-coverage theorem.  It combines the
previous studentization theorem with the portmanteau interval-coverage step.

The statement applies to arbitrary real-valued continuous treatments after FE
residualization, provided the design-specific CLT and standard-error consistency
assumptions hold.
-/
theorem continuous_did_fe_ccv_interval_coverage_tendsto_to_nominal
    {Ω Ωlim : Type _} [MeasurableSpace Ω] [MeasurableSpace Ωlim]
    {P : Measure Ω} {Plim : Measure Ωlim}
    [IsProbabilityMeasure P] [IsProbabilityMeasure Plim]
    (betaHat seDesign seHat : ℕ → Ω → ℝ) (beta : ℝ)
    (Z : Ωlim → ℝ) (z : ℝ) (target : NNReal)
    (hCLT :
      TendstoInDistribution
        (fun n ω => (betaHat n ω - beta) / seDesign n ω)
        Filter.atTop Z (fun _ : ℕ => P) Plim)
    (hSE :
      TendstoInMeasure P
        (fun n ω => seDesign n ω / seHat n ω)
        Filter.atTop (fun _ : Ω => (1 : ℝ)))
    (hSEmeas :
      ∀ n : ℕ, AEMeasurable (fun ω : Ω => seDesign n ω / seHat n ω) P)
    (hNonzero :
      ∀ n : ℕ, ∀ᵐ ω ∂P, seDesign n ω ≠ 0 ∧ seHat n ω ≠ 0)
    (hfrontier :
      (Plim.map Z) (frontier (Set.Icc (-z) z)) = 0)
    (hlimit :
      ((Plim.map Z) (Set.Icc (-z) z)).toNNReal = target) :
    Filter.Tendsto
      (fun n =>
        ((P.map (fun ω : Ω => (betaHat n ω - beta) / seHat n ω))
          (Set.Icc (-z) z)).toNNReal)
      Filter.atTop
      (nhds target) := by
  have hT :
      TendstoInDistribution
        (fun n ω => (betaHat n ω - beta) / seHat n ω)
        Filter.atTop Z (fun _ : ℕ => P) Plim :=
    continuous_did_fe_ccv_studentized_tendstoInDistribution
      betaHat seDesign seHat beta Z hCLT hSE hSEmeas hNonzero
  exact interval_coverage_tendsto_to_nominal_of_tendstoInDistribution
    (fun n ω => (betaHat n ω - beta) / seHat n ω)
    Z z target hT hfrontier hlimit

/--
Defensible asymptotic optimality criterion: if a feasible variance estimator is
ratio-consistent for the design variance, its relative squared error converges to
zero.  Since squared error is nonnegative, zero is the smallest possible
asymptotic loss under this criterion.
-/
theorem relative_sq_error_tendsto_zero_of_ratio_consistency
    (vHat vDesign : ℕ → ℝ)
    (hRatio :
      Filter.Tendsto (fun n : ℕ => vHat n / vDesign n) Filter.atTop (nhds 1)) :
    Filter.Tendsto
      (fun n : ℕ => ((vHat n / vDesign n) - 1) ^ 2)
      Filter.atTop
      (nhds 0) := by
  have hsub :
      Filter.Tendsto (fun n : ℕ => vHat n / vDesign n - 1)
        Filter.atTop (nhds (1 - 1)) :=
    hRatio.sub tendsto_const_nhds
  have hsq :
      Filter.Tendsto (fun n : ℕ => (vHat n / vDesign n - 1) ^ 2)
        Filter.atTop (nhds ((1 - 1) ^ 2)) :=
    hsub.pow 2
  simpa using hsq

/-!
### Extension roadmap

This roadmap incorporates the design notes in `Lean/lean4_extensions.txt`.  The
items below are not unproved Lean declarations; they are scoped implementation
targets showing how future econometric extensions should attach to the current
formalization.

* Multi-way clustered standard errors.  Add two finite clustering dimensions,
  e.g. `G` and `T`, plus the intersection partition `G x T`.  The target theorem
  is the finite inclusion-exclusion identity
  `V_multi = V_G + V_T - V_(G x T)`, built from `ClusterLevel.clusterVar` and
  finite-sum algebra.
* Multi-dimensional continuous treatments.  Replace scalar FWL division by a
  matrix system over a finite coefficient index `K`:
  `(D' D)^{-1} D' u`.  This requires a matrix version of
  `ObservationLevel.betaHat_error` and a matrix sandwich version of
  `DesignCovariance.quad`.
* Spatial HAC / network weak dependence.  Replace hard cluster membership with a
  distance-weighted covariance kernel, using a metric or network distance and a
  bandwidth kernel.  The current `DesignCovariance.quad` can already consume the
  resulting finite kernel once the kernel entries are defined.
* Small-sample cluster corrections (`CR2` / `CR3`).  Formalize the projection
  matrix `H`, the annihilator `M = I - H`, and block corrections such as
  `M_gg^{-1/2}`.  The first Lean target should be positive semidefiniteness of
  the corrected sandwich quadratic form.
* Dyadic network clustering.  Specialize the coordinate type to `V x V` and
  define a covariance-support predicate saying two dyads are related when they
  share a vertex.  The finite design-covariance proof can then be reused with
  this restricted kernel.
* Instrumental variables / 2SLS sandwich.  Add an instrument matrix/vector and a
  first-stage projection `P_Z`.  The first target is the 2SLS estimation-error
  identity, analogous to `ObservationLevel.betaHat_error`, followed by the 2SLS
  sandwich variance.
* Growing covariance dimensions.  Replace the fixed finite coordinate type
  `alpha` with a sequence `alpha : Nat -> Type _` equipped with
  `[forall n, Fintype (alpha n)]`.  This requires triangular-array convergence,
  uniform kernel bounds, and nondegenerate limiting variance; it is outside the
  current fixed-`alpha` theorem.
* Serially correlated panel errors / panel HAC.  Model observations as a product
  of unit and time indices and define covariance entries by temporal distance.
  Once the autocovariance kernel is built, the existing quadratic-form algebra
  applies.
* Two-way fixed effects demeaning.  Add product-domain residualization over
  `I x T` and prove the double-demeaned treatment identity
  `D_it - D_i. - D_.t + D_..`.  The proof belongs near the existing
  `BinaryReduction` and FWL algebra.
* Stratified and block-randomized designs.  Add a stratum partition on the
  observation index and impose zero-sum covariance constraints within strata.
  The natural target is a comparison theorem showing how those constraints
  reduce the design covariance relative to the unrestricted kernel.
* Infinite state-space design inference.  The asymptotic section already uses
  measure-theoretic probability spaces.  A remaining extension is replacing the
  finite-state `FiniteDesign.expectation` and variance identities by integral
  versions over a general probability space.
* Nonlinear link functions / GMM / GLM.  Replace the OLS score by the gradient of
  a differentiable criterion, prove a Taylor expansion around the target
  parameter, and reuse the studentization/coverage layer with a Hessian-based
  sandwich.
* Heterogeneous treatment effects / random coefficients.  Allow the structural
  coefficient to vary by observation, `beta : alpha -> Real`, and prove that the
  OLS error equals the usual score term plus a heterogeneity-weighting term.
* Staggered continuous event studies.  Add a cohort/event-time index and define
  treatment as a time-varying dose switched on after adoption.  The expected
  variance target should decompose into cohort-specific design-covariance terms.
* Group-level continuous treatment / Moulton correction.  Add the structural
  assumption that treatment is constant within cluster, then prove the ratio of
  clustered to robust variance under constant cluster size and intraclass
  correlation assumptions.
* Endogenous cluster selection.  Make cluster membership random on the design
  probability space.  The target theorem should condition on cluster fixed
  effects or a conditional-independence assignment mechanism before applying the
  design covariance kernel.

The immediate proof gaps for the current continuous-treatment CCV result remain:
an appropriate non-iid or cluster/block CLT for the treatment-design score, and
consistency of the concrete covariance numerator for the intended DiD estimator.
-/

end Asymptotics

end ContinuousCCV

end
