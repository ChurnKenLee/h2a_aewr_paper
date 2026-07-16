To convert your document to standard, clean Markdown, I have replaced the non-standard block equations with valid LaTeX math blocks (using `$$`), corrected the formatting of lists, and applied consistent header levels.

---

# Formalizing the Continuous-Dose CCV Estimator

Here is the formal version. The punchline is:

$$
\widehat V_{\text{CCV},D} = \widehat\lambda_D \widehat V_{\text{cluster}} + (1-\widehat\lambda_D)\widehat V_{\text{robust}}
$$

where $D$ is the continuous AEWR-bite treatment after residualizing covariates and fixed effects, and

$$
\widehat\lambda_D = 1 - q \frac{\left(G^{-1}\sum_{g=1}^G \widehat\Omega_g\right)^2}{G^{-1}\sum_{g=1}^G \widehat\Omega_g^2}, \qquad \widehat\Omega_g = \frac{1}{n_g} \sum_{i\in g} \widetilde D_{gi}^2
$$

This is the clean continuous-dose analog of the AAIW fixed-effect CCV formula. AAIW’s published fixed-effect CCV is explicitly a convex combination of robust and clustered variance estimators, with the weight based on within-cluster treatment variation ($\bar W_g(1-\bar W_g)$). They also emphasize that the fixed-effect estimator and this variance estimator require within-cluster treatment variation.

## 1. Setup

Let $g=1,\dots,G$ index counties or county-like clusters, and let $i=1,\dots,n_g$ index observations within county $g$. Let $Y_{gi}$ be the outcome, and let $D_{gi}$ be the continuous AEWR treatment intensity, for example:

$$
D_{gi} = \max\{0, \log(\text{AEWR}_{gi})-\log(w^0_{gi})\}
$$

Let $Z_{gi}$ collect all controls and fixed effects other than $D_{gi}$. The regression is:

$$
Y_{gi} = \beta D_{gi} + Z_{gi}'\theta + u_{gi}
$$

Stack the data and write $Y = D\beta + Z\theta + u$. Let $M_Z = I - Z(Z'Z)^{-1}Z'$ be the residual-maker. Define:

$$
\widetilde Y = M_ZY, \quad \widetilde D = M_ZD, \quad \widetilde u = M_Zu
$$

By Frisch–Waugh–Lovell:

$$
\widehat\beta = \frac{\widetilde D'\widetilde Y}{\widetilde D'\widetilde D}
$$

Let $S = \sum_{g=1}^G \sum_{i=1}^{n_g} \widetilde D_{gi}^2$. Then:

$$
\widehat\beta - \beta = S^{-1} \sum_{g=1}^G \sum_{i=1}^{n_g} \widetilde D_{gi}\widetilde u_{gi}
$$

Define the observation-level score $\psi_{gi} = \widetilde D_{gi}\widetilde u_{gi}$, and the county-level score $\Psi_g = \sum_{i=1}^{n_g}\psi_{gi}$. Thus:

$$
\widehat\beta - \beta = S^{-1}\sum_{g=1}^G \Psi_g
$$

## 2. Robust and clustered variance estimators

The Eicker–Huber–White variance estimator is:

$$
\widehat V_{\text{robust}} = S^{-2} \sum_{g=1}^G \sum_{i=1}^{n_g} \widehat\psi_{gi}^2
$$

The usual Liang–Zeger county-clustered estimator is:

$$
\widehat V_{\text{cluster}} = S^{-2} \sum_{g=1}^G \left( \sum_{i=1}^{n_g}\widehat\psi_{gi} \right)^2
$$

Expanding the cluster estimator shows the difference:

$$
\widehat V_{\text{cluster}} - \widehat V_{\text{robust}} = S^{-2} \sum_{g=1}^G \sum_{i \neq j} \widehat\psi_{gi}\widehat\psi_{gj}
$$

## 3. Continuous-treatment analog of the AAIW fixed-effect weight

For continuous treatment $D_{gi}$, the exact analog of $\bar W_g(1-\bar W_g)$ is the within-county residualized treatment variance $\widehat\Omega_g = \frac{1}{n_g} \sum_{i=1}^{n_g} \widetilde D_{gi}^2$.

Defining $\widehat\lambda_D$ as above, the continuous-treatment CCV estimator is:

$$
\widehat V_{\text{CCV},D} = \widehat\lambda_D \widehat V_{\text{cluster}} + (1-\widehat\lambda_D) \widehat V_{\text{robust}}
$$

If panel lengths $n_g$ are highly unequal and your estimand is unit-weighted, replace $\widehat\Omega_g$ with $\widehat S_g = \sum_{i=1}^{n_g}\widetilde D_{gi}^2$ and use the corresponding unit-weighted $\widehat\lambda_D^{\text{unit}}$.

## 4. Proof that this reduces to AAIW in the binary case

Setting $D_{gi} = W_{gi} \in \{0,1\}$, we have $\widetilde D_{gi} = W_{gi} - \bar W_g$. Since $W_{gi}^2 = W_{gi}$ for binary variables:

$$
\widehat\Omega_g = \frac{1}{n_g} \sum_i (W_{gi} - \bar W_g)^2 = \bar W_g(1 - \bar W_g)
$$

Plugging this into the continuous formula yields the AAIW fixed-effect CCV weight, confirming it is a strict extension.

## 5. Proof that the CCV estimator is a convex combination

Assuming $0 \leq q \leq 1$ and non-zero variation, Cauchy–Schwarz implies:

$$
\left( G^{-1}\sum_g \widehat\Omega_g \right)^2 \leq G^{-1}\sum_g \widehat\Omega_g^2
$$

It follows that $0 \leq \widehat\lambda_D \leq 1$, confirming that $\widehat V_{\text{CCV},D}$ is a convex combination of the clustered and robust variance estimators.

## 6. Consistency argument

As $G \to \infty$ and $\max_g \frac{S_g}{S} \to 0$, under standard regularity conditions and the assumption that the residualized treatment has non-zero limiting variation:

$$
\widehat\lambda_D \to_p \lambda_D = 1 - q \frac{\Omega_1^2}{\Omega_2}
$$

By Slutsky’s theorem, $\widehat V_{\text{CCV},D}$ consistently estimates the corresponding continuous-dose CCV asymptotic variance.

## 7. Limitations

This formula is exact under a continuous-dose assignment model where the cluster-level identifying-variation object $\Omega_g$ is sufficient (e.g., a common-shape location-scale assignment process). For arbitrary continuous treatments, higher moments and slope heterogeneity may affect the variance.
