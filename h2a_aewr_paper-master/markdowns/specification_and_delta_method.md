# Reading the Mundlak–Chamberlain specification and its delta-method effects

This note is a guided reading of the implemented multilevel
Mundlak–Chamberlain design. Its two goals are to make the regression
specification visually legible and to show exactly how the fitted regression
is converted into average treatment effects and standard errors.

The central idea is:

\[
\underbrace{\text{estimate a rich conditional mean}}_{\text{regression step}}
\quad\Longrightarrow\quad
\overbrace{\text{change one AEWR-growth coordinate}}^{\text{counterfactual step}}
\quad\Longrightarrow\quad
\underbrace{\text{average fitted changes}}_{\text{causal estimand}}
\quad\Longrightarrow\quad
\overbrace{\text{propagate coefficient uncertainty}}^{\text{delta method}}.
\]

The primary model is Gaussian with an identity link. Consequently, the
reported effects are linear combinations of the estimated coefficients once
the observed doses, moderators, sample, and counterfactual change are fixed.
The delta-method variance calculation is therefore exact with respect to the
estimated coefficient vector for this implemented model. It would become a
first-order approximation under a nonlinear link.

## 1. What varies, and at what level?

Counties are indexed by \(i\), calendar years by \(t\), and AEWR regions by
\(r\). County \(i\) belongs to AEWR region \(r(i)\). The annual treatment is

\[
\underbrace{W_{r,t}}_{\substack{\text{AEWR-growth dose}\\
\text{in log percentage points}}}
=
\overbrace{100}^{\text{percentage-point scale}}
\left[
  \underbrace{\log(\operatorname{AEWR}_{r,t})}_{\text{current log AEWR}}
  -
  \underbrace{\log(\operatorname{AEWR}_{r,t-1})}_{\text{lagged log AEWR}}
\right].
\]

Thus treatment varies at the **AEWR-region-by-year** level even though the
outcome is observed at the **county-by-year** level. Thousands of counties do
not create thousands of independent AEWR policy paths: there are 17 regional
paths.

The three causal horizons are

\[
\underbrace{W_{r(i),t}}_{p=0:\ \text{current growth}},
\qquad
\overbrace{W_{r(i),t-1}}^{p=1:\ \text{one-year lag}},
\qquad
\underbrace{W_{r(i),t-2}}_{p=2:\ \text{two-year lag}}.
\]

The primary treatment basis is quadratic:

\[
\underbrace{b_1(w)=w}_{\text{linear response}}
\qquad\text{and}\qquad
\overbrace{b_2(w)=\frac{w^2}{25}}^{\substack{\text{curvature}\\
\text{scaled around a 5-point change}}}.
\]

Dividing the quadratic by \(25=5^2\) improves numerical conditioning. It does
**not** scale an outcome. In particular, the employer outcome remains the raw
balanced-linkage count:

\[
\underbrace{Y_{it}^{\text{employers}}}_{\text{modeled outcome}}
=
\overbrace{\operatorname{nbr\_employers\_balanced\_start\_year}_{it}}
^{\text{raw integer count}},
\qquad
\underbrace{\frac{Y_{it}^{\text{employers}}}
{\operatorname{farm\ employment}_{i,2011}}}_{\text{not used}}.
\]

## 2. A brace map of the complete conditional mean

Let \(d_{ts}=1[t=s]\), let \(s_0=2013\) be the reference year, and let \(Z_i\)
be the standardized pre-policy AEWR bite. For outcome \(j\), the implemented
primary conditional mean is

\[
\begin{aligned}
E[Y_{it}^{j}\mid\mathcal H_i]
&=
\underbrace{\theta_t}_{\text{unrestricted year effect}}
+
\overbrace{\alpha_{r(i)}}^{\text{AEWR-region intercept}}
+
\underbrace{H_i'\lambda}_{\substack{\text{Chamberlain intercept projection}\\
\text{from predetermined histories}}}
\\[3pt]
&\quad+
\overbrace{
\sum_{s\ne s_0}\psi_s Z_i d_{ts}
}^{\substack{\text{baseline bite}\\\times\ \text{calendar-year trend}}}
\\[3pt]
&\quad+
\underbrace{
\sum_{p=0}^{2}\sum_{m=1}^{2}\sum_{s=2013}^{2022}
\left[
  \overbrace{\beta_{pms}^{(0)}}^{\text{effect at }Z_i=0}
  +
  \underbrace{\beta_{pms}^{(1)}Z_i}_{\text{effect heterogeneity}}
\right]
\overbrace{b_m(W_{r(i),t-p})}^{\text{dose basis}}
\underbrace{d_{ts}}_{\text{year cell}}
}_{\textbf{causal current-and-lag dose-response block}}
\\[3pt]
&\quad+
\overbrace{
\sum_{s\ne s_0}\sum_{h\in\mathcal A_s}
\left[
  \underbrace{\rho_{sh}^{(0)}}_{\text{history main effect}}
  +
  \overbrace{\rho_{sh}^{(1)}Z_i}^{\text{history}\times\text{bite}}
\right]
\underbrace{\widetilde W_{r(i),h}}_{\text{non-focal AEWR history}}
d_{ts}
}^{\substack{\text{Chamberlain treatment-history projection}\\
\text{held fixed in a focal counterfactual}}}
\\[3pt]
&\quad+
\underbrace{
\sum_{\ell\in\{C,M,S\}}\sum_{k=1}^{K}\sum_{s\ne s_0}
\left[
  \overbrace{\delta_{k\ell s}^{(0)}}^{\text{baseline trend}}
  +
  \underbrace{\delta_{k\ell s}^{(1)}Z_i}_{\text{trend}\times\text{bite}}
\right]
\overbrace{X_{ik}^{\ell}}^{\substack{\text{county, market, or state}\\
\text{baseline component}}}
d_{ts}
}_{\textbf{hierarchical predetermined-trend projection}}.
\end{aligned}
\tag{1}
\]

This looks forbidding because it protects the causal block from several forms
of selection. Reading one line at a time makes its role clearer.

### Fixed time and place

\[
\underbrace{\theta_t}_{\substack{\text{common shocks}\\
\text{in calendar year }t}}
+
\overbrace{\alpha_{r(i)}}^{\substack{\text{persistent differences}\\
\text{across AEWR regions}}}.
\]

These terms absorb common calendar shocks and time-invariant differences
across the 17 treatment-assignment regions.

### Predetermined latent type

\[
\underbrace{H_i'\lambda}_{\text{county latent-type projection}}
=
\overbrace{
\text{separate 2008, 2009, and 2010 histories}
}^{\text{Chamberlain components}}
+
\underbrace{
\text{pre-period means and trends}
}_{\text{Mundlak summaries}}.
\]

The hierarchy decomposes baseline variables into county, market, state, and
region pieces. This permits, for example, a county to differ from its local
market while the local market differs from its state.

### Heterogeneous causal response

For a particular horizon \(p\), polynomial degree \(m\), and outcome year
\(s\), the causal coefficient is

\[
\underbrace{
\beta_{pms}^{(0)}+\beta_{pms}^{(1)}Z_i
}_{\substack{\text{county-specific response}\\
\text{indexed by predetermined AEWR bite}}}
=
\overbrace{\beta_{pms}^{(0)}}^{\text{response at mean bite}}
+
\underbrace{\beta_{pms}^{(1)}Z_i}_{\text{bite-gradient in response}}.
\]

The response may therefore differ by horizon, calendar year, dose level, and
predetermined local binding intensity.

## 3. Why focal treatment histories are excluded

The nuisance history set for outcome year \(s\) is

\[
\underbrace{\mathcal A_s}_{\text{allowed nuisance histories}}
=
\overbrace{\{2011,\ldots,2022\}}^{\text{complete treatment history}}
\setminus
\underbrace{\{s,s-1,s-2\}}_{\text{current and two focal lags}}.
\tag{2}
\]

This exclusion is an identification requirement, not a convenience. Within
year \(s\), \(d_{ts}\) forces \(t=s\), so

\[
\underbrace{W_{r(i),t-p}d_{ts}}_{\text{causal regressor}}
=
\overbrace{W_{r(i),s-p}d_{ts}}^{\text{same numerical column}}
=
\underbrace{
\left.\widetilde W_{r(i),h}d_{ts}\right|_{h=s-p}
}_{\substack{\text{focal nuisance-history regressor}\\
\text{up to standardization and a year constant}}}.
\]

Including both would ask the data to distinguish two coefficients multiplying
the same column. The same problem occurs after interacting each copy with
\(Z_i\):

\[
\underbrace{W_{r(i),t-p}Z_i d_{ts}}_{\text{causal heterogeneity}}
=
\overbrace{
\left.\widetilde W_{r(i),h}Z_i d_{ts}\right|_{h=s-p}
}^{\text{duplicated nuisance heterogeneity}}.
\]

No estimator, sample-size increase, covariance correction, or delta method
can separately identify coefficients on duplicated columns.

The 17-region assignment support also disciplines polynomial complexity:

\[
\underbrace{1}_{\text{intercept}}
+
\overbrace{3\times2}^{\substack{\text{three horizons}\\\times\text{two powers}}}
+
\underbrace{9}_{\text{non-focal histories}}
=
\overbrace{16}^{\text{region-level columns per year}}
<
\underbrace{17}_{\text{AEWR regions}}.
\]

A cellwise cubic would instead exceed the available region-level rank once
the non-focal history projection is retained.

## 4. What is held fixed in a treatment contrast?

Choose a horizon \(p\) and change its dose by \(\delta\) log percentage points.
Here, the unsubscripted \(\delta\) denotes a treatment change. It is unrelated
to the nuisance-trend coefficients \(\delta_{k\ell s}^{(0)}\) and
\(\delta_{k\ell s}^{(1)}\) in equation (1):

\[
\underbrace{W_{r(i),t-p}}_{\text{observed focal coordinate}}
\quad\longrightarrow\quad
\overbrace{W_{r(i),t-p}+\delta}^{\text{counterfactual focal coordinate}}.
\]

The contrast changes the focal dose basis and its \(Z_i\) interaction while
holding fixed

\[
\underbrace{H_i}_{\text{predetermined type}},\quad
\overbrace{Z_i}^{\text{baseline moderator}},\quad
\underbrace{\widetilde W_{r,h},\ h\in\mathcal A_s}_{\text{non-focal history}},
\quad
\overbrace{X_{ik}^{\ell}}^{\text{baseline hierarchy}},\quad
\underbrace{\theta_t,\alpha_{r(i)}}_{\text{time and region}}.
\]

This is a **coordinate intervention** on a continuous treatment history. It
is not a claim that all elements of the historical AEWR path change together.

## 5. From the quadratic regression to a unit-level effect

For brevity, fix \(p\) and \(s\), write the observed focal dose as \(w\), and
write the county's moderator as \(z\). The relevant fitted causal response is

\[
\underbrace{
\left(\beta_{p,1,s}^{(0)}+\beta_{p,1,s}^{(1)}z\right)w
}_{\text{linear dose contribution}}
+
\overbrace{
\left(\beta_{p,2,s}^{(0)}+\beta_{p,2,s}^{(1)}z\right)
\frac{w^2}{25}
}^{\text{quadratic dose contribution}}.
\]

### Finite treatment change

Replacing \(w\) with \(w+\delta\) and subtracting the observed prediction gives

\[
\begin{aligned}
\Delta_{p,s}(w,z;\delta)
&=
\underbrace{
\delta
\left(\beta_{p,1,s}^{(0)}+\beta_{p,1,s}^{(1)}z\right)
}_{\text{linear-basis change}}
\\[3pt]
&\quad+
\overbrace{
\frac{2w\delta+\delta^2}{25}
\left(\beta_{p,2,s}^{(0)}+\beta_{p,2,s}^{(1)}z\right)
}^{\text{quadratic-basis change}}.
\end{aligned}
\tag{3}
\]

The factor \(2w\delta+\delta^2\) comes from

\[
\underbrace{(w+\delta)^2}_{\text{counterfactual square}}
-
\overbrace{w^2}^{\text{observed square}}
=
\underbrace{2w\delta+\delta^2}_{\text{change in the square}}.
\]

### Marginal treatment effect

Taking the derivative with respect to the focal dose gives

\[
\operatorname{ME}_{p,s}(w,z)
=
\underbrace{
\beta_{p,1,s}^{(0)}+\beta_{p,1,s}^{(1)}z
}_{\text{linear-basis slope}}
+
\overbrace{
\frac{2w}{25}
\left(\beta_{p,2,s}^{(0)}+\beta_{p,2,s}^{(1)}z\right)
}^{\text{dose-dependent curvature slope}}.
\tag{4}
\]

The marginal effect changes with \(w\), \(z\), \(s\), and \(p\). A single raw
regression coefficient is therefore not the design's average treatment
effect.

## 6. Averaging unit-level effects

Let \(\mathcal S\) denote the requested estimation subset and let
\(\omega_{it}\ge0\) be its standardization weight. The average finite treatment
effect is

\[
\widehat{\operatorname{ATE}}_{p}(\delta)
=
\underbrace{
\frac{1}{\sum_{(i,t)\in\mathcal S}\omega_{it}}
}_{\text{normalization}}
\sum_{(i,t)\in\mathcal S}
\overbrace{\omega_{it}}^{\text{standardization weight}}
\underbrace{
\widehat\Delta_{p,t}
\left(W_{r(i),t-p},Z_i;\delta\right)
}_{\text{county-year fitted treatment effect}}.
\tag{5}
\]

Here “ATE” means the average effect of a common \(\delta\)-point shift in one
continuous-treatment coordinate. It is not a binary treated-versus-control
contrast.

The corresponding average marginal effect is

\[
\widehat{\operatorname{AME}}_{p}
=
\overbrace{
\frac{1}{\sum_{(i,t)\in\mathcal S}\omega_{it}}
}^{\text{average rather than total}}
\sum_{(i,t)\in\mathcal S}
\underbrace{\omega_{it}}_{\text{chosen target population}}
\overbrace{
\widehat{\operatorname{ME}}_{p,t}
\left(W_{r(i),t-p},Z_i\right)
}^{\text{effect evaluated at the observed dose}}.
\tag{6}
\]

The retained employer-count standardizations are:

\[
\underbrace{\omega_{it}=1}_{\text{equal county-year average}},
\qquad
\overbrace{
\omega_{it}=\frac{1}{n_{r(i),t}}
}^{\text{equal region-year average}}.
\]

There is no farm-employment weight or farm-employment divisor for the employer
outcome. The sample-period employer total instead uses

\[
\underbrace{\omega_{it}=1}_{\text{one raw employer-count effect per row}},
\qquad
\overbrace{\text{normalization}=1}^{\text{sum, do not average}},
\]

so that

\[
\underbrace{
\widehat{\operatorname{TotalEffect}}_{p}^{\text{employers}}(\delta)
}_{\text{employer units over the sample period}}
=
\overbrace{
\sum_{(i,t)\in\mathcal S}
\widehat\Delta_{p,t}^{\text{employers}}(\delta)
}^{\text{direct sum of raw-count effects}}.
\]

## 7. Why a delta method is needed

The regression returns a coefficient vector, not an ATE:

\[
\underbrace{\widehat{\boldsymbol\gamma}}_{\text{all fitted coefficients}}
=
\overbrace{
\left(\widehat\theta,\widehat\alpha,\widehat\lambda,
\widehat\psi,\widehat\beta,\widehat\rho,\widehat\delta\right)
}^{\text{922 retained coefficients in the primary full-sample model}}.
\]

An ATE or AME is a function of that vector:

\[
\underbrace{\widehat q}_{\substack{\text{reported ATE, AME,}\\
\text{year effect, or subgroup effect}}}
=
\overbrace{h(\widehat{\boldsymbol\gamma})}^{\text{postestimation map}}.
\]

The delta method asks how sensitive \(h\) is to every coefficient:

\[
\underbrace{\widehat{\mathbf g}}_{\text{named gradient}}
=
\overbrace{
\frac{\partial h(\widehat{\boldsymbol\gamma})}
{\partial\widehat{\boldsymbol\gamma}'}
}^{\text{effect sensitivity to the coefficient vector}}.
\tag{7}
\]

For the identity-link model, the finite effect in (5) can be written as

\[
\widehat q
=
\sum_s
\left[
\underbrace{g_{p,1,s}^{(0)}\widehat\beta_{p,1,s}^{(0)}}
_{\text{linear main-effect contribution}}
+
\overbrace{g_{p,1,s}^{(1)}\widehat\beta_{p,1,s}^{(1)}}
^{\text{linear }Z_i\text{ contribution}}
+
\underbrace{g_{p,2,s}^{(0)}\widehat\beta_{p,2,s}^{(0)}}
_{\text{quadratic main-effect contribution}}
+
\overbrace{g_{p,2,s}^{(1)}\widehat\beta_{p,2,s}^{(1)}}
^{\text{quadratic }Z_i\text{ contribution}}
\right].
\tag{8}
\]

For a finite change, define

\[
\underbrace{a_{it}^{(1)}(\delta)=\delta}_{\text{linear-basis change}},
\qquad
\overbrace{
a_{it}^{(2)}(\delta)
=\frac{2W_{r(i),t-p}\delta+\delta^2}{25}
}^{\text{quadratic-basis change}}.
\]

Then the gradient entries are weighted sample moments:

\[
\underbrace{g_{p,m,s}^{(0)}}_{\text{gradient for }\beta_{pms}^{(0)}}
=
\overbrace{
\frac{\sum_{(i,t)\in\mathcal S}
\omega_{it}d_{ts}a_{it}^{(m)}(\delta)}
{\sum_{(i,t)\in\mathcal S}\omega_{it}}
}^{\text{average basis change in year }s},
\]

\[
\underbrace{g_{p,m,s}^{(1)}}_{\text{gradient for }\beta_{pms}^{(1)}}
=
\overbrace{
\frac{\sum_{(i,t)\in\mathcal S}
\omega_{it}d_{ts}a_{it}^{(m)}(\delta)Z_i}
{\sum_{(i,t)\in\mathcal S}\omega_{it}}
}^{\text{moderator-weighted average basis change}}.
\]

For an AME, replace the finite-change basis terms with derivatives:

\[
\underbrace{a_{it}^{(1)}}_{\text{linear derivative}}=1,
\qquad
\overbrace{a_{it}^{(2)}}^{\text{quadratic derivative}}
=\frac{2W_{r(i),t-p}}{25}.
\]

All nuisance-coordinate entries are zero in the direct effect gradient:

\[
\underbrace{
\frac{\partial h}{\partial(\theta,\alpha,\lambda,\psi,\rho,\delta)}
}_{\substack{\text{direct gradient entries for fixed}\\
\text{nuisance histories and baseline type}}}
=
\overbrace{\mathbf 0}^{\text{held fixed in the counterfactual}}.
\]

Those nuisance regressors still matter: estimating them changes the covariance
of the relevant \(\widehat\beta\) coefficients.

## 8. Delta-method variance with design-covariance CCV

Let \(\widehat{\mathbf V}_{\mathrm{dcCCV}}\) be the covariance matrix of the
complete identified coefficient vector. The reported variance is

\[
\underbrace{
\widehat{\operatorname{Var}}(\widehat q)
}_{\text{uncertainty in the reported treatment effect}}
=
\overbrace{\widehat{\mathbf g}'}^{\text{left sensitivity}}
\underbrace{\widehat{\mathbf V}_{\mathrm{dcCCV}}}
_{\substack{\text{coefficient covariance from}\\
\text{17 balanced path assignments}}}
\overbrace{\widehat{\mathbf g}}^{\text{right sensitivity}}.
\tag{9}
\]

Thus

\[
\underbrace{\widehat{\operatorname{SE}}(\widehat q)}_{\text{reported CCV SE}}
=
\overbrace{
\sqrt{
\widehat{\mathbf g}'
\widehat{\mathbf V}_{\mathrm{dcCCV}}
\widehat{\mathbf g}}
}^{\text{delta-method standard error}},
\]

and the 95 percent interval is

\[
\underbrace{\widehat q}_{\text{point estimate}}
\ \pm\
\overbrace{t_{0.975,16}}^{\substack{\text{critical value based on}\\
\text{17 independent policy paths}}}
\underbrace{\widehat{\operatorname{SE}}(\widehat q)}_{\text{delta-method CCV SE}}.
\tag{10}
\]

The covariance itself is formed by cyclically assigning the 17 observed AEWR
paths to the 17 region labels. If \(a=0,\ldots,16\) indexes these equally likely
assignment states, then

\[
\underbrace{\widehat{\mathbf V}_{\mathrm{dcCCV}}}_{\text{reporting covariance}}
=
\overbrace{\frac{1}{17}}^{\text{finite-design probability weight}}
\sum_{a=0}^{16}
\underbrace{
\left(\widehat{\mathbf b}_{e,a}-\overline{\mathbf b}_e\right)
\left(\widehat{\mathbf b}_{e,a}-\overline{\mathbf b}_e\right)'
}_{\text{state-specific coefficient-error outer product}}.
\tag{11}
\]

The delta method and the design covariance perform different jobs:

\[
\underbrace{\widehat{\mathbf V}_{\mathrm{dcCCV}}}
_{\text{How uncertain are the regression coefficients?}}
\qquad+qquad
\overbrace{\widehat{\mathbf g}}^{\text{How does the estimand use them?}}
\qquad\Longrightarrow\qquad
\underbrace{\widehat{\operatorname{Var}}(\widehat q)}
_{\text{How uncertain is the ATE or AME?}}.
\]

## 9. A two-coefficient miniature example

Suppose, only for illustration, that the fitted response were linear:

\[
\underbrace{\widehat Y_i(w)}_{\text{fitted raw employer count}}
=
\overbrace{\widehat\beta_0}^{\text{baseline}}
+
\underbrace{\widehat\beta_1 w}_{\text{dose response}}.
\]

A \(\delta=5\) change gives every row

\[
\underbrace{\widehat Y_i(w+5)-\widehat Y_i(w)}_{\text{finite treatment effect}}
=
\overbrace{5\widehat\beta_1}^{\text{coefficient combination}}.
\]

Therefore

\[
\underbrace{\widehat q}_{\text{average effect}}=5\widehat\beta_1,
\qquad
\overbrace{\widehat{\mathbf g}}^{\text{gradient}}
=
\underbrace{(0,5)'}_{\substack{\text{zero sensitivity to }\beta_0\\
\text{five-unit sensitivity to }\beta_1}},
\]

and

\[
\underbrace{\widehat{\operatorname{Var}}(\widehat q)}_{\text{effect variance}}
=
\overbrace{(0,5)}^{\text{left gradient}}
\underbrace{
\begin{pmatrix}
V_{00}&V_{01}\\V_{10}&V_{11}
\end{pmatrix}
}_{\text{coefficient covariance}}
\overbrace{(0,5)'}^{\text{right gradient}}
=
\underbrace{25V_{11}}_{\text{propagated variance}}.
\]

The implemented model follows exactly this logic with hundreds of nuisance
coefficients and year-specific linear and quadratic treatment coefficients.

## 10. How the implementation checks the algebra

The production code constructs a sparse named gradient by placing the sample
moments from (8) into the appropriate year-by-horizon coefficient coordinates.
As a validation, it also builds the complete observed and counterfactual model
matrices:

\[
\underbrace{\mathbf X_0}_{\text{observed design matrix}}
=
\overbrace{\mathbf X(W)}^{\text{observed focal dose}},
\qquad
\underbrace{\mathbf X_1}_{\text{counterfactual design matrix}}
=
\overbrace{\mathbf X(W+\delta)}^{\text{shifted focal dose}}.
\]

For an equally weighted average, the direct formula-matrix gradient is

\[
\underbrace{\widehat{\mathbf g}_{\mathrm{direct}}}_{\text{full-matrix check}}
=
\overbrace{
\frac{1}{N}\sum_{i,t}
\left(\mathbf x_{1,it}-\mathbf x_{0,it}\right)
}^{\text{average counterfactual change in every regressor}}.
\]

The employer-outcome validation requires

\[
\underbrace{
\max_k\left|
\widehat g_{\mathrm{named},k}
-
\widehat g_{\mathrm{direct},k}
\right|
}_{\text{largest delta-gradient discrepancy}}
\approx
\overbrace{5.55\times10^{-17}}^{\text{observed numerical error}}.
\]

This check is especially useful because it jointly tests the year cells,
quadratic rescaling, \(Z_i\) interactions, counterfactual basis refresh, and
coefficient-name alignment.

## 11. Interpretation checklist

Before interpreting an employer-count effect, verify each layer:

\[
\underbrace{Y_{it}}_{\text{raw balanced employer count}}
\quad\xleftarrow{\text{modeled by}}\quad
\overbrace{E[Y_{it}\mid\mathcal H_i]}^{\text{rich identity-link projection}}
\quad\xrightarrow{W_{t-p}\mapsto W_{t-p}+\delta}\quad
\underbrace{\Delta_{it}(\delta)}_{\text{county-year fitted effect}}
\quad\xrightarrow{\text{average}}\quad
\overbrace{\widehat{\operatorname{ATE}}(\delta)}^{\text{reported estimand}}
\quad\xrightarrow{\text{delta method}}\quad
\underbrace{\widehat{\operatorname{SE}}}_{\text{dcCCV uncertainty}}.
\]

In words:

1. The employer outcome is a raw count, not divided by farm employment.
2. Treatment is annual AEWR growth measured in log percentage points.
3. A counterfactual shifts one current-or-lag dose coordinate while holding
   predetermined type and non-focal history fixed.
4. County-year fitted changes are averaged using a declared standardization.
5. A named gradient maps coefficient covariance into ATE or AME covariance.
6. The causal interpretation still requires the conditional parallel-trends
   and treatment-history assumptions described in the main design README.

## 12. Code map

| Mathematical object | Implementation |
| --- | --- |
| Outcomes, horizons, basis, and design version | [`design.R`](design.R) |
| Baseline histories, hierarchy, treatment columns, raw employer outcome | [`01_build_panel.R`](01_build_panel.R) |
| Formula assembly | [`helpers.R`](helpers.R), especially `mc_build_formula()` |
| Four-model estimation ladder and coefficient covariance | [`02_estimate_models.R`](02_estimate_models.R) |
| Named finite-change and AME gradients | [`helpers.R`](helpers.R), especially `mc_master_sample_effect()` |
| Standardizations and sample-period totals | [`03_postestimation.R`](03_postestimation.R) |
| Raw-count and direct delta-gradient checks | [`05_validate.R`](05_validate.R) |

The main design assumptions, rank argument, CCV reference law, and artifact
inventory remain documented in [`README.md`](README.md).
