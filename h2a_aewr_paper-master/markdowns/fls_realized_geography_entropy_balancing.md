# FLS Realized-Geography Entropy Balancing

The procedure estimates a public-data analog of each FLS region-year's
geographic composition. It does not recover NASS's confidential sample or
survey weights.

The baseline prior \(q_i\) is each OEWS area's share of the region's Census
hired-worker frame analog. County mass is allocated to OEWS areas using
within-county township shares. Areas remain in the frame regardless of
whether an OEWS wage is observed.

The calibration targets are published FLS worker counts crossed by reference
quarter and duration, plus the annual FLS combined field-and-livestock wage
used to set the following year's AEWR. Counts are normalized into a joint
composition \(\tau\). Public area analogs combine strict QCEW agricultural
employment, QWI fills and persistence, and Census duration shares. The
resulting area compositions are converted to prior-standardized orthonormal
Helmert contrasts. The corresponding area wage is the employment-weighted mean
over the retained OEWS agricultural occupations. The preliminary FLS release
used by DOL to set the following year's AEWR is preferred, with the revised
value as a fallback. The wage is standardized under the same frame prior and
appended as one additional moment.

For a prior \(p\), the estimator solves

$$
\min_{\mathbf w}
D_{KL}(\mathbf w\Vert\mathbf p)
+
\frac{\rho}{2}
\left\|
\widetilde{\mathbf Z}'\mathbf w-\widetilde{\boldsymbol\tau}
\right\|_2^2
$$

subject to \(w_i\geq0\) and \(\sum_iw_i=1\). The KL term keeps the recovered
distribution near the frame prior; the second term rewards agreement with
the FLS composition and wage moments. Larger \(\rho\) pursues those moments
more strongly. The primary value is \(\rho=0.10\).

The deterministic center uses \(p=q\). To represent plausible realized
sampling variation, the procedure also draws

$$
\mathbf q^{(b)}
\sim
\operatorname{Dirichlet}(\kappa\mathbf q),
\qquad
\kappa=m\left(\frac{1}{\sum_iq_i^2}\right),
$$

and solves the same entropy problem for every draw. The primary specification
uses \(m=10\), 999 draws, and seed `20260726`; fixed \(m\) and \(\rho\)
alternatives provide sensitivity checks.

FLS and OEWS wages enter only through the declared wage moment. AEWR values,
first-stage estimates, and outcomes never enter fitting. Reported draw
percentiles are simulation envelopes for the predeclared ensemble.

The dissimilarity IV uses the primary deterministic center, expands it to
counties with fixed within-area frame shares, and then restricts it to the
predeclared donor clusters. The Census-frame instrument remains a benchmark.
Simulation draws are not propagated into the IV panel.
