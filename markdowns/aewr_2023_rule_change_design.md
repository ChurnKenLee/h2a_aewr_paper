# 2023 AEWR rule-change design

## Question and estimand

Estimate how H-2A demand responds when a regulatory formula changes the wage floor for non-field-and-livestock occupations.

This identifies an elasticity for the relatively small group of occupations moved to occupation-specific OEWS AEWRs. It does not directly identify the response of the main field-and-livestock workforce.

## Policy variation

The 2023 rule took effect on March 30, 2023. Six field-and-livestock SOC codes retained the state or regional FLS AEWR. Other non-range occupations became subject to statewide occupation-specific OEWS AEWRs. A job containing multiple SOCs is subject to the highest applicable AEWR.

The new methodology applies according to the ETA-790/790A job-order submission date. [DOL implementation guidance](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/2023-AEWR-Final-Rule-FAQs_implementation_FINAL.pdf) provides the cutoff and wage rules.

## Predicted wage shock

For job type \(j\), state \(s\), occupation \(o\), and AEWR region \(r\), construct

\[
Shock^0_{jso}
=
\log W^{new}_{so}
-
\log W^{old}_{r},
\]

where both terms incorporate other applicable wage floors. Set the shock to zero for jobs remaining under the FLS formula.

Classify occupation using a pre-rule employer-job cell or a task-text classifier trained only on pre-rule cases. Do not define treatment from the post-rule SOC, because employers may change task descriptions or occupation coding in response to the rule.

## Empirical design

Create a balanced employer-by-worksite-by-pre-rule-job-type monthly panel, including zero-application months. Let

\[
Post_{jt}
=
\mathbf 1\{\text{job order submitted on or after March 30, 2023}\}.
\]

Use \(Post_{jt}\times Shock^0_{jso}\) as an instrument for the log applicable wage floor. Include:

- employer-worksite-job-type fixed effects;
- state-by-month or AEWR-region-by-month fixed effects;
- SOC-by-month effects;
- intended-start-month and crop-season controls.

Estimate requested positions and applications with a zero-preserving count model. Report certified positions as a secondary outcome.

Base inference on the state-by-pre-rule-SOC shock cells. Report state-clustered and employer-clustered sensitivity estimates.

## Identifying assumption

Absent the rule, high- and low-shock job cells would have followed parallel relative trends after conditioning on local-time and occupation-time shocks. The mechanical formula difference must be unrelated to occupation-specific demand changes except through the wage floor.

## Main outcomes

- Applications and requested positions.
- Employer entry, exit, and contract size.
- Withdrawals and amendments.
- SOC recoding, task relabeling, and job bundling.
- Offered wage and the first-stage pass-through.

Avoidance responses are part of the policy effect, but stable pre-rule job cells should be the main elasticity sample.

## Required diagnostics

- Event-study coefficients before March 2023.
- Filing-date density and bunching around the announced cutoff.
- Results excluding a narrow cutoff window.
- Comparisons with both unchanged FLS occupations and low-shock OEWS occupations.
- Stable-employer and stable-task samples.
- Placebos using March 30 in earlier years.
- Alternative occupation classifications and highest-SOC treatment rules.

DOL reported that roughly 97% of initially certified positions remained under the FLS rate, so power and external validity are the central limitations. Public OFLC files through FY2025 are available from the [DOL performance-data page](https://www.dol.gov/agencies/eta/foreign-labor/performance).
