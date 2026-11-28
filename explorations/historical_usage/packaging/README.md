# Did near-threshold applications appear in pairs?

Among 124 late-period contractor applications requesting one to five positions below 150, 200, 250 or 300, **51 had a companion application by the same filer in the same worksite state for the same requested start day**. Only 15 had a companion requesting 1–10 positions that would take the pair across the threshold. Requiring close agreement on location, occupation, receipt date and certification status leaves **one** such focal application. These are comparisons of requested capacity. The filings do not reveal the staffing plan the employer would have chosen without the bond step. [Companion counts](companions.csv)

## Same-day matches

All counts below refer to focal applications one to five positions below a new step. A focal application can belong to more than one column. Two focal applications can count one another. “Contractor” uses the explicit application flag, and comparison employers also filed as individual employers. The early receipt window is April 2020–March 2022, the late window April 2023–March 2025. The sample retains decisions within 0–180 days of receipt.

| Era and filer | Near-step cases | Any companion | Companion of 1–10 positions taking sum across step | Identically sized companion |
|---|---:|---:|---:|---:|
| Early contractor | 27 | 4 | 0 | 0 |
| Late contractor | 124 | 51 | 15 | 17 |
| Early other employer | 22 | 7 | 3 | 0 |
| Late other employer | 34 | 18 | 6 | 0 |

- Among the 89 late contractor applications requesting the step size through four positions above it, 27 had any same-date companion, eight had a companion of 1–10 and two had an identically sized companion. Simultaneous applications were common among other employers too. Among late contractors just below, widening the permitted start-date difference to seven days gives 66 of 124 with a companion; fourteen days gives 73. These wider windows mix more unrelated work orders.
- The 17 late below-step contractor cases with an identically sized companion span eight filers. Requiring both cases to be certified, in the same specified worksite city and nonblank occupation code, with receipt dates within seven days, leaves nine. Conservatively collapsing distinct case numbers with the same filer, role, city, state, start and end dates, SOC, job title, requested number, and receipt date leaves four such cases. The corresponding broad identical-size count becomes 12 across six filers. Matching the disclosure fields cannot distinguish a duplicate, a replacement and a genuine repeat job. [Lookalike audit](lookalikes.csv) · [Companion counts](companions.csv)
- The 15 late contractor cases with small crossing companions span 12 filers. Applying the status, city, SOC and receipt-date checks leaves one. The same checks leave four of the six broad matches among late comparison employers. There is little direct support for splitting a small residual job off a single otherwise identical order.

## What survives combining requests?

For every filer, worksite state and exact requested start day, add all eligible requests in that era, including those initially outside the threshold bins. Treat each resulting group as one observation. Sixteen mixed contractor-status groups across both eras were omitted from the group-level analysis. Compare the share just below the four steps among all observations within the local bands.

| Unit, five-position band on each side | Early contractor | Late contractor | Early other | Late other | Difference in changes |
|---|---:|---:|---:|---:|---:|
| Applications | 27/126 | 124/213 | 22/68 | 34/80 | +26.6 pp |
| Filer–state–start groups | 30/127 | 92/178 | 19/60 | 27/78 | +25.1 pp |

The exact-start grouping leaves most of the excess share below the threshold. With the potential duplicate cases collapsed first, the application and group estimates are both +26.0 percentage points after rounding. Using just the two integers immediately below and at the threshold gives +35.1 points per application and +34.0 per group before the duplicate sensitivity. The post hoc group analysis has a different conditional denominator: grouping can move applications far from a threshold or bring groups of smaller applications into a band. These figures provide a mechanism diagnostic and carry no independent confirmatory test. [Group counts for both widths and duplicate treatments](bundle_sizes.csv)

The 2022 rule scales surety by workers under a certification. A contractor requesting two applications for different clients, worksites or occupations can create the same timing pattern as a split. Some potential companions can also use different states, filers, application roles, decision windows or unobserved job orders and escape the match. The full-vintage R7 identifier can link one filer across clients. A date-level sum gives neither realized employment nor the actual number of distinct workers. Credible evidence of intentional splitting would require tracing paired applications to the same underlying work order or overlapping client need. [Final rule, § 655.132(c)](https://public-inspection.federalregister.gov/2022-20506.pdf) · [Filer linkage audit](../../../markdowns/employer_panel_step1.md)

## Reproduction

```sh
source scripts/pipeline_helpers.sh
configure_r_library
Rscript --vanilla explorations/historical_usage/packaging/packaging.R
```

The script uses the same distinct-case filters, role, windows and worker counts as the [application-level investigation](../policy/bond_thresholds.R). It reads the retained disclosure and [policy_2025 filer IDs](../../../code/designs/policy_2025/README.md). Requested dates must be parseable and within fiscal year ±1, and the filer and state must be identified. Five of 73,431 baseline cases lack supported dates; all 487 applications in the five-position bands survive. The CSVs contain aggregates only. No pipeline stage or manuscript estimate is changed. [Coverage audit](coverage.csv)
