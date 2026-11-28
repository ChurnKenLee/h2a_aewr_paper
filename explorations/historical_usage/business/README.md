# Who accounted for the increase in certified positions?

In the five annual transitions from FY2020 to FY2025, positions in reconciled H-2A cases increased by **122,649**. Filers with positive positions in consecutive years added **88,447 net positions**, or 72.1% of growth. The tighter and looser filer identifiers give 67.4% and 73.1%. Between FY2011 and FY2019, the comparable share was 52.7%. A filer may be a grower, association or labor contractor. Each transition classifies participants afresh, so the figures include growth by a new filer after its first year. [Annual growth components](flows.csv)

## What changes when the worksite is the unit?

The figures below sum the annual changes between the FY2020 and FY2025 endpoints. The bottom three rows use **identical positions** with a named worksite business. Dollar amounts and wage rates play no role.

| Identity used | Net change in positions | Continuing IDs | First appearance since FY2008 | Return after a gap | Absent this year | Continuing share of net change |
|---|---:|---:|---:|---:|---:|---:|
| Filer, R7, all positions | 122,649 | 88,447 | 139,481 | 25,584 | −130,863 | 72.1% |
| Filer, R7, named positions | 97,059 | 70,014 | 110,796 | 21,575 | −105,326 | 72.1% |
| Worksite business, balanced | 97,059 | 39,961 | 360,849 | 56,786 | −360,537 | 41.2% |
| Worksite business, high recall | 97,059 | 57,000 | 253,017 | 42,202 | −255,160 | 58.7% |

Comparing rows two and three shows that the same growth can look like expansion by returning filers and turnover among the businesses they name. Linkage choices materially affect how much turnover can be attributed to businesses. For FY2025, 72.4% of named positions belonged to the same filer–business pair as the previous year, or 78.2% using high recall. Another 17.7% belonged to a previously active filer naming a business absent in the previous year, or 10.9% using high recall. The figures for new pairings of two separately active parties were only 2.8% and 3.8%. [Annual relationship accounts](relationships.csv)

Among R7 filers first observed with positive positions in FY2021 and FY2022, 65.8% of each cohort had positive positions again exactly three years later. The same calculation for balanced business IDs gave 20.1% and 20.7%. In the FY2022 cohort, positions three years later totaled 72.6% of the initial cohort capacity for filers and 35.2% for balanced worksite IDs, counting absences as zero. Positive cohort capacity includes subsequent expansion or contraction by observed returners. A missed link, ranch renaming or changed reporting can make a continuing business appear to leave. The FY2008 start also left-censors first appearances. [All cohorts with three years of follow-up](cohorts.csv) · [Earlier filer and naming audit](../../../markdowns/employer_panel_step1.md)

In FY2025, the largest 1% of active R7 filers held 42.0% of reconciled positions. Among named positions, the corresponding shares were 42.6% for R7 filers and 36.4% or 41.7% for the two business linkage tiers. The cutoff uses the ceiling of 1% of annual active units, so the number of units differs across definitions. These are shares of certified capacity and give no measure of product-market power. [Yearly counts and concentration](yearly.csv)

## Contractor contribution

From FY2021 to FY2025, both endpoints have an observed yes/no contractor response on all positive allocations. Positions filed on cases reporting “yes” rose from 135,400 to 172,131. The increase of 36,731 accounted for **45.7%** of the 80,440-position increase in reconciled cases. FY2020 had 21,577 positions with unknown contractor status. Assigning none or all of them to the FY2020 contractor total gives a 41.6–59.2% contribution over FY2020–2025. This partitions reported volume and allows filers to change classification. [Contractor-flag results](contractors.csv)

## Where the history is comparable

- FY2019 to FY2020 spans the Addendum B form change. All reconciled positions rose by 17,743 while positions assigned to a named business fell by 18,701. Balanced business counts rose from 9,105 to 29,262. That discontinuity confounds trends in measured worksite participation. [Coverage](coverage.csv) · [Annual units](yearly.csv)
- The FY2025 filer account includes all 398,059 positions in reconciled allocations. Business IDs cover 335,965, or 84.4%. Of the remaining 62,094 positions, 57,928 belong to individual master rows used as aliases, 2,891 to blank addendum records and 1,275 to other excluded master rows. This preserves the producer's assignment and roles. A case with an addendum may allocate capacity to a master row outside the worksite-business universe. [Source role totals](roles.csv) · [Linkage contract](../../../content/contracts/employer-linkage.md)
- The baseline follows positive allocations on reconciled cases, including two FY2008 withdrawn cases carrying 44 positive positions. A sensitivity retains cases whose source statuses all indicate full or partial certification, including expired certifications. The FY2020–2025 R7 continuing increment becomes 88,421 of 122,657 (72.1%). FY2020 has two quarantined cases. [Annual status and quarantine audit](coverage.csv) · [Status sensitivity](flows.csv)
- The owner’s FY2008 reconciled total of 90,167 exceeds the Department of Labor's **82,099 positions certified** by 8,068 (9.8%). The FY2008 source workbook itself gives 90,123 on certified-status master rows with a nonblank organization flag; their decision dates fall within FY2008. This locates 8,024 of the difference before allocation or employer linkage. FY2009 and FY2010 reconciled totals differ from published positions by −29 and +227. FY2025 differs by −199. The agency tables distinguish positions certified from applications certified. The FY2008 discrepancy remains unresolved; use it as historical exposure to prior activity and use later endpoints for growth comparisons. [Benchmark calculations](official_benchmarks.csv) · [OFLC FY2010 Annual Report, p. 31](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/oflc_2010_annual_report_master.pdf) · [OFLC FY2025 Selected Statistics, p. 1](https://www.dol.gov/sites/dolgov/files/ETA/oflc/pdfs/H-2A_Selected_Statistics_FY2025_Q4.pdf)

These are descriptive fiscal-year accounts. The wage elasticity, causal effect of intermediation and number of distinct people who worked are separate estimands. Full-vintage IDs can use later identity evidence, including partially reviewed FY2026 data, but all position outcomes here stop in FY2025. In the early years, a short observable history inflates first appearances relative to long-history cohorts. The allocation is the producer's proportional division of case certifications, which may differ from staffing at individual worksites. [Allocation contract](../../../content/contracts/h2a-worksite-exposure.md)

## Reproduction

Run from the repository root using the project library.

```sh
source scripts/pipeline_helpers.sh
configure_r_library
Rscript --vanilla explorations/historical_usage/business/growth.R
```

The script regenerates every CSV here, emitting aggregates only. Inputs are the retained worksite exposure and case reconciliation audits, case allocation source, worksite employer membership, normalized H-2A disclosure, raw FY2008 workbook and `data/intermediate/policy_2025/filer/filer_ids.parquet`. The filer file is a retained product of the [policy_2025 branch](../../../code/designs/policy_2025/README.md). If absent, its standalone runner `./scripts/run_policy_2025.sh` reconstructs it from cached local disclosures and publishes it after validation. This exploration does not require historical scratch files. The FY2008–10 and FY2025 official benchmarks are transcribed in the script from the two primary sources linked above.

For each identity rule, the script checks that the four signed annual flow terms sum to the observed change. It also checks row and case keys, complete annual coverage, unique balanced-to-high-recall nesting, complete filer linkage on positive allocations, and that the dyad accounts conserve the named position total. Position quantities can be fractional at a worksite. Table entries are rounded individually.
