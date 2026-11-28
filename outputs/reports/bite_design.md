# Full-Panel FE-PPML Bite Design

Generated 2026-11-26 10:25:55.901617 from `data/processed/county_year_panel.parquet` (SHA-256 `4e2e56d0884715ea0ff26b8251339c18748cc1966692a5ab0eac31171cf51b92`).

## Identification & Estimand Boundary

- **Panel**: Balanced 3,108 contiguous U.S. counties across 48 states and 17 AEWR regions (`2008-2025`, 55,944 county-years). Estimation spans `2011-2025` (excluding `2009`, where statutory AEWR is strictly NULL under `AEWR-001`; `2025` start-year H-2A outcomes remain provisional pending FY2027 decisions).
- **Frozen Baseline Exposure (`2008/2010`)**: Primary exposure $E_{p25}$ standardizes the 2008/2010 average of $\ln(\text{AEWR}^{\text{PPI}} / w_{p25}^{\text{PPI}})$ to mean 0, SD 1 within each AEWR region. Algebraically, within-region standardization cancels both the regional AEWR level and the national PPI deflator identically ($E_{p25} \equiv -\text{std}_{r}(\overline{\ln w_{p25}})$). Alternative frozen exposures use 2008/2010 OEWS Big-Six wages ($E_{\text{oews}}$) and the all-real five-quantile agricultural shortfall ($E_{\text{sf}}$).
- **Estimand**: Coefficients on $E_c \times \ln(\text{AEWR}_{rt})$ under `County + Region x Year FE` (`1b-PPML`) measure the **relative differential log-point (semi-elasticity) response** of high-bite vs low-bite counties within the same AEWR region per $+1$ SD of baseline exposure per unit increase in $\ln(\text{AEWR}_{rt})$. They are **not** aggregate AEWR elasticities.
- **Inference**: Every regression reports both region-clustered CRV1 SEs and exact delete-one-cluster jackknife (**CV3J**) SEs with $t(G-1)$ critical values and MDE $= 2.8 \times \text{CV3J SE}$.

## 1. Main Full-Panel Bite Estimates (`2011-2025`)

| spec_id | estimator | outcome | fe | estimate | se_crv1 | se_cv3j | p_cv3j | mde_cv3j | loo_min | loo_max | n_obs | n_counties | n_clusters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main_p25_hours | fepois | man_hours_certified_start_year | County + Region x Year | 0.0026 | 0.1286 | 0.1353 | 0.9849 | 0.3788 | -0.05859 | 0.09077 | 44805 |  2987 |    17 |
| main_p25_workers | fepois | nbr_workers_certified_start_year | County + Region x Year | -0.01046 | 0.1133 | 0.1189 | 0.931 | 0.3329 | -0.0591 | 0.05924 | 44805 |  2987 |    17 |
| main_p25_any_ge5 | feols | any_h2a_ge5 | County + Region x Year | -0.02427 | 0.03097 | 0.03162 | 0.454 | 0.08854 | -0.03911 | -0.003596 | 46620 |  3108 |    17 |
| main_p25_hours_per_pos | fepois_diff | implied_hours_per_position | County + Region x Year | 0.01306 | — | 0.05159 | 0.8034 | 0.1444 | -0.01726 | 0.04248 | 44805 |  2987 |    17 |
| state_year_p25_hours | fepois | man_hours_certified_start_year | County + State x Year | -0.04463 | 0.1493 | 0.1579 | 0.7811 | 0.4422 | -0.1009 | 0.05266 | 44805 |  2987 |    17 |
| state_year_p25_workers | fepois | nbr_workers_certified_start_year | County + State x Year | -0.0466 | 0.1417 | 0.1484 | 0.7576 | 0.4155 | -0.1048 | 0.0295 | 44805 |  2987 |    17 |
| xpre_p25_hours | fepois | man_hours_certified_start_year | County + Region x Year + X_pre x Year | 0.02905 | 0.1218 | 0.1299 | 0.8259 | 0.3638 | -0.02831 | 0.1154 | 44535 |  2969 |    17 |
| xpre_p25_workers | fepois | nbr_workers_certified_start_year | County + Region x Year + X_pre x Year | -0.00108 | 0.1037 | 0.1104 | 0.9923 | 0.3091 | -0.03219 | 0.07001 | 44535 |  2969 |    17 |
| oews_hours | fepois | man_hours_certified_start_year | County + Region x Year | 0.0411 | 0.1327 | 0.1336 | 0.7624 | 0.3741 | -0.03729 | 0.1381 | 44220 |  2948 |    17 |
| oews_workers | fepois | nbr_workers_certified_start_year | County + Region x Year | 0.1052 | 0.1567 | 0.1619 | 0.5251 | 0.4534 | 0.02675 | 0.2088 | 44220 |  2948 |    17 |
| sf_log_hours | fepois | man_hours_certified_start_year | County + Region x Year | 0.008126 | 0.1198 | 0.1237 | 0.9485 | 0.3465 | -0.06411 | 0.07303 | 44805 |  2987 |    17 |
| sf_log_workers | fepois | nbr_workers_certified_start_year | County + Region x Year | -0.009281 | 0.1174 | 0.1218 | 0.9402 | 0.3411 | -0.0961 | 0.04216 | 44805 |  2987 |    17 |
| spec1a_p25_hours | fepois | man_hours_certified_start_year | County + Region x Year + E_p25 x Year | -0.2143 | 0.5797 | 0.7522 | 0.7794 | 2.106 | -0.5815 | 0.3694 | 44805 |  2987 |    17 |
| spec1a_p25_workers | fepois | nbr_workers_certified_start_year | County + Region x Year + E_p25 x Year | -0.07281 | 0.5502 | 0.7237 | 0.9211 | 2.026 | -0.3102 | 0.5356 | 44805 |  2987 |    17 |

## 2. Identifying-Variation Split, Effective Clusters & Influence

| weighting | var_total | sd_total | var_resid_C_RY | sd_resid_C_RY | r2_C_RY | var_resid_C_RY_EY | sd_resid_C_RY_EY | r2_C_RY_EY | share_1a_of_1b | var_resid_C_SY | sd_resid_C_SY | r2_C_SY | share_within_SY_of_1b |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unweighted | 6.455 | 2.541 | 0.0292 | 0.1709 | 0.9955 | 0.0007619 | 0.0276 | 0.9999 | 0.02609 | 0.02513 | 0.1585 | 0.9961 | 0.8606 |
| ppml_fitted_hours | 6.348 |  2.52 | 0.02237 | 0.1496 | 0.9965 | 0.0007695 | 0.02774 | 0.9999 | 0.03439 | 0.01931 | 0.139 | 0.997 | 0.8631 |

- **Effective clusters ($G_{\text{eff}}$, Fisher information)**: `13.36` (hours; CA Region 17 share = `12.1%`, Hessian identity rel. err = `1.35e-09`), `12.25` (positions; CA Region 17 share = `13.1%`).
- **State-year residual variance ratio vs county noise (deviance residuals)**: Hours `MS_SY / MS_county = 2.85` (`R2_SY_within_RY = 0.0291`, `p = 1.43e-81`); Positions `MS_SY / MS_county = 2.29` (`R2_SY_within_RY = 0.0236`, `p = 2.95e-48`).

## 3. Event Study (`2008-2025`, Ref. = `2011`) & Joint Pre-Trend Tests

- **Certified Hours joint `2008-2010` pre-trend**: 2008-2010 weekly hours are pooled-mean imputed; F_CV3J(3,16)=0.531 (p=0.6674), F_CRV1(3,16)=0.541 (p=0.6610)
- **Certified Positions joint `2008-2010` pre-trend**: Positions directly observed 2008-2025; F_CV3J(3,16)=0.460 (p=0.7142), F_CRV1(3,16)=0.462 (p=0.7127)

| outcome | year | estimate | se_crv1 | se_cv3j | ci95_low_cv3j | ci95_high_cv3j | p_cv3j | hours_imputation_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| man_hours_certified_start_year |  2008 | -0.04214 | 0.07011 | 0.07182 | -0.1944 | 0.1101 | 0.5655 | 2008-2010_weekly_hours_pooled_mean_imputed |
| man_hours_certified_start_year |  2009 | -0.05841 | 0.06191 | 0.06313 | -0.1922 | 0.07542 | 0.3686 | 2008-2010_weekly_hours_pooled_mean_imputed |
| man_hours_certified_start_year |  2010 | -0.01875 | 0.06893 | 0.0707 | -0.1686 | 0.1311 | 0.7942 | 2008-2010_weekly_hours_pooled_mean_imputed |
| man_hours_certified_start_year |  2011 | 0 | 0 | 0 | 0 | 0 | — | reference_year |
| man_hours_certified_start_year |  2012 | 0.01956 | 0.04912 | 0.04776 | -0.08168 | 0.1208 | 0.6875 | observed |
| man_hours_certified_start_year |  2013 | -0.001814 | 0.06078 | 0.05979 | -0.1286 | 0.1249 | 0.9762 | observed |
| man_hours_certified_start_year |  2014 | 0.02971 | 0.05819 | 0.05735 | -0.09187 | 0.1513 | 0.6116 | observed |
| man_hours_certified_start_year |  2015 | 0.03374 | 0.06454 | 0.06426 | -0.1025 |  0.17 | 0.6068 | observed |
| man_hours_certified_start_year |  2016 | 0.03644 | 0.07569 | 0.07698 | -0.1268 | 0.1996 | 0.6423 | observed |
| man_hours_certified_start_year |  2017 | 0.05926 | 0.08595 | 0.08801 | -0.1273 | 0.2458 | 0.5103 | observed |
| man_hours_certified_start_year |  2018 | 0.04078 | 0.0983 | 0.102 | -0.1755 | 0.257 | 0.6946 | observed |
| man_hours_certified_start_year |  2019 | 0.02397 | 0.09756 | 0.1007 | -0.1894 | 0.2374 | 0.8148 | observed |
| man_hours_certified_start_year |  2020 | 0.0515 | 0.0847 | 0.08613 | -0.1311 | 0.2341 | 0.5583 | observed |
| man_hours_certified_start_year |  2021 | 0.02867 | 0.09676 | 0.09959 | -0.1825 | 0.2398 | 0.7772 | observed |
| man_hours_certified_start_year |  2022 | 0.03137 | 0.0913 | 0.09413 | -0.1682 | 0.2309 | 0.7433 | observed |
| man_hours_certified_start_year |  2023 | 0.02788 | 0.09358 | 0.09681 | -0.1774 | 0.2331 | 0.7771 | observed |
| man_hours_certified_start_year |  2024 | 0.03892 | 0.09866 | 0.102 | -0.1773 | 0.2551 | 0.7077 | observed |
| man_hours_certified_start_year |  2025 | 0.03381 | 0.09993 | 0.1037 | -0.186 | 0.2537 | 0.7486 | provisional_start_year_pending_fy2027 |
| nbr_workers_certified_start_year |  2008 | -0.03864 | 0.07881 | 0.07918 | -0.2065 | 0.1292 | 0.6322 | observed |
| nbr_workers_certified_start_year |  2009 | -0.05311 | 0.07354 | 0.07376 | -0.2095 | 0.1032 | 0.4819 | observed |
| nbr_workers_certified_start_year |  2010 | -0.01216 | 0.08142 | 0.08266 | -0.1874 | 0.1631 | 0.8849 | observed |
| nbr_workers_certified_start_year |  2011 | 0 | 0 | 0 | 0 | 0 | — | reference_year |
| nbr_workers_certified_start_year |  2012 | 0.03061 | 0.06304 | 0.06171 | -0.1002 | 0.1614 | 0.6266 | observed |
| nbr_workers_certified_start_year |  2013 | 0.0012 | 0.06361 | 0.06259 | -0.1315 | 0.1339 | 0.9849 | observed |
| nbr_workers_certified_start_year |  2014 | 0.04677 | 0.06294 | 0.0614 | -0.0834 | 0.1769 | 0.4574 | observed |
| nbr_workers_certified_start_year |  2015 | 0.03754 | 0.06144 | 0.05996 | -0.08957 | 0.1647 | 0.5401 | observed |
| nbr_workers_certified_start_year |  2016 | 0.03796 | 0.06956 | 0.06907 | -0.1085 | 0.1844 | 0.5902 | observed |
| nbr_workers_certified_start_year |  2017 | 0.04393 | 0.08171 | 0.08295 | -0.1319 | 0.2198 | 0.6036 | observed |
| nbr_workers_certified_start_year |  2018 | 0.03626 | 0.09539 | 0.09862 | -0.1728 | 0.2453 | 0.718 | observed |
| nbr_workers_certified_start_year |  2019 | 0.02215 | 0.09298 | 0.09468 | -0.1786 | 0.2229 | 0.818 | observed |
| nbr_workers_certified_start_year |  2020 | 0.0411 | 0.08825 | 0.08807 | -0.1456 | 0.2278 | 0.647 | observed |
| nbr_workers_certified_start_year |  2021 | 0.02442 | 0.09493 | 0.09582 | -0.1787 | 0.2276 | 0.8021 | observed |
| nbr_workers_certified_start_year |  2022 | 0.03216 | 0.08691 | 0.08751 | -0.1533 | 0.2177 | 0.7181 | observed |
| nbr_workers_certified_start_year |  2023 | 0.01895 | 0.08924 | 0.09034 | -0.1726 | 0.2105 | 0.8365 | observed |
| nbr_workers_certified_start_year |  2024 | 0.03339 | 0.09422 | 0.0954 | -0.1689 | 0.2356 | 0.7309 | observed |
| nbr_workers_certified_start_year |  2025 | 0.02784 | 0.09428 | 0.09599 | -0.1757 | 0.2313 | 0.7755 | provisional_start_year_pending_fy2027 |

## 4. Diagnostics, Farm Outcomes & Standalone Domestic-Floor Margin

| spec_id | spec_group | outcome | term | estimate | se_crv1 | se_cv3j | p_cv3j | mde_cv3j | n_obs | n_clusters | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pretrend_joint_hours | pretrend_joint_test | man_hours_certified_start_year | joint_2008_2010_F_cv3j | 0.5311 | 0.5411 | — | 0.6674 | — | 53910 |    17 | 2008-2010 weekly hours are pooled-mean imputed; F_CV3J(3,16)=0.531 (p=0.6674), F_CRV1(3,16)=0.541 (p=0.6610) |
| pretrend_joint_workers | pretrend_joint_test | nbr_workers_certified_start_year | joint_2008_2010_F_cv3j | 0.4599 | 0.462 | — | 0.7142 | — | 53910 |    17 | Positions directly observed 2008-2025; F_CV3J(3,16)=0.460 (p=0.7142), F_CRV1(3,16)=0.462 (p=0.7127) |
| spt_quartile_hours | quartile_spt | man_hours_certified_start_year | E_p25_quartile::Q2:ln_aewr | 0.2274 | 0.289 | 0.3188 | 0.4859 | 0.8925 | 44805 |    17 |  |
| spt_quartile_hours | quartile_spt | man_hours_certified_start_year | E_p25_quartile::Q3:ln_aewr | -0.3165 | 0.3859 | 0.4333 | 0.4756 | 1.213 | 44805 |    17 |  |
| spt_quartile_hours | quartile_spt | man_hours_certified_start_year | E_p25_quartile::Q4:ln_aewr | 0.09432 | 0.4416 | 0.4955 | 0.8514 | 1.387 | 44805 |    17 |  |
| spt_quartile_workers | quartile_spt | nbr_workers_certified_start_year | E_p25_quartile::Q2:ln_aewr | 0.2354 | 0.2479 | 0.2766 | 0.4073 | 0.7745 | 44805 |    17 |  |
| spt_quartile_workers | quartile_spt | nbr_workers_certified_start_year | E_p25_quartile::Q3:ln_aewr | -0.2349 | 0.4179 | 0.4733 | 0.6264 | 1.325 | 44805 |    17 |  |
| spt_quartile_workers | quartile_spt | nbr_workers_certified_start_year | E_p25_quartile::Q4:ln_aewr | 0.0241 | 0.3846 | 0.4335 | 0.9564 | 1.214 | 44805 |    17 |  |
| ppi_decouple_hours | ppi_decoupling | man_hours_certified_start_year | E_p25_x_ln_aewr | 0.03681 | 0.1749 | 0.1834 | 0.8435 | 0.5135 | 44805 |    17 |  |
| ppi_decouple_hours | ppi_decoupling | man_hours_certified_start_year | E_p25_x_ln_ppi_2012 | -0.04845 | 0.09818 | 0.1019 | 0.6409 | 0.2853 | 44805 |    17 |  |
| ppi_decouple_workers | ppi_decoupling | nbr_workers_certified_start_year | E_p25_x_ln_aewr | 0.006784 | 0.1621 | 0.1704 | 0.9687 | 0.4771 | 44805 |    17 |  |
| ppi_decouple_workers | ppi_decoupling | nbr_workers_certified_start_year | E_p25_x_ln_ppi_2012 | -0.02409 | 0.09321 | 0.09797 | 0.8089 | 0.2743 | 44805 |    17 |  |
| addendum_b_gt0_2011_2025 | addendum_b_extensive | any_h2a_gt0 | E_p25_x_ln_aewr | 0.04013 | 0.02028 | 0.0204 | 0.06675 | 0.05712 | 46620 |    17 |  |
| addendum_b_gt0_2011_2019 | addendum_b_extensive | any_h2a_gt0 | E_p25_x_ln_aewr | 0.01919 | 0.05905 | 0.05996 | 0.7531 | 0.1679 | 27972 |    17 |  |
| addendum_b_ge1_2011_2025 | addendum_b_extensive | any_h2a_ge1 | E_p25_x_ln_aewr | 0.02907 | 0.0239 | 0.02403 | 0.2439 | 0.06728 | 46620 |    17 |  |
| addendum_b_ge1_2011_2019 | addendum_b_extensive | any_h2a_ge1 | E_p25_x_ln_aewr | 0.03109 | 0.0566 | 0.05737 | 0.5953 | 0.1606 | 27972 |    17 |  |
| addendum_b_ge5_2011_2025 | addendum_b_extensive | any_h2a_ge5 | E_p25_x_ln_aewr | -0.02427 | 0.03097 | 0.03162 | 0.454 | 0.08854 | 46620 |    17 |  |
| addendum_b_ge5_2011_2019 | addendum_b_extensive | any_h2a_ge5 | E_p25_x_ln_aewr | -0.05242 | 0.05131 | 0.05132 | 0.3222 | 0.1437 | 27972 |    17 |  |
| employers_balanced_2011_2019 | addendum_b_extensive | nbr_employers_balanced_start_year | E_p25_x_ln_aewr | 0.3455 | 0.2546 | 0.2687 | 0.2167 | 0.7522 | 22392 |    17 |  |
| employers_balanced_2020_2025 | addendum_b_extensive | nbr_employers_balanced_start_year | E_p25_x_ln_aewr | 0.08267 |  0.15 | 0.1604 | 0.6134 | 0.4492 | 17784 |    17 |  |
| drop_top1pct_hours | concentration_robustness | man_hours_certified_start_year | E_p25_x_ln_aewr | 0.1612 | 0.1208 | 0.1248 | 0.2147 | 0.3493 | 44325 |    17 |  |
| drop_top1pct_workers | concentration_robustness | nbr_workers_certified_start_year | E_p25_x_ln_aewr | 0.1002 | 0.1183 | 0.1219 | 0.4229 | 0.3413 | 44325 |    17 |  |
| geff_fisher_hours | concentration_metrics | man_hours_certified_start_year | G_eff_fisher | 13.36 | — | — | — | — | 44805 |    17 | G_eff_outcome=13.09, G_eff_county=13.46, hessian_rel_err=1.35e-09 |
| ca_fisher_share_hours | concentration_metrics | man_hours_certified_start_year | ca_fisher_share | 0.1207 | — | — | — | — | 44805 |    17 | CA hours share=0.0826 |
| geff_fisher_workers | concentration_metrics | nbr_workers_certified_start_year | G_eff_fisher | 12.25 | — | — | — | — | 44805 |    17 | G_eff_outcome=12.03, hessian_rel_err=4.85e-08 |
| ca_fisher_share_workers | concentration_metrics | nbr_workers_certified_start_year | ca_fisher_share | 0.1311 | — | — | — | — | 44805 |    17 | Region 17 share of total FE-PPML Fisher information for positions |
| var_split_1a_unweighted | variance_decomposition | E_p25_x_ln_aewr | share_1a_of_1b | 0.02609 | — | — | — | — | 46620 |    17 | R2(C+RY)=0.995476, R2(C+RY+E*Y)=0.999882, R2(C+SY)=0.996106, within_SY_share=0.8606 |
| var_split_1a_ppml_weighted | variance_decomposition | E_p25_x_ln_aewr | share_1a_of_1b | 0.03439 | — | — | — | — | 44805 |    17 | R2(C+RY)=0.996476, R2(C+RY+E*Y)=0.999879, R2(C+SY)=0.996958, within_SY_share=0.8631 |
| state_year_noise_ratio_hours | variance_decomposition | man_hours_certified_start_year | ms_ratio_sy_to_county | 2.846 | — | — | — | — | 44805 |    17 | R2_SY_within_RY=0.0291, RE_var_ratio=0.0297, df=(465,44085), p=1.432e-81 |
| state_year_noise_ratio_workers | variance_decomposition | nbr_workers_certified_start_year | ms_ratio_sy_to_county | 2.292 | — | — | — | — | 44805 |    17 | R2_SY_within_RY=0.0236, RE_var_ratio=0.0208, df=(465,44085), p=2.951e-48 |
| farm_fisher_price | farm_outcomes | fisher_index_ppi | E_p25_x_ln_aewr | -5.68 | 4.353 |  4.28 | 0.2032 | 11.98 | 43359 |    17 |  |
| farm_labor_share | farm_outcomes | share_farm_laborexp_prodexp | E_p25_x_ln_aewr | 0.004303 | 0.003146 | 0.003162 | 0.1924 | 0.008852 | 36492 |    17 |  |
| farm_emp_ppml | farm_outcomes | emp_farm | E_p25_x_ln_aewr | 0.00135 | 0.01485 | 0.01462 | 0.9276 | 0.04094 | 36492 |    17 |  |
| farm_emp_ols | farm_outcomes | emp_farm | E_p25_x_ln_aewr | 7.538 | 14.46 |    14 | 0.5976 | 39.19 | 36948 |    17 |  |
| domestic_floor_hours | standalone_domestic_floor | man_hours_certified_start_year | ln_prevailing_ag_min_wage | 0.3764 | 0.2276 | 0.3736 | 0.3189 | 1.046 | 44805 |    48 | Standalone domestic-floor margin (NOT an AEWR elasticity): all 48 contiguous states retained through 2025 (46 multi-state-region states provide within-region variation; CA and FL absorbed by Region x Year FE) |
| domestic_floor_workers | standalone_domestic_floor | nbr_workers_certified_start_year | ln_prevailing_ag_min_wage | 0.3134 | 0.1416 | 0.2123 | 0.1465 | 0.5944 | 44805 |    48 | Standalone domestic-floor margin (NOT an AEWR elasticity): all 48 contiguous states retained through 2025 (46 multi-state-region states provide within-region variation; CA and FL absorbed by Region x Year FE) |

## 5. Leave-One-Region-Out (`1b-PPML` Primary Hours & Positions)

| spec_id | omitted_cluster | states | n_counties_region | hours_share_2011_2025 | fisher_info_share_hours | estimate |
| --- | --- | --- | --- | --- | --- | --- |
| main_p25_hours | 1 | OR,WA |    75 | 0.09111 | 0.04777 | -0.0187 |
| main_p25_hours | 2 | ID,MT,WY |   123 | 0.03514 | 0.05665 | 0.01741 |
| main_p25_hours | 3 | KS,ND,NE,SD |   317 | 0.04278 | 0.04271 | 0.006932 |
| main_p25_hours | 4 | MI,MN,WI |   242 | 0.04249 | 0.03827 | -0.0105 |
| main_p25_hours | 5 | CT,MA,ME,NH,NY,RI,VT |   129 | 0.04156 | 0.06684 | 0.002339 |
| main_p25_hours | 6 | DE,MD,NJ,PA |   115 | 0.02417 | 0.01499 | -0.006467 |
| main_p25_hours | 7 | IL,IN,OH |   282 | 0.02791 | 0.0254 | 0.01109 |
| main_p25_hours | 8 | KY,TN,WV |   270 | 0.05809 | 0.0742 | -0.009573 |
| main_p25_hours | 9 | NC,VA |   234 | 0.1044 | 0.06329 | -0.02885 |
| main_p25_hours | 10 | AL,GA,SC |   272 | 0.1029 | 0.07878 | 0.01454 |
| main_p25_hours | 11 | FL |    67 | 0.124 | 0.1237 | 0.09077 |
| main_p25_hours | 12 | AR,LA,MS |   221 | 0.09103 | 0.08765 | -0.05859 |
| main_p25_hours | 13 | IA,MO |   214 | 0.01665 | 0.01702 | -0.01063 |
| main_p25_hours | 14 | OK,TX |   331 | 0.04668 | 0.05236 | -0.009117 |
| main_p25_hours | 15 | AZ,NM |    48 | 0.03311 | 0.0597 | -0.01025 |
| main_p25_hours | 16 | CO,NV,UT |   110 | 0.03544 | 0.03001 | -0.000527 |
| main_p25_hours | 17 | CA |    58 | 0.08256 | 0.1207 | 0.07437 |
| main_p25_workers | 1 | OR,WA |    75 | 0.09111 | 0.04777 | -0.04102 |
| main_p25_workers | 2 | ID,MT,WY |   123 | 0.03514 | 0.05665 | 0.01596 |
| main_p25_workers | 3 | KS,ND,NE,SD |   317 | 0.04278 | 0.04271 | -0.008123 |
| main_p25_workers | 4 | MI,MN,WI |   242 | 0.04249 | 0.03827 | -0.02306 |
| main_p25_workers | 5 | CT,MA,ME,NH,NY,RI,VT |   129 | 0.04156 | 0.06684 | -0.0001424 |
| main_p25_workers | 6 | DE,MD,NJ,PA |   115 | 0.02417 | 0.01499 | -0.01753 |
| main_p25_workers | 7 | IL,IN,OH |   282 | 0.02791 | 0.0254 | -0.003897 |
| main_p25_workers | 8 | KY,TN,WV |   270 | 0.05809 | 0.0742 | -0.02655 |
| main_p25_workers | 9 | NC,VA |   234 | 0.1044 | 0.06329 | -0.0591 |
| main_p25_workers | 10 | AL,GA,SC |   272 | 0.1029 | 0.07878 | 0.01416 |
| main_p25_workers | 11 | FL |    67 | 0.124 | 0.1237 | 0.04829 |
| main_p25_workers | 12 | AR,LA,MS |   221 | 0.09103 | 0.08765 | -0.04133 |
| main_p25_workers | 13 | IA,MO |   214 | 0.01665 | 0.01702 | -0.02605 |
| main_p25_workers | 14 | OK,TX |   331 | 0.04668 | 0.05236 | -0.02332 |
| main_p25_workers | 15 | AZ,NM |    48 | 0.03311 | 0.0597 | -0.01114 |
| main_p25_workers | 16 | CO,NV,UT |   110 | 0.03544 | 0.03001 | -0.02184 |
| main_p25_workers | 17 | CA |    58 | 0.08256 | 0.1207 | 0.05924 |
