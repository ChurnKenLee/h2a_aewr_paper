# Border study: public-data plan and DOL archival request

Non-public NASS data and custom tabulations are unavailable by user instruction. The former NASS access/tabulation proposal is withdrawn. No institution, eligible project lead or secure-access application is needed for the current county analysis.

## Current county analysis

Use the existing public BEA farm-job denominators and published 2007 Census hired-worker totals. These county totals cover agricultural activity beyond H-2A applicants; they do not require identifying which individual farms participate. The county comparison is already runnable from the retained public inputs, with its identification and precision limitations reported separately.

Nonparticipant farm records would be needed for a farm-level probability of applying or adopting H-2A: the denominator would be all eligible farms, including those never observed in application files. That is a different outcome from certifications per county farm job. It is not a prerequisite for this project and is not pursued through restricted NASS access.

## Public-data spatial extension, if pursued

One possible different estimand is allocated certifications or position-days per predetermined agricultural acre. It would use actual worksite geocodes, a geographic frame defined independently of subsequent H-2A use, and public land-use data. Areas with no certifications would remain in the frame, while incomplete geocoding or source coverage must not be coded as zero. This measures geographic utilization intensity, not the share of farms adopting H-2A. No such finer estimator is implemented or selected by the current registry.

## DOL/OFLC archival records request — draft, not sent

For a longer historical extension, ask first whether downloadable electronic FY2000--2007 case/job-order records and dictionaries exist, including worksite records where retained. Relevant fields are case/job-order IDs, status/revision history, locations, requested/certified positions and date intervals. Request documentation of changes in reporting units and coverage and release/amendment IDs to avoid counting administrative versions twice.

For 2009--early 2010, seek the OES wage determinations actually used: area definitions/crosswalks, SOC, skill, rate and validity dates, applicable rule/job-order cohort, and recorded prevailing/minimum/CBA overrides. Application-wage averages in the 2010 final rule cannot supply those rates. For 2023--2024, seek archived occupation lookups and job-order duties/cohort fields. Do not assert that the requested records were retained. These gaps do not prevent the current 2011--2022 county comparisons from running.

The [OFLC performance page](https://www.dol.gov/agencies/eta/foreign-labor/performance) is the public release entry point. The [SeasonalJobs feeds](https://seasonaljobs.dol.gov/feeds) cover a recent update window and cannot reconstruct this historical panel. Report generation does not submit this draft.

## Public worksite geocoding

The generated batches contain actual worksite streets from public disclosures. `H2A_RUN_PUBLIC_GEOCODER=1` enables the Census batch requests; default execution prepares requests and reads caches. The source README documents the 10,000-address batches, 50-batch cap, cache and failure behavior. No restricted NASS permission is involved. The earlier execution environment failed to resolve the public Census/FLAG hostnames; this was a programmatic download failure, not an institutional data-access requirement. The geocoding manifest records the latest run and coverage.

Address-range matches still need outcome-linkage, coordinate-quality and boundary-vintage review. See the [Census API documentation](https://geocoding.geo.census.gov/geocoder/Geocoding_Services_API.html). Employer mailing addresses or filing-agent locations cannot substitute for worksites.
