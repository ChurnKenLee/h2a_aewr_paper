# Complete chapter-direction specification ledger

Generated together with the [adviser report](chapter_directions_exploration.md). Unavailable entries are retained. Administrative `tau=2013` names the event; `term` identifies the year contrast. Intervals are state jackknife t intervals. `se_crv1` is secondary. `h0` means baseline certified hours; baseline-position weights use the 2011/2012 mean. The three heterogeneity intervals also have simultaneous and Holm-adjusted results. All earnings rows here are pre-policy.

## Wage extensions

[Full CSV with formula and inference fields](/tmp/h2a-chapter-directions/wage_estimates.csv)

| specification | tau | tier | weights | term | status | n | states | estimate | se_jackknife | low | high | p | p_holm | low_simultaneous | high_simultaneous |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| occupation_dw | 2024 | R7 | h0 | dose | estimated | 9492 | 46 | 0.017211771 | 0.02457291 | -0.03228060 | 0.0667041 | 0.4872603549709 | — | — | — |
| occupation_reduce | 2024 | R7 | h0 | dose | estimated | 9492 | 46 | 0.001172689 | 0.00183556 | -0.00252431 | 0.0048697 | 0.5261437215017 | — | — | — |
| occupation_retain | 2024 | R7 | h0 | dose | estimated | 9492 | 46 | 0.000532708 | 0.00485275 | -0.00924123 | 0.0103066 | 0.9130763196675 | — | — | — |
| occupation_dp | 2024 | R7 | h0 | dose | estimated | 9492 | 46 | 0.017211771 | 0.02457291 | -0.03228060 | 0.0667041 | 0.4872603549709 | — | — | — |
| occupation_dnear | 2024 | R7 | h0 | dose | estimated | 9492 | 46 | 0.005684271 | 0.00825664 | -0.01094546 | 0.0223140 | 0.4947047413314 | — | — | — |
| occupation_dcdf0 | 2024 | R7 | h0 | dose | estimated | 9492 | 46 | -0.002422553 | 0.00275644 | -0.00797431 | 0.0031292 | 0.3841408461077 | — | — | — |
| occupation_dcdf25 | 2024 | R7 | h0 | dose | estimated | 9492 | 46 | -0.002247268 | 0.00252621 | -0.00733531 | 0.0028408 | 0.3784214362396 | — | — | — |
| occupation_dcdf100 | 2024 | R7 | h0 | dose | estimated | 9492 | 46 | -0.002446289 | 0.00275649 | -0.00799815 | 0.0031056 | 0.3795473272293 | — | — | — |
| occupation_dw | 2024 | R7 | equal | dose | estimated | 9492 | 46 | 0.008609923 | 0.02569422 | -0.04314088 | 0.0603607 | 0.7391128802588 | — | — | — |
| occupation_reduce | 2024 | R7 | equal | dose | estimated | 9492 | 46 | 0.000491153 | 0.00084439 | -0.00120953 | 0.0021918 | 0.5636929464142 | — | — | — |
| occupation_retain | 2024 | R7 | equal | dose | estimated | 9492 | 46 | 0.006657276 | 0.00838758 | -0.01023618 | 0.0235507 | 0.4315320785330 | — | — | — |
| occupation_dp | 2024 | R7 | equal | dose | estimated | 9492 | 46 | 0.008609923 | 0.02569422 | -0.04314088 | 0.0603607 | 0.7391128802588 | — | — | — |
| occupation_dnear | 2024 | R7 | equal | dose | estimated | 9492 | 46 | -0.007180910 | 0.00964528 | -0.02660750 | 0.0122457 | 0.4604446000062 | — | — | — |
| occupation_dcdf0 | 2024 | R7 | equal | dose | estimated | 9492 | 46 | -0.001391322 | 0.00334005 | -0.00811853 | 0.0053359 | 0.6789829011910 | — | — | — |
| occupation_dcdf25 | 2024 | R7 | equal | dose | estimated | 9492 | 46 | -0.002327015 | 0.00218894 | -0.00673576 | 0.0020817 | 0.2934177808634 | — | — | — |
| occupation_dcdf100 | 2024 | R7 | equal | dose | estimated | 9492 | 46 | -0.004531848 | 0.00361799 | -0.01181886 | 0.0027552 | 0.2168267248091 | — | — | — |
| schedule_envelope_dnear_candidate | 2024 | R7 | h0 | dose | estimated | 7850 | 44 | 0.007046165 | 0.00910404 | -0.01131388 | 0.0254062 | 0.4431888823635 | — | — | — |
| schedule_envelope_dpremium_lower | 2024 | R7 | h0 | dose | estimated | 7850 | 44 | 0.003174859 | 0.02142327 | -0.04002929 | 0.0463790 | 0.8828802793318 | — | — | — |
| schedule_envelope_dnear_candidate | 2024 | R7 | equal | dose | estimated | 7850 | 44 | -0.006705845 | 0.01026956 | -0.02741639 | 0.0140047 | 0.5172438046779 | — | — | — |
| schedule_envelope_dpremium_lower | 2024 | R7 | equal | dose | estimated | 7850 | 44 | -0.013225316 | 0.02317583 | -0.05996383 | 0.0335132 | 0.5712063348597 | — | — | — |
| filer_state_unrestricted | 2024 | R7 | h0 | dose | estimated | 10647 | 46 | -0.015080377 | 0.04313831 | -0.10196539 | 0.0718046 | 0.7282831643943 | — | — | — |
| filer_state_fixed_composition | 2024 | R7 | h0 | dose | estimated | 9152 | 46 | 0.020152618 | 0.02773925 | -0.03571711 | 0.0760223 | 0.4712935012276 | — | — | — |
| filer_state_unrestricted_common_sample | 2024 | R7 | h0 | dose | estimated | 9152 | 46 | 0.007096420 | 0.02787706 | -0.04905087 | 0.0632437 | 0.8002217145100 | — | — | — |
| filer_state_unrestricted | 2024 | R7 | equal | dose | estimated | 10647 | 46 | 0.003481000 | 0.03343957 | -0.06386975 | 0.0708317 | 0.9175539869720 | — | — | — |
| filer_state_fixed_composition | 2024 | R7 | equal | dose | estimated | 9152 | 46 | 0.008690603 | 0.02320249 | -0.03804162 | 0.0554228 | 0.7097510274887 | — | — | — |
| filer_state_unrestricted_common_sample | 2024 | R7 | equal | dose | estimated | 9152 | 46 | 0.006298517 | 0.02876425 | -0.05163565 | 0.0642327 | 0.8276633097606 | — | — | — |
| within_filer_across_state | 2024 | R7 | h0 | dose | estimated | 1572 | 46 | -0.051035227 | 0.16954280 | -0.39251195 | 0.2904415 | 0.7647882179946 | — | — | — |
| within_filer_across_state | 2024 | R7 | equal | dose | estimated | 1572 | 46 | 0.005057150 | 0.05158615 | -0.09884268 | 0.1089570 | 0.9223413741749 | — | — | — |
| selection_stay | 2024 | R7 | h0 | dose | estimated | 12332 | 46 | -0.001341381 | 0.00543281 | -0.01228362 | 0.0096009 | 0.8061056706260 | — | — | — |
| selection_any_cert | 2024 | R7 | h0 | dose | estimated | 12332 | 46 | -0.000664655 | 0.00522960 | -0.01119760 | 0.0098683 | 0.8994314754582 | — | — | — |
| selection_stay | 2024 | R7 | equal | dose | estimated | 12332 | 46 | -0.000410651 | 0.00636828 | -0.01323703 | 0.0124157 | 0.9488706395709 | — | — | — |
| selection_any_cert | 2024 | R7 | equal | dose | estimated | 12332 | 46 | 0.000399979 | 0.00557333 | -0.01082528 | 0.0116252 | 0.9431057162046 | — | — | — |
| selection_baseline_premium | 2024 | R7 | equal | stay | estimated | 12332 | 46 | -0.339137810 | 0.11949967 | -0.57982249 | -0.0984531 | 0.0067868442620 | — | — | — |
| selection_baseline_premium | 2024 | R7 | equal | dose:stay | estimated | 12332 | 46 | -0.047478311 | 0.02613751 | -0.10012196 | 0.0051653 | 0.0759610337646 | — | — | — |
| selection_prechange | 2024 | R7 | equal | stay | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| selection_prechange | 2024 | R7 | equal | dose:stay | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| occupation_dw | 2025 | R7 | h0 | dose | estimated | 11430 | 46 | 0.013896467 | 0.02499807 | -0.03645223 | 0.0642452 | 0.5810317498118 | — | — | — |
| occupation_reduce | 2025 | R7 | h0 | dose | estimated | 11430 | 46 | -0.000960280 | 0.00373449 | -0.00848194 | 0.0065614 | 0.7982444109599 | — | — | — |
| occupation_retain | 2025 | R7 | h0 | dose | estimated | 11430 | 46 | 0.000828508 | 0.00626856 | -0.01179701 | 0.0134540 | 0.8954399987725 | — | — | — |
| occupation_dp | 2025 | R7 | h0 | dose | estimated | 11430 | 46 | 0.013896467 | 0.02499807 | -0.03645223 | 0.0642452 | 0.5810317498118 | — | — | — |
| occupation_dnear | 2025 | R7 | h0 | dose | estimated | 11430 | 46 | -0.003774386 | 0.01674353 | -0.03749758 | 0.0299488 | 0.8226703798075 | — | — | — |
| occupation_dcdf0 | 2025 | R7 | h0 | dose | estimated | 11430 | 46 | -0.003160801 | 0.00570958 | -0.01466049 | 0.0083389 | 0.5825959358297 | — | — | — |
| occupation_dcdf25 | 2025 | R7 | h0 | dose | estimated | 11430 | 46 | -0.002665835 | 0.00538990 | -0.01352164 | 0.0081900 | 0.6232897059462 | — | — | — |
| occupation_dcdf100 | 2025 | R7 | h0 | dose | estimated | 11430 | 46 | 0.000446092 | 0.00329393 | -0.00618822 | 0.0070804 | 0.8928769930054 | — | — | — |
| occupation_dw | 2025 | R7 | equal | dose | estimated | 11430 | 46 | -0.002563839 | 0.01205677 | -0.02684743 | 0.0217197 | 0.8325625319722 | — | — | — |
| occupation_reduce | 2025 | R7 | equal | dose | estimated | 11430 | 46 | 0.000515412 | 0.00442756 | -0.00840214 | 0.0094330 | 0.9078454717471 | — | — | — |
| occupation_retain | 2025 | R7 | equal | dose | estimated | 11430 | 46 | 0.001464110 | 0.00742596 | -0.01349255 | 0.0164208 | 0.8445893762998 | — | — | — |
| occupation_dp | 2025 | R7 | equal | dose | estimated | 11430 | 46 | -0.002563839 | 0.01205677 | -0.02684743 | 0.0217197 | 0.8325625319722 | — | — | — |
| occupation_dnear | 2025 | R7 | equal | dose | estimated | 11430 | 46 | 0.008376173 | 0.03086018 | -0.05377941 | 0.0705318 | 0.7873067450400 | — | — | — |
| occupation_dcdf0 | 2025 | R7 | equal | dose | estimated | 11430 | 46 | -0.002560632 | 0.00701126 | -0.01668204 | 0.0115608 | 0.7166604201997 | — | — | — |
| occupation_dcdf25 | 2025 | R7 | equal | dose | estimated | 11430 | 46 | -0.002018990 | 0.00567627 | -0.01345158 | 0.0094136 | 0.7237349277023 | — | — | — |
| occupation_dcdf100 | 2025 | R7 | equal | dose | estimated | 11430 | 46 | 0.000531783 | 0.00139426 | -0.00227639 | 0.0033400 | 0.7046944586567 | — | — | — |
| schedule_envelope_dnear_candidate | 2025 | R7 | h0 | dose | estimated | 9168 | 44 | -0.000921544 | 0.02123262 | -0.04374120 | 0.0418981 | 0.9655818012912 | — | — | — |
| schedule_envelope_dpremium_lower | 2025 | R7 | h0 | dose | estimated | 9168 | 44 | 0.001696371 | 0.00959991 | -0.01766369 | 0.0210564 | 0.8605682289434 | — | — | — |
| schedule_envelope_dnear_candidate | 2025 | R7 | equal | dose | estimated | 9168 | 44 | 0.009281518 | 0.03639332 | -0.06411260 | 0.0826756 | 0.7999130509706 | — | — | — |
| schedule_envelope_dpremium_lower | 2025 | R7 | equal | dose | estimated | 9168 | 44 | -0.003616212 | 0.01642573 | -0.03674185 | 0.0295094 | 0.8267920709332 | — | — | — |
| filer_state_unrestricted | 2025 | R7 | h0 | dose | estimated | 11477 | 46 | 0.026978175 | 0.03258657 | -0.03865456 | 0.0926109 | 0.4121007061747 | — | — | — |
| filer_state_fixed_composition | 2025 | R7 | h0 | dose | estimated | 10775 | 46 | 0.014739582 | 0.02359344 | -0.03278005 | 0.0622592 | 0.5353018817178 | — | — | — |
| filer_state_unrestricted_common_sample | 2025 | R7 | h0 | dose | estimated | 10775 | 46 | 0.029536304 | 0.03087442 | -0.03264797 | 0.0917206 | 0.3438506883489 | — | — | — |
| filer_state_unrestricted | 2025 | R7 | equal | dose | estimated | 11477 | 46 | -0.012251723 | 0.02109301 | -0.05473523 | 0.0302318 | 0.5642442801905 | — | — | — |
| filer_state_fixed_composition | 2025 | R7 | equal | dose | estimated | 10775 | 46 | -0.005460651 | 0.01336091 | -0.03237091 | 0.0214496 | 0.6846958180063 | — | — | — |
| filer_state_unrestricted_common_sample | 2025 | R7 | equal | dose | estimated | 10775 | 46 | -0.004542367 | 0.01505713 | -0.03486898 | 0.0257842 | 0.7642891851808 | — | — | — |
| within_filer_across_state | 2025 | R7 | h0 | dose | estimated | 1761 | 46 | 0.041630067 | 0.07743556 | -0.11433315 | 0.1975933 | 0.5934957398354 | — | — | — |
| within_filer_across_state | 2025 | R7 | equal | dose | estimated | 1761 | 46 | 0.029413855 | 0.03027152 | -0.03155612 | 0.0903838 | 0.3364101429382 | — | — | — |
| selection_stay | 2025 | R7 | h0 | dose | estimated | 13301 | 46 | -0.007714324 | 0.00732275 | -0.02246310 | 0.0070345 | 0.2977479009614 | — | — | — |
| selection_any_cert | 2025 | R7 | h0 | dose | estimated | 13301 | 46 | -0.007904410 | 0.00728454 | -0.02257623 | 0.0067674 | 0.2836591857117 | — | — | — |
| selection_stay | 2025 | R7 | equal | dose | estimated | 13301 | 46 | -0.008092581 | 0.00518232 | -0.01853031 | 0.0023452 | 0.1253943306038 | — | — | — |
| selection_any_cert | 2025 | R7 | equal | dose | estimated | 13301 | 46 | -0.008446220 | 0.00539575 | -0.01931381 | 0.0024214 | 0.1245073820786 | — | — | — |
| selection_baseline_premium | 2025 | R7 | equal | stay | estimated | 13301 | 46 | -0.067909278 | 0.12828025 | -0.32627897 | 0.1904604 | 0.5991423904397 | — | — | — |
| selection_baseline_premium | 2025 | R7 | equal | dose:stay | estimated | 13301 | 46 | 0.035814930 | 0.02937616 | -0.02335169 | 0.0949815 | 0.2291268070961 | — | — | — |
| selection_prechange | 2025 | R7 | equal | stay | estimated | 10647 | 46 | 0.133970863 | 0.12028850 | -0.10830261 | 0.3762443 | 0.2713017537702 | — | — | — |
| selection_prechange | 2025 | R7 | equal | dose:stay | estimated | 10647 | 46 | 0.056877696 | 0.03297090 | -0.00952911 | 0.1232845 | 0.0913744055973 | — | — | — |
| occupation_dw | 2026 | R7 | h0 | dose | estimated | 12386 | 46 | 0.640361725 | 0.09606015 | 0.44688664 | 0.8338368 | 0.0000000319015 | — | — | — |
| occupation_reduce | 2026 | R7 | h0 | dose | estimated | 12386 | 46 | -0.010130933 | 0.02944835 | -0.06944294 | 0.0491811 | 0.7324307834563 | — | — | — |
| occupation_retain | 2026 | R7 | h0 | dose | estimated | 12386 | 46 | -0.012077465 | 0.01573934 | -0.04377813 | 0.0196232 | 0.4468855846018 | — | — | — |
| occupation_dp | 2026 | R7 | h0 | dose | estimated | 12386 | 46 | -0.359638275 | 0.09606015 | -0.55311336 | -0.1661632 | 0.0005123040934 | — | — | — |
| occupation_dnear | 2026 | R7 | h0 | dose | estimated | 12386 | 46 | 0.025682853 | 0.03282889 | -0.04043793 | 0.0918036 | 0.4381211829466 | — | — | — |
| occupation_dcdf0 | 2026 | R7 | h0 | dose | estimated | 12386 | 46 | 0.018551399 | 0.02409119 | -0.02997075 | 0.0670735 | 0.4452946403783 | — | — | — |
| occupation_dcdf25 | 2026 | R7 | h0 | dose | estimated | 12386 | 46 | 0.020668797 | 0.02145127 | -0.02253628 | 0.0638739 | 0.3404344426595 | — | — | — |
| occupation_dcdf100 | 2026 | R7 | h0 | dose | estimated | 12386 | 46 | 0.046771069 | 0.03258537 | -0.01885923 | 0.1124014 | 0.1581049728724 | — | — | — |
| occupation_dw | 2026 | R7 | equal | dose | estimated | 12386 | 46 | 0.601347669 | 0.17551311 | 0.24784612 | 0.9548492 | 0.0013174045806 | — | — | — |
| occupation_reduce | 2026 | R7 | equal | dose | estimated | 12386 | 46 | -0.024784670 | 0.03022079 | -0.08565247 | 0.0360831 | 0.4164708538281 | — | — | — |
| occupation_retain | 2026 | R7 | equal | dose | estimated | 12386 | 46 | -0.011820412 | 0.01563346 | -0.04330782 | 0.0196670 | 0.4535311812401 | — | — | — |
| occupation_dp | 2026 | R7 | equal | dose | estimated | 12386 | 46 | -0.398652331 | 0.17551311 | -0.75215388 | -0.0451508 | 0.0279578338346 | — | — | — |
| occupation_dnear | 2026 | R7 | equal | dose | estimated | 12386 | 46 | -0.008852861 | 0.02679120 | -0.06281312 | 0.0451074 | 0.7426017935425 | — | — | — |
| occupation_dcdf0 | 2026 | R7 | equal | dose | estimated | 12386 | 46 | 0.012766748 | 0.02305722 | -0.03367288 | 0.0592064 | 0.5825262012525 | — | — | — |
| occupation_dcdf25 | 2026 | R7 | equal | dose | estimated | 12386 | 46 | 0.012762108 | 0.02359301 | -0.03475666 | 0.0602809 | 0.5912253810837 | — | — | — |
| occupation_dcdf100 | 2026 | R7 | equal | dose | estimated | 12386 | 46 | 0.017537349 | 0.03024127 | -0.04337169 | 0.0784464 | 0.5648647234528 | — | — | — |
| schedule_envelope_dnear_candidate | 2026 | R7 | h0 | dose | estimated | 8272 | 39 | 0.036546686 | 0.06060910 | -0.08615002 | 0.1592434 | 0.5500974546064 | — | — | — |
| schedule_envelope_dpremium_lower | 2026 | R7 | h0 | dose | estimated | 8272 | 39 | -0.304846757 | 0.26676512 | -0.84488450 | 0.2351910 | 0.2602899712305 | — | — | — |
| schedule_envelope_dnear_candidate | 2026 | R7 | equal | dose | estimated | 8272 | 39 | 0.005796562 | 0.05285668 | -0.10120620 | 0.1127993 | 0.9132516535978 | — | — | — |
| schedule_envelope_dpremium_lower | 2026 | R7 | equal | dose | estimated | 8272 | 39 | -0.443536930 | 0.33174965 | -1.11512899 | 0.2280551 | 0.1891839632838 | — | — | — |
| filer_state_unrestricted | 2026 | R7 | h0 | dose | estimated | 12313 | 46 | 0.649946323 | 0.08955126 | 0.46958083 | 0.8303118 | 0.0000000042316 | — | — | — |
| filer_state_fixed_composition | 2026 | R7 | h0 | dose | estimated | 11691 | 46 | 0.639310665 | 0.09710821 | 0.44372468 | 0.8348966 | 0.0000000423547 | — | — | — |
| filer_state_unrestricted_common_sample | 2026 | R7 | h0 | dose | estimated | 11691 | 46 | 0.644946815 | 0.09489213 | 0.45382425 | 0.8360694 | 0.0000000204191 | — | — | — |
| filer_state_unrestricted | 2026 | R7 | equal | dose | estimated | 12313 | 46 | 0.615430470 | 0.16246402 | 0.28821113 | 0.9426498 | 0.0004479152076 | — | — | — |
| filer_state_fixed_composition | 2026 | R7 | equal | dose | estimated | 11691 | 46 | 0.621805608 | 0.16604712 | 0.28736954 | 0.9562417 | 0.0005109585434 | — | — | — |
| filer_state_unrestricted_common_sample | 2026 | R7 | equal | dose | estimated | 11691 | 46 | 0.622673873 | 0.16550920 | 0.28932123 | 0.9560265 | 0.0004846642290 | — | — | — |
| within_filer_across_state | 2026 | R7 | h0 | dose | estimated | 1903 | 46 | 0.407829017 | 0.13529606 | 0.13532876 | 0.6803293 | 0.0042208499936 | — | — | — |
| within_filer_across_state | 2026 | R7 | equal | dose | estimated | 1903 | 46 | 0.183214904 | 0.05284308 | 0.07678348 | 0.2896463 | 0.0011689472002 | — | — | — |
| selection_stay | 2026 | R7 | h0 | dose | estimated | 14046 | 46 | -0.003610656 | 0.00549800 | -0.01468419 | 0.0074629 | 0.5147043444111 | — | — | — |
| selection_any_cert | 2026 | R7 | h0 | dose | estimated | 14046 | 46 | -0.004022901 | 0.00570983 | -0.01552310 | 0.0074773 | 0.4847169005346 | — | — | — |
| selection_stay | 2026 | R7 | equal | dose | estimated | 14046 | 46 | -0.003700231 | 0.00479284 | -0.01335350 | 0.0059530 | 0.4441304727735 | — | — | — |
| selection_any_cert | 2026 | R7 | equal | dose | estimated | 14046 | 46 | -0.004559232 | 0.00590133 | -0.01644512 | 0.0073267 | 0.4438115536096 | — | — | — |
| selection_baseline_premium | 2026 | R7 | equal | stay | estimated | 14046 | 46 | -0.404242864 | 0.29865346 | -1.00576182 | 0.1972761 | 0.1826395625116 | — | — | — |
| selection_baseline_premium | 2026 | R7 | equal | dose:stay | estimated | 14046 | 46 | -0.024256460 | 0.07003321 | -0.16531058 | 0.1167977 | 0.7306890819429 | — | — | — |
| selection_prechange | 2026 | R7 | equal | stay | estimated | 11477 | 46 | 0.426866018 | 0.22824641 | -0.03284584 | 0.8865779 | 0.0679695084222 | — | — | — |
| selection_prechange | 2026 | R7 | equal | dose:stay | estimated | 11477 | 46 | 0.086095140 | 0.04780067 | -0.01018036 | 0.1823706 | 0.0783859136363 | — | — | — |
| heterogeneity_highpremium | 2026 | R7 | h0 | dose:highpremium | estimated | 12386 | 46 | -0.204558237 | 0.50784954 | -1.22741972 | 0.8183032 | 0.6890077598597 | 1.0000 | -1.46747 | 1.05835 |
| heterogeneity_contractor | 2026 | R7 | h0 | dose:contractor | estimated | 12386 | 46 | 0.270429007 | 0.16266491 | -0.05719494 | 0.5980530 | 0.1033654617223 | 0.3101 | -0.13408 | 0.67494 |
| heterogeneity_large | 2026 | R7 | h0 | dose:large | estimated | 12386 | 46 | 0.032383222 | 0.11183746 | -0.19286898 | 0.2576354 | 0.7734857928814 | 1.0000 | -0.24573 | 0.31050 |
| receipt_aligned_single_state_common_SOC | 2026 | R7 | h0 | dose | estimated | 9982 | 46 | 0.694941103 | 0.09220634 | 0.50922801 | 0.8806542 | 0.0000000016413 | — | — | — |
| receipt_aligned_single_state_common_SOC | 2026 | R7 | equal | dose | estimated | 9982 | 46 | 0.701721437 | 0.14779161 | 0.40405385 | 0.9993890 | 0.0000211797210 | — | — | — |
| occupation_dw | 2024 | R5 | h0 | dose | estimated | 9398 | 46 | 0.018655088 | 0.02515126 | -0.03200215 | 0.0693123 | 0.4621132304160 | — | — | — |
| occupation_reduce | 2024 | R5 | h0 | dose | estimated | 9398 | 46 | 0.001012276 | 0.00165457 | -0.00232020 | 0.0043448 | 0.5437449483626 | — | — | — |
| occupation_retain | 2024 | R5 | h0 | dose | estimated | 9398 | 46 | 0.000239825 | 0.00493823 | -0.00970627 | 0.0101859 | 0.9614807899024 | — | — | — |
| occupation_dp | 2024 | R5 | h0 | dose | estimated | 9398 | 46 | 0.018655088 | 0.02515126 | -0.03200215 | 0.0693123 | 0.4621132304160 | — | — | — |
| occupation_dnear | 2024 | R5 | h0 | dose | estimated | 9398 | 46 | 0.006158012 | 0.00850880 | -0.01097958 | 0.0232956 | 0.4729813501827 | — | — | — |
| occupation_dcdf0 | 2024 | R5 | h0 | dose | estimated | 9398 | 46 | -0.002423368 | 0.00268992 | -0.00784115 | 0.0029944 | 0.3724341403635 | — | — | — |
| occupation_dcdf25 | 2024 | R5 | h0 | dose | estimated | 9398 | 46 | -0.002468243 | 0.00250281 | -0.00750917 | 0.0025727 | 0.3293142134205 | — | — | — |
| occupation_dcdf100 | 2024 | R5 | h0 | dose | estimated | 9398 | 46 | -0.002703711 | 0.00290625 | -0.00855720 | 0.0031498 | 0.3571753314847 | — | — | — |
| occupation_dw | 2024 | R5 | equal | dose | estimated | 9398 | 46 | 0.010201841 | 0.02589850 | -0.04196041 | 0.0623641 | 0.6955033018580 | — | — | — |
| occupation_reduce | 2024 | R5 | equal | dose | estimated | 9398 | 46 | 0.000325010 | 0.00061929 | -0.00092231 | 0.0015723 | 0.6022916488031 | — | — | — |
| occupation_retain | 2024 | R5 | equal | dose | estimated | 9398 | 46 | 0.006088834 | 0.00800610 | -0.01003628 | 0.0222139 | 0.4509078216716 | — | — | — |
| occupation_dp | 2024 | R5 | equal | dose | estimated | 9398 | 46 | 0.010201841 | 0.02589850 | -0.04196041 | 0.0623641 | 0.6955033018580 | — | — | — |
| occupation_dnear | 2024 | R5 | equal | dose | estimated | 9398 | 46 | -0.006262897 | 0.00944771 | -0.02529156 | 0.0127658 | 0.5107751781458 | — | — | — |
| occupation_dcdf0 | 2024 | R5 | equal | dose | estimated | 9398 | 46 | -0.001279681 | 0.00341472 | -0.00815728 | 0.0055979 | 0.7096035295476 | — | — | — |
| occupation_dcdf25 | 2024 | R5 | equal | dose | estimated | 9398 | 46 | -0.002510580 | 0.00205501 | -0.00664958 | 0.0016284 | 0.2281869955442 | — | — | — |
| occupation_dcdf100 | 2024 | R5 | equal | dose | estimated | 9398 | 46 | -0.004827825 | 0.00382910 | -0.01254004 | 0.0028844 | 0.2138703461054 | — | — | — |
| schedule_envelope_dnear_candidate | 2024 | R5 | h0 | dose | estimated | 7781 | 44 | 0.007624624 | 0.00935499 | -0.01124151 | 0.0264908 | 0.4195448876042 | — | — | — |
| schedule_envelope_dpremium_lower | 2024 | R5 | h0 | dose | estimated | 7781 | 44 | 0.004055766 | 0.02156638 | -0.03943699 | 0.0475485 | 0.8517146231139 | — | — | — |
| schedule_envelope_dnear_candidate | 2024 | R5 | equal | dose | estimated | 7781 | 44 | -0.005492316 | 0.01011861 | -0.02589844 | 0.0149138 | 0.5900745205644 | — | — | — |
| schedule_envelope_dpremium_lower | 2024 | R5 | equal | dose | estimated | 7781 | 44 | -0.012163263 | 0.02273677 | -0.05801632 | 0.0336898 | 0.5954332315209 | — | — | — |
| filer_state_unrestricted | 2024 | R5 | h0 | dose | estimated | 10552 | 46 | -0.013602380 | 0.04309918 | -0.10040859 | 0.0732038 | 0.7537603540530 | — | — | — |
| filer_state_fixed_composition | 2024 | R5 | h0 | dose | estimated | 9068 | 46 | 0.021842398 | 0.02810615 | -0.03476630 | 0.0784511 | 0.4411429207988 | — | — | — |
| filer_state_unrestricted_common_sample | 2024 | R5 | h0 | dose | estimated | 9068 | 46 | 0.008412671 | 0.02854438 | -0.04907866 | 0.0659040 | 0.7695612719187 | — | — | — |
| filer_state_unrestricted | 2024 | R5 | equal | dose | estimated | 10552 | 46 | 0.005991679 | 0.03399543 | -0.06247864 | 0.0744620 | 0.8608887242125 | — | — | — |
| filer_state_fixed_composition | 2024 | R5 | equal | dose | estimated | 9068 | 46 | 0.010464438 | 0.02330366 | -0.03647154 | 0.0574004 | 0.6555529426867 | — | — | — |
| filer_state_unrestricted_common_sample | 2024 | R5 | equal | dose | estimated | 9068 | 46 | 0.008603865 | 0.03028716 | -0.05239760 | 0.0696053 | 0.7776549615131 | — | — | — |
| occupation_dw | 2025 | R5 | h0 | dose | estimated | 11324 | 46 | 0.013962510 | 0.02520082 | -0.03679455 | 0.0647196 | 0.5822877437692 | — | — | — |
| occupation_reduce | 2025 | R5 | h0 | dose | estimated | 11324 | 46 | -0.000936021 | 0.00377124 | -0.00853168 | 0.0066596 | 0.8051090053317 | — | — | — |
| occupation_retain | 2025 | R5 | h0 | dose | estimated | 11324 | 46 | 0.000962482 | 0.00628877 | -0.01170376 | 0.0136287 | 0.8790449991132 | — | — | — |
| occupation_dp | 2025 | R5 | h0 | dose | estimated | 11324 | 46 | 0.013962510 | 0.02520082 | -0.03679455 | 0.0647196 | 0.5822877437692 | — | — | — |
| occupation_dnear | 2025 | R5 | h0 | dose | estimated | 11324 | 46 | -0.003890462 | 0.01632994 | -0.03678066 | 0.0289997 | 0.8127757421515 | — | — | — |
| occupation_dcdf0 | 2025 | R5 | h0 | dose | estimated | 11324 | 46 | -0.003145399 | 0.00571739 | -0.01466081 | 0.0083700 | 0.5849396603576 | — | — | — |
| occupation_dcdf25 | 2025 | R5 | h0 | dose | estimated | 11324 | 46 | -0.002622331 | 0.00539528 | -0.01348898 | 0.0082443 | 0.6292959815555 | — | — | — |
| occupation_dcdf100 | 2025 | R5 | h0 | dose | estimated | 11324 | 46 | 0.000447455 | 0.00333423 | -0.00626803 | 0.0071629 | 0.8938425879789 | — | — | — |
| occupation_dw | 2025 | R5 | equal | dose | estimated | 11324 | 46 | -0.002017897 | 0.01208823 | -0.02636485 | 0.0223291 | 0.8681724636223 | — | — | — |
| occupation_reduce | 2025 | R5 | equal | dose | estimated | 11324 | 46 | 0.000576729 | 0.00441726 | -0.00832009 | 0.0094736 | 0.8967033747026 | — | — | — |
| occupation_retain | 2025 | R5 | equal | dose | estimated | 11324 | 46 | 0.001746566 | 0.00764588 | -0.01365302 | 0.0171462 | 0.8203449464338 | — | — | — |
| occupation_dp | 2025 | R5 | equal | dose | estimated | 11324 | 46 | -0.002017897 | 0.01208823 | -0.02636485 | 0.0223291 | 0.8681724636223 | — | — | — |
| occupation_dnear | 2025 | R5 | equal | dose | estimated | 11324 | 46 | 0.008532368 | 0.03049860 | -0.05289496 | 0.0699597 | 0.7809415694192 | — | — | — |
| occupation_dcdf0 | 2025 | R5 | equal | dose | estimated | 11324 | 46 | -0.002563675 | 0.00704307 | -0.01674915 | 0.0116218 | 0.7175631324426 | — | — | — |
| occupation_dcdf25 | 2025 | R5 | equal | dose | estimated | 11324 | 46 | -0.001919793 | 0.00558103 | -0.01316057 | 0.0093210 | 0.7324597023831 | — | — | — |
| occupation_dcdf100 | 2025 | R5 | equal | dose | estimated | 11324 | 46 | 0.000595466 | 0.00135800 | -0.00213968 | 0.0033306 | 0.6631302481596 | — | — | — |
| schedule_envelope_dnear_candidate | 2025 | R5 | h0 | dose | estimated | 9078 | 44 | -0.000967810 | 0.02072488 | -0.04276352 | 0.0408279 | 0.9629701595070 | — | — | — |
| schedule_envelope_dpremium_lower | 2025 | R5 | h0 | dose | estimated | 9078 | 44 | 0.001815690 | 0.00962250 | -0.01758993 | 0.0212213 | 0.8512218973175 | — | — | — |
| schedule_envelope_dnear_candidate | 2025 | R5 | equal | dose | estimated | 9078 | 44 | 0.009342829 | 0.03589432 | -0.06304498 | 0.0817306 | 0.7958847234423 | — | — | — |
| schedule_envelope_dpremium_lower | 2025 | R5 | equal | dose | estimated | 9078 | 44 | -0.002984122 | 0.01678598 | -0.03683628 | 0.0308680 | 0.8597347894013 | — | — | — |
| filer_state_unrestricted | 2025 | R5 | h0 | dose | estimated | 11379 | 46 | 0.027496904 | 0.03285124 | -0.03866890 | 0.0936627 | 0.4070087599261 | — | — | — |
| filer_state_fixed_composition | 2025 | R5 | h0 | dose | estimated | 10679 | 46 | 0.014624406 | 0.02383426 | -0.03338027 | 0.0626291 | 0.5425773159154 | — | — | — |
| filer_state_unrestricted_common_sample | 2025 | R5 | h0 | dose | estimated | 10679 | 46 | 0.030051577 | 0.03112331 | -0.03263399 | 0.0927371 | 0.3394226012773 | — | — | — |
| filer_state_unrestricted | 2025 | R5 | equal | dose | estimated | 11379 | 46 | -0.011139373 | 0.02137972 | -0.05420035 | 0.0319216 | 0.6049039290843 | — | — | — |
| filer_state_fixed_composition | 2025 | R5 | equal | dose | estimated | 10679 | 46 | -0.004830302 | 0.01356071 | -0.03214296 | 0.0224824 | 0.7233565882033 | — | — | — |
| filer_state_unrestricted_common_sample | 2025 | R5 | equal | dose | estimated | 10679 | 46 | -0.003245758 | 0.01564515 | -0.03475672 | 0.0282652 | 0.8365859494742 | — | — | — |
| occupation_dw | 2026 | R5 | h0 | dose | estimated | 12287 | 46 | 0.639001395 | 0.09586732 | 0.44591469 | 0.8320881 | 0.0000000319869 | — | — | — |
| occupation_reduce | 2026 | R5 | h0 | dose | estimated | 12287 | 46 | -0.009693283 | 0.02895879 | -0.06801929 | 0.0486327 | 0.7393864577000 | — | — | — |
| occupation_retain | 2026 | R5 | h0 | dose | estimated | 12287 | 46 | -0.012331895 | 0.01579959 | -0.04415390 | 0.0194901 | 0.4391715223429 | — | — | — |
| occupation_dp | 2026 | R5 | h0 | dose | estimated | 12287 | 46 | -0.360998605 | 0.09586732 | -0.55408531 | -0.1679119 | 0.0004796336167 | — | — | — |
| occupation_dnear | 2026 | R5 | h0 | dose | estimated | 12287 | 46 | 0.026009047 | 0.03286610 | -0.04018668 | 0.0922048 | 0.4328832003630 | — | — | — |
| occupation_dcdf0 | 2026 | R5 | h0 | dose | estimated | 12287 | 46 | 0.018838466 | 0.02389680 | -0.02929215 | 0.0669691 | 0.4346393964457 | — | — | — |
| occupation_dcdf25 | 2026 | R5 | h0 | dose | estimated | 12287 | 46 | 0.020951158 | 0.02128373 | -0.02191648 | 0.0638188 | 0.3301947477787 | — | — | — |
| occupation_dcdf100 | 2026 | R5 | h0 | dose | estimated | 12287 | 46 | 0.047411526 | 0.03309825 | -0.01925178 | 0.1140748 | 0.1589261306403 | — | — | — |
| occupation_dw | 2026 | R5 | equal | dose | estimated | 12287 | 46 | 0.597844038 | 0.18231645 | 0.23063986 | 0.9650482 | 0.0020133091915 | — | — | — |
| occupation_reduce | 2026 | R5 | equal | dose | estimated | 12287 | 46 | -0.024088882 | 0.02898022 | -0.08245804 | 0.0342803 | 0.4102393606682 | — | — | — |
| occupation_retain | 2026 | R5 | equal | dose | estimated | 12287 | 46 | -0.012273554 | 0.01570857 | -0.04391223 | 0.0193651 | 0.4387006985125 | — | — | — |
| occupation_dp | 2026 | R5 | equal | dose | estimated | 12287 | 46 | -0.402155962 | 0.18231645 | -0.76936014 | -0.0349518 | 0.0325460180149 | — | — | — |
| occupation_dnear | 2026 | R5 | equal | dose | estimated | 12287 | 46 | -0.008606024 | 0.02725390 | -0.06349820 | 0.0462862 | 0.7536353819462 | — | — | — |
| occupation_dcdf0 | 2026 | R5 | equal | dose | estimated | 12287 | 46 | 0.013097344 | 0.02415546 | -0.03555425 | 0.0617489 | 0.5903486264864 | — | — | — |
| occupation_dcdf25 | 2026 | R5 | equal | dose | estimated | 12287 | 46 | 0.013005099 | 0.02457435 | -0.03649018 | 0.0625004 | 0.5992578516109 | — | — | — |
| occupation_dcdf100 | 2026 | R5 | equal | dose | estimated | 12287 | 46 | 0.018232362 | 0.03154607 | -0.04530467 | 0.0817694 | 0.5661722735104 | — | — | — |
| schedule_envelope_dnear_candidate | 2026 | R5 | h0 | dose | estimated | 8207 | 39 | 0.036958903 | 0.06070874 | -0.08593951 | 0.1598573 | 0.5462857176563 | — | — | — |
| schedule_envelope_dpremium_lower | 2026 | R5 | h0 | dose | estimated | 8207 | 39 | -0.307076452 | 0.26622412 | -0.84601900 | 0.2318661 | 0.2559290053038 | — | — | — |
| schedule_envelope_dnear_candidate | 2026 | R5 | equal | dose | estimated | 8207 | 39 | 0.005977898 | 0.05296795 | -0.10125011 | 0.1132059 | 0.9107366342935 | — | — | — |
| schedule_envelope_dpremium_lower | 2026 | R5 | equal | dose | estimated | 8207 | 39 | -0.450923390 | 0.32962899 | -1.11822239 | 0.2163756 | 0.1793562609142 | — | — | — |
| filer_state_unrestricted | 2026 | R5 | h0 | dose | estimated | 12227 | 46 | 0.648565397 | 0.08898849 | 0.46933338 | 0.8277974 | 0.0000000038161 | — | — | — |
| filer_state_fixed_composition | 2026 | R5 | h0 | dose | estimated | 11606 | 46 | 0.637949705 | 0.09707581 | 0.44242899 | 0.8334704 | 0.0000000441047 | — | — | — |
| filer_state_unrestricted_common_sample | 2026 | R5 | h0 | dose | estimated | 11606 | 46 | 0.642992295 | 0.09469936 | 0.45225800 | 0.8337266 | 0.0000000208999 | — | — | — |
| filer_state_unrestricted | 2026 | R5 | equal | dose | estimated | 12227 | 46 | 0.610211416 | 0.16606015 | 0.27574911 | 0.9446737 | 0.0006313874223 | — | — | — |
| filer_state_fixed_composition | 2026 | R5 | equal | dose | estimated | 11606 | 46 | 0.617736843 | 0.17097912 | 0.27336723 | 0.9621065 | 0.0007595602097 | — | — | — |
| filer_state_unrestricted_common_sample | 2026 | R5 | equal | dose | estimated | 11606 | 46 | 0.617712103 | 0.17103886 | 0.27322215 | 0.9622021 | 0.0007627551191 | — | — | — |
| occupation_dw | 2024 | R8 | h0 | dose | estimated | 9589 | 46 | 0.017690117 | 0.02527758 | -0.03322154 | 0.0686018 | 0.4876330634437 | — | — | — |
| occupation_reduce | 2024 | R8 | h0 | dose | estimated | 9589 | 46 | 0.001264926 | 0.00177346 | -0.00230701 | 0.0048369 | 0.4793720567107 | — | — | — |
| occupation_retain | 2024 | R8 | h0 | dose | estimated | 9589 | 46 | 0.000436450 | 0.00472527 | -0.00908073 | 0.0099536 | 0.9268179897378 | — | — | — |
| occupation_dp | 2024 | R8 | h0 | dose | estimated | 9589 | 46 | 0.017690117 | 0.02527758 | -0.03322154 | 0.0686018 | 0.4876330634437 | — | — | — |
| occupation_dnear | 2024 | R8 | h0 | dose | estimated | 9589 | 46 | 0.005693028 | 0.00823263 | -0.01088833 | 0.0222744 | 0.4927911564201 | — | — | — |
| occupation_dcdf0 | 2024 | R8 | h0 | dose | estimated | 9589 | 46 | -0.002512658 | 0.00287548 | -0.00830418 | 0.0032789 | 0.3868554191063 | — | — | — |
| occupation_dcdf25 | 2024 | R8 | h0 | dose | estimated | 9589 | 46 | -0.002320304 | 0.00267077 | -0.00769951 | 0.0030589 | 0.3895787623809 | — | — | — |
| occupation_dcdf100 | 2024 | R8 | h0 | dose | estimated | 9589 | 46 | -0.002681550 | 0.00302252 | -0.00876921 | 0.0034061 | 0.3796932727834 | — | — | — |
| occupation_dw | 2024 | R8 | equal | dose | estimated | 9589 | 46 | 0.011100622 | 0.02929579 | -0.04790413 | 0.0701054 | 0.7065328148799 | — | — | — |
| occupation_reduce | 2024 | R8 | equal | dose | estimated | 9589 | 46 | 0.000607693 | 0.00094779 | -0.00130126 | 0.0025166 | 0.5246659133103 | — | — | — |
| occupation_retain | 2024 | R8 | equal | dose | estimated | 9589 | 46 | 0.006323216 | 0.00837972 | -0.01055441 | 0.0232008 | 0.4544289226310 | — | — | — |
| occupation_dp | 2024 | R8 | equal | dose | estimated | 9589 | 46 | 0.011100622 | 0.02929579 | -0.04790413 | 0.0701054 | 0.7065328148799 | — | — | — |
| occupation_dnear | 2024 | R8 | equal | dose | estimated | 9589 | 46 | -0.007104483 | 0.01004675 | -0.02733968 | 0.0131307 | 0.4831241387833 | — | — | — |
| occupation_dcdf0 | 2024 | R8 | equal | dose | estimated | 9589 | 46 | -0.001661580 | 0.00313386 | -0.00797349 | 0.0046503 | 0.5985779000785 | — | — | — |
| occupation_dcdf25 | 2024 | R8 | equal | dose | estimated | 9589 | 46 | -0.002576211 | 0.00228354 | -0.00717550 | 0.0020231 | 0.2652291262915 | — | — | — |
| occupation_dcdf100 | 2024 | R8 | equal | dose | estimated | 9589 | 46 | -0.004950261 | 0.00428080 | -0.01357223 | 0.0036717 | 0.2536247686320 | — | — | — |
| schedule_envelope_dnear_candidate | 2024 | R8 | h0 | dose | estimated | 7913 | 44 | 0.007164856 | 0.00907223 | -0.01113103 | 0.0254607 | 0.4340031741026 | — | — | — |
| schedule_envelope_dpremium_lower | 2024 | R8 | h0 | dose | estimated | 7913 | 44 | 0.003655663 | 0.02191539 | -0.04054093 | 0.0478523 | 0.8683030327861 | — | — | — |
| schedule_envelope_dnear_candidate | 2024 | R8 | equal | dose | estimated | 7913 | 44 | -0.006133721 | 0.01098142 | -0.02827987 | 0.0160124 | 0.5793631116473 | — | — | — |
| schedule_envelope_dpremium_lower | 2024 | R8 | equal | dose | estimated | 7913 | 44 | -0.011237404 | 0.02623113 | -0.06413751 | 0.0416627 | 0.6704985939188 | — | — | — |
| filer_state_unrestricted | 2024 | R8 | h0 | dose | estimated | 10752 | 46 | -0.018068814 | 0.04846788 | -0.11568814 | 0.0795505 | 0.7110477344682 | — | — | — |
| filer_state_fixed_composition | 2024 | R8 | h0 | dose | estimated | 9242 | 46 | 0.019760621 | 0.02766857 | -0.03596674 | 0.0754880 | 0.4787979997179 | — | — | — |
| filer_state_unrestricted_common_sample | 2024 | R8 | h0 | dose | estimated | 9242 | 46 | 0.007332310 | 0.02776809 | -0.04859550 | 0.0632601 | 0.7929429305759 | — | — | — |
| filer_state_unrestricted | 2024 | R8 | equal | dose | estimated | 10752 | 46 | 0.007326228 | 0.03708654 | -0.06736990 | 0.0820224 | 0.8442913789907 | — | — | — |
| filer_state_fixed_composition | 2024 | R8 | equal | dose | estimated | 9242 | 46 | 0.011576489 | 0.02754365 | -0.04389928 | 0.0670523 | 0.6762694353704 | — | — | — |
| filer_state_unrestricted_common_sample | 2024 | R8 | equal | dose | estimated | 9242 | 46 | 0.011779669 | 0.03248763 | -0.05365378 | 0.0772131 | 0.7186091476611 | — | — | — |
| occupation_dw | 2025 | R8 | h0 | dose | estimated | 11580 | 46 | 0.013581424 | 0.02519426 | -0.03716243 | 0.0643253 | 0.5924970537440 | — | — | — |
| occupation_reduce | 2025 | R8 | h0 | dose | estimated | 11580 | 46 | -0.000873229 | 0.00364332 | -0.00821125 | 0.0064648 | 0.8116671891669 | — | — | — |
| occupation_retain | 2025 | R8 | h0 | dose | estimated | 11580 | 46 | 0.000752807 | 0.00622005 | -0.01177501 | 0.0132806 | 0.9042068095504 | — | — | — |
| occupation_dp | 2025 | R8 | h0 | dose | estimated | 11580 | 46 | 0.013581424 | 0.02519426 | -0.03716243 | 0.0643253 | 0.5924970537440 | — | — | — |
| occupation_dnear | 2025 | R8 | h0 | dose | estimated | 11580 | 46 | -0.004626035 | 0.01682275 | -0.03850879 | 0.0292567 | 0.7845849963826 | — | — | — |
| occupation_dcdf0 | 2025 | R8 | h0 | dose | estimated | 11580 | 46 | -0.003386641 | 0.00569159 | -0.01485008 | 0.0080768 | 0.5548067382277 | — | — | — |
| occupation_dcdf25 | 2025 | R8 | h0 | dose | estimated | 11580 | 46 | -0.002891267 | 0.00534603 | -0.01365873 | 0.0078762 | 0.5912955726688 | — | — | — |
| occupation_dcdf100 | 2025 | R8 | h0 | dose | estimated | 11580 | 46 | 0.000176169 | 0.00297726 | -0.00582034 | 0.0061727 | 0.9530775313843 | — | — | — |
| occupation_dw | 2025 | R8 | equal | dose | estimated | 11580 | 46 | -0.002665086 | 0.01223471 | -0.02730705 | 0.0219769 | 0.8285463137730 | — | — | — |
| occupation_reduce | 2025 | R8 | equal | dose | estimated | 11580 | 46 | 0.000789659 | 0.00410188 | -0.00747194 | 0.0090513 | 0.8482075700968 | — | — | — |
| occupation_retain | 2025 | R8 | equal | dose | estimated | 11580 | 46 | 0.001180803 | 0.00739862 | -0.01372079 | 0.0160824 | 0.8739122531220 | — | — | — |
| occupation_dp | 2025 | R8 | equal | dose | estimated | 11580 | 46 | -0.002665086 | 0.01223471 | -0.02730705 | 0.0219769 | 0.8285463137730 | — | — | — |
| occupation_dnear | 2025 | R8 | equal | dose | estimated | 11580 | 46 | 0.008029261 | 0.03043524 | -0.05327045 | 0.0693290 | 0.7931271454351 | — | — | — |
| occupation_dcdf0 | 2025 | R8 | equal | dose | estimated | 11580 | 46 | -0.002626657 | 0.00700708 | -0.01673964 | 0.0114863 | 0.7095273513157 | — | — | — |
| occupation_dcdf25 | 2025 | R8 | equal | dose | estimated | 11580 | 46 | -0.002040731 | 0.00564218 | -0.01340467 | 0.0093232 | 0.7192751081264 | — | — | — |
| occupation_dcdf100 | 2025 | R8 | equal | dose | estimated | 11580 | 46 | 0.000436517 | 0.00141269 | -0.00240878 | 0.0032818 | 0.7587494878909 | — | — | — |
| schedule_envelope_dnear_candidate | 2025 | R8 | h0 | dose | estimated | 9269 | 44 | -0.001228952 | 0.02122526 | -0.04403376 | 0.0415759 | 0.9540961584364 | — | — | — |
| schedule_envelope_dpremium_lower | 2025 | R8 | h0 | dose | estimated | 9269 | 44 | 0.001052653 | 0.00898477 | -0.01706686 | 0.0191722 | 0.9072790429156 | — | — | — |
| schedule_envelope_dnear_candidate | 2025 | R8 | equal | dose | estimated | 9269 | 44 | 0.009061227 | 0.03599093 | -0.06352141 | 0.0816439 | 0.8024227807776 | — | — | — |
| schedule_envelope_dpremium_lower | 2025 | R8 | equal | dose | estimated | 9269 | 44 | -0.005082439 | 0.01370633 | -0.03272388 | 0.0225590 | 0.7125998654136 | — | — | — |
| filer_state_unrestricted | 2025 | R8 | h0 | dose | estimated | 11607 | 46 | 0.026117654 | 0.03345632 | -0.04126684 | 0.0935022 | 0.4390962020931 | — | — | — |
| filer_state_fixed_composition | 2025 | R8 | h0 | dose | estimated | 10902 | 46 | 0.014449805 | 0.02378254 | -0.03345069 | 0.0623503 | 0.5465198429491 | — | — | — |
| filer_state_unrestricted_common_sample | 2025 | R8 | h0 | dose | estimated | 10902 | 46 | 0.029146428 | 0.03164624 | -0.03459238 | 0.0928852 | 0.3619579024977 | — | — | — |
| filer_state_unrestricted | 2025 | R8 | equal | dose | estimated | 11607 | 46 | -0.015028577 | 0.02053819 | -0.05639461 | 0.0263375 | 0.4681221158140 | — | — | — |
| filer_state_fixed_composition | 2025 | R8 | equal | dose | estimated | 10902 | 46 | -0.006415728 | 0.01169829 | -0.02997729 | 0.0171458 | 0.5861053749238 | — | — | — |
| filer_state_unrestricted_common_sample | 2025 | R8 | equal | dose | estimated | 10902 | 46 | -0.006287786 | 0.01329191 | -0.03305906 | 0.0204835 | 0.6384615367432 | — | — | — |
| occupation_dw | 2026 | R8 | h0 | dose | estimated | 12485 | 46 | 0.638373421 | 0.09599604 | 0.44502746 | 0.8317194 | 0.0000000337278 | — | — | — |
| occupation_reduce | 2026 | R8 | h0 | dose | estimated | 12485 | 46 | -0.009900070 | 0.02952466 | -0.06936578 | 0.0495656 | 0.7389454515462 | — | — | — |
| occupation_retain | 2026 | R8 | h0 | dose | estimated | 12485 | 46 | -0.011936195 | 0.01569204 | -0.04354158 | 0.0196692 | 0.4508317997325 | — | — | — |
| occupation_dp | 2026 | R8 | h0 | dose | estimated | 12485 | 46 | -0.361626579 | 0.09599604 | -0.55497254 | -0.1682806 | 0.0004774640289 | — | — | — |
| occupation_dnear | 2026 | R8 | h0 | dose | estimated | 12485 | 46 | 0.026520514 | 0.03258046 | -0.03909990 | 0.0921409 | 0.4199312573237 | — | — | — |
| occupation_dcdf0 | 2026 | R8 | h0 | dose | estimated | 12485 | 46 | 0.018879768 | 0.02393393 | -0.02932565 | 0.0670852 | 0.4343486398159 | — | — | — |
| occupation_dcdf25 | 2026 | R8 | h0 | dose | estimated | 12485 | 46 | 0.021017299 | 0.02129863 | -0.02188033 | 0.0639149 | 0.3290212616435 | — | — | — |
| occupation_dcdf100 | 2026 | R8 | h0 | dose | estimated | 12485 | 46 | 0.047458357 | 0.03264659 | -0.01829525 | 0.1132120 | 0.1529685152250 | — | — | — |
| occupation_dw | 2026 | R8 | equal | dose | estimated | 12485 | 46 | 0.599594352 | 0.17243187 | 0.25229873 | 0.9468900 | 0.0011347372158 | — | — | — |
| occupation_reduce | 2026 | R8 | equal | dose | estimated | 12485 | 46 | -0.024847893 | 0.03066502 | -0.08661042 | 0.0369146 | 0.4220319078105 | — | — | — |
| occupation_retain | 2026 | R8 | equal | dose | estimated | 12485 | 46 | -0.011764181 | 0.01552017 | -0.04302340 | 0.0194950 | 0.4524065511504 | — | — | — |
| occupation_dp | 2026 | R8 | equal | dose | estimated | 12485 | 46 | -0.400405648 | 0.17243187 | -0.74770127 | -0.0531100 | 0.0248102388283 | — | — | — |
| occupation_dnear | 2026 | R8 | equal | dose | estimated | 12485 | 46 | -0.008708261 | 0.02683006 | -0.06274677 | 0.0453302 | 0.7470097419737 | — | — | — |
| occupation_dcdf0 | 2026 | R8 | equal | dose | estimated | 12485 | 46 | 0.012899082 | 0.02280776 | -0.03303811 | 0.0588363 | 0.5745045283211 | — | — | — |
| occupation_dcdf25 | 2026 | R8 | equal | dose | estimated | 12485 | 46 | 0.012964864 | 0.02339225 | -0.03414955 | 0.0600793 | 0.5821603802585 | — | — | — |
| occupation_dcdf100 | 2026 | R8 | equal | dose | estimated | 12485 | 46 | 0.017502617 | 0.02951997 | -0.04195364 | 0.0769589 | 0.5562110145848 | — | — | — |
| schedule_envelope_dnear_candidate | 2026 | R8 | h0 | dose | estimated | 8326 | 39 | 0.036895227 | 0.05992478 | -0.08441615 | 0.1582066 | 0.5417680316029 | — | — | — |
| schedule_envelope_dpremium_lower | 2026 | R8 | h0 | dose | estimated | 8326 | 39 | -0.306588660 | 0.26319751 | -0.83940415 | 0.2262268 | 0.2513357494179 | — | — | — |
| schedule_envelope_dnear_candidate | 2026 | R8 | equal | dose | estimated | 8326 | 39 | 0.006151410 | 0.05308868 | -0.10132100 | 0.1136238 | 0.9083653706378 | — | — | — |
| schedule_envelope_dpremium_lower | 2026 | R8 | equal | dose | estimated | 8326 | 39 | -0.436721370 | 0.33180509 | -1.10842565 | 0.2349829 | 0.1959919715500 | — | — | — |
| filer_state_unrestricted | 2026 | R8 | h0 | dose | estimated | 12406 | 46 | 0.648371369 | 0.08933999 | 0.46843139 | 0.8283114 | 0.0000000042383 | — | — | — |
| filer_state_fixed_composition | 2026 | R8 | h0 | dose | estimated | 11781 | 46 | 0.638087885 | 0.09716260 | 0.44239237 | 0.8337834 | 0.0000000447817 | — | — | — |
| filer_state_unrestricted_common_sample | 2026 | R8 | h0 | dose | estimated | 11781 | 46 | 0.643883572 | 0.09508812 | 0.45236627 | 0.8354009 | 0.0000000222568 | — | — | — |
| filer_state_unrestricted | 2026 | R8 | equal | dose | estimated | 12406 | 46 | 0.617741757 | 0.15938175 | 0.29673043 | 0.9387531 | 0.0003424469406 | — | — | — |
| filer_state_fixed_composition | 2026 | R8 | equal | dose | estimated | 11781 | 46 | 0.623013520 | 0.16416224 | 0.29237380 | 0.9536532 | 0.0004384578975 | — | — | — |
| filer_state_unrestricted_common_sample | 2026 | R8 | equal | dose | estimated | 11781 | 46 | 0.624642635 | 0.16329283 | 0.29575400 | 0.9535313 | 0.0003998691272 | — | — | — |
| occupation_dw | 2024 | FEIN | h0 | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| occupation_reduce | 2024 | FEIN | h0 | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| occupation_retain | 2024 | FEIN | h0 | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| occupation_dp | 2024 | FEIN | h0 | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| occupation_dnear | 2024 | FEIN | h0 | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| occupation_dcdf0 | 2024 | FEIN | h0 | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| occupation_dcdf25 | 2024 | FEIN | h0 | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| occupation_dcdf100 | 2024 | FEIN | h0 | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| occupation_dw | 2024 | FEIN | equal | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| occupation_reduce | 2024 | FEIN | equal | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| occupation_retain | 2024 | FEIN | equal | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| occupation_dp | 2024 | FEIN | equal | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| occupation_dnear | 2024 | FEIN | equal | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| occupation_dcdf0 | 2024 | FEIN | equal | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| occupation_dcdf25 | 2024 | FEIN | equal | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| occupation_dcdf100 | 2024 | FEIN | equal | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| schedule_envelope_dnear_candidate | 2024 | FEIN | h0 | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| schedule_envelope_dpremium_lower | 2024 | FEIN | h0 | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| schedule_envelope_dnear_candidate | 2024 | FEIN | equal | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| schedule_envelope_dpremium_lower | 2024 | FEIN | equal | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| filer_state_unrestricted | 2024 | FEIN | h0 | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| filer_state_fixed_composition | 2024 | FEIN | h0 | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| filer_state_unrestricted_common_sample | 2024 | FEIN | h0 | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| filer_state_unrestricted | 2024 | FEIN | equal | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| filer_state_fixed_composition | 2024 | FEIN | equal | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| filer_state_unrestricted_common_sample | 2024 | FEIN | equal | dose | insufficient_support | 0 | 0 | — | — | — | — | — | — | — | — |
| occupation_dw | 2025 | FEIN | h0 | dose | estimated | 11671 | 46 | 0.013202912 | 0.02408134 | -0.03529940 | 0.0617052 | 0.5862210454302 | — | — | — |
| occupation_reduce | 2025 | FEIN | h0 | dose | estimated | 11671 | 46 | 0.000013881 | 0.00261006 | -0.00524306 | 0.0052708 | 0.9957801160411 | — | — | — |
| occupation_retain | 2025 | FEIN | h0 | dose | estimated | 11671 | 46 | 0.000227114 | 0.00618041 | -0.01222087 | 0.0126751 | 0.9708489849105 | — | — | — |
| occupation_dp | 2025 | FEIN | h0 | dose | estimated | 11671 | 46 | 0.013202912 | 0.02408134 | -0.03529940 | 0.0617052 | 0.5862210454302 | — | — | — |
| occupation_dnear | 2025 | FEIN | h0 | dose | estimated | 11671 | 46 | -0.005187316 | 0.01631071 | -0.03803877 | 0.0276641 | 0.7519324648294 | — | — | — |
| occupation_dcdf0 | 2025 | FEIN | h0 | dose | estimated | 11671 | 46 | -0.003146770 | 0.00529083 | -0.01380306 | 0.0075095 | 0.5549837271954 | — | — | — |
| occupation_dcdf25 | 2025 | FEIN | h0 | dose | estimated | 11671 | 46 | -0.002603984 | 0.00493327 | -0.01254011 | 0.0073321 | 0.6002030499922 | — | — | — |
| occupation_dcdf100 | 2025 | FEIN | h0 | dose | estimated | 11671 | 46 | 0.000120524 | 0.00288977 | -0.00569977 | 0.0059408 | 0.9669166331053 | — | — | — |
| occupation_dw | 2025 | FEIN | equal | dose | estimated | 11671 | 46 | -0.003252030 | 0.01262095 | -0.02867192 | 0.0221679 | 0.7978368512461 | — | — | — |
| occupation_reduce | 2025 | FEIN | equal | dose | estimated | 11671 | 46 | 0.000687183 | 0.00409591 | -0.00756241 | 0.0089368 | 0.8675137275443 | — | — | — |
| occupation_retain | 2025 | FEIN | equal | dose | estimated | 11671 | 46 | 0.001538398 | 0.00847162 | -0.01552433 | 0.0186011 | 0.8567166739838 | — | — | — |
| occupation_dp | 2025 | FEIN | equal | dose | estimated | 11671 | 46 | -0.003252030 | 0.01262095 | -0.02867192 | 0.0221679 | 0.7978368512461 | — | — | — |
| occupation_dnear | 2025 | FEIN | equal | dose | estimated | 11671 | 46 | 0.006688674 | 0.03027643 | -0.05429118 | 0.0676685 | 0.8261539177850 | — | — | — |
| occupation_dcdf0 | 2025 | FEIN | equal | dose | estimated | 11671 | 46 | -0.002855115 | 0.00720842 | -0.01737363 | 0.0116634 | 0.6939176354585 | — | — | — |
| occupation_dcdf25 | 2025 | FEIN | equal | dose | estimated | 11671 | 46 | -0.002033995 | 0.00553479 | -0.01318164 | 0.0091136 | 0.7149744025931 | — | — | — |
| occupation_dcdf100 | 2025 | FEIN | equal | dose | estimated | 11671 | 46 | 0.000474325 | 0.00125604 | -0.00205547 | 0.0030041 | 0.7074773745445 | — | — | — |
| schedule_envelope_dnear_candidate | 2025 | FEIN | h0 | dose | estimated | 9328 | 44 | -0.001835532 | 0.02067641 | -0.04353349 | 0.0398624 | 0.9296735382082 | — | — | — |
| schedule_envelope_dpremium_lower | 2025 | FEIN | h0 | dose | estimated | 9328 | 44 | 0.000631054 | 0.00914234 | -0.01780623 | 0.0190683 | 0.9452893960847 | — | — | — |
| schedule_envelope_dnear_candidate | 2025 | FEIN | equal | dose | estimated | 9328 | 44 | 0.007802634 | 0.03590716 | -0.06461105 | 0.0802163 | 0.8290021675635 | — | — | — |
| schedule_envelope_dpremium_lower | 2025 | FEIN | equal | dose | estimated | 9328 | 44 | -0.005516478 | 0.01555460 | -0.03688531 | 0.0258524 | 0.7245833795432 | — | — | — |
| filer_state_unrestricted | 2025 | FEIN | h0 | dose | estimated | 11703 | 46 | 0.027475580 | 0.03182687 | -0.03662703 | 0.0915782 | 0.3925601284757 | — | — | — |
| filer_state_fixed_composition | 2025 | FEIN | h0 | dose | estimated | 10997 | 46 | 0.014229085 | 0.02242258 | -0.03093231 | 0.0593905 | 0.5289108631624 | — | — | — |
| filer_state_unrestricted_common_sample | 2025 | FEIN | h0 | dose | estimated | 10997 | 46 | 0.028440405 | 0.03069877 | -0.03339009 | 0.0902709 | 0.3591623055802 | — | — | — |
| filer_state_unrestricted | 2025 | FEIN | equal | dose | estimated | 11703 | 46 | -0.012172707 | 0.02105803 | -0.05458575 | 0.0302403 | 0.5661084131580 | — | — | — |
| filer_state_fixed_composition | 2025 | FEIN | equal | dose | estimated | 10997 | 46 | -0.006668315 | 0.01303803 | -0.03292826 | 0.0195916 | 0.6115358884609 | — | — | — |
| filer_state_unrestricted_common_sample | 2025 | FEIN | equal | dose | estimated | 10997 | 46 | -0.006275753 | 0.01413507 | -0.03474525 | 0.0221937 | 0.6591815639430 | — | — | — |
| occupation_dw | 2026 | FEIN | h0 | dose | estimated | 12507 | 46 | 0.639568909 | 0.09630548 | 0.44559972 | 0.8335381 | 0.0000000347778 | — | — | — |
| occupation_reduce | 2026 | FEIN | h0 | dose | estimated | 12507 | 46 | -0.009368652 | 0.02891723 | -0.06761094 | 0.0488736 | 0.7474530004560 | — | — | — |
| occupation_retain | 2026 | FEIN | h0 | dose | estimated | 12507 | 46 | -0.012686813 | 0.01499871 | -0.04289577 | 0.0175221 | 0.4021062543352 | — | — | — |
| occupation_dp | 2026 | FEIN | h0 | dose | estimated | 12507 | 46 | -0.360431091 | 0.09630548 | -0.55440028 | -0.1664619 | 0.0005143333314 | — | — | — |
| occupation_dnear | 2026 | FEIN | h0 | dose | estimated | 12507 | 46 | 0.025929260 | 0.03193246 | -0.03838601 | 0.0902445 | 0.4210644912709 | — | — | — |
| occupation_dcdf0 | 2026 | FEIN | h0 | dose | estimated | 12507 | 46 | 0.018361207 | 0.02368179 | -0.02933636 | 0.0660588 | 0.4421999898111 | — | — | — |
| occupation_dcdf25 | 2026 | FEIN | h0 | dose | estimated | 12507 | 46 | 0.020486830 | 0.02110762 | -0.02202610 | 0.0629998 | 0.3369411068454 | — | — | — |
| occupation_dcdf100 | 2026 | FEIN | h0 | dose | estimated | 12507 | 46 | 0.047437573 | 0.03330736 | -0.01964689 | 0.1145220 | 0.1612758795630 | — | — | — |
| occupation_dw | 2026 | FEIN | equal | dose | estimated | 12507 | 46 | 0.600872933 | 0.17300707 | 0.25241881 | 0.9493271 | 0.0011487008899 | — | — | — |
| occupation_reduce | 2026 | FEIN | equal | dose | estimated | 12507 | 46 | -0.024908031 | 0.03080832 | -0.08695917 | 0.0371431 | 0.4230657939082 | — | — | — |
| occupation_retain | 2026 | FEIN | equal | dose | estimated | 12507 | 46 | -0.012479380 | 0.01493072 | -0.04255140 | 0.0175926 | 0.4076729526484 | — | — | — |
| occupation_dp | 2026 | FEIN | equal | dose | estimated | 12507 | 46 | -0.399127067 | 0.17300707 | -0.74758119 | -0.0506729 | 0.0257124415996 | — | — | — |
| occupation_dnear | 2026 | FEIN | equal | dose | estimated | 12507 | 46 | -0.008369104 | 0.02749483 | -0.06374653 | 0.0470083 | 0.7622352977340 | — | — | — |
| occupation_dcdf0 | 2026 | FEIN | equal | dose | estimated | 12507 | 46 | 0.013074311 | 0.02372445 | -0.03470918 | 0.0608578 | 0.5842978090742 | — | — | — |
| occupation_dcdf25 | 2026 | FEIN | equal | dose | estimated | 12507 | 46 | 0.013144110 | 0.02436006 | -0.03591958 | 0.0622078 | 0.5921494149664 | — | — | — |
| occupation_dcdf100 | 2026 | FEIN | equal | dose | estimated | 12507 | 46 | 0.017334635 | 0.03020302 | -0.04349738 | 0.0781666 | 0.5688681784854 | — | — | — |
| schedule_envelope_dnear_candidate | 2026 | FEIN | h0 | dose | estimated | 8346 | 39 | 0.028806996 | 0.05753161 | -0.08765965 | 0.1452736 | 0.6194589786431 | — | — | — |
| schedule_envelope_dpremium_lower | 2026 | FEIN | h0 | dose | estimated | 8346 | 39 | -0.301894603 | 0.26184641 | -0.83197494 | 0.2281857 | 0.2561339037294 | — | — | — |
| schedule_envelope_dnear_candidate | 2026 | FEIN | equal | dose | estimated | 8346 | 39 | 0.007759978 | 0.05154981 | -0.09659715 | 0.1121171 | 0.8811398749807 | — | — | — |
| schedule_envelope_dpremium_lower | 2026 | FEIN | equal | dose | estimated | 8346 | 39 | -0.432834979 | 0.33758360 | -1.11623724 | 0.2505673 | 0.2075549561304 | — | — | — |
| filer_state_unrestricted | 2026 | FEIN | h0 | dose | estimated | 12446 | 46 | 0.648020032 | 0.08946400 | 0.46783029 | 0.8282098 | 0.0000000044449 | — | — | — |
| filer_state_fixed_composition | 2026 | FEIN | h0 | dose | estimated | 11813 | 46 | 0.638203162 | 0.09748163 | 0.44186509 | 0.8345412 | 0.0000000480078 | — | — | — |
| filer_state_unrestricted_common_sample | 2026 | FEIN | h0 | dose | estimated | 11813 | 46 | 0.645592672 | 0.09482899 | 0.45459727 | 0.8365881 | 0.0000000196424 | — | — | — |
| filer_state_unrestricted | 2026 | FEIN | equal | dose | estimated | 12446 | 46 | 0.615760905 | 0.15748627 | 0.29856728 | 0.9329545 | 0.0003083441013 | — | — | — |
| filer_state_fixed_composition | 2026 | FEIN | equal | dose | estimated | 11813 | 46 | 0.623295448 | 0.16256164 | 0.29587950 | 0.9507114 | 0.0003891084857 | — | — | — |
| filer_state_unrestricted_common_sample | 2026 | FEIN | equal | dose | estimated | 11813 | 46 | 0.625188518 | 0.16158300 | 0.29974364 | 0.9506334 | 0.0003495851794 | — | — | — |

## Administrative contrasts

[Full CSV with formula and inference fields](/tmp/h2a-chapter-directions/admin_estimates.csv)

| specification | tau | tier | weights | term | status | n | states | estimate | se_jackknife | low | high | p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary_frame_timely | 2013 | R7 | equal | year::2011:exposure | estimated | 7320 | 46 | -0.01103483671909167955494 | 0.04411240351844902124 | -0.09988177813727923404 | 0.07781210469909587146 | 0.803608 |
| primary_frame_timely | 2013 | R7 | equal | year::2013:exposure | estimated | 7320 | 46 | -0.02356597800212217874716 | 0.10246810041281484427 | -0.22994732629575542759 | 0.18281537029151107010 | 0.819147 |
| primary_frame_timely | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 7320 | 46 | — | — | — | — | — |
| primary_frame_timely | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 7320 | 46 | — | — | — | — | — |
| primary_frame_observed_filing | 2013 | R7 | equal | year::2011:exposure | estimated | 7320 | 46 | 0.00000000000000000954098 | 0.00000000000000016595 | -0.00000000000000032469 | 0.00000000000000034378 | 0.954406 |
| primary_frame_observed_filing | 2013 | R7 | equal | year::2013:exposure | estimated | 7320 | 46 | 0.00474953537598679989551 | 0.14387616541415382265 | -0.28503193696384165889 | 0.29453100771581530726 | 0.973812 |
| primary_frame_observed_filing | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 7320 | 46 | — | — | — | — | — |
| primary_frame_observed_filing | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 7320 | 46 | — | — | — | — | — |
| primary_frame_certified_positions | 2013 | R7 | equal | year::2011:exposure | estimated | 7158 | 46 | -1.09806072211100902435987 | 2.67955991649376557007 | -6.49497143063038073763 | 4.29884998640836180073 | 0.683903 |
| primary_frame_certified_positions | 2013 | R7 | equal | year::2013:exposure | estimated | 7158 | 46 | 5.01116019117726985143690 | 5.51548653558192736313 | -6.09759993146496981353 | 16.11992031381950951641 | 0.368422 |
| primary_frame_certified_positions | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 7158 | 46 | — | — | — | — | — |
| primary_frame_certified_positions | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 7158 | 46 | — | — | — | — | — |
| primary_frame_timing | 2013 | R7 | equal | year::2011:exposure | estimated | 6567 | 46 | 1.24306198230402831228503 | 1.84468596776001292348 | -2.47232627678235772350 | 4.95845024139041434807 | 0.503847 |
| primary_frame_timing | 2013 | R7 | equal | year::2013:exposure | estimated | 6567 | 46 | 3.15281042934763489071770 | 2.96959479962752892490 | -2.82826052018510987196 | 9.13388137888038009748 | 0.294038 |
| primary_frame_timing | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 6567 | 46 | — | — | — | — | — |
| primary_frame_timing | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 6567 | 46 | — | — | — | — | — |
| primary_frame_observed_denial | 2013 | R7 | equal | year::2011:exposure | estimated | 7320 | 46 | 0.01666109968260238993754 | 0.04076673675123101970 | -0.06544732296166536067 | 0.09876952232687014055 | 0.684703 |
| primary_frame_observed_denial | 2013 | R7 | equal | year::2013:exposure | estimated | 7320 | 46 | 0.00352116024219942995790 | 0.08973820932629275604 | -0.17722087127398558626 | 0.18426319175838448694 | 0.968874 |
| primary_frame_observed_denial | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 7320 | 46 | — | — | — | — | — |
| primary_frame_observed_denial | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 7320 | 46 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R7 | equal | year::2011:exposure | fit_failed | 12310 | 49 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R7 | equal | year::2013:exposure | fit_failed | 12310 | 49 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R7 | equal | year::2014:exposure | fit_failed | 12310 | 49 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R7 | equal | year::2015:exposure | fit_failed | 12310 | 49 | — | — | — | — | — |
| primary_frame_timely | 2013 | R7 | baseline_positions | year::2011:exposure | estimated | 7098 | 46 | 0.01447729447133992014551 | 0.01694540185358121842 | -0.01965249682790579100 | 0.04860708577058563823 | 0.397437 |
| primary_frame_timely | 2013 | R7 | baseline_positions | year::2013:exposure | estimated | 7098 | 46 | 0.13639328608556861022194 | 0.31439818882035375225 | -0.49683717147550615634 | 0.76962374364664332127 | 0.666490 |
| primary_frame_timely | 2013 | R7 | baseline_positions | year::2014:exposure | term_not_identified | 7098 | 46 | — | — | — | — | — |
| primary_frame_timely | 2013 | R7 | baseline_positions | year::2015:exposure | term_not_identified | 7098 | 46 | — | — | — | — | — |
| primary_frame_observed_filing | 2013 | R7 | baseline_positions | year::2011:exposure | estimated | 7098 | 46 | 0.00000000000000011102230 | 0.00000000000000015788 | -0.00000000000000020697 | 0.00000000000000042901 | 0.485557 |
| primary_frame_observed_filing | 2013 | R7 | baseline_positions | year::2013:exposure | estimated | 7098 | 46 | 0.07306483930674333160393 | 0.27396339557535748233 | -0.47872576415088807789 | 0.62485544276437476885 | 0.790922 |
| primary_frame_observed_filing | 2013 | R7 | baseline_positions | year::2014:exposure | term_not_identified | 7098 | 46 | — | — | — | — | — |
| primary_frame_observed_filing | 2013 | R7 | baseline_positions | year::2015:exposure | term_not_identified | 7098 | 46 | — | — | — | — | — |
| primary_frame_certified_positions | 2013 | R7 | baseline_positions | year::2011:exposure | estimated | 7098 | 46 | 1.74559785624088648070540 | 3.70107889027250003267 | -5.70875767917231868864 | 9.19995339165409120596 | 0.639459 |
| primary_frame_certified_positions | 2013 | R7 | baseline_positions | year::2013:exposure | estimated | 7098 | 46 | 23.19532148696860929248942 | 22.67518491107555078656 | -22.47484528592849883921 | 68.86548825986571387148 | 0.311805 |
| primary_frame_certified_positions | 2013 | R7 | baseline_positions | year::2014:exposure | term_not_identified | 7098 | 46 | — | — | — | — | — |
| primary_frame_certified_positions | 2013 | R7 | baseline_positions | year::2015:exposure | term_not_identified | 7098 | 46 | — | — | — | — | — |
| primary_frame_timing | 2013 | R7 | baseline_positions | year::2011:exposure | estimated | 6399 | 46 | -0.80905649314205585476145 | 3.52892386882510322721 | -7.91667401644520474235 | 6.29856103016109258874 | 0.819702 |
| primary_frame_timing | 2013 | R7 | baseline_positions | year::2013:exposure | estimated | 6399 | 46 | -5.37860517319360287302743 | 9.32348362464894719892 | -24.15706513777412922650 | 13.39985479138692348045 | 0.566890 |
| primary_frame_timing | 2013 | R7 | baseline_positions | year::2014:exposure | term_not_identified | 6399 | 46 | — | — | — | — | — |
| primary_frame_timing | 2013 | R7 | baseline_positions | year::2015:exposure | term_not_identified | 6399 | 46 | — | — | — | — | — |
| primary_frame_observed_denial | 2013 | R7 | baseline_positions | year::2011:exposure | estimated | 7098 | 46 | -0.00161384593700600992815 | 0.00649331726999106962 | -0.01469205825557355974 | 0.01146436638156153945 | 0.804848 |
| primary_frame_observed_denial | 2013 | R7 | baseline_positions | year::2013:exposure | estimated | 7098 | 46 | -0.04171606204103320281185 | 0.09356438158694752039 | -0.23016440007384492139 | 0.14673227599177851577 | 0.657841 |
| primary_frame_observed_denial | 2013 | R7 | baseline_positions | year::2014:exposure | term_not_identified | 7098 | 46 | — | — | — | — | — |
| primary_frame_observed_denial | 2013 | R7 | baseline_positions | year::2015:exposure | term_not_identified | 7098 | 46 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R7 | baseline_positions | year::2011:exposure | fit_failed | 11940 | 49 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R7 | baseline_positions | year::2013:exposure | fit_failed | 11940 | 49 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R7 | baseline_positions | year::2014:exposure | fit_failed | 11940 | 49 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R7 | baseline_positions | year::2015:exposure | fit_failed | 11940 | 49 | — | — | — | — | — |
| month_comparison_timely | 2013 | R7 | equal | year::2011:exposure | estimated | 7077 | 45 | -0.04879898136958880222602 | 0.10858414321098600053 | -0.26763594269576790330 | 0.17003797995659028497 | 0.655341 |
| month_comparison_timely | 2013 | R7 | equal | year::2013:exposure | estimated | 7077 | 45 | -0.16219269923340784433918 | 0.06152199863143582165 | -0.28618214039017719763 | -0.03820325807663849105 | 0.011535 |
| month_comparison_timely | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 7077 | 45 | — | — | — | — | — |
| month_comparison_timely | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 7077 | 45 | — | — | — | — | — |
| aug_dec_timely | 2013 | R7 | equal | year::2011:exposure | estimated | 1215 | 33 | -0.01103483671909167955494 | 0.04390424283246110310 | -0.10046485286390087477 | 0.07839517942571751219 | 0.803159 |
| aug_dec_timely | 2013 | R7 | equal | year::2013:exposure | estimated | 1215 | 33 | -0.02356597800212217874716 | 0.10200902467406372520 | -0.23135156169456680408 | 0.18421960569032244659 | 0.818771 |
| aug_dec_timely | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 1215 | 33 | — | — | — | — | — |
| aug_dec_timely | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 1215 | 33 | — | — | — | — | — |
| exclude_largest_timely | 2013 | R7 | equal | year::2011:exposure | estimated | 7308 | 46 | -0.01103483671909167955494 | 0.04411240351844902124 | -0.09988177813727923404 | 0.07781210469909587146 | 0.803608 |
| exclude_largest_timely | 2013 | R7 | equal | year::2013:exposure | estimated | 7308 | 46 | -0.02356597800212217874716 | 0.10246810041281484427 | -0.22994732629575542759 | 0.18281537029151107010 | 0.819147 |
| exclude_largest_timely | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 7308 | 46 | — | — | — | — | — |
| exclude_largest_timely | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 7308 | 46 | — | — | — | — | — |
| calendar_placebo_2012_timely | 2013 | R7 | equal | year::2012:exposure | estimated | 4880 | 46 | 0.01103483671909178016890 | 0.04411240351844909757 | -0.07781210469909594085 | 0.09988177813727949772 | 0.803608 |
| calendar_placebo_2014_timely | 2013 | R7 | equal | year::2014:exposure | fit_failed | 2462 | 49 | — | — | — | — | — |
| calendar_placebo_2015_timely | 2013 | R7 | equal | year::2015:exposure | insufficient_support | 0 | 0 | — | — | — | — | — |
| month_comparison_observed_filing | 2013 | R7 | equal | year::2011:exposure | estimated | 7077 | 45 | 0.00000000000000012490009 | 0.00000000000000020627 | -0.00000000000000029082 | 0.00000000000000054062 | 0.547956 |
| month_comparison_observed_filing | 2013 | R7 | equal | year::2013:exposure | estimated | 7077 | 45 | -0.10752736446840494011212 | 0.11465374078454390605 | -0.33859679593425517918 | 0.12354206699744532671 | 0.353444 |
| month_comparison_observed_filing | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 7077 | 45 | — | — | — | — | — |
| month_comparison_observed_filing | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 7077 | 45 | — | — | — | — | — |
| aug_dec_observed_filing | 2013 | R7 | equal | year::2011:exposure | estimated | 1215 | 33 | 0.00000000000000000954098 | 0.00000000000000016333 | -0.00000000000000032316 | 0.00000000000000034224 | 0.953782 |
| aug_dec_observed_filing | 2013 | R7 | equal | year::2013:exposure | estimated | 1215 | 33 | 0.00474953537598679989551 | 0.14320900379314610573 | -0.28695765953397661674 | 0.29645673028595026510 | 0.973749 |
| aug_dec_observed_filing | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 1215 | 33 | — | — | — | — | — |
| aug_dec_observed_filing | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 1215 | 33 | — | — | — | — | — |
| exclude_largest_observed_filing | 2013 | R7 | equal | year::2011:exposure | estimated | 7308 | 46 | 0.00000000000000000954098 | 0.00000000000000016595 | -0.00000000000000032469 | 0.00000000000000034378 | 0.954406 |
| exclude_largest_observed_filing | 2013 | R7 | equal | year::2013:exposure | estimated | 7308 | 46 | 0.00474953537598679989551 | 0.14387616541415382265 | -0.28503193696384165889 | 0.29453100771581530726 | 0.973812 |
| exclude_largest_observed_filing | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 7308 | 46 | — | — | — | — | — |
| exclude_largest_observed_filing | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 7308 | 46 | — | — | — | — | — |
| calendar_placebo_2012_observed_filing | 2013 | R7 | equal | year::2012:exposure | fit_failed | 4924 | 49 | — | — | — | — | — |
| calendar_placebo_2014_observed_filing | 2013 | R7 | equal | year::2014:exposure | fit_failed | 2462 | 49 | — | — | — | — | — |
| calendar_placebo_2015_observed_filing | 2013 | R7 | equal | year::2015:exposure | insufficient_support | 0 | 0 | — | — | — | — | — |
| month_comparison_certified_positions | 2013 | R7 | equal | year::2011:exposure | estimated | 6912 | 44 | 2.17408657783388425954740 | 2.11240039261909018364 | -2.08597481560682718182 | 6.43414797127459614501 | 0.309139 |
| month_comparison_certified_positions | 2013 | R7 | equal | year::2013:exposure | estimated | 6912 | 44 | 2.76677207113043799324714 | 4.61148480313753594828 | -6.53317335821468780921 | 12.06671750047556379570 | 0.551672 |
| month_comparison_certified_positions | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 6912 | 44 | — | — | — | — | — |
| month_comparison_certified_positions | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 6912 | 44 | — | — | — | — | — |
| aug_dec_certified_positions | 2013 | R7 | equal | year::2011:exposure | estimated | 1212 | 33 | -1.09806072211100991253829 | 2.66732515164385342032 | -6.53122426134414624244 | 4.33510281712212730554 | 0.683325 |
| aug_dec_certified_positions | 2013 | R7 | equal | year::2013:exposure | estimated | 1212 | 33 | 5.01116019117726718690164 | 5.49128942843656719930 | -6.17423034419514049631 | 16.19655072654967398194 | 0.368296 |
| aug_dec_certified_positions | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 1212 | 33 | — | — | — | — | — |
| aug_dec_certified_positions | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 1212 | 33 | — | — | — | — | — |
| exclude_largest_certified_positions | 2013 | R7 | equal | year::2011:exposure | estimated | 7146 | 46 | -1.09806072211101080071671 | 2.67955991649376734642 | -6.49497143063038606670 | 4.29884998640836357708 | 0.683903 |
| exclude_largest_certified_positions | 2013 | R7 | equal | year::2013:exposure | estimated | 7146 | 46 | 5.01116019117726807508006 | 5.51548653558193002766 | -6.09759993146497691896 | 16.11992031381951306912 | 0.368422 |
| exclude_largest_certified_positions | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 7146 | 46 | — | — | — | — | — |
| exclude_largest_certified_positions | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 7146 | 46 | — | — | — | — | — |
| calendar_placebo_2012_certified_positions | 2013 | R7 | equal | year::2012:exposure | estimated | 4744 | 46 | 1.28148352845027257984611 | 2.65746754358624670544 | -4.07093085692764589112 | 6.63389791382819105081 | 0.631987 |
| calendar_placebo_2014_certified_positions | 2013 | R7 | equal | year::2014:exposure | fit_failed | 2453 | 49 | — | — | — | — | — |
| calendar_placebo_2015_certified_positions | 2013 | R7 | equal | year::2015:exposure | insufficient_support | 0 | 0 | — | — | — | — | — |
| month_comparison_timing | 2013 | R7 | equal | year::2011:exposure | estimated | 6342 | 45 | 4.66994105817346927267408 | 2.45495595686354262099 | -0.27769757397687744316 | 9.61757969032381687668 | 0.063697 |
| month_comparison_timing | 2013 | R7 | equal | year::2013:exposure | estimated | 6342 | 45 | 4.82378360013004403583636 | 3.86893104984458657825 | -2.97353458548540405815 | 12.62110178574549124164 | 0.219067 |
| month_comparison_timing | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 6342 | 45 | — | — | — | — | — |
| month_comparison_timing | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 6342 | 45 | — | — | — | — | — |
| aug_dec_timing | 2013 | R7 | equal | year::2011:exposure | estimated | 1089 | 33 | 1.24306198230402609183898 | 1.83656987653408720895 | -2.49790843680265872706 | 4.98403240141071091074 | 0.503372 |
| aug_dec_timing | 2013 | R7 | equal | year::2013:exposure | estimated | 1089 | 33 | 3.15281042934763178209323 | 2.95629655732474194707 | -2.86896860142344234390 | 9.17458946011870857262 | 0.294190 |
| aug_dec_timing | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 1089 | 33 | — | — | — | — | — |
| aug_dec_timing | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 1089 | 33 | — | — | — | — | — |
| exclude_largest_timing | 2013 | R7 | equal | year::2011:exposure | estimated | 6555 | 46 | 1.24306198230402897841884 | 1.84468596776001247939 | -2.47232627678235594715 | 4.95845024139041434807 | 0.503847 |
| exclude_largest_timing | 2013 | R7 | equal | year::2013:exposure | estimated | 6555 | 46 | 3.15281042934763577889612 | 2.96959479962752759263 | -2.82826052018510587516 | 9.13388137888037832113 | 0.294038 |
| exclude_largest_timing | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 6555 | 46 | — | — | — | — | — |
| exclude_largest_timing | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 6555 | 46 | — | — | — | — | — |
| calendar_placebo_2012_timing | 2013 | R7 | equal | year::2012:exposure | estimated | 4756 | 46 | -1.24306198230402098481306 | 1.84468596776001247939 | -4.95845024139040635447 | 2.47232627678236394075 | 0.503847 |
| calendar_placebo_2014_timing | 2013 | R7 | equal | year::2014:exposure | fit_failed | 1817 | 49 | — | — | — | — | — |
| calendar_placebo_2015_timing | 2013 | R7 | equal | year::2015:exposure | insufficient_support | 0 | 0 | — | — | — | — | — |
| catchup7 | 2013 | R7 | equal | year::2011:exposure | estimated | 7320 | 46 | -0.03088633505225655065396 | 0.04757783881813591742 | -0.12671302145149079821 | 0.06494035134697770384 | 0.519525 |
| catchup7 | 2013 | R7 | equal | year::2013:exposure | estimated | 7320 | 46 | -0.02357874423899242871672 | 0.10681975704439623798 | -0.23872477890153959157 | 0.19156729042355474801 | 0.826298 |
| catchup7 | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 7320 | 46 | — | — | — | — | — |
| catchup7 | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 7320 | 46 | — | — | — | — | — |
| catchup14 | 2013 | R7 | equal | year::2011:exposure | estimated | 7320 | 46 | -0.02475854135455941157362 | 0.04499448737774763840 | -0.11538209086303739814 | 0.06586500815391860275 | 0.584864 |
| catchup14 | 2013 | R7 | equal | year::2013:exposure | estimated | 7320 | 46 | -0.02357874423899250157510 | 0.10681975704439627961 | -0.23872477890153970259 | 0.19156729042355469250 | 0.826298 |
| catchup14 | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 7320 | 46 | — | — | — | — | — |
| catchup14 | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 7320 | 46 | — | — | — | — | — |
| catchup30 | 2013 | R7 | equal | year::2011:exposure | estimated | 7320 | 46 | -0.02475854135455936994026 | 0.04499448737774770779 | -0.11538209086303752304 | 0.06586500815391875541 | 0.584864 |
| catchup30 | 2013 | R7 | equal | year::2013:exposure | estimated | 7320 | 46 | -0.03870673493018240984842 | 0.09677860672715167534 | -0.23362885471050531816 | 0.15621538485014047071 | 0.691085 |
| catchup30 | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 7320 | 46 | — | — | — | — | — |
| catchup30 | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 7320 | 46 | — | — | — | — | — |
| catchup60 | 2013 | R7 | equal | year::2011:exposure | estimated | 7320 | 46 | -0.02475854135455936994026 | 0.04499448737774770779 | -0.11538209086303752304 | 0.06586500815391875541 | 0.584864 |
| catchup60 | 2013 | R7 | equal | year::2013:exposure | estimated | 7320 | 46 | -0.03870673493018240984842 | 0.09677860672715167534 | -0.23362885471050531816 | 0.15621538485014047071 | 0.691085 |
| catchup60 | 2013 | R7 | equal | year::2014:exposure | term_not_identified | 7320 | 46 | — | — | — | — | — |
| catchup60 | 2013 | R7 | equal | year::2015:exposure | term_not_identified | 7320 | 46 | — | — | — | — | — |
| primary_frame_timely | 2013 | R5 | equal | year::2011:exposure | estimated | 7113 | 46 | -0.00201982464663232009691 | 0.04356233849477515047 | -0.08975887823653352915 | 0.08571922894326886033 | 0.963223 |
| primary_frame_timely | 2013 | R5 | equal | year::2013:exposure | estimated | 7113 | 46 | -0.03373550181748422027717 | 0.09209879202264856235 | -0.21923199094213297045 | 0.15176098730716450214 | 0.715860 |
| primary_frame_timely | 2013 | R5 | equal | year::2014:exposure | term_not_identified | 7113 | 46 | — | — | — | — | — |
| primary_frame_timely | 2013 | R5 | equal | year::2015:exposure | term_not_identified | 7113 | 46 | — | — | — | — | — |
| primary_frame_observed_filing | 2013 | R5 | equal | year::2011:exposure | estimated | 7113 | 46 | 0.00000000000000005030698 | 0.00000000000000018563 | -0.00000000000000032357 | 0.00000000000000042418 | 0.787622 |
| primary_frame_observed_filing | 2013 | R5 | equal | year::2013:exposure | estimated | 7113 | 46 | -0.01559399585295380072369 | 0.13521199236493852691 | -0.28792492789250762053 | 0.25673693618660003990 | 0.908697 |
| primary_frame_observed_filing | 2013 | R5 | equal | year::2014:exposure | term_not_identified | 7113 | 46 | — | — | — | — | — |
| primary_frame_observed_filing | 2013 | R5 | equal | year::2015:exposure | term_not_identified | 7113 | 46 | — | — | — | — | — |
| primary_frame_certified_positions | 2013 | R5 | equal | year::2011:exposure | estimated | 6964 | 46 | -2.16810970094323263168690 | 3.07147213370863836346 | -8.35437213429888636540 | 4.01815273241242110203 | 0.483897 |
| primary_frame_certified_positions | 2013 | R5 | equal | year::2013:exposure | estimated | 6964 | 46 | 4.27538670717013857824895 | 4.85270736175248718069 | -5.49846763538257832238 | 14.04924104972285547888 | 0.382982 |
| primary_frame_certified_positions | 2013 | R5 | equal | year::2014:exposure | term_not_identified | 6964 | 46 | — | — | — | — | — |
| primary_frame_certified_positions | 2013 | R5 | equal | year::2015:exposure | term_not_identified | 6964 | 46 | — | — | — | — | — |
| primary_frame_timing | 2013 | R5 | equal | year::2011:exposure | estimated | 6364 | 46 | 2.08357646140753427488335 | 1.69319599654972030578 | -1.32669533328274003026 | 5.49384825609780858002 | 0.224882 |
| primary_frame_timing | 2013 | R5 | equal | year::2013:exposure | estimated | 6364 | 46 | 4.24099117414111148605116 | 2.89741106615232180133 | -1.59469427317714540493 | 10.07667662145936837703 | 0.150222 |
| primary_frame_timing | 2013 | R5 | equal | year::2014:exposure | term_not_identified | 6364 | 46 | — | — | — | — | — |
| primary_frame_timing | 2013 | R5 | equal | year::2015:exposure | term_not_identified | 6364 | 46 | — | — | — | — | — |
| primary_frame_observed_denial | 2013 | R5 | equal | year::2011:exposure | estimated | 7113 | 46 | 0.00792636147229332059527 | 0.03903538051949285298 | -0.07069493071827061270 | 0.08654765366285725736 | 0.840007 |
| primary_frame_observed_denial | 2013 | R5 | equal | year::2013:exposure | estimated | 7113 | 46 | -0.00749919229115181013717 | 0.09291830539510752274 | -0.19464626607650328816 | 0.17964788149419966268 | 0.936033 |
| primary_frame_observed_denial | 2013 | R5 | equal | year::2014:exposure | term_not_identified | 7113 | 46 | — | — | — | — | — |
| primary_frame_observed_denial | 2013 | R5 | equal | year::2015:exposure | term_not_identified | 7113 | 46 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R5 | equal | year::2011:exposure | fit_failed | 11975 | 49 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R5 | equal | year::2013:exposure | fit_failed | 11975 | 49 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R5 | equal | year::2014:exposure | fit_failed | 11975 | 49 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R5 | equal | year::2015:exposure | fit_failed | 11975 | 49 | — | — | — | — | — |
| primary_frame_timely | 2013 | R5 | baseline_positions | year::2011:exposure | estimated | 6906 | 46 | 0.01676083507209041867370 | 0.01830962584510009011 | -0.02011664439166601856 | 0.05363831453584685938 | 0.364855 |
| primary_frame_timely | 2013 | R5 | baseline_positions | year::2013:exposure | estimated | 6906 | 46 | 0.13115365477631596435870 | 0.26110637555760363160 | -0.39474158109264861904 | 0.65704889064528049225 | 0.617906 |
| primary_frame_timely | 2013 | R5 | baseline_positions | year::2014:exposure | term_not_identified | 6906 | 46 | — | — | — | — | — |
| primary_frame_timely | 2013 | R5 | baseline_positions | year::2015:exposure | term_not_identified | 6906 | 46 | — | — | — | — | — |
| primary_frame_observed_filing | 2013 | R5 | baseline_positions | year::2011:exposure | estimated | 6906 | 46 | 0.00000000000000002081668 | 0.00000000000000022141 | -0.00000000000000042513 | 0.00000000000000046676 | 0.925513 |
| primary_frame_observed_filing | 2013 | R5 | baseline_positions | year::2013:exposure | estimated | 6906 | 46 | 0.05608617459361078805280 | 0.21479607100817812992 | -0.37653531994225175605 | 0.48870766912947327665 | 0.795196 |
| primary_frame_observed_filing | 2013 | R5 | baseline_positions | year::2014:exposure | term_not_identified | 6906 | 46 | — | — | — | — | — |
| primary_frame_observed_filing | 2013 | R5 | baseline_positions | year::2015:exposure | term_not_identified | 6906 | 46 | — | — | — | — | — |
| primary_frame_certified_positions | 2013 | R5 | baseline_positions | year::2011:exposure | estimated | 6906 | 46 | -0.04821557247653274203003 | 5.73008521268766735091 | -11.58919961792678954282 | 11.49276847297372405876 | 0.993324 |
| primary_frame_certified_positions | 2013 | R5 | baseline_positions | year::2013:exposure | estimated | 6906 | 46 | 21.84810422807227681119002 | 22.51101476331066919556 | -23.49140689385851032966 | 67.18761535000307105747 | 0.336960 |
| primary_frame_certified_positions | 2013 | R5 | baseline_positions | year::2014:exposure | term_not_identified | 6906 | 46 | — | — | — | — | — |
| primary_frame_certified_positions | 2013 | R5 | baseline_positions | year::2015:exposure | term_not_identified | 6906 | 46 | — | — | — | — | — |
| primary_frame_timing | 2013 | R5 | baseline_positions | year::2011:exposure | estimated | 6207 | 46 | -0.29478528765750988682726 | 3.26112347023107673749 | -6.86302512060878910916 | 6.27345454529376755914 | 0.928375 |
| primary_frame_timing | 2013 | R5 | baseline_positions | year::2013:exposure | estimated | 6207 | 46 | -4.94822464814638074415143 | 10.01833552637520341477 | -25.12618818276406074119 | 15.22973888647129570018 | 0.623767 |
| primary_frame_timing | 2013 | R5 | baseline_positions | year::2014:exposure | term_not_identified | 6207 | 46 | — | — | — | — | — |
| primary_frame_timing | 2013 | R5 | baseline_positions | year::2015:exposure | term_not_identified | 6207 | 46 | — | — | — | — | — |
| primary_frame_observed_denial | 2013 | R5 | baseline_positions | year::2011:exposure | estimated | 6906 | 46 | -0.00212486191368649020642 | 0.00722867529230966036 | -0.01668416131704663144 | 0.01243443748967363975 | 0.770148 |
| primary_frame_observed_denial | 2013 | R5 | baseline_positions | year::2013:exposure | estimated | 6906 | 46 | -0.04939363193502759041831 | 0.11472244358939656661 | -0.28045649434912295339 | 0.18166923047906780031 | 0.668852 |
| primary_frame_observed_denial | 2013 | R5 | baseline_positions | year::2014:exposure | term_not_identified | 6906 | 46 | — | — | — | — | — |
| primary_frame_observed_denial | 2013 | R5 | baseline_positions | year::2015:exposure | term_not_identified | 6906 | 46 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R5 | baseline_positions | year::2011:exposure | fit_failed | 11630 | 49 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R5 | baseline_positions | year::2013:exposure | fit_failed | 11630 | 49 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R5 | baseline_positions | year::2014:exposure | fit_failed | 11630 | 49 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R5 | baseline_positions | year::2015:exposure | fit_failed | 11630 | 49 | — | — | — | — | — |
| primary_frame_timely | 2013 | R8 | equal | year::2011:exposure | estimated | 7401 | 46 | -0.01091140817224709944067 | 0.04343686278595604994 | -0.09839774071179352433 | 0.07657492436729930463 | 0.802802 |
| primary_frame_timely | 2013 | R8 | equal | year::2013:exposure | estimated | 7401 | 46 | -0.01853769062113347007581 | 0.10174901848151844674 | -0.22347073356005964229 | 0.18639535231779269520 | 0.856252 |
| primary_frame_timely | 2013 | R8 | equal | year::2014:exposure | term_not_identified | 7401 | 46 | — | — | — | — | — |
| primary_frame_timely | 2013 | R8 | equal | year::2015:exposure | term_not_identified | 7401 | 46 | — | — | — | — | — |
| primary_frame_observed_filing | 2013 | R8 | equal | year::2011:exposure | estimated | 7401 | 46 | -0.00000000000000000086736 | 0.00000000000000014767 | -0.00000000000000029830 | 0.00000000000000029656 | 0.995340 |
| primary_frame_observed_filing | 2013 | R8 | equal | year::2013:exposure | estimated | 7401 | 46 | -0.00685289170031638025099 | 0.14036224024492208184 | -0.28955695544852150247 | 0.27585117204788872636 | 0.961276 |
| primary_frame_observed_filing | 2013 | R8 | equal | year::2014:exposure | term_not_identified | 7401 | 46 | — | — | — | — | — |
| primary_frame_observed_filing | 2013 | R8 | equal | year::2015:exposure | term_not_identified | 7401 | 46 | — | — | — | — | — |
| primary_frame_certified_positions | 2013 | R8 | equal | year::2011:exposure | estimated | 7239 | 46 | -2.69119692958613843103421 | 3.20860226399948800236 | -9.15365362307826480048 | 3.77125976390598705024 | 0.406046 |
| primary_frame_certified_positions | 2013 | R8 | equal | year::2013:exposure | estimated | 7239 | 46 | 3.93232002640992250874774 | 4.12255551680048437646 | -4.37093301082736296337 | 12.23557306364720886904 | 0.345253 |
| primary_frame_certified_positions | 2013 | R8 | equal | year::2014:exposure | term_not_identified | 7239 | 46 | — | — | — | — | — |
| primary_frame_certified_positions | 2013 | R8 | equal | year::2015:exposure | term_not_identified | 7239 | 46 | — | — | — | — | — |
| primary_frame_timing | 2013 | R8 | equal | year::2011:exposure | estimated | 6643 | 46 | 1.27422061126531271924023 | 1.80334984592125113778 | -2.35791242474243167848 | 4.90635364727305756105 | 0.483467 |
| primary_frame_timing | 2013 | R8 | equal | year::2013:exposure | estimated | 6643 | 46 | 3.07340402496616960092979 | 2.88615099223365589864 | -2.73960246931345441368 | 8.88641051924579272736 | 0.292611 |
| primary_frame_timing | 2013 | R8 | equal | year::2014:exposure | term_not_identified | 6643 | 46 | — | — | — | — | — |
| primary_frame_timing | 2013 | R8 | equal | year::2015:exposure | term_not_identified | 6643 | 46 | — | — | — | — | — |
| primary_frame_observed_denial | 2013 | R8 | equal | year::2011:exposure | estimated | 7401 | 46 | 0.03223826259578507769099 | 0.04893620163997052658 | -0.06632430696623602129 | 0.13080083215780619055 | 0.513393 |
| primary_frame_observed_denial | 2013 | R8 | equal | year::2013:exposure | estimated | 7401 | 46 | 0.00453267636759962025406 | 0.08890700564337128409 | -0.17453522499396303957 | 0.18360057772916230090 | 0.959565 |
| primary_frame_observed_denial | 2013 | R8 | equal | year::2014:exposure | term_not_identified | 7401 | 46 | — | — | — | — | — |
| primary_frame_observed_denial | 2013 | R8 | equal | year::2015:exposure | term_not_identified | 7401 | 46 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R8 | equal | year::2011:exposure | fit_failed | 12440 | 49 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R8 | equal | year::2013:exposure | fit_failed | 12440 | 49 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R8 | equal | year::2014:exposure | fit_failed | 12440 | 49 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R8 | equal | year::2015:exposure | fit_failed | 12440 | 49 | — | — | — | — | — |
| primary_frame_timely | 2013 | R8 | baseline_positions | year::2011:exposure | estimated | 7179 | 46 | 0.01484198570909664970097 | 0.01727468177195189136 | -0.01995100938962983059 | 0.04963498080782315081 | 0.394797 |
| primary_frame_timely | 2013 | R8 | baseline_positions | year::2013:exposure | estimated | 7179 | 46 | 0.14102659600842387677133 | 0.31726054378312179560 | -0.49796894038334199584 | 0.78002213240018969387 | 0.658802 |
| primary_frame_timely | 2013 | R8 | baseline_positions | year::2014:exposure | term_not_identified | 7179 | 46 | — | — | — | — | — |
| primary_frame_timely | 2013 | R8 | baseline_positions | year::2015:exposure | term_not_identified | 7179 | 46 | — | — | — | — | — |
| primary_frame_observed_filing | 2013 | R8 | baseline_positions | year::2011:exposure | estimated | 7179 | 46 | 0.00000000000000009020562 | 0.00000000000000016367 | -0.00000000000000023944 | 0.00000000000000041985 | 0.584259 |
| primary_frame_observed_filing | 2013 | R8 | baseline_positions | year::2013:exposure | estimated | 7179 | 46 | 0.07413283367554962532964 | 0.27559431826204394689 | -0.48094261669233967504 | 0.62920828404343898121 | 0.789165 |
| primary_frame_observed_filing | 2013 | R8 | baseline_positions | year::2014:exposure | term_not_identified | 7179 | 46 | — | — | — | — | — |
| primary_frame_observed_filing | 2013 | R8 | baseline_positions | year::2015:exposure | term_not_identified | 7179 | 46 | — | — | — | — | — |
| primary_frame_certified_positions | 2013 | R8 | baseline_positions | year::2011:exposure | estimated | 7179 | 46 | -5.63005358593681215495508 | 4.05235834495702196278 | -13.79192226147432975836 | 2.53181508960070544845 | 0.171570 |
| primary_frame_certified_positions | 2013 | R8 | baseline_positions | year::2013:exposure | estimated | 7179 | 46 | 16.54003379560855080399051 | 15.64993095642760323472 | -14.98054517948355268686 | 48.06061277070065784756 | 0.296209 |
| primary_frame_certified_positions | 2013 | R8 | baseline_positions | year::2014:exposure | term_not_identified | 7179 | 46 | — | — | — | — | — |
| primary_frame_certified_positions | 2013 | R8 | baseline_positions | year::2015:exposure | term_not_identified | 7179 | 46 | — | — | — | — | — |
| primary_frame_timing | 2013 | R8 | baseline_positions | year::2011:exposure | estimated | 6475 | 46 | -0.78063284464498616799233 | 3.48424019310507127045 | -7.79825282525276541890 | 6.23698713596279397109 | 0.823735 |
| primary_frame_timing | 2013 | R8 | baseline_positions | year::2013:exposure | estimated | 6475 | 46 | -5.19642772532095609250291 | 9.07271348915281272696 | -23.46981071016860909140 | 13.07695525952669513003 | 0.569663 |
| primary_frame_timing | 2013 | R8 | baseline_positions | year::2014:exposure | term_not_identified | 6475 | 46 | — | — | — | — | — |
| primary_frame_timing | 2013 | R8 | baseline_positions | year::2015:exposure | term_not_identified | 6475 | 46 | — | — | — | — | — |
| primary_frame_observed_denial | 2013 | R8 | baseline_positions | year::2011:exposure | estimated | 7179 | 46 | -0.00121560308400636999868 | 0.00619346797451530984 | -0.01368988792040265988 | 0.01125868175238992075 | 0.845281 |
| primary_frame_observed_denial | 2013 | R8 | baseline_positions | year::2013:exposure | estimated | 7179 | 46 | -0.04152990078039690835521 | 0.09348307895901714071 | -0.22981448691476907831 | 0.14675468535397528935 | 0.658991 |
| primary_frame_observed_denial | 2013 | R8 | baseline_positions | year::2014:exposure | term_not_identified | 7179 | 46 | — | — | — | — | — |
| primary_frame_observed_denial | 2013 | R8 | baseline_positions | year::2015:exposure | term_not_identified | 7179 | 46 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R8 | baseline_positions | year::2011:exposure | fit_failed | 12070 | 49 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R8 | baseline_positions | year::2013:exposure | fit_failed | 12070 | 49 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R8 | baseline_positions | year::2014:exposure | fit_failed | 12070 | 49 | — | — | — | — | — |
| primary_frame_unknown_dates_or_status | 2013 | R8 | baseline_positions | year::2015:exposure | fit_failed | 12070 | 49 | — | — | — | — | — |

## Pre-policy earnings

[Full CSV with formula and inference fields](/tmp/h2a-chapter-directions/earnings_pre_estimates.csv)

| specification | tau | tier | weights | term | status | n | states | estimate | se_jackknife | low | high | p | percent_low | percent_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crop_primary | 2016 | county | equal | D | estimated | 1328 | 47 | 0.00372590 | 0.0096319 | -0.015662190 | 0.023113985 | 0.70066780 | -2.7355650 | 1.8972359 |
| crop_employment_weighted | 2016 | county | emp_l | D | estimated | 1328 | 47 | -0.00369495 | 0.0140143 | -0.031904322 | 0.024514412 | 0.79322285 | -2.8988818 | 3.9027507 |
| crop_cash_dose | 2016 | county | equal | D_h2a | estimated | 1328 | 47 | 0.00155873 | 0.0119646 | -0.022524799 | 0.025642263 | 0.89691418 | -3.0302116 | 2.7398376 |
| crop_commuting_zone | 2016 | county | equal | D_cz | estimated | 1328 | 47 | -0.01664432 | 0.0101547 | -0.037084629 | 0.003795980 | 0.10801919 | -0.4544817 | 4.5506603 |
| crop_no_intensity_control | 2016 | county | equal | D | estimated | 1328 | 47 | -0.00863007 | 0.0022374 | -0.013133636 | -0.004126507 | 0.00035508 | 0.4964089 | 1.5885213 |
| crop_UI_covered | 2016 | county | equal | D | insufficient_support | 76 | 2 | — | — | — | — | — | — | — |
| crop_UI_excluded | 2016 | county | equal | D | estimated | 120 | 3 | -0.10475519 | 0.1671284 | -0.823850845 | 0.614340462 | 0.59480366 | -52.1552386 | 168.7525854 |
| crop_UI_ambiguous | 2016 | county | equal | D | estimated | 1132 | 42 | 0.00523620 | 0.0096297 | -0.014211303 | 0.024683698 | 0.58955345 | -2.9186052 | 1.7199806 |
| animal_primary | 2016 | county | equal | D | estimated | 1152 | 47 | -0.00197521 | 0.0171591 | -0.036514772 | 0.032564351 | 0.90885797 | -3.8323556 | 4.4791899 |
| all_private_primary | 2016 | county | equal | D | estimated | 3017 | 48 | 0.00486230 | 0.0021936 | 0.000449392 | 0.009275217 | 0.03152586 | -1.1068549 | -0.0539125 |
| crop_primary | 2017 | county | equal | D | estimated | 1358 | 47 | 0.00201210 | 0.0108437 | -0.019815164 | 0.023839368 | 0.85360976 | -2.8201929 | 2.4063152 |
| crop_employment_weighted | 2017 | county | emp_l | D | estimated | 1358 | 47 | -0.00265666 | 0.0111322 | -0.025064667 | 0.019751355 | 0.81244047 | -2.3422948 | 3.0534500 |
| crop_cash_dose | 2017 | county | equal | D_h2a | estimated | 1358 | 47 | 0.00913384 | 0.0126214 | -0.016271801 | 0.034539483 | 0.47292859 | -4.0600182 | 1.9718044 |
| crop_commuting_zone | 2017 | county | equal | D_cz | estimated | 1358 | 47 | -0.02980277 | 0.0178078 | -0.065648090 | 0.006042554 | 0.10100191 | -0.7224839 | 8.1963784 |
| crop_no_intensity_control | 2017 | county | equal | D | estimated | 1358 | 47 | -0.00110547 | 0.0069613 | -0.015117828 | 0.012906893 | 0.87452006 | -1.5368945 | 1.8306948 |
| crop_UI_covered | 2017 | county | equal | D | insufficient_support | 79 | 2 | — | — | — | — | — | — | — |
| crop_UI_excluded | 2017 | county | equal | D | estimated | 120 | 3 | -0.00151745 | 0.3894078 | -1.677003971 | 1.673969080 | 0.99724455 | -86.5845465 | 648.1286470 |
| crop_UI_ambiguous | 2017 | county | equal | D | estimated | 1159 | 42 | 0.00614466 | 0.0124947 | -0.019088991 | 0.031378303 | 0.62549815 | -3.6953867 | 2.3171165 |
| animal_primary | 2017 | county | equal | D | estimated | 1158 | 47 | 0.01371946 | 0.0210842 | -0.028720907 | 0.056159835 | 0.51847887 | -6.5171138 | 3.5065892 |
| all_private_primary | 2017 | county | equal | D | estimated | 3018 | 48 | 0.00251645 | 0.0028437 | -0.003204254 | 0.008237162 | 0.38069691 | -0.9835902 | 0.3852507 |
| crop_primary | 2018 | county | equal | D | estimated | 1370 | 48 | 0.00928246 | 0.0073482 | -0.005500308 | 0.024065230 | 0.21274048 | -2.8465284 | 0.6622200 |
| crop_employment_weighted | 2018 | county | emp_l | D | estimated | 1370 | 48 | 0.00120400 | 0.0099424 | -0.018797572 | 0.021205563 | 0.90413060 | -2.5125638 | 2.2813422 |
| crop_cash_dose | 2018 | county | equal | D_h2a | estimated | 1370 | 48 | 0.01711149 | 0.0098625 | -0.002729292 | 0.036952263 | 0.08929365 | -4.3373950 | 0.3280519 |
| crop_commuting_zone | 2018 | county | equal | D_cz | estimated | 1370 | 48 | 0.01857215 | 0.0119723 | -0.005513101 | 0.042657400 | 0.12754909 | -4.9900802 | 0.6637653 |
| crop_no_intensity_control | 2018 | county | equal | D | estimated | 1370 | 48 | -0.00156297 | 0.0024421 | -0.006475832 | 0.003349883 | 0.52527118 | -0.4011791 | 0.7801271 |
| crop_UI_covered | 2018 | county | equal | D | insufficient_support | 79 | 2 | — | — | — | — | — | — | — |
| crop_UI_excluded | 2018 | county | equal | D | estimated | 128 | 3 | 0.03694763 | 0.1919881 | -0.789110653 | 0.863005905 | 0.86516185 | -64.4989206 | 157.7790878 |
| crop_UI_ambiguous | 2018 | county | equal | D | estimated | 1163 | 43 | 0.00887867 | 0.0075236 | -0.006304632 | 0.024061972 | 0.24459957 | -2.8461485 | 0.7594249 |
| animal_primary | 2018 | county | equal | D | estimated | 1175 | 47 | -0.00565301 | 0.0108962 | -0.027585866 | 0.016279852 | 0.60638262 | -1.9346234 | 3.3657041 |
| all_private_primary | 2018 | county | equal | D | estimated | 3019 | 48 | -0.00046818 | 0.0028492 | -0.006200059 | 0.005263697 | 0.87018443 | -0.6296530 | 0.7467817 |
| crop_primary | 2019 | county | equal | D | estimated | 1386 | 48 | -0.01108567 | 0.0108468 | -0.032906538 | 0.010735189 | 0.31200031 | -1.2799606 | 4.0277855 |
| crop_employment_weighted | 2019 | county | emp_l | D | estimated | 1386 | 48 | -0.00973344 | 0.0172583 | -0.044452708 | 0.024985833 | 0.57544577 | -2.9537968 | 5.4791639 |
| crop_cash_dose | 2019 | county | equal | D_h2a | estimated | 1386 | 48 | -0.01199838 | 0.0091683 | -0.030442535 | 0.006445770 | 0.19700352 | -0.7705087 | 3.7206500 |
| crop_commuting_zone | 2019 | county | equal | D_cz | estimated | 1386 | 48 | -0.00762798 | 0.0189392 | -0.045728768 | 0.030472808 | 0.68895016 | -3.5906859 | 5.6408049 |
| crop_no_intensity_control | 2019 | county | equal | D | estimated | 1386 | 48 | -0.00339735 | 0.0029108 | -0.009253151 | 0.002458449 | 0.24903794 | -0.2945792 | 1.1165657 |
| crop_UI_covered | 2019 | county | equal | D | insufficient_support | 83 | 2 | — | — | — | — | — | — | — |
| crop_UI_excluded | 2019 | county | equal | D | estimated | 133 | 3 | -0.08992677 | 0.5308372 | -2.373934835 | 2.194081296 | 0.88106247 | -92.8130087 | 1626.5698756 |
| crop_UI_ambiguous | 2019 | county | equal | D | estimated | 1170 | 43 | -0.00843245 | 0.0105685 | -0.029760588 | 0.012895691 | 0.42942641 | -1.5355709 | 3.6358064 |
| animal_primary | 2019 | county | equal | D | estimated | 1168 | 47 | 0.02406712 | 0.0340349 | -0.044441662 | 0.092575898 | 0.48305295 | -10.5142756 | 5.4777658 |
| all_private_primary | 2019 | county | equal | D | estimated | 3019 | 48 | -0.00254098 | 0.0018916 | -0.006346433 | 0.001264482 | 0.18563284 | -0.1516228 | 0.7644792 |
| crop_primary | 2020 | county | equal | D | estimated | 1412 | 48 | 0.01024408 | 0.0105636 | -0.011007086 | 0.031495252 | 0.33713266 | -3.7089011 | 1.3296121 |
| crop_employment_weighted | 2020 | county | emp_l | D | estimated | 1412 | 48 | 0.00014628 | 0.0126583 | -0.025319025 | 0.025611589 | 0.99082865 | -3.0266421 | 3.0849099 |
| crop_cash_dose | 2020 | county | equal | D_h2a | estimated | 1412 | 48 | 0.01606682 | 0.0134954 | -0.011082415 | 0.043216055 | 0.23981240 | -5.0537520 | 1.3387721 |
| crop_commuting_zone | 2020 | county | equal | D_cz | estimated | 1412 | 48 | 0.01265612 | 0.0192136 | -0.025996688 | 0.051308926 | 0.51329891 | -5.9713546 | 3.1687721 |
| crop_no_intensity_control | 2020 | county | equal | D | estimated | 1412 | 48 | -0.00297942 | 0.0022966 | -0.007599516 | 0.001640667 | 0.20084640 | -0.1966864 | 0.9161128 |
| crop_UI_covered | 2020 | county | equal | D | insufficient_support | 82 | 2 | — | — | — | — | — | — | — |
| crop_UI_excluded | 2020 | county | equal | D | estimated | 134 | 3 | -0.00139194 | 0.1710854 | -0.737512906 | 0.734729026 | 0.99424712 | -58.5911198 | 142.3021918 |
| crop_UI_ambiguous | 2020 | county | equal | D | estimated | 1196 | 43 | 0.01208686 | 0.0114394 | -0.010998736 | 0.035172452 | 0.29673346 | -4.1328630 | 1.3285967 |
| animal_primary | 2020 | county | equal | D | estimated | 1177 | 46 | 0.04191806 | 0.0388002 | -0.036229551 | 0.120065667 | 0.28573960 | -13.4180481 | 4.4434364 |
| all_private_primary | 2020 | county | equal | D | estimated | 3023 | 48 | 0.00258589 | 0.0038689 | -0.005197428 | 0.010369200 | 0.50716693 | -1.2365945 | 0.6256403 |
| crop_primary | 2021 | county | equal | D | estimated | 1449 | 48 | -0.00410886 | 0.0075838 | -0.019365426 | 0.011147709 | 0.59051807 | -1.3288173 | 2.3510629 |
| crop_employment_weighted | 2021 | county | emp_l | D | estimated | 1449 | 48 | -0.01992607 | 0.0098656 | -0.039773118 | -0.000079022 | 0.04913102 | 0.0094831 | 4.8885049 |
| crop_cash_dose | 2021 | county | equal | D_h2a | estimated | 1449 | 48 | 0.00019934 | 0.0098179 | -0.019551719 | 0.019950394 | 0.98388724 | -2.3656173 | 2.3739462 |
| crop_commuting_zone | 2021 | county | equal | D_cz | estimated | 1449 | 48 | 0.00668501 | 0.0162697 | -0.026045493 | 0.039415514 | 0.68302352 | -4.6197466 | 3.1748145 |
| crop_no_intensity_control | 2021 | county | equal | D | estimated | 1449 | 48 | 0.00067257 | 0.0014078 | -0.002159549 | 0.003504690 | 0.63504465 | -0.4196796 | 0.2594819 |
| crop_UI_covered | 2021 | county | equal | D | insufficient_support | 82 | 2 | — | — | — | — | — | — | — |
| crop_UI_excluded | 2021 | county | equal | D | estimated | 141 | 3 | -0.03711173 | 0.1926833 | -0.866160969 | 0.791937509 | 0.86505358 | -61.3384651 | 182.7500660 |
| crop_UI_ambiguous | 2021 | county | equal | D | estimated | 1226 | 43 | -0.00069568 | 0.0080399 | -0.016920876 | 0.015529521 | 0.93145790 | -1.8462859 | 2.0512601 |
| animal_primary | 2021 | county | equal | D | estimated | 1209 | 46 | -0.02998348 | 0.0100116 | -0.050147938 | -0.009819027 | 0.00445145 | 1.1852523 | 6.2025066 |
| all_private_primary | 2021 | county | equal | D | estimated | 3022 | 48 | 0.00761475 | 0.0037955 | -0.000020814 | 0.015250322 | 0.05060259 | -1.8133951 | 0.0024978 |
| crop_primary | 2022 | county | equal | D | estimated | 1471 | 48 | -0.00717473 | 0.0147371 | -0.036821873 | 0.022472416 | 0.62862792 | -2.6606539 | 4.5176998 |
| crop_employment_weighted | 2022 | county | emp_l | D | estimated | 1471 | 48 | -0.00056040 | 0.0172016 | -0.035165573 | 0.034044777 | 0.97414889 | -4.0030468 | 4.3101710 |
| crop_cash_dose | 2022 | county | equal | D_h2a | estimated | 1471 | 48 | 0.00341317 | 0.0146198 | -0.025998085 | 0.032824425 | 0.81641649 | -3.8623637 | 3.1689451 |
| crop_commuting_zone | 2022 | county | equal | D_cz | estimated | 1471 | 48 | -0.01443112 | 0.0113383 | -0.037240886 | 0.008378654 | 0.20935878 | -1.0004008 | 4.5702662 |
| crop_no_intensity_control | 2022 | county | equal | D | estimated | 1471 | 48 | -0.00233488 | 0.0027987 | -0.007965163 | 0.003295406 | 0.40835000 | -0.3946679 | 0.9604021 |
| crop_UI_covered | 2022 | county | equal | D | insufficient_support | 82 | 2 | — | — | — | — | — | — | — |
| crop_UI_excluded | 2022 | county | equal | D | estimated | 140 | 3 | 0.08781460 | 0.7558079 | -3.164164154 | 3.339793359 | 0.91811967 | -98.1825796 | 4356.7147499 |
| crop_UI_ambiguous | 2022 | county | equal | D | estimated | 1249 | 43 | -0.00659547 | 0.0168698 | -0.040640108 | 0.027449171 | 0.69780086 | -3.2402424 | 4.9976864 |
| animal_primary | 2022 | county | equal | D | estimated | 1222 | 46 | 0.04811026 | 0.0214454 | 0.004916961 | 0.091303557 | 0.02984054 | -10.3775436 | -0.5882980 |
| all_private_primary | 2022 | county | equal | D | estimated | 3015 | 48 | 0.00177608 | 0.0019623 | -0.002171550 | 0.005723716 | 0.37002676 | -0.6844925 | 0.2609258 |
| crop_primary | 2023 | county | equal | D | estimated | 1465 | 48 | 0.00960317 | 0.0106721 | -0.011866254 | 0.031072598 | 0.37279188 | -3.6600514 | 1.4341370 |
| crop_employment_weighted | 2023 | county | emp_l | D | estimated | 1465 | 48 | 0.02334830 | 0.0138476 | -0.004509535 | 0.051206132 | 0.09840758 | -5.9597551 | 0.5426110 |
| crop_cash_dose | 2023 | county | equal | D_h2a | estimated | 1465 | 48 | -0.01918450 | 0.0184547 | -0.056310585 | 0.017941575 | 0.30386959 | -2.1299777 | 6.9908041 |
| crop_commuting_zone | 2023 | county | equal | D_cz | estimated | 1465 | 48 | 0.02217013 | 0.0185515 | -0.015150772 | 0.059491034 | 0.23806089 | -6.8900601 | 1.8347206 |
| crop_no_intensity_control | 2023 | county | equal | D | estimated | 1465 | 48 | -0.00115797 | 0.0038084 | -0.008819425 | 0.006503489 | 0.76242541 | -0.7773813 | 1.0639512 |
| crop_UI_covered | 2023 | county | equal | D | insufficient_support | 84 | 2 | — | — | — | — | — | — | — |
| crop_UI_excluded | 2023 | county | equal | D | estimated | 142 | 3 | -0.09831581 | 0.9906140 | -4.360583884 | 4.163952270 | 0.92999371 | -99.3240070 | 18629.7949371 |
| crop_UI_ambiguous | 2023 | county | equal | D | estimated | 1239 | 43 | 0.00303518 | 0.0111497 | -0.019465841 | 0.025536202 | 0.78678621 | -3.0178692 | 2.3633968 |
| animal_primary | 2023 | county | equal | D | estimated | 1213 | 47 | 0.03056040 | 0.0337169 | -0.037308228 | 0.098429030 | 0.36945724 | -11.1405995 | 4.5787169 |
| all_private_primary | 2023 | county | equal | D | estimated | 3017 | 48 | -0.00039087 | 0.0022467 | -0.004910696 | 0.004128952 | 0.86263288 | -0.4942488 | 0.5910232 |
| crop_primary | 2024 | county | equal | D | estimated | 1483 | 47 | -0.00812106 | 0.0092614 | -0.026763276 | 0.010521146 | 0.38511209 | -1.2546010 | 3.2637213 |
| crop_employment_weighted | 2024 | county | emp_l | D | estimated | 1483 | 47 | -0.00216047 | 0.0099855 | -0.022260183 | 0.017939251 | 0.82966308 | -2.1297046 | 2.7072189 |
| crop_cash_dose | 2024 | county | equal | D_h2a | estimated | 1483 | 47 | -0.00416678 | 0.0119480 | -0.028216857 | 0.019883299 | 0.72887366 | -2.3577561 | 3.4440012 |
| crop_commuting_zone | 2024 | county | equal | D_cz | estimated | 1483 | 47 | -0.01629600 | 0.0168483 | -0.050209872 | 0.017617878 | 0.33849260 | -2.0919540 | 6.2104000 |
| crop_no_intensity_control | 2024 | county | equal | D | estimated | 1483 | 47 | -0.00311926 | 0.0029942 | -0.009146272 | 0.002907760 | 0.30296489 | -0.3483231 | 1.1035978 |
| crop_UI_covered | 2024 | county | equal | D | insufficient_support | 85 | 2 | — | — | — | — | — | — | — |
| crop_UI_excluded | 2024 | county | equal | D | estimated | 145 | 3 | -0.01415730 | 0.3369759 | -1.464047529 | 1.435732933 | 0.97030557 | -82.1448735 | 479.4197786 |
| crop_UI_ambiguous | 2024 | county | equal | D | estimated | 1253 | 42 | -0.00575564 | 0.0095055 | -0.024952344 | 0.013441073 | 0.54817956 | -1.5999908 | 3.0395606 |
| animal_primary | 2024 | county | equal | D | estimated | 1219 | 46 | -0.01055187 | 0.0254670 | -0.061845140 | 0.040741393 | 0.68059769 | -4.7713812 | 7.7037447 |
| all_private_primary | 2024 | county | equal | D | estimated | 3017 | 47 | -0.00196867 | 0.0035416 | -0.009097496 | 0.005160163 | 0.58099189 | -0.6173064 | 1.0976803 |
| QWI_stable_monthly_primary | 2016 | county | equal | D | estimated | 1748 | 47 | 0.00452901 | 0.0052453 | -0.006029228 | 0.015087249 | 0.39237439 | -1.7941793 | 0.7261310 |
| QWI_stable_monthly_employment_weighted | 2016 | county | emp_l | D | estimated | 1748 | 47 | 0.01048513 | 0.0061396 | -0.001873281 | 0.022843535 | 0.09441835 | -2.7039937 | 0.2250466 |
| QWI_stable_monthly_cash_dose | 2016 | county | equal | D_h2a | estimated | 1748 | 47 | 0.00415881 | 0.0071740 | -0.010281610 | 0.018599226 | 0.56493928 | -2.2071843 | 1.2414359 |
| QWI_stable_monthly_commuting_zone | 2016 | county | equal | D_cz | estimated | 1748 | 47 | -0.02188919 | 0.0108680 | -0.043765430 | -0.000012944 | 0.04987014 | 0.0015533 | 5.3922076 |
| QWI_stable_monthly_no_intensity_control | 2016 | county | equal | D | estimated | 1748 | 47 | -0.00657778 | 0.0025896 | -0.011790291 | -0.001365264 | 0.01452300 | 0.1639659 | 1.4248911 |
| QWI_stable_monthly_primary | 2017 | county | equal | D | estimated | 1769 | 47 | 0.00974436 | 0.0072534 | -0.004855913 | 0.024344640 | 0.18572442 | -2.8790977 | 0.5844106 |
| QWI_stable_monthly_employment_weighted | 2017 | county | emp_l | D | estimated | 1769 | 47 | -0.01175700 | 0.0123317 | -0.036579504 | 0.013065502 | 0.34537396 | -1.5556333 | 4.4873060 |
| QWI_stable_monthly_cash_dose | 2017 | county | equal | D_h2a | estimated | 1769 | 47 | 0.01551722 | 0.0110782 | -0.006781936 | 0.037816383 | 0.16801215 | -4.4365403 | 0.8171529 |
| QWI_stable_monthly_commuting_zone | 2017 | county | equal | D_cz | estimated | 1769 | 47 | 0.01339495 | 0.0111323 | -0.009013209 | 0.035803107 | 0.23503551 | -4.2053864 | 1.0874553 |
| QWI_stable_monthly_no_intensity_control | 2017 | county | equal | D | estimated | 1769 | 47 | 0.00328058 | 0.0014100 | 0.000442467 | 0.006118683 | 0.02443916 | -0.7315530 | -0.0530819 |
| QWI_stable_monthly_primary | 2018 | county | equal | D | estimated | 1787 | 47 | 0.00053892 | 0.0052455 | -0.010019798 | 0.011097647 | 0.91861626 | -1.3228895 | 1.2096334 |
| QWI_stable_monthly_employment_weighted | 2018 | county | emp_l | D | estimated | 1787 | 47 | -0.01142712 | 0.0081441 | -0.027820374 | 0.004966134 | 0.16729712 | -0.5941639 | 3.3947963 |
| QWI_stable_monthly_cash_dose | 2018 | county | equal | D_h2a | estimated | 1787 | 47 | -0.00267171 | 0.0052768 | -0.013293333 | 0.007949907 | 0.61505412 | -0.9494527 | 1.6079912 |
| QWI_stable_monthly_commuting_zone | 2018 | county | equal | D_cz | estimated | 1787 | 47 | -0.01567210 | 0.0087833 | -0.033351911 | 0.002007709 | 0.08096939 | -0.2406350 | 4.0833978 |
| QWI_stable_monthly_no_intensity_control | 2018 | county | equal | D | estimated | 1787 | 47 | -0.00190905 | 0.0019129 | -0.005759574 | 0.001941482 | 0.32351504 | -0.2327067 | 0.6935429 |
| QWI_stable_monthly_primary | 2019 | county | equal | D | estimated | 1794 | 47 | -0.00532812 | 0.0104477 | -0.026358204 | 0.015701958 | 0.61250268 | -1.8665942 | 3.2135384 |
| QWI_stable_monthly_employment_weighted | 2019 | county | emp_l | D | estimated | 1794 | 47 | -0.00186796 | 0.0165247 | -0.035130545 | 0.031394619 | 0.91049046 | -3.6972724 | 4.3057865 |
| QWI_stable_monthly_cash_dose | 2019 | county | equal | D_h2a | estimated | 1794 | 47 | -0.00923387 | 0.0101143 | -0.029592929 | 0.011125189 | 0.36602877 | -1.3261508 | 3.6149579 |
| QWI_stable_monthly_commuting_zone | 2019 | county | equal | D_cz | estimated | 1794 | 47 | 0.00161556 | 0.0110882 | -0.020703798 | 0.023934913 | 0.88479417 | -2.8313343 | 2.5155756 |
| QWI_stable_monthly_no_intensity_control | 2019 | county | equal | D | estimated | 1794 | 47 | -0.00075552 | 0.0022551 | -0.005294810 | 0.003783770 | 0.73912795 | -0.4530231 | 0.6374000 |
| QWI_stable_monthly_primary | 2020 | county | equal | D | estimated | 1810 | 47 | 0.00076236 | 0.0120975 | -0.023588543 | 0.025113269 | 0.95002484 | -2.9686363 | 2.8710680 |
| QWI_stable_monthly_employment_weighted | 2020 | county | emp_l | D | estimated | 1810 | 47 | -0.00323779 | 0.0067015 | -0.016727118 | 0.010251539 | 0.63128290 | -1.2226489 | 2.0275350 |
| QWI_stable_monthly_cash_dose | 2020 | county | equal | D_h2a | estimated | 1810 | 47 | 0.00492576 | 0.0132859 | -0.021817400 | 0.031668913 | 0.71252545 | -3.7289654 | 2.6526610 |
| QWI_stable_monthly_commuting_zone | 2020 | county | equal | D_cz | estimated | 1810 | 47 | 0.01005993 | 0.0177796 | -0.025728497 | 0.045848353 | 0.57427026 | -5.3531911 | 3.1355747 |
| QWI_stable_monthly_no_intensity_control | 2020 | county | equal | D | estimated | 1810 | 47 | -0.00524576 | 0.0035932 | -0.012478519 | 0.001986996 | 0.15111128 | -0.2381555 | 1.5086898 |
| QWI_stable_monthly_primary | 2021 | county | equal | D | estimated | 1836 | 47 | 0.00287034 | 0.0085106 | -0.014260670 | 0.020001358 | 0.73745076 | -2.3715881 | 1.7260067 |
| QWI_stable_monthly_employment_weighted | 2021 | county | emp_l | D | estimated | 1836 | 47 | 0.00793157 | 0.0087144 | -0.009609625 | 0.025472768 | 0.36747894 | -3.0104865 | 1.1598295 |
| QWI_stable_monthly_cash_dose | 2021 | county | equal | D_h2a | estimated | 1836 | 47 | -0.00322719 | 0.0132014 | -0.029800247 | 0.023345876 | 0.80796357 | -2.7626269 | 3.6407386 |
| QWI_stable_monthly_commuting_zone | 2021 | county | equal | D_cz | estimated | 1836 | 47 | 0.00044430 | 0.0139383 | -0.027611980 | 0.028500583 | 0.97470857 | -3.3622466 | 3.3689434 |
| QWI_stable_monthly_no_intensity_control | 2021 | county | equal | D | estimated | 1836 | 47 | -0.00222623 | 0.0015677 | -0.005381776 | 0.000929318 | 0.16232340 | -0.1114560 | 0.6479030 |
| QWI_stable_monthly_primary | 2022 | county | equal | D | estimated | 1818 | 46 | 0.01801045 | 0.0112713 | -0.004691129 | 0.040712027 | 0.11706380 | -4.7680253 | 0.5645229 |
| QWI_stable_monthly_employment_weighted | 2022 | county | emp_l | D | estimated | 1818 | 46 | -0.00362881 | 0.0075331 | -0.018801210 | 0.011543590 | 0.63234173 | -1.3756806 | 2.2817886 |
| QWI_stable_monthly_cash_dose | 2022 | county | equal | D_h2a | estimated | 1818 | 46 | 0.03058903 | 0.0131675 | 0.004068251 | 0.057109808 | 0.02475419 | -6.6236206 | -0.4870004 |
| QWI_stable_monthly_commuting_zone | 2022 | county | equal | D_cz | estimated | 1818 | 46 | 0.02084143 | 0.0114004 | -0.002120058 | 0.043802911 | 0.07416245 | -5.1205922 | 0.2547309 |
| QWI_stable_monthly_no_intensity_control | 2022 | county | equal | D | estimated | 1818 | 46 | 0.00148570 | 0.0025606 | -0.003671617 | 0.006643011 | 0.56466428 | -0.7939924 | 0.4415660 |
| QWI_stable_monthly_primary | 2023 | county | equal | D | estimated | 1827 | 46 | 0.01188676 | 0.0072451 | -0.002705699 | 0.026479214 | 0.10784185 | -3.1275534 | 0.3252115 |
| QWI_stable_monthly_employment_weighted | 2023 | county | emp_l | D | estimated | 1827 | 46 | 0.00329969 | 0.0127897 | -0.022459998 | 0.029059370 | 0.79758586 | -3.4270248 | 2.7318487 |
| QWI_stable_monthly_cash_dose | 2023 | county | equal | D_h2a | estimated | 1827 | 46 | 0.00557510 | 0.0087114 | -0.011970617 | 0.023120814 | 0.52543380 | -2.7363620 | 1.4468409 |
| QWI_stable_monthly_commuting_zone | 2023 | county | equal | D_cz | estimated | 1827 | 46 | 0.02446036 | 0.0124878 | -0.000691413 | 0.049612140 | 0.05635840 | -5.7797037 | 0.0830040 |
| QWI_stable_monthly_no_intensity_control | 2023 | county | equal | D | estimated | 1827 | 46 | 0.00028941 | 0.0016101 | -0.002953518 | 0.003532336 | 0.85815959 | -0.4229833 | 0.3550509 |
| QWI_stable_monthly_primary | 2024 | county | equal | D | estimated | 1855 | 46 | -0.01381572 | 0.0075962 | -0.029115220 | 0.001483777 | 0.07560489 | -0.1778948 | 3.5555776 |
| QWI_stable_monthly_employment_weighted | 2024 | county | emp_l | D | estimated | 1855 | 46 | 0.00074212 | 0.0065617 | -0.012473831 | 0.013958067 | 0.91045557 | -1.6610185 | 1.5081187 |
| QWI_stable_monthly_cash_dose | 2024 | county | equal | D_h2a | estimated | 1855 | 46 | -0.00829541 | 0.0122321 | -0.032932113 | 0.016341285 | 0.50113774 | -1.9418526 | 4.0309781 |
| QWI_stable_monthly_commuting_zone | 2024 | county | equal | D_cz | estimated | 1855 | 46 | -0.02708515 | 0.0224349 | -0.072271294 | 0.018100995 | 0.23363226 | -2.1486987 | 9.0597328 |
| QWI_stable_monthly_no_intensity_control | 2024 | county | equal | D | estimated | 1855 | 46 | -0.00241489 | 0.0021666 | -0.006778624 | 0.001948836 | 0.27093632 | -0.2335871 | 0.8167523 |

## Original worksite-business benchmarks

| block | spec | id_tier | tau | outcome | term | weights | estimate | se_jackknife | p_jackknife | se_state_crv1 | n | units | states | mean_y |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1_county_decomposition | total | balanced | 2025 | total | dose | equal | 0.00303219 | 0.0573955 | 0.9581013262843 | 0.0305674 | 2655 | 2655 | 46 | 0.7025143 |
| P1_county_decomposition | within | balanced | 2025 | within | dose | equal | 0.01874792 | 0.0272328 | 0.4947144935829 | 0.0153917 | 2655 | 2655 | 46 | 0.6960483 |
| P1_county_decomposition | between | balanced | 2025 | between | dose | equal | -0.00352557 | 0.0055684 | 0.5298451084153 | 0.0035906 | 2655 | 2655 | 46 | -0.0097890 |
| P1_county_decomposition | entry | balanced | 2025 | entry | dose | equal | 0.01023406 | 0.0503214 | 0.8397596026537 | 0.0258717 | 2655 | 2655 | 46 | 0.0480415 |
| P1_county_decomposition | exit | balanced | 2025 | exit | dose | equal | -0.02830429 | 0.0308247 | 0.3633924788632 | 0.0181583 | 2655 | 2655 | 46 | -0.0375325 |
| P1_county_decomposition | no_continuer | balanced | 2025 | no_continuer | dose | equal | 0.00588006 | 0.0152945 | 0.7024513729449 | 0.0081599 | 2655 | 2655 | 46 | 0.0057460 |
| P2_continuer_pass_through | employer_state_units | balanced | 2025 | dw | dose | equal | -0.05600218 | 0.0786803 | 0.4802817740252 | 0.0266906 | 23832 | 23832 | 46 | 0.6399271 |
| P2_continuer_pass_through | employer_state_units | balanced | 2025 | dw | dose | h0 | 0.01151157 | 0.0328512 | 0.7276617440090 | 0.0158134 | 23832 | 23832 | 46 | 0.6793017 |
| P3_selection_margins | exit_of_base_year_incumbents | balanced | 2025 | exit_hourly | dose | equal | 0.00854112 | 0.0237950 | 0.7213147493410 | 0.0145163 | 37287 | 37287 | 46 | 0.3608496 |
| P3_selection_margins | exit_of_base_year_incumbents | balanced | 2025 | exit_any | dose | equal | 0.00791124 | 0.0232503 | 0.7352416300146 | 0.0141177 | 37287 | 37287 | 46 | 0.3585700 |
| P3_selection_margins | exit_of_base_year_incumbents | balanced | 2025 | exit_hourly | dose | h0 | 0.01100206 | 0.0108936 | 0.3179157237883 | 0.0060224 | 37287 | 37287 | 46 | 0.1472008 |
| P3_selection_margins | exit_of_base_year_incumbents | balanced | 2025 | exit_any | dose | h0 | 0.01109007 | 0.0108586 | 0.3125655981955 | 0.0059848 | 37287 | 37287 | 46 | 0.1454947 |
| P3_selection_margins | entrant_share_of_tau_units | balanced | 2025 | entrant | dose | equal | -0.00050707 | 0.0204559 | 0.9803330787713 | 0.0111978 | 40008 | 40008 | 46 | 0.4043191 |
| P3_selection_margins | entrant_share_of_tau_units | balanced | 2025 | entrant | dose | h1 | 0.00835387 | 0.0078905 | 0.2953732492946 | 0.0048948 | 40008 | 40008 | 46 | 0.1415603 |
| P4_selection_test | base_year_offer_minus_floor | balanced | 2025 | prem0 | dose | equal | -0.01811296 | 0.1338788 | 0.8929830433293 | 0.0794011 | 37287 | 37287 | 46 | 0.3263496 |
| P4_selection_test | base_year_offer_minus_floor | balanced | 2025 | prem0 | stay | equal | 0.14484883 | 0.1564730 | 0.3595340547780 | 0.1541329 | 37287 | 37287 | 46 | 0.3263496 |
| P4_selection_test | base_year_offer_minus_floor | balanced | 2025 | prem0 | stay_x_dose | equal | 0.07531655 | 0.0967545 | 0.4403900013208 | 0.0745526 | 37287 | 37287 | 46 | 0.3263496 |
| P4_selection_test | prior_year_wage_change | balanced | 2025 | dw_pre | dose | equal | -0.02283535 | 0.1085212 | 0.8342874493038 | 0.0727426 | 20666 | 20666 | 46 | 1.4059789 |
| P4_selection_test | prior_year_wage_change | balanced | 2025 | dw_pre | stay | equal | 0.11296622 | 0.1401331 | 0.4244048625388 | 0.1341537 | 20666 | 20666 | 46 | 1.4059789 |
| P4_selection_test | prior_year_wage_change | balanced | 2025 | dw_pre | stay_x_dose | equal | 0.03348540 | 0.0757995 | 0.6607764448293 | 0.0652212 | 20666 | 20666 | 46 | 1.4059789 |
| P1_county_decomposition | total | high_recall | 2025 | total | dose | equal | 0.00303219 | 0.0573955 | 0.9581013262843 | 0.0305674 | 2655 | 2655 | 46 | 0.7025143 |
| P1_county_decomposition | within | high_recall | 2025 | within | dose | equal | 0.02487233 | 0.0251674 | 0.3283025338574 | 0.0146000 | 2655 | 2655 | 46 | 0.6948609 |
| P1_county_decomposition | between | high_recall | 2025 | between | dose | equal | -0.00870783 | 0.0056689 | 0.1315203770202 | 0.0035237 | 2655 | 2655 | 46 | -0.0088196 |
| P1_county_decomposition | entry | high_recall | 2025 | entry | dose | equal | 0.00749128 | 0.0491863 | 0.8796279204948 | 0.0255594 | 2655 | 2655 | 46 | 0.0386456 |
| P1_county_decomposition | exit | high_recall | 2025 | exit | dose | equal | -0.02621301 | 0.0285154 | 0.3628620589474 | 0.0169196 | 2655 | 2655 | 46 | -0.0281080 |
| P1_county_decomposition | no_continuer | high_recall | 2025 | no_continuer | dose | equal | 0.00558943 | 0.0152288 | 0.7153165290816 | 0.0081271 | 2655 | 2655 | 46 | 0.0059354 |
| P2_continuer_pass_through | employer_state_units | high_recall | 2025 | dw | dose | equal | 0.00402474 | 0.0296327 | 0.8925686535643 | 0.0159312 | 21404 | 21404 | 46 | 0.6831412 |
| P2_continuer_pass_through | employer_state_units | high_recall | 2025 | dw | dose | h0 | 0.01177774 | 0.0419899 | 0.7803872796477 | 0.0188008 | 21404 | 21404 | 46 | 0.6779776 |
| P3_selection_margins | exit_of_base_year_incumbents | high_recall | 2025 | exit_hourly | dose | equal | 0.01664007 | 0.0259707 | 0.5249507752586 | 0.0154459 | 32071 | 32071 | 46 | 0.3326058 |
| P3_selection_margins | exit_of_base_year_incumbents | high_recall | 2025 | exit_any | dose | equal | 0.01693868 | 0.0261547 | 0.5205122580025 | 0.0154955 | 32071 | 32071 | 46 | 0.3303919 |
| P3_selection_margins | exit_of_base_year_incumbents | high_recall | 2025 | exit_hourly | dose | h0 | 0.01374385 | 0.0104152 | 0.1936451673531 | 0.0050534 | 32071 | 32071 | 46 | 0.1168805 |
| P3_selection_margins | exit_of_base_year_incumbents | high_recall | 2025 | exit_any | dose | h0 | 0.01369541 | 0.0104717 | 0.1975671848643 | 0.0050500 | 32071 | 32071 | 46 | 0.1152075 |
| P3_selection_margins | entrant_share_of_tau_units | high_recall | 2025 | entrant | dose | equal | 0.00908667 | 0.0215332 | 0.6750460119982 | 0.0116577 | 34145 | 34145 | 46 | 0.3731439 |
| P3_selection_margins | entrant_share_of_tau_units | high_recall | 2025 | entrant | dose | h1 | 0.00797307 | 0.0066979 | 0.2401346453703 | 0.0039108 | 34145 | 34145 | 46 | 0.1094521 |
| P1_county_decomposition | total | balanced | 2026 | total | dose | equal | 0.66233208 | 0.0945617 | 0.0000000100437 | 0.0613290 | 2686 | 2686 | 46 | -2.1759091 |
| P1_county_decomposition | within | balanced | 2026 | within | dose | equal | 0.61583937 | 0.0944880 | 0.0000000530691 | 0.0587357 | 2686 | 2686 | 46 | -2.1378344 |
| P1_county_decomposition | between | balanced | 2026 | between | dose | equal | -0.00793887 | 0.0155172 | 0.6114202426842 | 0.0083347 | 2686 | 2686 | 46 | 0.0058617 |
| P1_county_decomposition | entry | balanced | 2026 | entry | dose | equal | 0.02077710 | 0.0330233 | 0.5324223769026 | 0.0173268 | 2686 | 2686 | 46 | 0.0098098 |
| P1_county_decomposition | exit | balanced | 2026 | exit | dose | equal | 0.02669601 | 0.0330721 | 0.4237939446767 | 0.0163309 | 2686 | 2686 | 46 | -0.0226746 |
| P1_county_decomposition | no_continuer | balanced | 2026 | no_continuer | dose | equal | 0.00695847 | 0.0129084 | 0.5924995809366 | 0.0075019 | 2686 | 2686 | 46 | -0.0310716 |
| P2_continuer_pass_through | employer_state_units | balanced | 2026 | dw | dose | equal | 0.59512732 | 0.1670890 | 0.0008845244747 | 0.0897172 | 25424 | 25424 | 46 | -2.2295453 |
| P2_continuer_pass_through | employer_state_units | balanced | 2026 | dw | dose | h0 | 0.64404439 | 0.0907482 | 0.0000000073172 | 0.0531087 | 25424 | 25424 | 46 | -2.3792203 |
| P3_selection_margins | exit_of_base_year_incumbents | balanced | 2026 | exit_hourly | dose | equal | 0.01501246 | 0.0319454 | 0.6406663706780 | 0.0145262 | 40008 | 40008 | 46 | 0.3645271 |
| P3_selection_margins | exit_of_base_year_incumbents | balanced | 2026 | exit_any | dose | equal | 0.01684241 | 0.0324449 | 0.6062289975471 | 0.0146816 | 40008 | 40008 | 46 | 0.3612777 |
| P3_selection_margins | exit_of_base_year_incumbents | balanced | 2026 | exit_hourly | dose | h0 | 0.01438341 | 0.0164021 | 0.3851862638365 | 0.0068507 | 40008 | 40008 | 46 | 0.1147426 |
| P3_selection_margins | exit_of_base_year_incumbents | balanced | 2026 | exit_any | dose | h0 | 0.01490807 | 0.0164498 | 0.3696168434330 | 0.0068741 | 40008 | 40008 | 46 | 0.1124931 |
| P3_selection_margins | entrant_share_of_tau_units | balanced | 2026 | entrant | dose | equal | 0.03758760 | 0.0544169 | 0.4932804879548 | 0.0235645 | 43647 | 43647 | 46 | 0.4175086 |
| P3_selection_margins | entrant_share_of_tau_units | balanced | 2026 | entrant | dose | h1 | 0.00235887 | 0.0176653 | 0.8943684872386 | 0.0077352 | 43647 | 43647 | 46 | 0.1394703 |
| P4_selection_test | base_year_offer_minus_floor | balanced | 2026 | prem0 | dose | equal | -0.00329053 | 0.1559275 | 0.9832568416815 | 0.0957787 | 40008 | 40008 | 46 | 0.3678590 |
| P4_selection_test | base_year_offer_minus_floor | balanced | 2026 | prem0 | stay | equal | 0.09698768 | 0.3204597 | 0.7635499350261 | 0.2929698 | 40008 | 40008 | 46 | 0.3678590 |
| P4_selection_test | base_year_offer_minus_floor | balanced | 2026 | prem0 | stay_x_dose | equal | -0.00440415 | 0.1285541 | 0.9728220552178 | 0.0920239 | 40008 | 40008 | 46 | 0.3678590 |
| P4_selection_test | prior_year_wage_change | balanced | 2026 | dw_pre | dose | equal | -0.05169154 | 0.0632037 | 0.4177485920739 | 0.0380196 | 23832 | 23832 | 46 | 0.6399271 |
| P4_selection_test | prior_year_wage_change | balanced | 2026 | dw_pre | stay | equal | -0.01666060 | 0.0613296 | 0.7871283094671 | 0.0552664 | 23832 | 23832 | 46 | 0.6399271 |
| P4_selection_test | prior_year_wage_change | balanced | 2026 | dw_pre | stay_x_dose | equal | -0.00547559 | 0.0548371 | 0.9209053636679 | 0.0443903 | 23832 | 23832 | 46 | 0.6399271 |
| P1_county_decomposition | total | high_recall | 2026 | total | dose | equal | 0.66233208 | 0.0945617 | 0.0000000100437 | 0.0613290 | 2686 | 2686 | 46 | -2.1759091 |
| P1_county_decomposition | within | high_recall | 2026 | within | dose | equal | 0.60766683 | 0.0957479 | 0.0000000953698 | 0.0587999 | 2686 | 2686 | 46 | -2.1407311 |
| P1_county_decomposition | between | high_recall | 2026 | between | dose | equal | -0.00807153 | 0.0132439 | 0.5452889483278 | 0.0071959 | 2686 | 2686 | 46 | 0.0087580 |
| P1_county_decomposition | entry | high_recall | 2026 | entry | dose | equal | 0.02582393 | 0.0275611 | 0.3537754771973 | 0.0152087 | 2686 | 2686 | 46 | 0.0092098 |
| P1_county_decomposition | exit | high_recall | 2026 | exit | dose | equal | 0.03093539 | 0.0326662 | 0.3486885142636 | 0.0162155 | 2686 | 2686 | 46 | -0.0281034 |
| P1_county_decomposition | no_continuer | high_recall | 2026 | no_continuer | dose | equal | 0.00597746 | 0.0097628 | 0.5434417279362 | 0.0060556 | 2686 | 2686 | 46 | -0.0250424 |
| P2_continuer_pass_through | employer_state_units | high_recall | 2026 | dw | dose | equal | 0.62906864 | 0.1564390 | 0.0002183808584 | 0.0841964 | 22932 | 22932 | 46 | -2.1503322 |
| P2_continuer_pass_through | employer_state_units | high_recall | 2026 | dw | dose | h0 | 0.64653648 | 0.0968500 | 0.0000000308920 | 0.0565382 | 22932 | 22932 | 46 | -2.3818065 |
| P3_selection_margins | exit_of_base_year_incumbents | high_recall | 2026 | exit_hourly | dose | equal | 0.01492848 | 0.0237384 | 0.5326109731361 | 0.0118825 | 34145 | 34145 | 46 | 0.3283936 |
| P3_selection_margins | exit_of_base_year_incumbents | high_recall | 2026 | exit_any | dose | equal | 0.01677079 | 0.0243938 | 0.4952945549699 | 0.0121226 | 34145 | 34145 | 46 | 0.3246449 |
| P3_selection_margins | exit_of_base_year_incumbents | high_recall | 2026 | exit_hourly | dose | h0 | 0.00979263 | 0.0097951 | 0.3227799711305 | 0.0039910 | 34145 | 34145 | 46 | 0.0915740 |
| P3_selection_margins | exit_of_base_year_incumbents | high_recall | 2026 | exit_any | dose | h0 | 0.01031655 | 0.0099312 | 0.3044487474472 | 0.0040737 | 34145 | 34145 | 46 | 0.0892946 |
| P3_selection_margins | entrant_share_of_tau_units | high_recall | 2026 | entrant | dose | equal | 0.03644381 | 0.0469134 | 0.4413226932820 | 0.0207448 | 37348 | 37348 | 46 | 0.3859912 |
| P3_selection_margins | entrant_share_of_tau_units | high_recall | 2026 | entrant | dose | h1 | 0.00135911 | 0.0130482 | 0.9175050250732 | 0.0057212 | 37348 | 37348 | 46 | 0.1146288 |

## All state-deletion ranges

[Every deletion and omitted state: wage_state_deletions.csv](/tmp/h2a-chapter-directions/wage_state_deletions.csv)

| specification | tau | tier | weights | term | deletions | unidentified | minimum | maximum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| filer_state_fixed_composition | 2024 | R5 | equal | dose | 46 | 0 | 0.0063663037 | 0.026120919 |
| filer_state_fixed_composition | 2024 | R5 | h0 | dose | 46 | 0 | 0.0137779277 | 0.045715433 |
| filer_state_fixed_composition | 2024 | R7 | equal | dose | 46 | 0 | 0.0046833623 | 0.024211725 |
| filer_state_fixed_composition | 2024 | R7 | h0 | dose | 46 | 0 | 0.0120717610 | 0.043978953 |
| filer_state_fixed_composition | 2024 | R8 | equal | dose | 46 | 0 | 0.0059175733 | 0.031179660 |
| filer_state_fixed_composition | 2024 | R8 | h0 | dose | 46 | 0 | 0.0113509574 | 0.043590164 |
| filer_state_fixed_composition | 2025 | FEIN | equal | dose | 46 | 0 | -0.0102841588 | 0.003995357 |
| filer_state_fixed_composition | 2025 | FEIN | h0 | dose | 46 | 0 | -0.0031725564 | 0.021677587 |
| filer_state_fixed_composition | 2025 | R5 | equal | dose | 46 | 0 | -0.0077584484 | 0.006575905 |
| filer_state_fixed_composition | 2025 | R5 | h0 | dose | 46 | 0 | -0.0034448337 | 0.022257611 |
| filer_state_fixed_composition | 2025 | R7 | equal | dose | 46 | 0 | -0.0084683301 | 0.005653682 |
| filer_state_fixed_composition | 2025 | R7 | h0 | dose | 46 | 0 | -0.0032798807 | 0.022366410 |
| filer_state_fixed_composition | 2025 | R8 | equal | dose | 46 | 0 | -0.0104915163 | 0.002536154 |
| filer_state_fixed_composition | 2025 | R8 | h0 | dose | 46 | 0 | -0.0037443082 | 0.022083299 |
| filer_state_fixed_composition | 2026 | FEIN | equal | dose | 46 | 0 | 0.5016121917 | 0.699813369 |
| filer_state_fixed_composition | 2026 | FEIN | h0 | dose | 46 | 0 | 0.5985863548 | 0.678912448 |
| filer_state_fixed_composition | 2026 | R5 | equal | dose | 46 | 0 | 0.4828215580 | 0.693686847 |
| filer_state_fixed_composition | 2026 | R5 | h0 | dose | 46 | 0 | 0.5968780459 | 0.678137982 |
| filer_state_fixed_composition | 2026 | R7 | equal | dose | 46 | 0 | 0.4937643291 | 0.697572282 |
| filer_state_fixed_composition | 2026 | R7 | h0 | dose | 46 | 0 | 0.5997647688 | 0.680218715 |
| filer_state_fixed_composition | 2026 | R8 | equal | dose | 46 | 0 | 0.4985080014 | 0.699500924 |
| filer_state_fixed_composition | 2026 | R8 | h0 | dose | 46 | 0 | 0.5983792790 | 0.679290700 |
| filer_state_unrestricted | 2024 | R5 | equal | dose | 46 | 0 | -0.0163895694 | 0.019365405 |
| filer_state_unrestricted | 2024 | R5 | h0 | dose | 46 | 0 | -0.0280863830 | 0.013148003 |
| filer_state_unrestricted | 2024 | R7 | equal | dose | 46 | 0 | -0.0186319317 | 0.016416839 |
| filer_state_unrestricted | 2024 | R7 | h0 | dose | 46 | 0 | -0.0293011043 | 0.011750451 |
| filer_state_unrestricted | 2024 | R8 | equal | dose | 46 | 0 | -0.0158211763 | 0.020491223 |
| filer_state_unrestricted | 2024 | R8 | h0 | dose | 46 | 0 | -0.0327033634 | 0.012290478 |
| filer_state_unrestricted | 2025 | FEIN | equal | dose | 46 | 0 | -0.0228383043 | -0.001305269 |
| filer_state_unrestricted | 2025 | FEIN | h0 | dose | 46 | 0 | 0.0022343944 | 0.039720204 |
| filer_state_unrestricted | 2025 | R5 | equal | dose | 46 | 0 | -0.0220660009 | -0.001102405 |
| filer_state_unrestricted | 2025 | R5 | h0 | dose | 46 | 0 | 0.0013205183 | 0.040002498 |
| filer_state_unrestricted | 2025 | R7 | equal | dose | 46 | 0 | -0.0232230092 | -0.002773810 |
| filer_state_unrestricted | 2025 | R7 | h0 | dose | 46 | 0 | 0.0009675000 | 0.039358436 |
| filer_state_unrestricted | 2025 | R8 | equal | dose | 46 | 0 | -0.0273501506 | -0.007543063 |
| filer_state_unrestricted | 2025 | R8 | h0 | dose | 46 | 0 | -0.0002507345 | 0.038431255 |
| filer_state_unrestricted | 2026 | FEIN | equal | dose | 46 | 0 | 0.5029080006 | 0.701891898 |
| filer_state_unrestricted | 2026 | FEIN | h0 | dose | 46 | 0 | 0.6234871107 | 0.689225288 |
| filer_state_unrestricted | 2026 | R5 | equal | dose | 46 | 0 | 0.4842206835 | 0.696575282 |
| filer_state_unrestricted | 2026 | R5 | h0 | dose | 46 | 0 | 0.6234337601 | 0.689700734 |
| filer_state_unrestricted | 2026 | R7 | equal | dose | 46 | 0 | 0.4943035886 | 0.701160734 |
| filer_state_unrestricted | 2026 | R7 | h0 | dose | 46 | 0 | 0.6251506961 | 0.691200797 |
| filer_state_unrestricted | 2026 | R8 | equal | dose | 46 | 0 | 0.5014598003 | 0.703343873 |
| filer_state_unrestricted | 2026 | R8 | h0 | dose | 46 | 0 | 0.6242679329 | 0.689442833 |
| filer_state_unrestricted_common_sample | 2024 | R5 | equal | dose | 46 | 0 | -0.0003731661 | 0.028134766 |
| filer_state_unrestricted_common_sample | 2024 | R5 | h0 | dose | 46 | 0 | -0.0030582151 | 0.030735405 |
| filer_state_unrestricted_common_sample | 2024 | R7 | equal | dose | 46 | 0 | -0.0028830066 | 0.024272468 |
| filer_state_unrestricted_common_sample | 2024 | R7 | h0 | dose | 46 | 0 | -0.0043635885 | 0.028796268 |
| filer_state_unrestricted_common_sample | 2024 | R8 | equal | dose | 46 | 0 | 0.0011462084 | 0.033121759 |
| filer_state_unrestricted_common_sample | 2024 | R8 | h0 | dose | 46 | 0 | -0.0043233809 | 0.028783212 |
| filer_state_unrestricted_common_sample | 2025 | FEIN | equal | dose | 46 | 0 | -0.0114085213 | 0.003883263 |
| filer_state_unrestricted_common_sample | 2025 | FEIN | h0 | dose | 46 | 0 | 0.0034743946 | 0.039953952 |
| filer_state_unrestricted_common_sample | 2025 | R5 | equal | dose | 46 | 0 | -0.0076715898 | 0.009617445 |
| filer_state_unrestricted_common_sample | 2025 | R5 | h0 | dose | 46 | 0 | 0.0045594421 | 0.042034964 |
| filer_state_unrestricted_common_sample | 2025 | R7 | equal | dose | 46 | 0 | -0.0091830165 | 0.007511384 |
| filer_state_unrestricted_common_sample | 2025 | R7 | h0 | dose | 46 | 0 | 0.0041939643 | 0.041413549 |
| filer_state_unrestricted_common_sample | 2025 | R8 | equal | dose | 46 | 0 | -0.0118589519 | 0.002249707 |
| filer_state_unrestricted_common_sample | 2025 | R8 | h0 | dose | 46 | 0 | 0.0036016848 | 0.040909825 |
| filer_state_unrestricted_common_sample | 2026 | FEIN | equal | dose | 46 | 0 | 0.5082709237 | 0.706885121 |
| filer_state_unrestricted_common_sample | 2026 | FEIN | h0 | dose | 46 | 0 | 0.6167455674 | 0.686228762 |
| filer_state_unrestricted_common_sample | 2026 | R5 | equal | dose | 46 | 0 | 0.4866022904 | 0.699459617 |
| filer_state_unrestricted_common_sample | 2026 | R5 | h0 | dose | 46 | 0 | 0.6141732325 | 0.684359980 |
| filer_state_unrestricted_common_sample | 2026 | R7 | equal | dose | 46 | 0 | 0.4992250433 | 0.704031986 |
| filer_state_unrestricted_common_sample | 2026 | R7 | h0 | dose | 46 | 0 | 0.6180717410 | 0.687117311 |
| filer_state_unrestricted_common_sample | 2026 | R8 | equal | dose | 46 | 0 | 0.5050392006 | 0.706533956 |
| filer_state_unrestricted_common_sample | 2026 | R8 | h0 | dose | 46 | 0 | 0.6161211167 | 0.686165697 |
| heterogeneity_contractor | 2026 | R7 | h0 | dose:contractor | 46 | 0 | 0.1698737819 | 0.341237467 |
| heterogeneity_highpremium | 2026 | R7 | h0 | dose:highpremium | 46 | 0 | -0.3653958128 | 0.255039171 |
| heterogeneity_large | 2026 | R7 | h0 | dose:large | 46 | 0 | -0.0649547591 | 0.056541413 |
| occupation_dcdf0 | 2024 | R5 | equal | dose | 46 | 0 | -0.0019672414 | 0.001623099 |
| occupation_dcdf0 | 2024 | R5 | h0 | dose | 46 | 0 | -0.0037234296 | -0.001066598 |
| occupation_dcdf0 | 2024 | R7 | equal | dose | 46 | 0 | -0.0020786526 | 0.001300587 |
| occupation_dcdf0 | 2024 | R7 | h0 | dose | 46 | 0 | -0.0037185682 | -0.001092950 |
| occupation_dcdf0 | 2024 | R8 | equal | dose | 46 | 0 | -0.0023602785 | 0.000608649 |
| occupation_dcdf0 | 2024 | R8 | h0 | dose | 46 | 0 | -0.0038425244 | -0.001099647 |
| occupation_dcdf0 | 2025 | FEIN | equal | dose | 46 | 0 | -0.0095152653 | -0.000499040 |
| occupation_dcdf0 | 2025 | FEIN | h0 | dose | 46 | 0 | -0.0071207490 | -0.000869010 |
| occupation_dcdf0 | 2025 | R5 | equal | dose | 46 | 0 | -0.0089264559 | -0.000118967 |
| occupation_dcdf0 | 2025 | R5 | h0 | dose | 46 | 0 | -0.0071983773 | -0.000545522 |
| occupation_dcdf0 | 2025 | R7 | equal | dose | 46 | 0 | -0.0089016575 | -0.000124787 |
| occupation_dcdf0 | 2025 | R7 | h0 | dose | 46 | 0 | -0.0072402468 | -0.000563884 |
| occupation_dcdf0 | 2025 | R8 | equal | dose | 46 | 0 | -0.0089723656 | -0.000139291 |
| occupation_dcdf0 | 2025 | R8 | h0 | dose | 46 | 0 | -0.0076489429 | -0.000783017 |
| occupation_dcdf0 | 2026 | FEIN | equal | dose | 46 | 0 | 0.0049560897 | 0.029252454 |
| occupation_dcdf0 | 2026 | FEIN | h0 | dose | 46 | 0 | 0.0092718542 | 0.030267177 |
| occupation_dcdf0 | 2026 | R5 | equal | dose | 46 | 0 | 0.0050995795 | 0.030237634 |
| occupation_dcdf0 | 2026 | R5 | h0 | dose | 46 | 0 | 0.0099801335 | 0.030806727 |
| occupation_dcdf0 | 2026 | R7 | equal | dose | 46 | 0 | 0.0048267662 | 0.028431809 |
| occupation_dcdf0 | 2026 | R7 | h0 | dose | 46 | 0 | 0.0090942719 | 0.030453571 |
| occupation_dcdf0 | 2026 | R8 | equal | dose | 46 | 0 | 0.0048559465 | 0.028112088 |
| occupation_dcdf0 | 2026 | R8 | h0 | dose | 46 | 0 | 0.0095602249 | 0.030825802 |
| occupation_dcdf100 | 2024 | R5 | equal | dose | 46 | 0 | -0.0079731158 | -0.003746337 |
| occupation_dcdf100 | 2024 | R5 | h0 | dose | 46 | 0 | -0.0040914800 | -0.001301384 |
| occupation_dcdf100 | 2024 | R7 | equal | dose | 46 | 0 | -0.0073616699 | -0.003461538 |
| occupation_dcdf100 | 2024 | R7 | h0 | dose | 46 | 0 | -0.0037837514 | -0.001051317 |
| occupation_dcdf100 | 2024 | R8 | equal | dose | 46 | 0 | -0.0083769805 | -0.003545517 |
| occupation_dcdf100 | 2024 | R8 | h0 | dose | 46 | 0 | -0.0040921586 | -0.001044904 |
| occupation_dcdf100 | 2025 | FEIN | equal | dose | 46 | 0 | -0.0000665841 | 0.001057083 |
| occupation_dcdf100 | 2025 | FEIN | h0 | dose | 46 | 0 | -0.0010516099 | 0.002076235 |
| occupation_dcdf100 | 2025 | R5 | equal | dose | 46 | 0 | 0.0001924172 | 0.001239254 |
| occupation_dcdf100 | 2025 | R5 | h0 | dose | 46 | 0 | -0.0010576620 | 0.002542153 |
| occupation_dcdf100 | 2025 | R7 | equal | dose | 46 | 0 | 0.0000484897 | 0.001144095 |
| occupation_dcdf100 | 2025 | R7 | h0 | dose | 46 | 0 | -0.0010417443 | 0.002518147 |
| occupation_dcdf100 | 2025 | R8 | equal | dose | 46 | 0 | -0.0001722725 | 0.001032973 |
| occupation_dcdf100 | 2025 | R8 | h0 | dose | 46 | 0 | -0.0010478697 | 0.002172320 |
| occupation_dcdf100 | 2026 | FEIN | equal | dose | 46 | 0 | 0.0091512226 | 0.042794278 |
| occupation_dcdf100 | 2026 | FEIN | h0 | dose | 46 | 0 | 0.0350688320 | 0.072806857 |
| occupation_dcdf100 | 2026 | R5 | equal | dose | 46 | 0 | 0.0103096039 | 0.045594247 |
| occupation_dcdf100 | 2026 | R5 | h0 | dose | 46 | 0 | 0.0350631806 | 0.072364107 |
| occupation_dcdf100 | 2026 | R7 | equal | dose | 46 | 0 | 0.0098419775 | 0.043376737 |
| occupation_dcdf100 | 2026 | R7 | h0 | dose | 46 | 0 | 0.0344496540 | 0.071050354 |
| occupation_dcdf100 | 2026 | R8 | equal | dose | 46 | 0 | 0.0098803752 | 0.042496623 |
| occupation_dcdf100 | 2026 | R8 | h0 | dose | 46 | 0 | 0.0350650219 | 0.071974709 |
| occupation_dcdf25 | 2024 | R5 | equal | dose | 46 | 0 | -0.0032122170 | -0.001445281 |
| occupation_dcdf25 | 2024 | R5 | h0 | dose | 46 | 0 | -0.0037669658 | -0.001504251 |
| occupation_dcdf25 | 2024 | R7 | equal | dose | 46 | 0 | -0.0030089039 | -0.001277841 |
| occupation_dcdf25 | 2024 | R7 | h0 | dose | 46 | 0 | -0.0034937927 | -0.001282655 |
| occupation_dcdf25 | 2024 | R8 | equal | dose | 46 | 0 | -0.0032682695 | -0.001203924 |
| occupation_dcdf25 | 2024 | R8 | h0 | dose | 46 | 0 | -0.0035964238 | -0.001125822 |
| occupation_dcdf25 | 2025 | FEIN | equal | dose | 46 | 0 | -0.0068057521 | 0.000246073 |
| occupation_dcdf25 | 2025 | FEIN | h0 | dose | 46 | 0 | -0.0056341577 | -0.000237744 |
| occupation_dcdf25 | 2025 | R5 | equal | dose | 46 | 0 | -0.0066279722 | 0.000444056 |
| occupation_dcdf25 | 2025 | R5 | h0 | dose | 46 | 0 | -0.0057355497 | 0.000069077 |
| occupation_dcdf25 | 2025 | R7 | equal | dose | 46 | 0 | -0.0068671430 | 0.000335474 |
| occupation_dcdf25 | 2025 | R7 | h0 | dose | 46 | 0 | -0.0058352831 | 0.000021517 |
| occupation_dcdf25 | 2025 | R8 | equal | dose | 46 | 0 | -0.0068500600 | 0.000366460 |
| occupation_dcdf25 | 2025 | R8 | h0 | dose | 46 | 0 | -0.0062533839 | -0.000197845 |
| occupation_dcdf25 | 2026 | FEIN | equal | dose | 46 | 0 | 0.0039531653 | 0.029758821 |
| occupation_dcdf25 | 2026 | FEIN | h0 | dose | 46 | 0 | 0.0137974170 | 0.030789269 |
| occupation_dcdf25 | 2026 | R5 | equal | dose | 46 | 0 | 0.0040069268 | 0.030340615 |
| occupation_dcdf25 | 2026 | R5 | h0 | dose | 46 | 0 | 0.0144911080 | 0.031278969 |
| occupation_dcdf25 | 2026 | R7 | equal | dose | 46 | 0 | 0.0040168700 | 0.028755011 |
| occupation_dcdf25 | 2026 | R7 | h0 | dose | 46 | 0 | 0.0136180605 | 0.030939395 |
| occupation_dcdf25 | 2026 | R8 | equal | dose | 46 | 0 | 0.0043185021 | 0.028608285 |
| occupation_dcdf25 | 2026 | R8 | h0 | dose | 46 | 0 | 0.0141055401 | 0.031355616 |
| occupation_dnear | 2024 | R5 | equal | dose | 46 | 0 | -0.0085596706 | 0.001178478 |
| occupation_dnear | 2024 | R5 | h0 | dose | 46 | 0 | 0.0017942816 | 0.010709715 |
| occupation_dnear | 2024 | R7 | equal | dose | 46 | 0 | -0.0095286763 | 0.000509757 |
| occupation_dnear | 2024 | R7 | h0 | dose | 46 | 0 | 0.0015384497 | 0.010219185 |
| occupation_dnear | 2024 | R8 | equal | dose | 46 | 0 | -0.0095586746 | 0.000870198 |
| occupation_dnear | 2024 | R8 | h0 | dose | 46 | 0 | 0.0015720664 | 0.010258174 |
| occupation_dnear | 2025 | FEIN | equal | dose | 46 | 0 | -0.0184167629 | 0.020045220 |
| occupation_dnear | 2025 | FEIN | h0 | dose | 46 | 0 | -0.0171690222 | 0.000340287 |
| occupation_dnear | 2025 | R5 | equal | dose | 46 | 0 | -0.0165143549 | 0.021962177 |
| occupation_dnear | 2025 | R5 | h0 | dose | 46 | 0 | -0.0155004206 | 0.001614903 |
| occupation_dnear | 2025 | R7 | equal | dose | 46 | 0 | -0.0172513174 | 0.021949777 |
| occupation_dnear | 2025 | R7 | h0 | dose | 46 | 0 | -0.0158342497 | 0.001881880 |
| occupation_dnear | 2025 | R8 | equal | dose | 46 | 0 | -0.0173747146 | 0.021453444 |
| occupation_dnear | 2025 | R8 | h0 | dose | 46 | 0 | -0.0167947705 | 0.000936208 |
| occupation_dnear | 2026 | FEIN | equal | dose | 46 | 0 | -0.0168683880 | 0.013952207 |
| occupation_dnear | 2026 | FEIN | h0 | dose | 46 | 0 | 0.0075802001 | 0.037130571 |
| occupation_dnear | 2026 | R5 | equal | dose | 46 | 0 | -0.0174914124 | 0.013859577 |
| occupation_dnear | 2026 | R5 | h0 | dose | 46 | 0 | 0.0064196421 | 0.037198460 |
| occupation_dnear | 2026 | R7 | equal | dose | 46 | 0 | -0.0176023379 | 0.013208211 |
| occupation_dnear | 2026 | R7 | h0 | dose | 46 | 0 | 0.0059046671 | 0.036812065 |
| occupation_dnear | 2026 | R8 | equal | dose | 46 | 0 | -0.0173514855 | 0.013288108 |
| occupation_dnear | 2026 | R8 | h0 | dose | 46 | 0 | 0.0080121728 | 0.037756873 |
| occupation_dp | 2024 | R5 | equal | dose | 46 | 0 | 0.0045699476 | 0.028085857 |
| occupation_dp | 2024 | R5 | h0 | dose | 46 | 0 | 0.0109949291 | 0.039904889 |
| occupation_dp | 2024 | R7 | equal | dose | 46 | 0 | 0.0030205814 | 0.026426712 |
| occupation_dp | 2024 | R7 | h0 | dose | 46 | 0 | 0.0095348187 | 0.038194161 |
| occupation_dp | 2024 | R8 | equal | dose | 46 | 0 | 0.0054895057 | 0.032701937 |
| occupation_dp | 2024 | R8 | h0 | dose | 46 | 0 | 0.0096704290 | 0.039446671 |
| occupation_dp | 2025 | FEIN | equal | dose | 46 | 0 | -0.0115169784 | 0.002481795 |
| occupation_dp | 2025 | FEIN | h0 | dose | 46 | 0 | -0.0059465535 | 0.021429885 |
| occupation_dp | 2025 | R5 | equal | dose | 46 | 0 | -0.0093550097 | 0.004043388 |
| occupation_dp | 2025 | R5 | h0 | dose | 46 | 0 | -0.0057285659 | 0.022396378 |
| occupation_dp | 2025 | R7 | equal | dose | 46 | 0 | -0.0099819996 | 0.003529202 |
| occupation_dp | 2025 | R7 | h0 | dose | 46 | 0 | -0.0057380667 | 0.022307939 |
| occupation_dp | 2025 | R8 | equal | dose | 46 | 0 | -0.0114315144 | 0.002156541 |
| occupation_dp | 2025 | R8 | h0 | dose | 46 | 0 | -0.0062367089 | 0.021995740 |
| occupation_dp | 2026 | FEIN | equal | dose | 46 | 0 | -0.5391261673 | -0.326988983 |
| occupation_dp | 2026 | FEIN | h0 | dose | 46 | 0 | -0.3956126015 | -0.319365148 |
| occupation_dp | 2026 | R5 | equal | dose | 46 | 0 | -0.5552411863 | -0.330508209 |
| occupation_dp | 2026 | R5 | h0 | dose | 46 | 0 | -0.3963192594 | -0.319573412 |
| occupation_dp | 2026 | R7 | equal | dose | 46 | 0 | -0.5433999902 | -0.327446269 |
| occupation_dp | 2026 | R7 | h0 | dose | 46 | 0 | -0.3932345863 | -0.317277473 |
| occupation_dp | 2026 | R8 | equal | dose | 46 | 0 | -0.5410087330 | -0.328892119 |
| occupation_dp | 2026 | R8 | h0 | dose | 46 | 0 | -0.3949671648 | -0.318862939 |
| occupation_dw | 2024 | R5 | equal | dose | 46 | 0 | 0.0045699476 | 0.028085857 |
| occupation_dw | 2024 | R5 | h0 | dose | 46 | 0 | 0.0109949291 | 0.039904889 |
| occupation_dw | 2024 | R7 | equal | dose | 46 | 0 | 0.0030205814 | 0.026426712 |
| occupation_dw | 2024 | R7 | h0 | dose | 46 | 0 | 0.0095348187 | 0.038194161 |
| occupation_dw | 2024 | R8 | equal | dose | 46 | 0 | 0.0054895057 | 0.032701937 |
| occupation_dw | 2024 | R8 | h0 | dose | 46 | 0 | 0.0096704290 | 0.039446671 |
| occupation_dw | 2025 | FEIN | equal | dose | 46 | 0 | -0.0115169784 | 0.002481795 |
| occupation_dw | 2025 | FEIN | h0 | dose | 46 | 0 | -0.0059465535 | 0.021429885 |
| occupation_dw | 2025 | R5 | equal | dose | 46 | 0 | -0.0093550097 | 0.004043388 |
| occupation_dw | 2025 | R5 | h0 | dose | 46 | 0 | -0.0057285659 | 0.022396378 |
| occupation_dw | 2025 | R7 | equal | dose | 46 | 0 | -0.0099819996 | 0.003529202 |
| occupation_dw | 2025 | R7 | h0 | dose | 46 | 0 | -0.0057380667 | 0.022307939 |
| occupation_dw | 2025 | R8 | equal | dose | 46 | 0 | -0.0114315144 | 0.002156541 |
| occupation_dw | 2025 | R8 | h0 | dose | 46 | 0 | -0.0062367089 | 0.021995740 |
| occupation_dw | 2026 | FEIN | equal | dose | 46 | 0 | 0.4608738327 | 0.673011017 |
| occupation_dw | 2026 | FEIN | h0 | dose | 46 | 0 | 0.6043873985 | 0.680634852 |
| occupation_dw | 2026 | R5 | equal | dose | 46 | 0 | 0.4447588137 | 0.669491791 |
| occupation_dw | 2026 | R5 | h0 | dose | 46 | 0 | 0.6036807406 | 0.680426588 |
| occupation_dw | 2026 | R7 | equal | dose | 46 | 0 | 0.4566000098 | 0.672553731 |
| occupation_dw | 2026 | R7 | h0 | dose | 46 | 0 | 0.6067654137 | 0.682722527 |
| occupation_dw | 2026 | R8 | equal | dose | 46 | 0 | 0.4589912670 | 0.671107881 |
| occupation_dw | 2026 | R8 | h0 | dose | 46 | 0 | 0.6050328352 | 0.681137061 |
| occupation_reduce | 2024 | R5 | equal | dose | 46 | 0 | 0.0001278700 | 0.000622769 |
| occupation_reduce | 2024 | R5 | h0 | dose | 46 | 0 | 0.0000051074 | 0.001478719 |
| occupation_reduce | 2024 | R7 | equal | dose | 46 | 0 | 0.0001560668 | 0.001014388 |
| occupation_reduce | 2024 | R7 | h0 | dose | 46 | 0 | 0.0000063470 | 0.001639664 |
| occupation_reduce | 2024 | R8 | equal | dose | 46 | 0 | 0.0002828234 | 0.001293180 |
| occupation_reduce | 2024 | R8 | h0 | dose | 46 | 0 | 0.0001295515 | 0.001731994 |
| occupation_reduce | 2025 | FEIN | equal | dose | 46 | 0 | -0.0024908145 | 0.002785776 |
| occupation_reduce | 2025 | FEIN | h0 | dose | 46 | 0 | -0.0010152346 | 0.001465319 |
| occupation_reduce | 2025 | R5 | equal | dose | 46 | 0 | -0.0029219748 | 0.002846022 |
| occupation_reduce | 2025 | R5 | h0 | dose | 46 | 0 | -0.0028072676 | 0.001055707 |
| occupation_reduce | 2025 | R7 | equal | dose | 46 | 0 | -0.0030429495 | 0.002716752 |
| occupation_reduce | 2025 | R7 | h0 | dose | 46 | 0 | -0.0028688609 | 0.001000275 |
| occupation_reduce | 2025 | R8 | equal | dose | 46 | 0 | -0.0022672868 | 0.003019305 |
| occupation_reduce | 2025 | R8 | h0 | dose | 46 | 0 | -0.0026607338 | 0.001081623 |
| occupation_reduce | 2026 | FEIN | equal | dose | 46 | 0 | -0.0434861399 | -0.011021276 |
| occupation_reduce | 2026 | FEIN | h0 | dose | 46 | 0 | -0.0258699554 | 0.000934042 |
| occupation_reduce | 2026 | R5 | equal | dose | 46 | 0 | -0.0409295858 | -0.010857839 |
| occupation_reduce | 2026 | R5 | h0 | dose | 46 | 0 | -0.0262942666 | 0.000526345 |
| occupation_reduce | 2026 | R7 | equal | dose | 46 | 0 | -0.0430531766 | -0.011230661 |
| occupation_reduce | 2026 | R7 | h0 | dose | 46 | 0 | -0.0269484628 | 0.000145924 |
| occupation_reduce | 2026 | R8 | equal | dose | 46 | 0 | -0.0434680274 | -0.011068771 |
| occupation_reduce | 2026 | R8 | h0 | dose | 46 | 0 | -0.0267621650 | 0.000415426 |
| occupation_retain | 2024 | R5 | equal | dose | 46 | 0 | 0.0002726176 | 0.007687484 |
| occupation_retain | 2024 | R5 | h0 | dose | 46 | 0 | -0.0039176675 | 0.001758045 |
| occupation_retain | 2024 | R7 | equal | dose | 46 | 0 | 0.0005618074 | 0.008278783 |
| occupation_retain | 2024 | R7 | h0 | dose | 46 | 0 | -0.0035291484 | 0.002050088 |
| occupation_retain | 2024 | R8 | equal | dose | 46 | 0 | 0.0000484616 | 0.008021166 |
| occupation_retain | 2024 | R8 | h0 | dose | 46 | 0 | -0.0034648980 | 0.001930891 |
| occupation_retain | 2025 | FEIN | equal | dose | 46 | 0 | -0.0064399676 | 0.003642192 |
| occupation_retain | 2025 | FEIN | h0 | dose | 46 | 0 | -0.0046123402 | 0.002795711 |
| occupation_retain | 2025 | R5 | equal | dose | 46 | 0 | -0.0053818223 | 0.003701977 |
| occupation_retain | 2025 | R5 | h0 | dose | 46 | 0 | -0.0035538385 | 0.003700614 |
| occupation_retain | 2025 | R7 | equal | dose | 46 | 0 | -0.0054361813 | 0.003373525 |
| occupation_retain | 2025 | R7 | h0 | dose | 46 | 0 | -0.0036930997 | 0.003538309 |
| occupation_retain | 2025 | R8 | equal | dose | 46 | 0 | -0.0056499445 | 0.003199839 |
| occupation_retain | 2025 | R8 | h0 | dose | 46 | 0 | -0.0037313526 | 0.003435993 |
| occupation_retain | 2026 | FEIN | equal | dose | 46 | 0 | -0.0157916170 | -0.002134451 |
| occupation_retain | 2026 | FEIN | h0 | dose | 46 | 0 | -0.0162659049 | -0.003047155 |
| occupation_retain | 2026 | R5 | equal | dose | 46 | 0 | -0.0156600246 | -0.001526237 |
| occupation_retain | 2026 | R5 | h0 | dose | 46 | 0 | -0.0159849573 | -0.002327650 |
| occupation_retain | 2026 | R7 | equal | dose | 46 | 0 | -0.0151772965 | -0.001154615 |
| occupation_retain | 2026 | R7 | h0 | dose | 46 | 0 | -0.0157400468 | -0.002100222 |
| occupation_retain | 2026 | R8 | equal | dose | 46 | 0 | -0.0150830867 | -0.001177757 |
| occupation_retain | 2026 | R8 | h0 | dose | 46 | 0 | -0.0155678324 | -0.001988989 |
| receipt_aligned_single_state_common_SOC | 2026 | R7 | equal | dose | 46 | 0 | 0.6076162101 | 0.771963281 |
| receipt_aligned_single_state_common_SOC | 2026 | R7 | h0 | dose | 46 | 0 | 0.6590272609 | 0.739072789 |
| schedule_envelope_dnear_candidate | 2024 | R5 | equal | dose | 44 | 0 | -0.0076519897 | 0.002575966 |
| schedule_envelope_dnear_candidate | 2024 | R5 | h0 | dose | 44 | 0 | 0.0030240146 | 0.012733298 |
| schedule_envelope_dnear_candidate | 2024 | R7 | equal | dose | 44 | 0 | -0.0089404384 | 0.001595001 |
| schedule_envelope_dnear_candidate | 2024 | R7 | h0 | dose | 44 | 0 | 0.0026904262 | 0.012098435 |
| schedule_envelope_dnear_candidate | 2024 | R8 | equal | dose | 44 | 0 | -0.0084839093 | 0.002672057 |
| schedule_envelope_dnear_candidate | 2024 | R8 | h0 | dose | 44 | 0 | 0.0028394093 | 0.012288302 |
| schedule_envelope_dnear_candidate | 2025 | FEIN | equal | dose | 44 | 0 | -0.0227306299 | 0.022426968 |
| schedule_envelope_dnear_candidate | 2025 | FEIN | h0 | dose | 44 | 0 | -0.0184161184 | 0.005992696 |
| schedule_envelope_dnear_candidate | 2025 | R5 | equal | dose | 44 | 0 | -0.0210953980 | 0.023959947 |
| schedule_envelope_dnear_candidate | 2025 | R5 | h0 | dose | 44 | 0 | -0.0173167613 | 0.006853171 |
| schedule_envelope_dnear_candidate | 2025 | R7 | equal | dose | 44 | 0 | -0.0218471783 | 0.024088559 |
| schedule_envelope_dnear_candidate | 2025 | R7 | h0 | dose | 44 | 0 | -0.0178036803 | 0.007045149 |
| schedule_envelope_dnear_candidate | 2025 | R8 | equal | dose | 44 | 0 | -0.0219660713 | 0.023739390 |
| schedule_envelope_dnear_candidate | 2025 | R8 | h0 | dose | 44 | 0 | -0.0180653959 | 0.006665269 |
| schedule_envelope_dnear_candidate | 2026 | FEIN | equal | dose | 39 | 0 | -0.0242651317 | 0.034691132 |
| schedule_envelope_dnear_candidate | 2026 | FEIN | h0 | dose | 39 | 0 | -0.0028267427 | 0.050715897 |
| schedule_envelope_dnear_candidate | 2026 | R5 | equal | dose | 39 | 0 | -0.0269127015 | 0.033798316 |
| schedule_envelope_dnear_candidate | 2026 | R5 | h0 | dose | 39 | 0 | 0.0051285265 | 0.059862429 |
| schedule_envelope_dnear_candidate | 2026 | R7 | equal | dose | 39 | 0 | -0.0269398388 | 0.033623260 |
| schedule_envelope_dnear_candidate | 2026 | R7 | h0 | dose | 39 | 0 | 0.0048439953 | 0.059419795 |
| schedule_envelope_dnear_candidate | 2026 | R8 | equal | dose | 39 | 0 | -0.0269104751 | 0.034317143 |
| schedule_envelope_dnear_candidate | 2026 | R8 | h0 | dose | 39 | 0 | 0.0057652133 | 0.059615330 |
| schedule_envelope_dpremium_lower | 2024 | R5 | equal | dose | 44 | 0 | -0.0152772389 | 0.004138785 |
| schedule_envelope_dpremium_lower | 2024 | R5 | h0 | dose | 44 | 0 | -0.0016756928 | 0.022271357 |
| schedule_envelope_dpremium_lower | 2024 | R7 | equal | dose | 44 | 0 | -0.0163782641 | 0.003553471 |
| schedule_envelope_dpremium_lower | 2024 | R7 | h0 | dose | 44 | 0 | -0.0025286479 | 0.021519174 |
| schedule_envelope_dpremium_lower | 2024 | R8 | equal | dose | 44 | 0 | -0.0154043758 | 0.008377839 |
| schedule_envelope_dpremium_lower | 2024 | R8 | h0 | dose | 44 | 0 | -0.0019633525 | 0.022660353 |
| schedule_envelope_dpremium_lower | 2025 | FEIN | equal | dose | 44 | 0 | -0.0081656029 | 0.008921318 |
| schedule_envelope_dpremium_lower | 2025 | FEIN | h0 | dose | 44 | 0 | -0.0023355085 | 0.006024597 |
| schedule_envelope_dpremium_lower | 2025 | R5 | equal | dose | 44 | 0 | -0.0056030115 | 0.012826417 |
| schedule_envelope_dpremium_lower | 2025 | R5 | h0 | dose | 44 | 0 | -0.0011624660 | 0.007420996 |
| schedule_envelope_dpremium_lower | 2025 | R7 | equal | dose | 44 | 0 | -0.0062016745 | 0.011749887 |
| schedule_envelope_dpremium_lower | 2025 | R7 | h0 | dose | 44 | 0 | -0.0013888036 | 0.007121887 |
| schedule_envelope_dpremium_lower | 2025 | R8 | equal | dose | 44 | 0 | -0.0074159761 | 0.007272476 |
| schedule_envelope_dpremium_lower | 2025 | R8 | h0 | dose | 44 | 0 | -0.0023943563 | 0.005386357 |
| schedule_envelope_dpremium_lower | 2026 | FEIN | equal | dose | 39 | 0 | -0.5669787333 | -0.182066757 |
| schedule_envelope_dpremium_lower | 2026 | FEIN | h0 | dose | 39 | 0 | -0.4358892713 | -0.186708421 |
| schedule_envelope_dpremium_lower | 2026 | R5 | equal | dose | 39 | 0 | -0.5783286502 | -0.204779502 |
| schedule_envelope_dpremium_lower | 2026 | R5 | h0 | dose | 39 | 0 | -0.4452925110 | -0.191672227 |
| schedule_envelope_dpremium_lower | 2026 | R7 | equal | dose | 39 | 0 | -0.5715569337 | -0.195624042 |
| schedule_envelope_dpremium_lower | 2026 | R7 | h0 | dose | 39 | 0 | -0.4420179980 | -0.187612128 |
| schedule_envelope_dpremium_lower | 2026 | R8 | equal | dose | 39 | 0 | -0.5665819468 | -0.188568346 |
| schedule_envelope_dpremium_lower | 2026 | R8 | h0 | dose | 39 | 0 | -0.4408972182 | -0.189625213 |
| selection_any_cert | 2024 | R7 | equal | dose | 46 | 0 | -0.0017286505 | 0.002826459 |
| selection_any_cert | 2024 | R7 | h0 | dose | 46 | 0 | -0.0024980662 | 0.001499547 |
| selection_any_cert | 2025 | R7 | equal | dose | 46 | 0 | -0.0106165536 | -0.005142588 |
| selection_any_cert | 2025 | R7 | h0 | dose | 46 | 0 | -0.0109520528 | -0.003736720 |
| selection_any_cert | 2026 | R7 | equal | dose | 46 | 0 | -0.0066978202 | -0.000832462 |
| selection_any_cert | 2026 | R7 | h0 | dose | 46 | 0 | -0.0072877261 | -0.002303155 |
| selection_baseline_premium | 2024 | R7 | equal | dose:stay | 46 | 0 | -0.0613478475 | -0.033897364 |
| selection_baseline_premium | 2024 | R7 | equal | stay | 46 | 0 | -0.4094959268 | -0.294449920 |
| selection_baseline_premium | 2025 | R7 | equal | dose:stay | 46 | 0 | 0.0198288837 | 0.051863484 |
| selection_baseline_premium | 2025 | R7 | equal | stay | 46 | 0 | -0.1194430672 | 0.012205550 |
| selection_baseline_premium | 2026 | R7 | equal | dose:stay | 46 | 0 | -0.0648563386 | 0.007973457 |
| selection_baseline_premium | 2026 | R7 | equal | stay | 46 | 0 | -0.6068809257 | -0.300969691 |
| selection_prechange | 2025 | R7 | equal | dose:stay | 46 | 0 | 0.0345873192 | 0.070445794 |
| selection_prechange | 2025 | R7 | equal | stay | 46 | 0 | 0.0645667409 | 0.201405369 |
| selection_prechange | 2026 | R7 | equal | dose:stay | 46 | 0 | 0.0599137206 | 0.118114215 |
| selection_prechange | 2026 | R7 | equal | stay | 46 | 0 | 0.2960435204 | 0.586506625 |
| selection_stay | 2024 | R7 | equal | dose | 46 | 0 | -0.0038138040 | 0.001994592 |
| selection_stay | 2024 | R7 | h0 | dose | 46 | 0 | -0.0038002077 | 0.000705845 |
| selection_stay | 2025 | R7 | equal | dose | 46 | 0 | -0.0103515325 | -0.005177331 |
| selection_stay | 2025 | R7 | h0 | dose | 46 | 0 | -0.0107984794 | -0.003524628 |
| selection_stay | 2026 | R7 | equal | dose | 46 | 0 | -0.0054897316 | -0.001154699 |
| selection_stay | 2026 | R7 | h0 | dose | 46 | 0 | -0.0067359144 | -0.001897771 |
| within_filer_across_state | 2024 | R7 | equal | dose | 46 | 0 | -0.0201984104 | 0.025710770 |
| within_filer_across_state | 2024 | R7 | h0 | dose | 46 | 0 | -0.1047922173 | 0.056472972 |
| within_filer_across_state | 2025 | R7 | equal | dose | 46 | 0 | 0.0101466894 | 0.038818137 |
| within_filer_across_state | 2025 | R7 | h0 | dose | 46 | 0 | -0.0165110841 | 0.062718673 |
| within_filer_across_state | 2026 | R7 | equal | dose | 46 | 0 | 0.1701865229 | 0.203326528 |
| within_filer_across_state | 2026 | R7 | h0 | dose | 46 | 0 | 0.3521237506 | 0.453695828 |

[Every deletion and omitted state: admin_state_deletions.csv](/tmp/h2a-chapter-directions/admin_state_deletions.csv)

| specification | tau | tier | weights | term | deletions | unidentified | minimum | maximum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aug_dec_certified_positions | 2013 | R7 | equal | year::2011:exposure | 33 | 0 | -2.5767873509650867092091 | 0.875938591338070615677 |
| aug_dec_certified_positions | 2013 | R7 | equal | year::2013:exposure | 33 | 0 | -0.0470899490460183686724 | 6.740004682253594836538 |
| aug_dec_certified_positions | 2013 | R7 | equal | year::2014:exposure | 33 | 33 | — | — |
| aug_dec_certified_positions | 2013 | R7 | equal | year::2015:exposure | 33 | 33 | — | — |
| aug_dec_observed_filing | 2013 | R7 | equal | year::2011:exposure | 33 | 0 | -0.0000000000000000277556 | 0.000000000000000116226 |
| aug_dec_observed_filing | 2013 | R7 | equal | year::2013:exposure | 33 | 0 | -0.0503082339326738278706 | 0.128280971341111410533 |
| aug_dec_observed_filing | 2013 | R7 | equal | year::2014:exposure | 33 | 33 | — | — |
| aug_dec_observed_filing | 2013 | R7 | equal | year::2015:exposure | 33 | 33 | — | — |
| aug_dec_timely | 2013 | R7 | equal | year::2011:exposure | 33 | 0 | -0.0407843644370803129484 | 0.012047734435235929809 |
| aug_dec_timely | 2013 | R7 | equal | year::2013:exposure | 33 | 0 | -0.0852645878259247619324 | 0.041063578513801018977 |
| aug_dec_timely | 2013 | R7 | equal | year::2014:exposure | 33 | 33 | — | — |
| aug_dec_timely | 2013 | R7 | equal | year::2015:exposure | 33 | 33 | — | — |
| aug_dec_timing | 2013 | R7 | equal | year::2011:exposure | 33 | 0 | 0.7178360148870690515821 | 2.759521413565404035495 |
| aug_dec_timing | 2013 | R7 | equal | year::2013:exposure | 33 | 0 | 1.7110580502943208980327 | 5.193796601301710502696 |
| aug_dec_timing | 2013 | R7 | equal | year::2014:exposure | 33 | 33 | — | — |
| aug_dec_timing | 2013 | R7 | equal | year::2015:exposure | 33 | 33 | — | — |
| calendar_placebo_2012_certified_positions | 2013 | R7 | equal | year::2012:exposure | 46 | 0 | -0.6957763438992333737687 | 2.687133844614653543914 |
| calendar_placebo_2012_timely | 2013 | R7 | equal | year::2012:exposure | 46 | 0 | -0.0120477344352358500118 | 0.040784364437080507237 |
| calendar_placebo_2012_timing | 2013 | R7 | equal | year::2012:exposure | 46 | 0 | -2.7595214135654040354950 | -0.717836014887069828738 |
| catchup14 | 2013 | R7 | equal | year::2011:exposure | 46 | 0 | -0.0619107045203067599126 | -0.002159135254580200197 |
| catchup14 | 2013 | R7 | equal | year::2013:exposure | 46 | 0 | -0.0923650972104914785854 | 0.041043926104421250833 |
| catchup14 | 2013 | R7 | equal | year::2014:exposure | 46 | 46 | — | — |
| catchup14 | 2013 | R7 | equal | year::2015:exposure | 46 | 46 | — | — |
| catchup30 | 2013 | R7 | equal | year::2011:exposure | 46 | 0 | -0.0619107045203068431793 | -0.002159135254580230121 |
| catchup30 | 2013 | R7 | equal | year::2013:exposure | 46 | 0 | -0.1005579926542222030106 | 0.017755820989422708978 |
| catchup30 | 2013 | R7 | equal | year::2014:exposure | 46 | 46 | — | — |
| catchup30 | 2013 | R7 | equal | year::2015:exposure | 46 | 46 | — | — |
| catchup60 | 2013 | R7 | equal | year::2011:exposure | 46 | 0 | -0.0619107045203068431793 | -0.002159135254580230121 |
| catchup60 | 2013 | R7 | equal | year::2013:exposure | 46 | 0 | -0.1005579926542222030106 | 0.017755820989422708978 |
| catchup60 | 2013 | R7 | equal | year::2014:exposure | 46 | 46 | — | — |
| catchup60 | 2013 | R7 | equal | year::2015:exposure | 46 | 46 | — | — |
| catchup7 | 2013 | R7 | equal | year::2011:exposure | 46 | 0 | -0.0713438610225846370216 | -0.008502667767242269928 |
| catchup7 | 2013 | R7 | equal | year::2013:exposure | 46 | 0 | -0.0923650972104914785854 | 0.041043926104421209200 |
| catchup7 | 2013 | R7 | equal | year::2014:exposure | 46 | 46 | — | — |
| catchup7 | 2013 | R7 | equal | year::2015:exposure | 46 | 46 | — | — |
| exclude_largest_certified_positions | 2013 | R7 | equal | year::2011:exposure | 46 | 0 | -2.5767873509650840446739 | 0.875938591338072614079 |
| exclude_largest_certified_positions | 2013 | R7 | equal | year::2013:exposure | 46 | 0 | -0.0470899490460190417451 | 6.740004682253596612895 |
| exclude_largest_certified_positions | 2013 | R7 | equal | year::2014:exposure | 46 | 46 | — | — |
| exclude_largest_certified_positions | 2013 | R7 | equal | year::2015:exposure | 46 | 46 | — | — |
| exclude_largest_observed_filing | 2013 | R7 | equal | year::2011:exposure | 46 | 0 | -0.0000000000000000277556 | 0.000000000000000116226 |
| exclude_largest_observed_filing | 2013 | R7 | equal | year::2013:exposure | 46 | 0 | -0.0503082339326738278706 | 0.128280971341111410533 |
| exclude_largest_observed_filing | 2013 | R7 | equal | year::2014:exposure | 46 | 46 | — | — |
| exclude_largest_observed_filing | 2013 | R7 | equal | year::2015:exposure | 46 | 46 | — | — |
| exclude_largest_timely | 2013 | R7 | equal | year::2011:exposure | 46 | 0 | -0.0407843644370803129484 | 0.012047734435235929809 |
| exclude_largest_timely | 2013 | R7 | equal | year::2013:exposure | 46 | 0 | -0.0852645878259247619324 | 0.041063578513801018977 |
| exclude_largest_timely | 2013 | R7 | equal | year::2014:exposure | 46 | 46 | — | — |
| exclude_largest_timely | 2013 | R7 | equal | year::2015:exposure | 46 | 46 | — | — |
| exclude_largest_timing | 2013 | R7 | equal | year::2011:exposure | 46 | 0 | 0.7178360148870697177159 | 2.759521413565404923673 |
| exclude_largest_timing | 2013 | R7 | equal | year::2013:exposure | 46 | 0 | 1.7110580502943224523449 | 5.193796601301711390875 |
| exclude_largest_timing | 2013 | R7 | equal | year::2014:exposure | 46 | 46 | — | — |
| exclude_largest_timing | 2013 | R7 | equal | year::2015:exposure | 46 | 46 | — | — |
| month_comparison_certified_positions | 2013 | R7 | equal | year::2011:exposure | 44 | 0 | 0.7456550458236725908989 | 3.181866824556788486689 |
| month_comparison_certified_positions | 2013 | R7 | equal | year::2013:exposure | 44 | 0 | 0.2771207702693841423702 | 6.399270159850850347993 |
| month_comparison_certified_positions | 2013 | R7 | equal | year::2014:exposure | 44 | 44 | — | — |
| month_comparison_certified_positions | 2013 | R7 | equal | year::2015:exposure | 44 | 44 | — | — |
| month_comparison_observed_filing | 2013 | R7 | equal | year::2011:exposure | 45 | 0 | -0.0000000000000000034694 | 0.000000000000000166533 |
| month_comparison_observed_filing | 2013 | R7 | equal | year::2013:exposure | 45 | 0 | -0.1425227853985882586763 | -0.008837412240688589937 |
| month_comparison_observed_filing | 2013 | R7 | equal | year::2014:exposure | 45 | 45 | — | — |
| month_comparison_observed_filing | 2013 | R7 | equal | year::2015:exposure | 45 | 45 | — | — |
| month_comparison_timely | 2013 | R7 | equal | year::2011:exposure | 45 | 0 | -0.1538217444766170338699 | -0.032308735180301119405 |
| month_comparison_timely | 2013 | R7 | equal | year::2013:exposure | 45 | 0 | -0.2012483571603685128526 | -0.140585722674014396150 |
| month_comparison_timely | 2013 | R7 | equal | year::2014:exposure | 45 | 45 | — | — |
| month_comparison_timely | 2013 | R7 | equal | year::2015:exposure | 45 | 45 | — | — |
| month_comparison_timing | 2013 | R7 | equal | year::2011:exposure | 45 | 0 | 3.7254857685809321665715 | 6.377124724962628832259 |
| month_comparison_timing | 2013 | R7 | equal | year::2013:exposure | 45 | 0 | 3.0970936575858871719902 | 6.736165631338651316184 |
| month_comparison_timing | 2013 | R7 | equal | year::2014:exposure | 45 | 45 | — | — |
| month_comparison_timing | 2013 | R7 | equal | year::2015:exposure | 45 | 45 | — | — |
| primary_frame_certified_positions | 2013 | R5 | baseline_positions | year::2011:exposure | 46 | 0 | -5.6762765060951272388934 | 1.297383767291693246193 |
| primary_frame_certified_positions | 2013 | R5 | baseline_positions | year::2013:exposure | 46 | 0 | -0.7248885069673431935655 | 24.288673475753839170466 |
| primary_frame_certified_positions | 2013 | R5 | baseline_positions | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_certified_positions | 2013 | R5 | baseline_positions | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_certified_positions | 2013 | R5 | equal | year::2011:exposure | 46 | 0 | -4.4205316934660583427785 | -0.086753563168105785053 |
| primary_frame_certified_positions | 2013 | R5 | equal | year::2013:exposure | 46 | 0 | 0.1506543447023873660839 | 6.541338891213135831038 |
| primary_frame_certified_positions | 2013 | R5 | equal | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_certified_positions | 2013 | R5 | equal | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_certified_positions | 2013 | R7 | baseline_positions | year::2011:exposure | 46 | 0 | -1.4723290239966253523107 | 3.008478728364346821422 |
| primary_frame_certified_positions | 2013 | R7 | baseline_positions | year::2013:exposure | 46 | 0 | 0.4014430585353618852196 | 25.417460366320970166498 |
| primary_frame_certified_positions | 2013 | R7 | baseline_positions | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_certified_positions | 2013 | R7 | baseline_positions | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_certified_positions | 2013 | R7 | equal | year::2011:exposure | 46 | 0 | -2.5767873509650813801386 | 0.875938591338072836123 |
| primary_frame_certified_positions | 2013 | R7 | equal | year::2013:exposure | 46 | 0 | -0.0470899490460152669868 | 6.740004682253601941966 |
| primary_frame_certified_positions | 2013 | R7 | equal | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_certified_positions | 2013 | R7 | equal | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_certified_positions | 2013 | R8 | baseline_positions | year::2011:exposure | 46 | 0 | -7.1279150940601176955624 | -2.238609938805242727256 |
| primary_frame_certified_positions | 2013 | R8 | baseline_positions | year::2013:exposure | 46 | 0 | 0.8455053913067742676901 | 18.556662191912963066898 |
| primary_frame_certified_positions | 2013 | R8 | baseline_positions | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_certified_positions | 2013 | R8 | baseline_positions | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_certified_positions | 2013 | R8 | equal | year::2011:exposure | 46 | 0 | -4.9765412213701623045381 | -0.796759990018222730157 |
| primary_frame_certified_positions | 2013 | R8 | equal | year::2013:exposure | 46 | 0 | 0.2121532811407418339655 | 5.076909675700349566796 |
| primary_frame_certified_positions | 2013 | R8 | equal | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_certified_positions | 2013 | R8 | equal | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_observed_denial | 2013 | R5 | baseline_positions | year::2011:exposure | 46 | 0 | -0.0065899199585471397375 | 0.003030084697578519851 |
| primary_frame_observed_denial | 2013 | R5 | baseline_positions | year::2013:exposure | 46 | 0 | -0.1531864629965814517121 | 0.003030084697578519851 |
| primary_frame_observed_denial | 2013 | R5 | baseline_positions | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_observed_denial | 2013 | R5 | baseline_positions | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_observed_denial | 2013 | R5 | equal | year::2011:exposure | 46 | 0 | -0.0169942905842242197867 | 0.038521045619341401367 |
| primary_frame_observed_denial | 2013 | R5 | equal | year::2013:exposure | 46 | 0 | -0.0330034049026964107498 | 0.080653439265496132404 |
| primary_frame_observed_denial | 2013 | R5 | equal | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_observed_denial | 2013 | R5 | equal | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_observed_denial | 2013 | R7 | baseline_positions | year::2011:exposure | 46 | 0 | -0.0048366446185774802230 | 0.003520830324010770004 |
| primary_frame_observed_denial | 2013 | R7 | baseline_positions | year::2013:exposure | 46 | 0 | -0.1250217027241960487949 | 0.003520830324010770004 |
| primary_frame_observed_denial | 2013 | R7 | baseline_positions | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_observed_denial | 2013 | R7 | baseline_positions | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_observed_denial | 2013 | R7 | equal | year::2011:exposure | 46 | 0 | -0.0062233898514373695715 | 0.049445461999439707657 |
| primary_frame_observed_denial | 2013 | R7 | equal | year::2013:exposure | 46 | 0 | -0.0198259419552933996433 | 0.088710975940171230003 |
| primary_frame_observed_denial | 2013 | R7 | equal | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_observed_denial | 2013 | R7 | equal | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_observed_denial | 2013 | R8 | baseline_positions | year::2011:exposure | 46 | 0 | -0.0036546074444108699042 | 0.003959511604478429744 |
| primary_frame_observed_denial | 2013 | R8 | baseline_positions | year::2013:exposure | 46 | 0 | -0.1248561200235464851982 | 0.003502466287761120140 |
| primary_frame_observed_denial | 2013 | R8 | baseline_positions | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_observed_denial | 2013 | R8 | baseline_positions | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_observed_denial | 2013 | R8 | equal | year::2011:exposure | 46 | 0 | 0.0101607230273471106641 | 0.072720092084483170525 |
| primary_frame_observed_denial | 2013 | R8 | equal | year::2013:exposure | 46 | 0 | -0.0185089927579241905753 | 0.088800702127735012259 |
| primary_frame_observed_denial | 2013 | R8 | equal | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_observed_denial | 2013 | R8 | equal | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_observed_filing | 2013 | R5 | baseline_positions | year::2011:exposure | 46 | 0 | -0.0000000000000000832667 | 0.000000000000000194289 |
| primary_frame_observed_filing | 2013 | R5 | baseline_positions | year::2013:exposure | 46 | 0 | -0.1547425610734356471454 | 0.108982720589438919490 |
| primary_frame_observed_filing | 2013 | R5 | baseline_positions | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_observed_filing | 2013 | R5 | baseline_positions | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_observed_filing | 2013 | R5 | equal | year::2011:exposure | 46 | 0 | -0.0000000000000000693889 | 0.000000000000000121431 |
| primary_frame_observed_filing | 2013 | R5 | equal | year::2013:exposure | 46 | 0 | -0.0535463717739872030466 | 0.103559249584675619471 |
| primary_frame_observed_filing | 2013 | R5 | equal | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_observed_filing | 2013 | R5 | equal | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_observed_filing | 2013 | R7 | baseline_positions | year::2011:exposure | 46 | 0 | -0.0000000000000000138778 | 0.000000000000000152656 |
| primary_frame_observed_filing | 2013 | R7 | baseline_positions | year::2013:exposure | 46 | 0 | -0.1975581917691049960162 | 0.130803101922188719231 |
| primary_frame_observed_filing | 2013 | R7 | baseline_positions | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_observed_filing | 2013 | R7 | baseline_positions | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_observed_filing | 2013 | R7 | equal | year::2011:exposure | 46 | 0 | -0.0000000000000000277556 | 0.000000000000000116226 |
| primary_frame_observed_filing | 2013 | R7 | equal | year::2013:exposure | 46 | 0 | -0.0503082339326738278706 | 0.128280971341111410533 |
| primary_frame_observed_filing | 2013 | R7 | equal | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_observed_filing | 2013 | R7 | equal | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_observed_filing | 2013 | R8 | baseline_positions | year::2011:exposure | 46 | 0 | 0.0000000000000000416334 | 0.000000000000000208167 |
| primary_frame_observed_filing | 2013 | R8 | baseline_positions | year::2013:exposure | 46 | 0 | -0.1982208053396815050640 | 0.131750211948985812560 |
| primary_frame_observed_filing | 2013 | R8 | baseline_positions | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_observed_filing | 2013 | R8 | baseline_positions | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_observed_filing | 2013 | R8 | equal | year::2011:exposure | 46 | 0 | -0.0000000000000000416334 | 0.000000000000000086736 |
| primary_frame_observed_filing | 2013 | R8 | equal | year::2013:exposure | 46 | 0 | -0.0629901701933972824055 | 0.108412947320626062986 |
| primary_frame_observed_filing | 2013 | R8 | equal | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_observed_filing | 2013 | R8 | equal | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_timely | 2013 | R5 | baseline_positions | year::2011:exposure | 46 | 0 | -0.0012471886294210300389 | 0.019586757101573259998 |
| primary_frame_timely | 2013 | R5 | baseline_positions | year::2013:exposure | 46 | 0 | -0.1277837630815592262490 | 0.184442478187814312696 |
| primary_frame_timely | 2013 | R5 | baseline_positions | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_timely | 2013 | R5 | baseline_positions | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_timely | 2013 | R5 | equal | year::2011:exposure | 46 | 0 | -0.0291582914757514688464 | 0.023124276237793121164 |
| primary_frame_timely | 2013 | R5 | equal | year::2013:exposure | 46 | 0 | -0.0781572394111111762305 | 0.026650911976615478810 |
| primary_frame_timely | 2013 | R5 | equal | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_timely | 2013 | R5 | equal | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_timely | 2013 | R7 | baseline_positions | year::2011:exposure | 46 | 0 | -0.0021008969982432199562 | 0.017077274366771519332 |
| primary_frame_timely | 2013 | R7 | baseline_positions | year::2013:exposure | 46 | 0 | -0.1761697939359196662945 | 0.193641479908861352532 |
| primary_frame_timely | 2013 | R7 | baseline_positions | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_timely | 2013 | R7 | baseline_positions | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_timely | 2013 | R7 | equal | year::2011:exposure | 46 | 0 | -0.0407843644370803129484 | 0.012047734435235929809 |
| primary_frame_timely | 2013 | R7 | equal | year::2013:exposure | 46 | 0 | -0.0852645878259247619324 | 0.041063578513801018977 |
| primary_frame_timely | 2013 | R7 | equal | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_timely | 2013 | R7 | equal | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_timely | 2013 | R8 | baseline_positions | year::2011:exposure | 46 | 0 | -0.0020981145037547300862 | 0.017434357864019971174 |
| primary_frame_timely | 2013 | R8 | baseline_positions | year::2013:exposure | 46 | 0 | -0.1744133681108810396410 | 0.198758256508095115800 |
| primary_frame_timely | 2013 | R8 | baseline_positions | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_timely | 2013 | R8 | baseline_positions | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_timely | 2013 | R8 | equal | year::2011:exposure | 46 | 0 | -0.0400863186783902425425 | 0.011908286806296750418 |
| primary_frame_timely | 2013 | R8 | equal | year::2013:exposure | 46 | 0 | -0.0786637671309763780725 | 0.047651558586626389935 |
| primary_frame_timely | 2013 | R8 | equal | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_timely | 2013 | R8 | equal | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_timing | 2013 | R5 | baseline_positions | year::2011:exposure | 46 | 0 | -0.8929122317625648452832 | 2.901277670483632498133 |
| primary_frame_timing | 2013 | R5 | baseline_positions | year::2013:exposure | 46 | 0 | -5.6214343630636678383894 | 5.162756611769459880179 |
| primary_frame_timing | 2013 | R5 | baseline_positions | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_timing | 2013 | R5 | baseline_positions | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_timing | 2013 | R5 | equal | year::2011:exposure | 46 | 0 | 1.5416976683757095401006 | 3.366624483977270987367 |
| primary_frame_timing | 2013 | R5 | equal | year::2013:exposure | 46 | 0 | 2.7966021897168578114190 | 6.323464660706398454693 |
| primary_frame_timing | 2013 | R5 | equal | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_timing | 2013 | R5 | equal | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_timing | 2013 | R7 | baseline_positions | year::2011:exposure | 46 | 0 | -1.3753941376068155033607 | 2.670008295395611241929 |
| primary_frame_timing | 2013 | R7 | baseline_positions | year::2013:exposure | 46 | 0 | -6.1094550623969086089460 | 3.996013967942184663684 |
| primary_frame_timing | 2013 | R7 | baseline_positions | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_timing | 2013 | R7 | baseline_positions | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_timing | 2013 | R7 | equal | year::2011:exposure | 46 | 0 | 0.7178360148870698287382 | 2.759521413565405811852 |
| primary_frame_timing | 2013 | R7 | equal | year::2013:exposure | 46 | 0 | 1.7110580502943222303003 | 5.193796601301714055410 |
| primary_frame_timing | 2013 | R7 | equal | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_timing | 2013 | R7 | equal | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_timing | 2013 | R8 | baseline_positions | year::2011:exposure | 46 | 0 | -1.3439126660096401000288 | 2.652562808587040521502 |
| primary_frame_timing | 2013 | R8 | baseline_positions | year::2013:exposure | 46 | 0 | -5.9193146848308826335483 | 3.922709076682412643322 |
| primary_frame_timing | 2013 | R8 | baseline_positions | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_timing | 2013 | R8 | baseline_positions | year::2015:exposure | 46 | 46 | — | — |
| primary_frame_timing | 2013 | R8 | equal | year::2011:exposure | 46 | 0 | 0.7527472668625198037873 | 2.747840084391073744996 |
| primary_frame_timing | 2013 | R8 | equal | year::2013:exposure | 46 | 0 | 1.7022786340904025870202 | 5.053313003031957961753 |
| primary_frame_timing | 2013 | R8 | equal | year::2014:exposure | 46 | 46 | — | — |
| primary_frame_timing | 2013 | R8 | equal | year::2015:exposure | 46 | 46 | — | — |

[Every deletion and omitted state: earnings_pre_state_deletions.csv](/tmp/h2a-chapter-directions/earnings_pre_state_deletions.csv)

| specification | tau | tier | weights | term | deletions | unidentified | minimum | maximum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QWI_stable_monthly_cash_dose | 2016 | county | equal | D_h2a | 47 | 0 | -0.000159436 | 0.007450352 |
| QWI_stable_monthly_cash_dose | 2017 | county | equal | D_h2a | 47 | 0 | 0.013147330 | 0.024452051 |
| QWI_stable_monthly_cash_dose | 2018 | county | equal | D_h2a | 47 | 0 | -0.005370817 | -0.001000718 |
| QWI_stable_monthly_cash_dose | 2019 | county | equal | D_h2a | 47 | 0 | -0.014685858 | -0.004828544 |
| QWI_stable_monthly_cash_dose | 2020 | county | equal | D_h2a | 47 | 0 | -0.002942676 | 0.013460595 |
| QWI_stable_monthly_cash_dose | 2021 | county | equal | D_h2a | 47 | 0 | -0.008683709 | 0.005382448 |
| QWI_stable_monthly_cash_dose | 2022 | county | equal | D_h2a | 46 | 0 | 0.022784900 | 0.037594987 |
| QWI_stable_monthly_cash_dose | 2023 | county | equal | D_h2a | 46 | 0 | 0.000776318 | 0.009189442 |
| QWI_stable_monthly_cash_dose | 2024 | county | equal | D_h2a | 46 | 0 | -0.016413690 | -0.004149534 |
| QWI_stable_monthly_commuting_zone | 2016 | county | equal | D_cz | 47 | 0 | -0.026363866 | -0.016220019 |
| QWI_stable_monthly_commuting_zone | 2017 | county | equal | D_cz | 47 | 0 | 0.008513933 | 0.018059091 |
| QWI_stable_monthly_commuting_zone | 2018 | county | equal | D_cz | 47 | 0 | -0.018881235 | -0.009901899 |
| QWI_stable_monthly_commuting_zone | 2019 | county | equal | D_cz | 47 | 0 | -0.002356521 | 0.006871976 |
| QWI_stable_monthly_commuting_zone | 2020 | county | equal | D_cz | 47 | 0 | 0.002959028 | 0.023509977 |
| QWI_stable_monthly_commuting_zone | 2021 | county | equal | D_cz | 47 | 0 | -0.004530630 | 0.010544933 |
| QWI_stable_monthly_commuting_zone | 2022 | county | equal | D_cz | 46 | 0 | 0.016685427 | 0.029041664 |
| QWI_stable_monthly_commuting_zone | 2023 | county | equal | D_cz | 46 | 0 | 0.019622608 | 0.031487293 |
| QWI_stable_monthly_commuting_zone | 2024 | county | equal | D_cz | 46 | 0 | -0.042308451 | -0.014170710 |
| QWI_stable_monthly_employment_weighted | 2016 | county | emp_l | D | 47 | 0 | 0.006154782 | 0.012926483 |
| QWI_stable_monthly_employment_weighted | 2017 | county | emp_l | D | 47 | 0 | -0.013593182 | 0.000050285 |
| QWI_stable_monthly_employment_weighted | 2018 | county | emp_l | D | 47 | 0 | -0.013334074 | -0.004259197 |
| QWI_stable_monthly_employment_weighted | 2019 | county | emp_l | D | 47 | 0 | -0.017610202 | 0.000477779 |
| QWI_stable_monthly_employment_weighted | 2020 | county | emp_l | D | 47 | 0 | -0.007630810 | 0.000159653 |
| QWI_stable_monthly_employment_weighted | 2021 | county | emp_l | D | 47 | 0 | 0.003544268 | 0.013111850 |
| QWI_stable_monthly_employment_weighted | 2022 | county | emp_l | D | 46 | 0 | -0.005940787 | 0.001019683 |
| QWI_stable_monthly_employment_weighted | 2023 | county | emp_l | D | 46 | 0 | -0.001360701 | 0.014974569 |
| QWI_stable_monthly_employment_weighted | 2024 | county | emp_l | D | 46 | 0 | -0.002979013 | 0.004398306 |
| QWI_stable_monthly_no_intensity_control | 2016 | county | equal | D | 47 | 0 | -0.007707017 | -0.004412796 |
| QWI_stable_monthly_no_intensity_control | 2017 | county | equal | D | 47 | 0 | 0.002597025 | 0.003890907 |
| QWI_stable_monthly_no_intensity_control | 2018 | county | equal | D | 47 | 0 | -0.003524260 | -0.001416551 |
| QWI_stable_monthly_no_intensity_control | 2019 | county | equal | D | 47 | 0 | -0.001589451 | 0.000743292 |
| QWI_stable_monthly_no_intensity_control | 2020 | county | equal | D | 47 | 0 | -0.007959933 | -0.003852090 |
| QWI_stable_monthly_no_intensity_control | 2021 | county | equal | D | 47 | 0 | -0.002809095 | -0.001217575 |
| QWI_stable_monthly_no_intensity_control | 2022 | county | equal | D | 46 | 0 | 0.000091065 | 0.002669661 |
| QWI_stable_monthly_no_intensity_control | 2023 | county | equal | D | 46 | 0 | -0.000490906 | 0.000985747 |
| QWI_stable_monthly_no_intensity_control | 2024 | county | equal | D | 46 | 0 | -0.003541739 | -0.001348462 |
| QWI_stable_monthly_primary | 2016 | county | equal | D | 47 | 0 | 0.003327146 | 0.007552404 |
| QWI_stable_monthly_primary | 2017 | county | equal | D | 47 | 0 | 0.006442482 | 0.013082437 |
| QWI_stable_monthly_primary | 2018 | county | equal | D | 47 | 0 | -0.001619575 | 0.002723469 |
| QWI_stable_monthly_primary | 2019 | county | equal | D | 47 | 0 | -0.010527443 | 0.000452395 |
| QWI_stable_monthly_primary | 2020 | county | equal | D | 47 | 0 | -0.009999876 | 0.004523253 |
| QWI_stable_monthly_primary | 2021 | county | equal | D | 47 | 0 | 0.000033593 | 0.007960514 |
| QWI_stable_monthly_primary | 2022 | county | equal | D | 46 | 0 | 0.011673467 | 0.022746481 |
| QWI_stable_monthly_primary | 2023 | county | equal | D | 46 | 0 | 0.007199471 | 0.014019608 |
| QWI_stable_monthly_primary | 2024 | county | equal | D | 46 | 0 | -0.018314652 | -0.011493511 |
| all_private_primary | 2016 | county | equal | D | 48 | 0 | 0.003787050 | 0.005967363 |
| all_private_primary | 2017 | county | equal | D | 48 | 0 | 0.000491415 | 0.004176042 |
| all_private_primary | 2018 | county | equal | D | 48 | 0 | -0.002716500 | 0.000820928 |
| all_private_primary | 2019 | county | equal | D | 48 | 0 | -0.003314375 | -0.001359076 |
| all_private_primary | 2020 | county | equal | D | 48 | 0 | -0.000523367 | 0.004091807 |
| all_private_primary | 2021 | county | equal | D | 48 | 0 | 0.004563447 | 0.008562561 |
| all_private_primary | 2022 | county | equal | D | 48 | 0 | 0.001031420 | 0.002598707 |
| all_private_primary | 2023 | county | equal | D | 48 | 0 | -0.001293288 | 0.001165911 |
| all_private_primary | 2024 | county | equal | D | 47 | 0 | -0.002963258 | 0.000868437 |
| animal_primary | 2016 | county | equal | D | 47 | 0 | -0.014083035 | 0.004018624 |
| animal_primary | 2017 | county | equal | D | 47 | 0 | 0.004482212 | 0.021831681 |
| animal_primary | 2018 | county | equal | D | 47 | 0 | -0.010137496 | -0.000636508 |
| animal_primary | 2019 | county | equal | D | 47 | 0 | -0.005324566 | 0.032269294 |
| animal_primary | 2020 | county | equal | D | 46 | 0 | 0.030038437 | 0.076609277 |
| animal_primary | 2021 | county | equal | D | 46 | 0 | -0.034467416 | -0.025632494 |
| animal_primary | 2022 | county | equal | D | 46 | 0 | 0.039132209 | 0.059420744 |
| animal_primary | 2023 | county | equal | D | 47 | 0 | 0.024001910 | 0.060790563 |
| animal_primary | 2024 | county | equal | D | 46 | 0 | -0.016637191 | 0.010169527 |
| crop_UI_ambiguous | 2016 | county | equal | D | 42 | 0 | -0.000284033 | 0.010698278 |
| crop_UI_ambiguous | 2017 | county | equal | D | 42 | 0 | 0.000957328 | 0.012596290 |
| crop_UI_ambiguous | 2018 | county | equal | D | 43 | 0 | 0.003549583 | 0.011901019 |
| crop_UI_ambiguous | 2019 | county | equal | D | 43 | 0 | -0.017122616 | -0.004585805 |
| crop_UI_ambiguous | 2020 | county | equal | D | 43 | 0 | 0.005412070 | 0.016329365 |
| crop_UI_ambiguous | 2021 | county | equal | D | 43 | 0 | -0.004995726 | 0.003146460 |
| crop_UI_ambiguous | 2022 | county | equal | D | 43 | 0 | -0.021507929 | -0.002105071 |
| crop_UI_ambiguous | 2023 | county | equal | D | 43 | 0 | -0.001943241 | 0.010354359 |
| crop_UI_ambiguous | 2024 | county | equal | D | 42 | 0 | -0.010410471 | -0.000273300 |
| crop_UI_excluded | 2016 | county | equal | D | 3 | 0 | -0.203368944 | 0.081535900 |
| crop_UI_excluded | 2017 | county | equal | D | 3 | 0 | -0.231009546 | 0.432816226 |
| crop_UI_excluded | 2018 | county | equal | D | 3 | 0 | -0.177157170 | 0.150126139 |
| crop_UI_excluded | 2019 | county | equal | D | 3 | 0 | -0.681692625 | 0.223228658 |
| crop_UI_excluded | 2020 | county | equal | D | 3 | 0 | -0.102305827 | 0.189344425 |
| crop_UI_excluded | 2021 | county | equal | D | 3 | 0 | -0.150890423 | 0.177577901 |
| crop_UI_excluded | 2022 | county | equal | D | 3 | 0 | -0.358433817 | 0.929996304 |
| crop_UI_excluded | 2023 | county | equal | D | 3 | 0 | -1.201887487 | 0.486818064 |
| crop_UI_excluded | 2024 | county | equal | D | 3 | 0 | -0.389476342 | 0.184968446 |
| crop_cash_dose | 2016 | county | equal | D_h2a | 47 | 0 | -0.004228852 | 0.006777678 |
| crop_cash_dose | 2017 | county | equal | D_h2a | 47 | 0 | 0.003549219 | 0.014650131 |
| crop_cash_dose | 2018 | county | equal | D_h2a | 48 | 0 | 0.011453293 | 0.021514659 |
| crop_cash_dose | 2019 | county | equal | D_h2a | 48 | 0 | -0.017718248 | -0.006785057 |
| crop_cash_dose | 2020 | county | equal | D_h2a | 48 | 0 | 0.009366035 | 0.023576316 |
| crop_cash_dose | 2021 | county | equal | D_h2a | 48 | 0 | -0.004213231 | 0.006630950 |
| crop_cash_dose | 2022 | county | equal | D_h2a | 48 | 0 | -0.004839807 | 0.008689056 |
| crop_cash_dose | 2023 | county | equal | D_h2a | 48 | 0 | -0.027939470 | -0.006453923 |
| crop_cash_dose | 2024 | county | equal | D_h2a | 47 | 0 | -0.012457973 | 0.001166976 |
| crop_commuting_zone | 2016 | county | equal | D_cz | 47 | 0 | -0.019515672 | -0.011210004 |
| crop_commuting_zone | 2017 | county | equal | D_cz | 47 | 0 | -0.038062429 | -0.021685504 |
| crop_commuting_zone | 2018 | county | equal | D_cz | 48 | 0 | 0.011119942 | 0.024022557 |
| crop_commuting_zone | 2019 | county | equal | D_cz | 48 | 0 | -0.022279220 | 0.000077732 |
| crop_commuting_zone | 2020 | county | equal | D_cz | 48 | 0 | 0.000286150 | 0.020306318 |
| crop_commuting_zone | 2021 | county | equal | D_cz | 48 | 0 | -0.006058066 | 0.011932104 |
| crop_commuting_zone | 2022 | county | equal | D_cz | 48 | 0 | -0.017359135 | -0.006574975 |
| crop_commuting_zone | 2023 | county | equal | D_cz | 48 | 0 | 0.009970492 | 0.032304589 |
| crop_commuting_zone | 2024 | county | equal | D_cz | 47 | 0 | -0.024650834 | -0.005032077 |
| crop_employment_weighted | 2016 | county | emp_l | D | 47 | 0 | -0.005333117 | 0.009987051 |
| crop_employment_weighted | 2017 | county | emp_l | D | 47 | 0 | -0.009915216 | 0.002952009 |
| crop_employment_weighted | 2018 | county | emp_l | D | 48 | 0 | -0.002550118 | 0.007676623 |
| crop_employment_weighted | 2019 | county | emp_l | D | 48 | 0 | -0.025883928 | -0.004765696 |
| crop_employment_weighted | 2020 | county | emp_l | D | 48 | 0 | -0.007662951 | 0.007090159 |
| crop_employment_weighted | 2021 | county | emp_l | D | 48 | 0 | -0.022730594 | -0.013014007 |
| crop_employment_weighted | 2022 | county | emp_l | D | 48 | 0 | -0.016180203 | 0.002144118 |
| crop_employment_weighted | 2023 | county | emp_l | D | 48 | 0 | 0.014403570 | 0.030750579 |
| crop_employment_weighted | 2024 | county | emp_l | D | 47 | 0 | -0.004618095 | 0.004704613 |
| crop_no_intensity_control | 2016 | county | equal | D | 47 | 0 | -0.009854302 | -0.007411229 |
| crop_no_intensity_control | 2017 | county | equal | D | 47 | 0 | -0.002599584 | 0.005572962 |
| crop_no_intensity_control | 2018 | county | equal | D | 48 | 0 | -0.003593207 | -0.000885720 |
| crop_no_intensity_control | 2019 | county | equal | D | 48 | 0 | -0.004160254 | -0.000818928 |
| crop_no_intensity_control | 2020 | county | equal | D | 48 | 0 | -0.004524119 | -0.001681380 |
| crop_no_intensity_control | 2021 | county | equal | D | 48 | 0 | 0.000215851 | 0.001543760 |
| crop_no_intensity_control | 2022 | county | equal | D | 48 | 0 | -0.004077652 | -0.000845454 |
| crop_no_intensity_control | 2023 | county | equal | D | 48 | 0 | -0.003220373 | 0.001698450 |
| crop_no_intensity_control | 2024 | county | equal | D | 47 | 0 | -0.003928326 | -0.000806724 |
| crop_primary | 2016 | county | equal | D | 47 | 0 | -0.002605708 | 0.008599724 |
| crop_primary | 2017 | county | equal | D | 47 | 0 | -0.002951320 | 0.006513130 |
| crop_primary | 2018 | county | equal | D | 48 | 0 | 0.004398718 | 0.011999859 |
| crop_primary | 2019 | county | equal | D | 48 | 0 | -0.019901594 | -0.007864954 |
| crop_primary | 2020 | county | equal | D | 48 | 0 | 0.003913488 | 0.014470789 |
| crop_primary | 2021 | county | equal | D | 48 | 0 | -0.008125795 | -0.001220692 |
| crop_primary | 2022 | county | equal | D | 48 | 0 | -0.020047350 | -0.003519158 |
| crop_primary | 2023 | county | equal | D | 48 | 0 | 0.006380069 | 0.014805833 |
| crop_primary | 2024 | county | equal | D | 47 | 0 | -0.012744504 | -0.002844748 |

## Coverage, exclusions and measurement audits

### wage_composition_coverage.csv

| tier | tau | continuing_cells | unrestricted_filer_states | common_occupation_filer_states | baseline_hours_coverage | current_hours_coverage | mean_unrestricted | mean_fixed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R7 | 2024 | 9492 | 10647 | 9152 | 0.92202 | 0.91299 | 1.15320 | 1.06517 |
| R7 | 2025 | 11430 | 11477 | 10775 | 0.95369 | 0.95020 | 0.66102 | 0.70995 |
| R7 | 2026 | 12386 | 12313 | 11691 | 0.95732 | 0.95914 | -2.37481 | -2.35168 |
| R5 | 2024 | 9398 | 10552 | 9068 | 0.92144 | 0.91246 | 1.15274 | 1.06426 |
| R5 | 2025 | 11324 | 11379 | 10679 | 0.95368 | 0.94976 | 0.65896 | 0.70882 |
| R5 | 2026 | 12287 | 12227 | 11606 | 0.95738 | 0.95923 | -2.37630 | -2.35460 |
| R8 | 2024 | 9589 | 10752 | 9242 | 0.91975 | 0.90815 | 1.15614 | 1.05830 |
| R8 | 2025 | 11580 | 11607 | 10902 | 0.95412 | 0.95164 | 0.67101 | 0.71468 |
| R8 | 2026 | 12485 | 12406 | 11781 | 0.95766 | 0.95893 | -2.37784 | -2.35341 |
| FEIN | 2024 | 0 | 0 | 0 | — | — | — | — |
| FEIN | 2025 | 11671 | 11703 | 10997 | 0.95402 | 0.95219 | 0.68002 | 0.71693 |
| FEIN | 2026 | 12507 | 12446 | 11813 | 0.95784 | 0.95788 | -2.38019 | -2.35520 |

### wage_floor_audit.csv

| fiscal_year | rows | cases | hourly_hours | missing_R7_rows | multistate_hours | common_SOC_hours | receipt_proxy_new_hours | missing_receipt_hours | resolved_effective_floors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | 165347 | 16056 | 256517516 | 0 | 0.10143 | 0.97166 | 0.00000 | 0 | 0 |
| 2024 | 201401 | 17480 | 266526316 | 0 | 0.10606 | 0.95707 | 0.00000 | 0 | 0 |
| 2025 | 210911 | 18454 | 276199095 | 0 | 0.10319 | 0.96220 | 0.00000 | 0 | 0 |
| 2026 | 235068 | 18652 | 322238712 | 0 | 0.10145 | 0.96713 | 0.98338 | 0 | 0 |

### wage_exclusions.csv

| fiscal_year | rows | nonhourly_or_invalid_wage | invalid_nonpositive_hours | unknown_occupation |
| --- | --- | --- | --- | --- |
| 2023 | 165347 | 1305 | 19 | 0 |
| 2024 | 201401 | 4626 | 10 | 0 |
| 2025 | 210911 | 2753 | 10 | 0 |
| 2026 | 235068 | 2472 | 7 | 0 |

### admin_exclusions.csv

| tier | baseline_cohorts | recurring | complete_recurring | unknown_baseline_positions |
| --- | --- | --- | --- | --- |
| R7 | 7408 | 2462 | 2462 | 247 |
| R5 | 7484 | 2395 | 2395 | 253 |
| R8 | 7369 | 2488 | 2488 | 247 |

### admin_master_sub_audit.csv

| fiscal_year | primary_sub | rows | cases |
| --- | --- | --- | --- |
| 2010 | — | 7394 | 4431 |
| 2011 | — | 7362 | 4941 |
| 2012 | — | 8047 | 5459 |
| 2013 | — | 8385 | 5973 |
| 2014 | — | 9405 | 6726 |
| 2015 | — | 10339 | 7567 |
| 2016 | PRI | 8684 | 8684 |
| 2016 | SUB | 2913 | 497 |

### admin_year_means.csv

| tier | year | exposed | cohorts | records | timely | observed_filing | positions | timing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R5 | 2011 | FALSE | 2265 | 2265 | 0.96291 | 1.00000 | 14.127 | -27.814 |
| R5 | 2011 | TRUE | 130 | 130 | 0.99231 | 1.00000 | 47.488 | -29.777 |
| R5 | 2012 | FALSE | 2265 | 2265 | 0.97837 | 1.00000 | 13.463 | -30.539 |
| R5 | 2012 | TRUE | 130 | 130 | 0.98462 | 1.00000 | 50.806 | -31.969 |
| R5 | 2013 | FALSE | 2265 | 1662 | 0.71921 | 0.73377 | 10.399 | -30.131 |
| R5 | 2013 | TRUE | 130 | 99 | 0.72308 | 0.76154 | 48.962 | -23.752 |
| R5 | 2014 | FALSE | 2265 | 1388 | — | — | — | — |
| R5 | 2014 | TRUE | 130 | 7 | — | — | — | — |
| R5 | 2015 | FALSE | 2265 | 21 | — | — | — | — |
| R5 | 2015 | TRUE | 130 | 53 | — | — | — | — |
| R7 | 2011 | FALSE | 2326 | 2326 | 0.96131 | 1.00000 | 14.204 | -27.765 |
| R7 | 2011 | TRUE | 136 | 136 | 0.99265 | 1.00000 | 45.807 | -29.757 |
| R7 | 2012 | FALSE | 2326 | 2326 | 0.97893 | 1.00000 | 13.587 | -30.491 |
| R7 | 2012 | TRUE | 136 | 136 | 0.97794 | 1.00000 | 49.163 | -31.896 |
| R7 | 2013 | FALSE | 2326 | 1726 | 0.72786 | 0.74205 | 10.507 | -30.130 |
| R7 | 2013 | TRUE | 136 | 103 | 0.72059 | 0.75735 | 47.059 | -24.183 |
| R7 | 2014 | FALSE | 2326 | 1455 | — | — | — | — |
| R7 | 2014 | TRUE | 136 | 7 | — | — | — | — |
| R7 | 2015 | FALSE | 2326 | 21 | — | — | — | — |
| R7 | 2015 | TRUE | 136 | 54 | — | — | — | — |
| R8 | 2011 | FALSE | 2348 | 2348 | 0.96082 | 1.00000 | 14.182 | -27.749 |
| R8 | 2011 | TRUE | 140 | 140 | 0.98571 | 1.00000 | 48.439 | -29.705 |
| R8 | 2012 | FALSE | 2348 | 2348 | 0.97828 | 1.00000 | 13.576 | -30.439 |
| R8 | 2012 | TRUE | 140 | 140 | 0.97143 | 1.00000 | 54.532 | -31.655 |
| R8 | 2013 | FALSE | 2348 | 1747 | 0.72956 | 0.74404 | 10.483 | -30.121 |
| R8 | 2013 | TRUE | 140 | 106 | 0.72143 | 0.75714 | 51.636 | -24.218 |
| R8 | 2014 | FALSE | 2348 | 1474 | — | — | — | — |
| R8 | 2014 | TRUE | 140 | 7 | — | — | — | — |
| R8 | 2015 | FALSE | 2348 | 21 | — | — | — | — |
| R8 | 2015 | TRUE | 140 | 59 | — | — | — | — |

### earnings_pre_coverage.csv

| series | tau | counties | states | mean_exposure | prior_employment | excluded_UI_states | ambiguous_UI_states |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crop | 2016 | 1328 | 47 | 0.19920 | 487723 | 3 | 42 |
| animal | 2016 | 1152 | 47 | 0.10391 | 193110 | 3 | 42 |
| all_private | 2016 | 3017 | 48 | 0.13445 | 106110981 | 3 | 43 |
| crop | 2017 | 1358 | 47 | 0.19647 | 483018 | 3 | 42 |
| animal | 2017 | 1158 | 47 | 0.10674 | 198795 | 3 | 42 |
| all_private | 2017 | 3018 | 48 | 0.13431 | 108294429 | 3 | 43 |
| crop | 2018 | 1370 | 48 | 0.19498 | 481484 | 3 | 43 |
| animal | 2018 | 1175 | 47 | 0.11072 | 206084 | 3 | 42 |
| all_private | 2018 | 3019 | 48 | 0.13437 | 110145586 | 3 | 43 |
| crop | 2019 | 1386 | 48 | 0.18952 | 485029 | 3 | 43 |
| animal | 2019 | 1168 | 47 | 0.11175 | 206191 | 3 | 42 |
| all_private | 2019 | 3019 | 48 | 0.13455 | 111981963 | 3 | 43 |
| crop | 2020 | 1412 | 48 | 0.18970 | 471921 | 3 | 43 |
| animal | 2020 | 1177 | 46 | 0.11140 | 207542 | 3 | 41 |
| all_private | 2020 | 3023 | 48 | 0.13447 | 113781616 | 3 | 43 |
| crop | 2021 | 1449 | 48 | 0.18898 | 472198 | 3 | 43 |
| animal | 2021 | 1209 | 46 | 0.11487 | 209619 | 3 | 41 |
| all_private | 2021 | 3022 | 48 | 0.13451 | 109692482 | 3 | 43 |
| crop | 2022 | 1471 | 48 | 0.19107 | 470415 | 3 | 43 |
| animal | 2022 | 1222 | 46 | 0.11793 | 207779 | 3 | 41 |
| all_private | 2022 | 3015 | 48 | 0.13468 | 108816192 | 3 | 43 |
| crop | 2023 | 1465 | 48 | 0.19021 | 473638 | 3 | 43 |
| animal | 2023 | 1213 | 47 | 0.11538 | 205970 | 3 | 42 |
| all_private | 2023 | 3017 | 48 | 0.13459 | 114181438 | 3 | 43 |
| crop | 2024 | 1483 | 47 | 0.18888 | 465719 | 3 | 42 |
| animal | 2024 | 1219 | 46 | 0.11721 | 204826 | 3 | 41 |
| all_private | 2024 | 3017 | 47 | 0.13437 | 115585539 | 3 | 42 |
| QWI_stable_monthly | 2016 | 1748 | 47 | 0.18474 | 377991 | 3 | 42 |
| QWI_stable_monthly | 2017 | 1769 | 47 | 0.18371 | 380377 | 3 | 42 |
| QWI_stable_monthly | 2018 | 1787 | 47 | 0.18165 | 381506 | 3 | 42 |
| QWI_stable_monthly | 2019 | 1794 | 47 | 0.18305 | 378908 | 3 | 42 |
| QWI_stable_monthly | 2020 | 1810 | 47 | 0.18077 | 374963 | 3 | 42 |
| QWI_stable_monthly | 2021 | 1836 | 47 | 0.17856 | 379811 | 3 | 42 |
| QWI_stable_monthly | 2022 | 1818 | 46 | 0.17399 | 376032 | 3 | 41 |
| QWI_stable_monthly | 2023 | 1827 | 46 | 0.17410 | 378511 | 3 | 41 |
| QWI_stable_monthly | 2024 | 1855 | 46 | 0.17164 | 373861 | 3 | 41 |

### earnings_pre_queue.csv

| specification | status | reason |
| --- | --- | --- |
| primary | awaiting_full_window | No 2025 Q4 or later earnings accessed; scheduled QCEW 2026 Q2 release 2026-12-02 |
| employment_weighted | awaiting_full_window | No 2025 Q4 or later earnings accessed; scheduled QCEW 2026 Q2 release 2026-12-02 |
| cash_dose | awaiting_full_window | No 2025 Q4 or later earnings accessed; scheduled QCEW 2026 Q2 release 2026-12-02 |
| commuting_zone | awaiting_full_window | No 2025 Q4 or later earnings accessed; scheduled QCEW 2026 Q2 release 2026-12-02 |
| no_intensity_control | awaiting_full_window | No 2025 Q4 or later earnings accessed; scheduled QCEW 2026 Q2 release 2026-12-02 |
| all_private_diagnostic | awaiting_full_window | No 2025 Q4 or later earnings accessed; scheduled QCEW 2026 Q2 release 2026-12-02 |
| animal_diagnostic | awaiting_full_window | No 2025 Q4 or later earnings accessed; scheduled QCEW 2026 Q2 release 2026-12-02 |
| prior_window_lead | unavailable_separate_pre_policy_slices | Needs public 2025Q1 and Q2 crop slices; local DNS unavailable, web CSV unavailable, Cloud tool disallowed by session approval policy, browser connection timed out. Annual archives containing post-policy earnings remain closed. |
| UI_splits | awaiting_full_window | No 2025 Q4 or later earnings accessed; scheduled QCEW 2026 Q2 release 2026-12-02 |
| QWI_secondary | awaiting_full_window | No 2025 Q4 or later earnings accessed; scheduled QCEW 2026 Q2 release 2026-12-02 |

## Input hashes and source vintages

These hashes retain the run's evidence record if temporary scratch is later removed. Cache modification time is local provenance, not a verified agency publication date. The official-source URLs are recorded with each QCEW archive. No post-policy archive was opened.

| Input | Bytes | SHA256 | Cache_mtime_UTC |
| --- | --- | --- | --- |
| /home/ken/Dropbox/projects/H-2A_Paper/data/processed/county_year_panel.parquet | 27381319 | bff3256f8c2e87afcd643575de8bf969a34788f5b645de30831f1ac0547345d6 | 2026-11-14 02:46:53.294564 |
| /home/ken/Dropbox/projects/H-2A_Paper/data/intermediate/h2a_allocated_worksites.parquet | 65008573 | 9bb0e71fb00800ffcec6aaab6e69420384d43adda62d9c407805efab00b694b2 | 2026-11-14 02:45:13.374039 |
| /home/ken/Dropbox/projects/H-2A_Paper/data/intermediate/h2a_disclosure_records.parquet | 11684555 | d351e20ddde9a742afbb72be930f8ff7bd11329296d09ec556208f7bcf36e265 | 2026-10-26 23:36:55.97732 |
| /home/ken/Dropbox/projects/H-2A_Paper/data/intermediate/h2a_employer_record_membership.parquet | 118246232 | 7642e1efc288a06c699482c4d178ef1df8e585e65ea8cb42747c9e80d94adf11 | 2026-10-26 23:44:37.444164 |
| /home/ken/Dropbox/projects/H-2A_Paper/data/intermediate/aewr_rate_schedule.parquet | 12544 | 42f309c7d63e79fe857c744295ddd9ad95942eeae62bc8e88632f09241b2d75b | 2026-11-14 02:27:51.433876 |
| /home/ken/Dropbox/projects/H-2A_Paper/data/intermediate/aewr_occupation_skill_rates.parquet | 1380937 | c8d952ac670b4874a67cbdbd63247e2dcace491ba824e1e96f2ab4602fe3175d | 2026-11-14 02:27:51.443876 |
| /home/ken/Dropbox/projects/H-2A_Paper/data/intermediate/state_real_minwages.parquet | 37741 | c9de6b5afbafc985b5dd3445b9d98a10215a52c4f94618b1ac2a1aeff930a2d8 | 2026-11-03 03:23:40.409748 |
| /home/ken/Dropbox/projects/H-2A_Paper/data/intermediate/qwi_county_ag_quarterly_earnings.parquet | 2762261 | 38d2986f2aef4f7a734e9d5404f9afe14fd312d8ab6920a3c9159e99ce6e2420 | 2026-11-10 01:25:33.933515 |
| /tmp/h2a-chapter-directions/filer/filer_ids.parquet | 7202306 | adfb4895c2d33bebb2988e18ecdb8e9ffab60d123dfd6516633f239615106cca | 2026-11-14 02:34:51.066323 |
| /tmp/h2a-chapter-directions/filer/filer_cases.parquet | 11004611 | caa718000917cb640ddafa0e93e2f5c30378ba87e7e0a764c7f9fd201ace5e06 | 2026-11-14 02:29:40.508897 |
| /tmp/h2a-chapter-directions/benchmark/track1_pre_panel.csv | 3589871 | b7053b6989e92f62f58a4ad909e5b6e080ffee85edd9405e971869eb3eb34383 | 2026-11-03 01:02:54.16722 |
| /tmp/h2a-chapter-directions/benchmark/track1_treatment_schedule.csv | 55545 | 75b6269310dd5f375afc4ddeb0a4663c52bba03c07c47b42106ae0afdb9402b7 | 2026-11-03 01:02:54.489222 |
| /tmp/h2a-chapter-directions/benchmark/track1_post_estimates.csv | 12598 | ba34f91ff376d00f88a32fb421733e5160d982e5321392957ea976cfb039a329 | 2026-11-03 02:36:27.698184 |
| /tmp/h2a-chapter-directions/benchmark/domestic_wage_exposure_frozen.csv | 375885 | 6bd476d1bf0c4cb8d98e3072d1b7b1d6375176d18dffc62a71d6bb6e0e25eda3 | 2026-11-15 05:59:55.422786 |
| /tmp/h2a-chapter-directions/benchmark/qcew_crop_county_quarter_2014_2024.parquet | 1019712 | 387ed6fc4a171ba4f09fb8f72c09604576521c740dd7999c17376465ce6208f1 | 2026-11-14 01:17:37.313743 |

| Input | SHA256 | Source | Cache_mtime_UTC |
| --- | --- | --- | --- |
| 2014_qtrly_singlefile.zip | 59cffadb9ea33ae2fd6a775514f458d4982108050acdefe4444b3116191bc6c3 | https://data.bls.gov/cew/data/files/2014/csv/2014_qtrly_singlefile.zip | 2026-07-01 04:09:31.807871 |
| 2015_qtrly_singlefile.zip | ee1f4b7ca537d735024a446f66847681650629489131aeeb383ff1cb2d26a0d3 | https://data.bls.gov/cew/data/files/2015/csv/2015_qtrly_singlefile.zip | 2026-07-01 04:09:30.767107 |
| 2016_qtrly_singlefile.zip | ea9c8671603ea0ec4c2a7d106bfbf7fcf30dcc626df143c7b5d7fdec397f9c36 | https://data.bls.gov/cew/data/files/2016/csv/2016_qtrly_singlefile.zip | 2026-07-01 04:09:12.571885 |
| 2017_qtrly_singlefile.zip | 8160295f7c94ef3c0d045e896a8cebd7d3bb1b162378e7d55ee7f3533d3b046d | https://data.bls.gov/cew/data/files/2017/csv/2017_qtrly_singlefile.zip | 2026-07-01 04:08:37.219702 |
| 2018_qtrly_singlefile.zip | 1f55d28705a40fc91d8425b9df0a8f9312422c44401e8388f609261248fb85b3 | https://data.bls.gov/cew/data/files/2018/csv/2018_qtrly_singlefile.zip | 2026-07-01 03:55:36.391392 |
| 2019_qtrly_singlefile.zip | 6e34c6e1ba43251ecc38fc1761db5707dc33d088230b1750fbeeeaedeca8346f | https://data.bls.gov/cew/data/files/2019/csv/2019_qtrly_singlefile.zip | 2026-07-01 03:55:37.47297 |
| 2020_qtrly_singlefile.zip | 09f5cdd51d489c44d92ed8d8da12192d79458e8585c7a7ee92886e2af1661c43 | https://data.bls.gov/cew/data/files/2020/csv/2020_qtrly_singlefile.zip | 2026-07-01 03:55:04.078611 |
| 2021_qtrly_singlefile.zip | 5521b535753d50bac0244f218ac84e4d442354b70fa5eec8c359e5f66fbce732 | https://data.bls.gov/cew/data/files/2021/csv/2021_qtrly_singlefile.zip | 2026-07-01 03:55:05.359747 |
| 2022_qtrly_singlefile.zip | cde0d0b23567d1ad35ca30ff956b0f8ef31a88a6b8c145ccc93acb2e3a0ac25d | https://data.bls.gov/cew/data/files/2022/csv/2022_qtrly_singlefile.zip | 2026-07-01 03:54:48.953436 |
| 2023_qtrly_singlefile.zip | 0a5f3aa6252ebd1369eef8ba4a3a34e9adc096b2e1a0b22177a3a6d907378960 | https://data.bls.gov/cew/data/files/2023/csv/2023_qtrly_singlefile.zip | 2026-07-01 03:54:45.771611 |
| 2024_qtrly_singlefile.zip | b0a67b5cab3eb662d8772f7802faec606f307674fb611846a64d38ce182923d4 | https://data.bls.gov/cew/data/files/2024/csv/2024_qtrly_singlefile.zip | 2026-07-01 03:45:10.722644 |

### Frozen declarations and earnings implementation

```text
86702463c43fc61eb461fb0a5d1f819e5360bca2bb55d96de9785157082c08a3  explorations/chapter_directions/registry.json
fd0745dcc108ec4b5052b9b7c94b5af91ef7a9c64407556ed53a630924d663ea  explorations/chapter_directions/05_extract_earnings.py
736af5975b785c8b1eee70f0b8de649f753f4aa8e7b6fdbd56b31564e6bcd50d  explorations/chapter_directions/06_estimate_earnings.R
1b712569687d44d956cdb09d11165ebc40d6df9cf813db625b850e4bd43362da  explorations/chapter_directions/helpers.R
b451e57f8ad9f802e0ec18a2cea144daeb79423a94c6209ec99215a9228a3096  explorations/chapter_directions/sources/ui_classification_review.json
6bd2dfdb2d67915072ca77b6095bfb980a36d85f4fab998b58490aea2abb6d8e  explorations/chapter_directions/measurement_amendment_2026_11_14.json
86702463c43fc61eb461fb0a5d1f819e5360bca2bb55d96de9785157082c08a3  explorations/chapter_directions/registry.json
4c600a20216cc73b99119a15b98253d1cc477cb4d1da86880e71d951d135df83  explorations/episode_2025/domestic_wage_registry.sha256
6bd476d1bf0c4cb8d98e3072d1b7b1d6375176d18dffc62a71d6bb6e0e25eda3  explorations/episode_2025/domestic_wage_exposure_frozen.csv
cd426756bc700bb90405ff36de18e644665e68f56085d798482321d75bc1377a  explorations/episode_2025/ep_helpers.R
34b4bc030665cc5fd0de8ff57afcce171872d2f78480f2c2b618505ef2734dca  explorations/identification/id_helpers.R
332b7eb54e645bbb0b55e7709e2c8011e4b141351bbe94e372dbbf3d67b64ded  code/a01_sources/04_02_qcew_quarterly_employment.py
cdaf24038245bf80d8c2f0116e88a334eb653f6f71312adde5c78268f715e2d5  code/c00_shared/geography.R
2575e5129ce2fc86649a60788d940b8262501dfc3269ca5f0cad5ead873c5752  src/h2a/geography.py
7604975a937cfb31c861561f9f4afce1cd8a097fe5802f1d5bc7fc559e572fd9  code/paths.R
92b67554170dfec18a4936e6bad1c71024c53a60329b7b2b494c01a7f352f2a9  src/h2a/paths.py
228e0515bfa26c9a755603e3560cfe2af7e1098bf327104f833026c2640114d8  explorations/chapter_directions/resume_earnings.sh
```

