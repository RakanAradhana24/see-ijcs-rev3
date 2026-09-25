# NUMBER_TRACE (preliminary; completed together with the manuscript in Phase 3)

Source workbook: `results/SEE_results_rev3.xlsx` (the same tables are in `results/csv/<sheet>.csv`).

## Table-level mapping
- Table I  <- sheet `T1_datasets` (columns projects_raw, projects_used, features, effort_unit)
- Table II <- sheet `T2_main`, row = (dataset, algorithm), columns `<metric>_mean` and `<metric>_sd`; bold = `<metric>_best`
- Table III <- sheet `T3_ablation`, columns `<spec>_mean10`, `<spec>_sd10` (and `<spec>_seed42`)
- Table IV <- sheet `T4_stability`
- Inference tables <- sheets `friedman`, `nemenyi`, `wilcoxon_holm`, `effect_sizes`, `inference_summary`
- SHAP <- sheet `shap_importance`

## Key in-text numbers

| Number in the manuscript | Cell | Value in this run |
|---|---|---|
| LR mean MAE, COCOMO81 (primary protocol) | `T2_main!C2` | 63,492 |
| LR across-fold median MAE, COCOMO81 | `T2_main!E2` | 315.756 |
| LR mean SA, COCOMO81 | `T2_main!W2` | -3,395.48 |
| LR MMRE / MdMRE, COCOMO81 | `T2_main!K2` | 11.4329 |
| RF MMRE, COCOMO81 | `T2_main!K4` | 0.755711 |
| RF SA, COCOMO81 | `T2_main!W4` | 65.1924 |
| RF MMRE, NASA93 | `T2_main!K9` | 0.639569 |
| RF SA, NASA93 | `T2_main!W9` | 65.8373 |
| Best PRED(25) of the study (Desharnais, LR) | `T2_main!S12` | 40.5357 |
| MAE of the failing fold, LR semi-log, COCOMO81 | `lr_case!C2` | 630,801 |
| Largest project size, COCOMO81 (KLOC) | `descriptives!J2` | 1,150 |
| Skewness of effort, COCOMO81 | `descriptives!G2` | 4.36784 |
| Skewness of ln(1+effort), COCOMO81 | `descriptives!H2` | 0.388152 |
| LR MAE over 10 partitions: mean, COCOMO81 | `T4_stability!C2` | 195,567 |
| LR MAE over 10 partitions: min, COCOMO81 | `T4_stability!F2` | 39,384.2 |
| LR MAE over 10 partitions: max, COCOMO81 | `T4_stability!G2` | 995,510 |
| RF MAE over 10 partitions: mean, COCOMO81 | `T4_stability!C4` | 442.831 |
| RF MAE over 10 partitions: SD, COCOMO81 | `T4_stability!D4` | 13.4041 |
| LR MAE over 10 partitions: mean, NASA93 | `T4_stability!C7` | 1,897.95 |
| Position of seed 42 among partitions, LR NASA93 (1 = lowest MAE) | `T4_stability!I7` | 1 |
| LR MAE, raw specification, mean of 10 partitions, COCOMO81 | `T3_ablation!D2` | 981.665 |
| LR MAE, log-log specification, mean of 10 partitions, COCOMO81 | `T3_ablation!J2` | 282.037 |
| LR MAE, log-log specification, mean of 10 partitions, NASA93 | `T3_ablation!J7` | 254.694 |
| LR MAE, log-log specification, mean of 10 partitions, Desharnais | `T3_ablation!J12` | 1,809.12 |
| Friedman p, COCOMO81, semi-log | `friedman!D2` | 0.037817 |
| Friedman p, NASA93, semi-log | `friedman!D3` | 0.0527048 |
| Friedman p, Desharnais, semi-log | `friedman!D4` | 0.0103388 |
| Friedman p, COCOMO81, log-log | `friedman!D5` | 0.000401412 |
| Nemenyi p, RF-KNN, COCOMO81, semi-log | `nemenyi!F9` | 0.0376961 |
| Wilcoxon per fold, LR-RF, COCOMO81: raw p | `wilcoxon_holm!E3` | 0.00390625 |
| Wilcoxon per fold, LR-RF, COCOMO81: Holm p | `wilcoxon_holm!F3` | 0.0390625 |
| Pairs significant after Holm, per project, COCOMO81, semi-log | `inference_summary!D2` | 3 |
| Top SHAP feature share (%), COCOMO81 | `shap_importance!E2` | 58.0896 |
| Top SHAP feature share (%), NASA93 | `shap_importance!E18` | 68.5435 |
| Top SHAP feature share (%), Desharnais | `shap_importance!E36` | 36.7148 |
