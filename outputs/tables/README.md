# Tables Guide

Use these files for writing results quickly.

## Start Here (minimal)
- `summary/human_key_metrics_table.csv` -> compact Human table with `Metric`, `Mean`, `95% CI`
- `summary/peak_sensitivity_compact_summary.csv` -> sensitivity stability summary across settings

## Human Baselines
- `human/human_per_image_behaviour.csv`
- `human/human_stats_by_category.csv`
- `human/human_consensus_peaks_by_category.csv`

## Confidence Intervals
- `ci/human_metric_confidence_intervals_overall.csv`
- `ci/human_metric_confidence_intervals_by_category.csv`
- `ci/model_metric_confidence_intervals.csv`

## Model Behaviour
- `behaviour/*_overall_behaviour.csv`
- `behaviour/*_per_category_behaviour.csv`
- `behaviour/*_per_image_behaviour.csv`

## Model-Human Delta
- `deltas/*_human_delta_*.csv`
- `deltas/all_models_human_delta_*.csv`

## Peak Sensitivity
- `sensitivity/*peak_sensitivity*.csv`
- `../peak_sensitivity_by_setting/` (per-setting split files)
