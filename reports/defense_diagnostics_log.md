# Defense Diagnostics Log

Все diagnostics ниже являются post-hoc анализом зафиксированных predictions на locked final holdout.

## Model comparison on final holdout

| model | rows | final_holdout_metric_percent |
| --- | --- | --- |
| Final decomposed replay | 201 | 9.544 |
| Boosting + replay features | 201 | 10.944 |
| Monthly replay | 201 | 11.693 |
| Static/history boosting | 201 | 30.893 |
| Zero baseline | 201 | 202.961 |

## Target-level error breakdown

| target | mean_actual | mean_predicted | mean_bias_pred_minus_actual | mean_abs_log_error | median_abs_error | p90_abs_error |
| --- | --- | --- | --- | --- | --- | --- |
| at_least_one | 0.06533 | 0.06771 | 0.00237 | 0.14108 | 0.00413 | 0.01731 |
| at_least_two | 0.02576 | 0.02634 | 0.00058 | 0.08196 | 0.00042 | 0.00906 |
| at_least_three | 0.01379 | 0.01344 | -0.00035 | 0.05043 | 0.00000 | 0.00494 |

## Worst final-holdout segments

| segment | segment_value | rows | metric_percent |
| --- | --- | --- | --- |
| Audience quartile | (299.999, 712.0] | 51 | 14.242 |
| Duration quartile | (18.0, 49.0] | 50 | 12.079 |
| Duration quartile | (49.0, 205.0] | 50 | 11.662 |
| CPM quartile | (130.0, 212.0] | 46 | 11.657 |
| Start hour bucket | night | 48 | 11.301 |
| CPM quartile | (70.0, 130.0] | 54 | 10.140 |
| Publisher count | 3+ publishers | 150 | 10.055 |
| Start hour bucket | day | 62 | 9.881 |

## Short interpretation

- Best final-holdout model in this comparison: `Final decomposed replay` with 9.54%.
- Locked decomposed replay metric: 9.54%.
- Segment diagnostics show where the model is less stable; this is useful as an honest limitation and future-work direction.
- Calibration bins are diagnostic only: no post-final calibration was fitted on final holdout.

## Generated files

- `outputs/defense_model_metrics.csv`
- `outputs/defense_target_error_breakdown.csv`
- `outputs/defense_segment_metrics.csv`
- `outputs/defense_calibration_bins.csv`
- `figures/08_segment_error_analysis.png`
- `figures/09_calibration_bins.png`
- `figures/10_target_error_breakdown.png`
- `figures/11_candidate_selection_curve.png`
- `figures/12_error_distribution.png`
