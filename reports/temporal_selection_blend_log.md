# Temporal selection-only blend

This experiment adds temporal-weighted residual sources to the previous
selection-only target blend. Source and blend choices are selected only on
the pre-final selection split, then all models are refit on pretest rows
and evaluated once on locked final holdout.

## Selected combination

{
  "at_least_one": {
    "name": "temporal_targetwise",
    "source_a": "temporal_targetwise",
    "source_b": null,
    "weight_b": 0.0
  },
  "at_least_two": {
    "name": "temporal_targetwise",
    "source_a": "temporal_targetwise",
    "source_b": null,
    "weight_b": 0.0
  },
  "at_least_three": {
    "name": "temporal_targetwise_0.65+hazard_targetwise_0.35",
    "source_a": "temporal_targetwise",
    "source_b": "hazard_targetwise",
    "weight_b": 0.35
  }
}

## Metrics

| model                    |   selection_metric_percent |   selection_metric_raw_percent |   final_holdout_metric_percent |   final_holdout_metric_raw_percent |
|:-------------------------|---------------------------:|-------------------------------:|-------------------------------:|-----------------------------------:|
| base_decomposed_replay   |                       7.96 |                        7.95906 |                           9.54 |                            9.5442  |
| temporal_selection_blend |                       7.85 |                        7.85368 |                           9.29 |                            9.28969 |

## Source metrics on selection

| source              |   selection_metric_percent |   selection_metric_raw_percent |   at_least_one_abs_log_error |   at_least_two_abs_log_error |   at_least_three_abs_log_error |
|:--------------------|---------------------------:|-------------------------------:|-----------------------------:|-----------------------------:|-------------------------------:|
| temporal_targetwise |                       7.85 |                        7.85371 |                    0.093656  |                    0.072439  |                      0.0607217 |
| targetwise          |                       7.92 |                        7.91747 |                    0.0953254 |                    0.0724728 |                      0.0607915 |
| hazard_targetwise   |                       7.92 |                        7.91898 |                    0.0953616 |                    0.072538  |                      0.060732  |
| temporal_ridge      |                       7.93 |                        7.93095 |                    0.0945376 |                    0.0726585 |                      0.0617682 |
| ridge               |                       7.94 |                        7.94244 |                    0.0957379 |                    0.0725491 |                      0.0609969 |
| hazard_ridge        |                       7.95 |                        7.9462  |                    0.0958725 |                    0.0726024 |                      0.0609133 |
| base                |                       7.96 |                        7.95906 |                    0.0957843 |                    0.0726739 |                      0.0612875 |