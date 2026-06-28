# Selection-only model blend

This experiment checks whether already selected residual variants are complementary.
Target options and blend weights are chosen on the selection split only, then refit
on all pretest rows and evaluated once on the locked final holdout.

## Selected combination

{
  "at_least_one": {
    "name": "targetwise_0.50+hazard_targetwise_0.50",
    "source_a": "targetwise",
    "source_b": "hazard_targetwise",
    "weight_b": 0.5
  },
  "at_least_two": {
    "name": "targetwise",
    "source_a": "targetwise",
    "source_b": null,
    "weight_b": 0.0
  },
  "at_least_three": {
    "name": "hazard_targetwise",
    "source_a": "hazard_targetwise",
    "source_b": null,
    "weight_b": 0.0
  }
}

## Metrics

| model                  |   selection_metric_percent |   selection_metric_raw_percent |   final_holdout_metric_percent |   final_holdout_metric_raw_percent |
|:-----------------------|---------------------------:|-------------------------------:|-------------------------------:|-----------------------------------:|
| base_decomposed_replay |                       7.96 |                        7.95906 |                           9.54 |                            9.5442  |
| selection_model_blend  |                       7.91 |                        7.91486 |                           9.46 |                            9.46203 |

## Source metrics on selection

| source            |   selection_metric_raw_percent |   selection_metric_percent |   at_least_one_abs_log_error |   at_least_two_abs_log_error |   at_least_three_abs_log_error |
|:------------------|-------------------------------:|---------------------------:|-----------------------------:|-----------------------------:|-------------------------------:|
| targetwise        |                        7.91747 |                       7.92 |                    0.0953254 |                    0.0724728 |                      0.0607915 |
| hazard_targetwise |                        7.91898 |                       7.92 |                    0.0953616 |                    0.072538  |                      0.060732  |
| ridge             |                        7.94244 |                       7.94 |                    0.0957379 |                    0.0725491 |                      0.0609969 |
| hazard_ridge      |                        7.9462  |                       7.95 |                    0.0958725 |                    0.0726024 |                      0.0609133 |
| base              |                        7.95906 |                       7.96 |                    0.0957843 |                    0.0726739 |                      0.0612875 |