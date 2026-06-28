# Hazard-feature residual calibration

This experiment uses the probabilistic exposure simulator only as an additional
feature block for the regularized log-residual ridge model.

## Leak-free protocol

- Hazard features are fitted from history before the forecast cutoff only.
- Ridge hyperparameters are selected on the later pretest split.
- Final holdout is evaluated only after config selection.

## Selected ridge config

{
  "feature_set": "all",
  "alpha": 300.0,
  "correction_weight": 0.08,
  "uncertainty_weight": 0.0,
  "correction_clip": 0.4
}

## Selected target-wise configs

{
  "at_least_one": {
    "feature_set": "static_replay",
    "alpha": 100.0,
    "correction_weight": 0.2,
    "uncertainty_weight": 2.0,
    "correction_clip": 0.08
  },
  "at_least_two": {
    "feature_set": "all",
    "alpha": 100.0,
    "correction_weight": 0.08,
    "uncertainty_weight": 0.0,
    "correction_clip": 0.4
  },
  "at_least_three": {
    "feature_set": "all",
    "alpha": 100.0,
    "correction_weight": 0.12,
    "uncertainty_weight": 0.0,
    "correction_clip": 0.4
  }
}

## Metrics

| model                                    |   selection_metric_percent |   final_holdout_metric_percent |
|:-----------------------------------------|---------------------------:|-------------------------------:|
| base_decomposed_replay                   |                       7.96 |                           9.54 |
| hazard_feature_ridge_residual            |                       7.95 |                           9.47 |
| hazard_feature_targetwise_ridge_residual |                     nan    |                           9.48 |