# Temporal-weighted residual Ridge

This experiment gives larger training weight to campaigns whose labels are closer
to the prediction cutoff. The goal is to adapt the residual layer to temporal
drift while keeping all features and labels past-only.

## Leak-free protocol

- Selection fit uses development rows only and recency is measured relative to calibration_start.
- Final fit uses pretest rows only and recency is measured relative to test_start.
- Final holdout is evaluated only after config and target-wise configs are selected.

## Selected config

{
  "feature_source": "standard",
  "feature_set": "all",
  "alpha": 1500.0,
  "correction_weight": 0.22,
  "uncertainty_weight": 0.0,
  "correction_clip": 0.15,
  "half_life_hours": 72.0
}

## Selected target-wise configs

{
  "at_least_one": {
    "feature_source": "standard",
    "feature_set": "no_decomp",
    "alpha": 300.0,
    "correction_weight": 0.35,
    "uncertainty_weight": 0.0,
    "correction_clip": 0.25,
    "half_life_hours": 72.0
  },
  "at_least_two": {
    "feature_source": "standard",
    "feature_set": "all",
    "alpha": 50.0,
    "correction_weight": 0.2,
    "uncertainty_weight": 0.0,
    "correction_clip": 0.08,
    "half_life_hours": "inf"
  },
  "at_least_three": {
    "feature_source": "hazard",
    "feature_set": "all",
    "alpha": 50.0,
    "correction_weight": 0.1,
    "uncertainty_weight": 0.0,
    "correction_clip": 0.6,
    "half_life_hours": "inf"
  }
}

## Metrics

| model                                       |   selection_metric_percent |   selection_metric_raw_percent |   final_holdout_metric_percent |   final_holdout_metric_raw_percent |
|:--------------------------------------------|---------------------------:|-------------------------------:|-------------------------------:|-----------------------------------:|
| base_decomposed_replay                      |                       7.96 |                        7.95906 |                           9.54 |                            9.5442  |
| temporal_weighted_ridge_residual            |                       7.93 |                        7.93095 |                           9.47 |                            9.4655  |
| temporal_weighted_targetwise_ridge_residual |                     nan    |                      nan       |                           9.29 |                            9.28932 |