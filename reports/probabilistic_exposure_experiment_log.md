# Probabilistic user-level exposure simulator

This experiment tries a deeper mechanistic decomposition than replay:
user activity rate, publisher preference, hour-of-week profile, empirical CPM
win curve and Poisson aggregation to 1+/2+/3+.

## Leak-free protocol

- Fit simulator statistics only on history before the forecast cutoff.
- Fit log-bias on development rows.
- Select simulator/blend hyperparameters on later pretest rows.
- Refit log-bias on all pretest rows before final holdout evaluation.

## Selected config

{
  "rate_smoothing_weeks": 0.5,
  "publisher_smoothing": 25.0,
  "hour_smoothing": 25.0,
  "cpm_smoothing": 0.0,
  "session_power": 0.5,
  "hazard_blend_weight": 0.0
}

## Metrics

| model                                  |   selection_metric_percent |   final_holdout_metric_percent |
|:---------------------------------------|---------------------------:|-------------------------------:|
| base_decomposed_replay                 |                       7.96 |                           9.54 |
| probabilistic_exposure_simulator_blend |                       7.96 |                           9.54 |