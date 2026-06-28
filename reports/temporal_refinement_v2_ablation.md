# Over-aggressive temporal refinement v2 ablation

This ablation expanded the temporal residual grid beyond the stable selected
region: shorter recency half-life for `at_least_one`, stronger correction
weights and lower Ridge alpha for `at_least_two` / `at_least_three`.

## Result

| Model | Selection, % | Selection raw, % | Final, % | Final raw, % |
|---|---:|---:|---:|---:|
| Base decomposed replay | 7.96 | 7.959060 | 9.54 | 9.544203 |
| Temporal-weighted Ridge residual | 7.93 | 7.930945 | 9.47 | 9.465500 |
| Over-aggressive target-wise temporal residual | n/a | n/a | 9.30 | 9.298835 |

## Interpretation

The stable temporal target-wise model reached `9.29%` final holdout
(`9.289324%` raw). The more aggressive v2 grid selected a very short
`24h` half-life and correction weight `0.50` for `at_least_one`, which is a
sign of possible local overfitting. It did not improve the locked final
holdout, so it is kept as a diagnostic ablation rather than the final model.
