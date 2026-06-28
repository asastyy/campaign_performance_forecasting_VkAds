# GitHub delta upload: VK Ads temporal experiments

Эта папка содержит файлы, которые нужно догрузить в репозиторий поверх предыдущей версии.

## Главное обновление

Финальная enhanced-модель после новых экспериментов:

- `temporal_weighted_targetwise_ridge_residual`
- locked final holdout: `9.29%`
- raw final holdout: `9.289324%`

Основная идея: decomposed replay остается базой, а поверх него добавляется
recency-weighted log-residual Ridge calibration. Для `1+`, `2+`, `3+` выбраны
отдельные конфигурации без доступа к final holdout.

## Что догружается

- новые скрипты экспериментов:
  - `run_probabilistic_exposure_simulator.py`
  - `run_hazard_feature_residual.py`
  - `run_selection_model_blend.py`
  - `run_temporal_weighted_residual.py`
  - `run_temporal_selection_blend.py`
- обновленный `README.md`;
- обновленный `generate_project_figures.py`;
- новые `outputs/` с метриками, конфигами и predictions;
- новые `reports/` с логами экспериментов и ablation;
- обновленные ключевые визуализации:
  - `figures/04_model_comparison.png`
  - `figures/06_predicted_vs_actual_final_holdout.png`
- обновленная сопроводительная записка:
  - `docs/Сопроводительная записка НИР Сергеева АВ VK Ads обновленная.docx`

## Финальная модель для описания в работе

Использовать как итог:

`outputs/validation_predictions_temporal_weighted_targetwise_ridge.tsv`

Не использовать как итог, но можно оставить как диагностику:

- `outputs/validation_predictions_temporal_selection_blend.tsv`
- `outputs/temporal_weighted_refinement_v2_metrics.csv`
