# VK Ads leak-free reach/frequency forecasting

Код курсовой работы по прогнозированию долей пользователей с 1+, 2+ и 3+ показами объявления.

## Что входит в репозиторий

- Python-код модели и экспериментов;
- таблицы метрик, предсказаний и model lock;
- готовые графики в папке `figures/`;
- тесты на отсутствие временных утечек

Датасет (`history.tsv`, `users.tsv`, `validate.tsv`, `validate_answers.tsv`) в репозиторий не включены. Их нужно скачать отдельно и перед запуском указать путь через `VK_ADS_DATA_DIR`.

## Основные файлы

- `src/vk_ads_solution.py` — sessionization, past-only auction replay, temporal splits, агрегаты и индикаторы.
- `select_strict_model.py` — выбор конфигурации только по кампаниям, завершённым до final holdout.
- `evaluate_locked_holdout.py` — отдельная оценка зафиксированного final holdout.
- `predict_future.py` — прогноз кампаний после конца доступной истории; файл ответов не читается.
- `run_experiments.py` — дополнительные past-only и purged OOF diagnostics.
- `run_boosting_baseline.py` — бустинг на агрегированных признаках как ML-baseline.
- `run_segment_residual_calibration.py` — дополнительный leak-free эксперимент:
  segment-aware residual calibration, uncertainty shrinkage и Ridge residual поверх replay.
- `run_probabilistic_exposure_simulator.py` — user-level hazard simulator:
  активность пользователя, площадки, hour-of-week, эмпирическая CPM win-curve и Poisson aggregation.
- `run_hazard_feature_residual.py` — добавление simulator diagnostics как признаков
  в регуляризованный residual Ridge поверх replay.
- `run_selection_model_blend.py` — selection-only target blend уже выбранных
  residual-моделей без доступа к final holdout.
- `run_temporal_weighted_residual.py` — temporal-local residual Ridge:
  recency-weighted обучение residual-слоя и отдельный подбор конфигураций для 1+/2+/3+.
- `run_temporal_selection_blend.py` — selection-only blend с temporal-weighted
  источниками; используется как ablation поверх финальной temporal-модели.
- `run_advance_like_baseline.py` — AdVance-inspired neural multi-task baseline:
  общий encoder и три residual-heads поверх replay-прогноза.
- `VK_Ads_reach_forecasting_solution.ipynb` — выполненный notebook с проверкой model lock.
- `VK_Ads_reach_forecasting_colab.ipynb` — самодостаточный notebook для Google Colab.
- `tests/test_leak_free_pipeline.py` — автоматические проверки временных границ.
- `docs/Сопроводительная записка НИР Сергеева АВ VK Ads расширенная.docx` — финальная сопроводительная записка.
- `docs/НИР VK Ads презентация Сергеева АВ.pptx` — презентация по работе.
- `figures/*.png` — EDA, сравнение моделей, абляции, predicted-vs-actual и PCA user representations.
- `generate_project_figures.py` — воспроизводимая генерация всех графиков из данных и outputs.
- `run_defense_diagnostics.py` — post-hoc диагностика финальной модели: сегменты, калибровка, target-level errors.
- `reports/controversial_questions_and_answers.md` — спорные вопросы к защите и короткие ответы.
- `reports/defense_diagnostics_log.md` — лог дополнительных diagnostics на locked final holdout.
- `reports/segment_residual_experiment_log.md` — лог дополнительного residual-эксперимента.
- `reports/probabilistic_exposure_experiment_log.md` — лог user-level simulator ablation.
- `reports/hazard_feature_residual_log.md` — лог residual-модели с simulator features.
- `reports/selection_model_blend_log.md` — лог финального target-wise selector-бленда.
- `reports/temporal_weighted_residual_log.md` — лог temporal-local residual Ridge.
- `reports/temporal_selection_blend_log.md` — лог selector-бленда с temporal sources.
- `reports/temporal_refinement_v2_ablation.md` — диагностика слишком агрессивного temporal refinement.
- `reports/advance_like_mlp_baseline_log.md` — лог нейросетевого baseline по мотивам AdVance.

## Строгий протокол

Validation делится полными группами `hour_start`:

| Зона | Назначение | Строк |
|---|---|---:|
| Development | ранняя проверка bias и устойчивости | 389 |
| Calibration | диагностика лагов и весов | 121 |
| Pretest | выбор итоговой конфигурации до test | 643 |
| Final holdout | однократная итоговая оценка | 201 |

Границы: `calibration_start=1107`, `test_start=1257`. В development и pretest допускаются только кампании, для которых соответственно выполнено `hour_end < calibration_start` и `hour_end < test_start`.

`select_strict_model.py` не оценивает final holdout. Итоговая конфигурация выбирается
по `pretest_metric_percent`: это все кампании, чьи target-окна полностью завершены
до `test_start`. Скрипт сохраняет:

- `outputs/strict_model_lock.json`;
- `outputs/strict_locked_predictions.tsv`;
- SHA-256 прогноза;
- таблицу кандидатов с calibration- и pretest-метриками.

`evaluate_locked_holdout.py` проверяет SHA-256 и только после этого вычисляет final-метрику.

## Зафиксированная модель

- monthly: одно прошлое окно с выравниванием по 31 дню;
- daily: геометрическое среднее 8 прошлых окон;
- weekly: геометрическое среднее до 5 прошлых окон;
- веса monthly/daily/weekly: `0.05 / 0.40 / 0.55`;
- дополнительный bias: отключён.

Каждое source-окно полностью заканчивается до forecast cutoff. Внутри окна моделируются CPM-аукцион, шестичасовые пользовательские сессии и Poisson-binomial агрегация `P(1+)`, `P(2+)`, `P(3+)`.

## Метрики

| Этап | Метрика, % |
|---|---:|
| Calibration diagnostic | 8.03 |
| Pretest selection | 8.17 |
| **Locked final temporal holdout, 201 строка** | **9.54** |

Метрики выше относятся к базовой зафиксированной decomposed replay-модели.
Итоговая enhanced-модель добавляет только leak-free residual-слой, выбранный
на pre-final selection split. Лучший итоговый результат работы:
`9.29%` на locked final temporal holdout (`9.289324%` без округления).

Бустинг на агрегированных признаках также проверялся как ML-baseline. В воспроизводимой
версии используется `HistGradientBoostingRegressor` из sklearn: CatBoost/LightGBM можно
подставить как аналог при наличии библиотек. На строгом final holdout бустинг оказался
хуже decomposed replay: `10.94%` для варианта с replay-признаками и `30.89%` для
варианта только со static/history features.

Дополнительно проверен второй слой поверх replay: сегментная log-residual calibration
и регуляризованный Ridge residual с shrinkage. Конфигурации выбираются без доступа к
final holdout, затем выбранный слой refit-ится на pretest-строках, завершенных до
`test_start`.

| Дополнительная модель | Selection, % | Final holdout, % |
|---|---:|---:|
| Base decomposed replay | 7.96 | 9.54 |
| Segment residual calibration + shrinkage | 7.96 | 9.51 |
| Ridge residual shrinkage | 7.94 | 9.47 |
| Target-wise Ridge residual shrinkage | target-wise selection | 9.46 |
| Probabilistic exposure simulator blend | 7.96 | 9.54 |
| Hazard-feature Ridge residual | 7.95 | 9.47 |
| Selection-only target blend | 7.91 | 9.46 |
| Temporal-weighted Ridge residual | 7.93 | 9.47 |
| **Temporal-weighted target-wise Ridge residual** | target-wise selection | **9.29** |
| Temporal selection-only blend | 7.85 | 9.29 |
| Over-aggressive temporal refinement v2 | diagnostic only | 9.30 |

Ключевое улучшение дал temporal-local residual layer: строки, ближайшие к
прогнозному cutoff, получают больший вес при обучении residual-коррекции.
Это отражает временной дрейф рекламного инвентаря и активности пользователей.
Для `1+`, `2+`, `3+` конфигурации выбираются отдельно на pre-final selection
split, затем refit-ятся на всех pretest-строках, завершенных до `test_start`.

Unrounded final для лучшей модели: `9.289324%`. Дополнительный temporal
selection blend снизил selection до `7.853684%`, но на final дал `9.289694%`,
то есть не улучшил raw holdout относительно самой temporal target-wise модели.
Более агрессивный refinement с `half_life=24h` и высокой correction ухудшил
final до `9.30%`, поэтому оставлен как диагностический ablation против overfit.

Отдельно проверенный user-level simulator как самостоятельный прогноз оказался
слишком грубым: даже малая примесь hazard-прогноза ухудшала selection-метрику,
поэтому в финальной схеме он используется только как слабый диагностический
feature block, а не как основная модель.

Для сравнения с нейросетевым индустриальным стилем также реализован
AdVance-inspired multi-task MLP residual baseline. Это не воспроизведение
AdVance: в открытом VK Ads датасете нет click sequence, creative/item ids,
full RTB auction graph, fatigue vector и других входов, необходимых статье.
Честный аналог на доступных данных — общий encoder по campaign/replay/user
признакам и три target-specific residual heads. На locked final holdout он
получил `9.52%`: лучше base replay `9.54%`, но хуже регуляризованного
temporal-weighted target-wise Ridge residual `9.29%`.
