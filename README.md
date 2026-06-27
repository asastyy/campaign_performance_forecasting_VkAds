# VK Ads leak-free reach/frequency forecasting

Код курсовой работы по прогнозированию долей пользователей с 1+, 2+ и 3+ показами объявления.

## Что входит в репозиторий

- воспроизводимый Python-код модели и экспериментов;
- Colab notebook для запуска решения в облаке;
- таблицы метрик, предсказаний и model lock;
- готовые графики для презентации и защиты в папке `figures/`;
- тесты на отсутствие временных утечек;
- сопроводительная записка и презентация в папке `docs/`.

Сырые файлы датасета (`history.tsv`, `users.tsv`, `validate.tsv`, `validate_answers.tsv`) в репозиторий не включены. Их нужно скачать отдельно и перед запуском указать путь через `VK_ADS_DATA_DIR`.

## Основные файлы

- `src/vk_ads_solution.py` — sessionization, past-only auction replay, temporal splits, агрегаты и индикаторы.
- `select_strict_model.py` — выбор конфигурации только по кампаниям, завершённым до final holdout.
- `evaluate_locked_holdout.py` — отдельная оценка зафиксированного final holdout.
- `predict_future.py` — прогноз кампаний после конца доступной истории; файл ответов не читается.
- `run_experiments.py` — дополнительные past-only и purged OOF diagnostics.
- `run_boosting_baseline.py` — бустинг на агрегированных признаках как ML-baseline.
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

Метрики на всём открытом validation относятся только к development diagnostics. Основная offline-оценка работы — `9.54%` на locked final holdout.

Бустинг на агрегированных признаках также проверялся как ML-baseline. В воспроизводимой
версии используется `HistGradientBoostingRegressor` из sklearn: CatBoost/LightGBM можно
подставить как аналог при наличии библиотек. На строгом final holdout бустинг оказался
хуже decomposed replay: `10.94%` для варианта с replay-признаками и `30.89%` для
варианта только со static/history features.

## Запуск

```bash
export VK_ADS_DATA_DIR="/path/to/data"

# 1. Выбор и фиксация модели без final answers
python select_strict_model.py

# 2. Однократная проверка зафиксированного holdout
python evaluate_locked_holdout.py

# 3. Автоматические проверки утечек
pytest -q tests/test_leak_free_pipeline.py

# 4. Прогноз реального будущего test
python predict_future.py future_campaigns.tsv predictions.tsv

# 5. Генерация графиков для презентации
python generate_project_figures.py

# 6. Post-hoc diagnostics для защиты
python run_defense_diagnostics.py
```

Для `predict_future.py` требуется, чтобы все `hour_start` будущих кампаний были строго больше максимального `hour` в `history.tsv`.
