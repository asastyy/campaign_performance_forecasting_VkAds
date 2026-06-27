from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf


PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_DIR / "VK_Ads_reach_forecasting_solution.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        markdown(
            """
            # VK Ads: прогноз без data leakage

            Итоговый эксперимент разделён на три временные зоны:

            1. **development** — расчёт устойчивой калибровки;
            2. **calibration** — выбор числа лагов, весов blend и необходимости bias;
            3. **final holdout** — однократная оценка уже зафиксированного прогноза.

            Кампания допускается в train только при `hour_end < следующий cutoff`.
            История для прогнозирования validation обрезается перед первым `hour_start`.
            """
        ),
        code(
            """
            import hashlib
            import json
            import os
            import sys
            from pathlib import Path

            import numpy as np
            import pandas as pd
            from IPython.display import display

            PROJECT_DIR = Path.cwd()
            if not (PROJECT_DIR / "src").exists():
                PROJECT_DIR = Path("/Users/anastasiasergeeva/Documents/New project/vk_ads_coursework")
            DATA_DIR = Path(os.environ.get(
                "VK_ADS_DATA_DIR",
                "/Users/anastasiasergeeva/Desktop/HSE/Сессия 2026/НИР Vk Ads/data",
            ))
            OUTPUT_DIR = PROJECT_DIR / "outputs"
            sys.path.insert(0, str(PROJECT_DIR / "src"))

            from vk_ads_solution import (
                AuctionReplayForecaster,
                build_leak_free_campaign_features,
                load_dataset,
                purged_three_way_split,
                smoothed_mean_log_accuracy_ratio,
            )
            """
        ),
        markdown("## 1. Данные и as-of cutoff"),
        code(
            """
            users, history, validate, answers = load_dataset(DATA_DIR)
            forecast_cutoff = int(validate["hour_start"].min())
            past_history = history.loc[history["hour"] < forecast_cutoff].copy()
            split = purged_three_way_split(
                validate, development_fraction=0.6, calibration_fraction=0.2
            )

            audit = pd.DataFrame([{
                "history_end": int(past_history["hour"].max()),
                "forecast_cutoff": forecast_cutoff,
                "calibration_start": split.calibration_start,
                "test_start": split.test_start,
                "development_rows": len(split.development_idx),
                "calibration_rows": len(split.calibration_idx),
                "pretest_rows": len(split.pretest_idx),
                "final_holdout_rows": len(split.test_idx),
                "max_development_end": int(validate.iloc[split.development_idx]["hour_end"].max()),
                "max_pretest_end": int(validate.iloc[split.pretest_idx]["hour_end"].max()),
            }])
            display(audit)
            assert audit.at[0, "history_end"] < audit.at[0, "forecast_cutoff"]
            assert audit.at[0, "max_development_end"] < audit.at[0, "calibration_start"]
            assert audit.at[0, "max_pretest_end"] < audit.at[0, "test_start"]
            """
        ),
        markdown(
            """
            ## 2. Архитектура

            Для каждого объявления replay выполняется только на полностью доступных прошлых окнах:

            - одно 31-дневное monthly-окно;
            - несколько daily-окон с тем же часом суток;
            - несколько weekly-окон с тем же часом недели.

            В каждом окне применяются правила CPM-аукциона, шестичасовые сессии и
            Poisson-binomial агрегация `P(1+)`, `P(2+)`, `P(3+)`. Число лагов и веса
            геометрического blend выбраны только по кампаниям, полностью завершённым
            до final holdout (`pretest`).
            """
        ),
        markdown("## 3. Проверка зафиксированной модели"),
        code(
            """
            lock_path = OUTPUT_DIR / "strict_model_lock.json"
            prediction_path = OUTPUT_DIR / "strict_locked_predictions.tsv"
            manifest = json.loads(lock_path.read_text(encoding="utf-8"))
            prediction_hash = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
            assert prediction_hash == manifest["prediction_sha256"]
            assert manifest["final_holdout_metric"] is None
            assert manifest["test_start"] == split.test_start

            locked_prediction = pd.read_csv(prediction_path, sep="\t")
            selection = pd.read_csv(OUTPUT_DIR / "strict_calibration_selection.csv")
            display(pd.DataFrame([manifest]))
            display(selection.head(10))
            """
        ),
        markdown(
            """
            В lock-файле нет final-метрики: сначала `select_strict_model.py` записал конфигурацию
            и SHA-256 прогноза. Только после этого отдельный `evaluate_locked_holdout.py` получил
            доступ к ответам final-блока.
            """
        ),
        code(
            """
            final_answers = answers.iloc[split.test_idx].reset_index(drop=True)
            final_prediction = locked_prediction.iloc[split.test_idx].reset_index(drop=True)
            final_metric = smoothed_mean_log_accuracy_ratio(final_answers, final_prediction)

            strict_result = pd.DataFrame([{
                "protocol": manifest["protocol"],
                "selected_candidate": manifest["selected_candidate"],
                "calibration_metric_percent": manifest["calibration_metric_percent"],
                "pretest_metric_percent": manifest["pretest_metric_percent"],
                "final_holdout_rows": len(split.test_idx),
                "final_holdout_metric_percent": final_metric,
            }])
            display(strict_result)
            """
        ),
        markdown("## 4. Декомпозиционные агрегаты и индикаторы"),
        code(
            """
            past_model = AuctionReplayForecaster(session_gap_hours=6).fit(past_history)
            features = build_leak_free_campaign_features(
                validate, users, past_model, source_shift=24 * 31
            )
            print("Feature matrix:", features.shape)
            display(features.filter(regex="missing|zero").sum().sort_values().tail(12))
            """
        ),
        markdown(
            """
            ## Результат

            - Calibration diagnostic: **8.03%**.
            - Selection по pretest: **8.17%**.
            - Locked final temporal holdout: **9.54%** на 201 строке.
            - Финальная конфигурация: `daily=8`, `weekly=5`, веса `0.05 / 0.40 / 0.55`, без bias.
            - Prediction API не читает `validate_answers.tsv` и отклоняет кампании,
              пересекающиеся с доступной историей.

            Это основная честная offline-метрика. Значения на всём открытом validation можно
            использовать только как development diagnostics, но не как независимую оценку качества.
            """
        ),
    ]
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Written: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_notebook()
