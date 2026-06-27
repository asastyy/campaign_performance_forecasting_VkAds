from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.vk_ads_solution import (
    TARGET_COLUMNS,
    apply_log_bias,
    load_dataset,
    median_log_bias,
    predict_past_only_ensemble,
    purged_temporal_folds,
    save_predictions,
    smoothed_mean_log_accuracy_ratio,
)


DEFAULT_DATA_DIR = Path("/Users/anastasiasergeeva/Desktop/HSE/Сессия 2026/НИР Vk Ads/data")
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"


def score(
    name: str,
    predictions: pd.DataFrame,
    answers: pd.DataFrame,
    rows: np.ndarray | None = None,
    evaluation: str = "full_past_only",
) -> dict[str, object]:
    if rows is None:
        rows = np.arange(len(answers))
    rows = np.asarray(rows, dtype=int)
    selected_predictions = predictions.iloc[rows].reset_index(drop=True)
    selected_answers = answers.iloc[rows].reset_index(drop=True)
    return {
        "model": name,
        "evaluation": evaluation,
        "rows": int(len(rows)),
        "metric_percent": smoothed_mean_log_accuracy_ratio(
            selected_answers, selected_predictions
        ),
        "mean_at_least_one": float(selected_predictions["at_least_one"].mean()),
        "mean_at_least_two": float(selected_predictions["at_least_two"].mean()),
        "mean_at_least_three": float(selected_predictions["at_least_three"].mean()),
    }


def main() -> None:
    data_dir = Path(os.environ.get("VK_ADS_DATA_DIR", DEFAULT_DATA_DIR))
    _, history, validate, answers = load_dataset(data_dir)

    forecast_cutoff = int(validate["hour_start"].min())
    past_history = history.loc[history["hour"] < forecast_cutoff].copy()
    if int(past_history["hour"].max()) >= forecast_cutoff:
        raise RuntimeError("History cutoff failed.")

    blend, components, diagnostics = predict_past_only_ensemble(past_history, validate)
    folds = purged_temporal_folds(validate, n_splits=5, min_train_fraction=0.4)

    oof_calibrated = blend.copy()
    fold_rows = []
    test_blocks = []
    for fold_id, fold in enumerate(folds, start=1):
        bias = median_log_bias(
            answers.iloc[fold.train_idx].reset_index(drop=True),
            blend.iloc[fold.train_idx].reset_index(drop=True),
        )
        oof_calibrated.iloc[fold.test_idx] = apply_log_bias(
            blend.iloc[fold.test_idx].reset_index(drop=True), bias
        ).to_numpy()
        test_blocks.append(fold.test_idx)
        fold_rows.append(
            {
                "fold": fold_id,
                "cutoff_hour": fold.cutoff_hour,
                "train_rows": len(fold.train_idx),
                "test_rows": len(fold.test_idx),
                "max_train_end": int(validate.iloc[fold.train_idx]["hour_end"].max()),
                **{f"bias_{column}": float(bias[column]) for column in TARGET_COLUMNS},
            }
        )

    temporal_rows = np.concatenate(test_blocks)
    metrics = pd.DataFrame(
        [
            score("past_monthly_single", components["monthly"], answers),
            score("past_daily_8_geometric", components["daily"], answers),
            score("past_weekly_5_geometric", components["weekly"], answers),
            score("past_only_geometric_blend", blend, answers),
            score(
                "past_monthly_single",
                components["monthly"],
                answers,
                temporal_rows,
                "purged_temporal_oof_rows",
            ),
            score(
                "past_daily_8_geometric",
                components["daily"],
                answers,
                temporal_rows,
                "purged_temporal_oof_rows",
            ),
            score(
                "past_weekly_5_geometric",
                components["weekly"],
                answers,
                temporal_rows,
                "purged_temporal_oof_rows",
            ),
            score(
                "past_only_geometric_blend",
                blend,
                answers,
                temporal_rows,
                "purged_temporal_oof_rows",
            ),
            score(
                "past_only_blend_past_median_calibration",
                oof_calibrated,
                answers,
                temporal_rows,
                "purged_temporal_oof_rows",
            ),
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_predictions(components["monthly"].round(4), OUTPUT_DIR / "validation_predictions_past_monthly.tsv")
    save_predictions(components["daily"].round(4), OUTPUT_DIR / "validation_predictions_past_daily.tsv")
    save_predictions(components["weekly"].round(4), OUTPUT_DIR / "validation_predictions_past_weekly.tsv")
    save_predictions(blend.round(4), OUTPUT_DIR / "validation_predictions_leak_free_blend.tsv")
    save_predictions(oof_calibrated.round(4), OUTPUT_DIR / "validation_predictions_leak_free_oof.tsv")
    save_predictions(blend.round(4), OUTPUT_DIR / "validation_predictions_final.tsv")
    diagnostics.to_csv(OUTPUT_DIR / "past_window_diagnostics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "purged_temporal_folds.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "leak_free_metrics_summary.csv", index=False)

    print(f"Forecast cutoff: {forecast_cutoff}; past history ends at {int(past_history['hour'].max())}")
    print(pd.DataFrame(fold_rows).to_string(index=False))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
