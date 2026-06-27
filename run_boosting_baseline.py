from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.vk_ads_solution import (
    AuctionReplayForecaster,
    EPSILON,
    PastOnlyEnsembleConfig,
    TARGET_COLUMNS,
    build_campaign_features,
    build_leak_free_campaign_features,
    enforce_target_constraints,
    load_dataset,
    predict_past_only_ensemble,
    purged_three_way_split,
    save_predictions,
    smoothed_mean_log_accuracy_ratio,
)


DEFAULT_DATA_DIR = Path("/Users/anastasiasergeeva/Desktop/HSE/Сессия 2026/НИР Vk Ads/data")
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"


def score(answers: pd.DataFrame, predictions: pd.DataFrame, rows: np.ndarray) -> float:
    return smoothed_mean_log_accuracy_ratio(
        answers.iloc[rows].reset_index(drop=True),
        predictions.iloc[rows].reset_index(drop=True),
    )


def fit_boosting(
    features: pd.DataFrame,
    answers: pd.DataFrame,
    train_idx: np.ndarray,
    predict_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    """Fit one gradient-boosting regressor per target in metric log-space."""

    if predict_idx is None:
        predict_idx = np.arange(len(features))

    x_train = features.iloc[train_idx].to_numpy(dtype=np.float32)
    x_pred = features.iloc[predict_idx].to_numpy(dtype=np.float32)
    result = pd.DataFrame(index=np.arange(len(predict_idx)), columns=TARGET_COLUMNS, dtype=float)

    for target in TARGET_COLUMNS:
        y_train = np.log(answers.iloc[train_idx][target].to_numpy(dtype=float) + EPSILON)
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.035,
            max_iter=120,
            max_leaf_nodes=8,
            min_samples_leaf=25,
            l2_regularization=1.0,
            random_state=42,
        )
        model.fit(x_train, y_train)
        result[target] = np.exp(model.predict(x_pred)) - EPSILON

    return enforce_target_constraints(result)


def main() -> None:
    data_dir = Path(os.environ.get("VK_ADS_DATA_DIR", DEFAULT_DATA_DIR))
    users, history, campaigns, answers = load_dataset(data_dir)
    if len(answers) != len(campaigns):
        raise RuntimeError("Boosting baseline needs validate_answers.tsv for supervised training.")

    split = purged_three_way_split(
        campaigns, development_fraction=0.6, calibration_fraction=0.2
    )
    forecast_cutoff = int(campaigns["hour_start"].min())
    past_history = history.loc[history["hour"] < forecast_cutoff].copy()

    replay_config = PastOnlyEnsembleConfig()
    replay_blend, replay_components, _ = predict_past_only_ensemble(
        past_history, campaigns, config=replay_config
    )
    past_model = AuctionReplayForecaster(session_gap_hours=6).fit(past_history)

    static_features = build_campaign_features(
        campaigns,
        users=users,
        history=past_history,
        extra_predictions=None,
        include_user_embeddings=True,
    )
    decomposition_features = build_leak_free_campaign_features(
        campaigns, users, past_model, source_shift=24 * 31
    )
    replay_features = build_campaign_features(
        campaigns,
        users=users,
        history=past_history,
        extra_predictions={
            "replay_monthly": replay_components["monthly"],
            "replay_daily": replay_components["daily"],
            "replay_weekly": replay_components["weekly"],
            "replay_blend": replay_blend,
        },
        include_user_embeddings=True,
    )
    replay_features = pd.concat(
        [replay_features.reset_index(drop=True), decomposition_features.reset_index(drop=True)],
        axis=1,
    )

    feature_sets = {
        "boosting_static_history": static_features,
        "boosting_with_replay_features": replay_features,
    }
    predictions: dict[str, pd.DataFrame] = {
        "replay_blend_locked": replay_blend.reset_index(drop=True)
    }
    rows = [
        {
            "model": "replay_blend_locked",
            "train_rows": 0,
            "features": 0,
            "calibration_metric_percent": score(answers, replay_blend, split.calibration_idx),
            "pretest_metric_percent": score(answers, replay_blend, split.pretest_idx),
            "final_holdout_metric_percent": score(answers, replay_blend, split.test_idx),
        }
    ]

    for name, features in feature_sets.items():
        features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        features = features.select_dtypes(include=[np.number]).astype(np.float32)

        dev_prediction = fit_boosting(features, answers, split.development_idx)
        pretest_prediction = fit_boosting(features, answers, split.pretest_idx)
        predictions[f"{name}_trained_on_development"] = dev_prediction
        predictions[f"{name}_trained_on_pretest"] = pretest_prediction

        rows.append(
            {
                "model": f"{name}_trained_on_development",
                "train_rows": len(split.development_idx),
                "features": features.shape[1],
                "calibration_metric_percent": score(
                    answers, dev_prediction, split.calibration_idx
                ),
                "pretest_metric_percent": score(answers, dev_prediction, split.pretest_idx),
                "final_holdout_metric_percent": score(
                    answers, dev_prediction, split.test_idx
                ),
            }
        )
        rows.append(
            {
                "model": f"{name}_trained_on_pretest",
                "train_rows": len(split.pretest_idx),
                "features": features.shape[1],
                "calibration_metric_percent": score(
                    answers, pretest_prediction, split.calibration_idx
                ),
                "pretest_metric_percent": score(answers, pretest_prediction, split.pretest_idx),
                "final_holdout_metric_percent": score(
                    answers, pretest_prediction, split.test_idx
                ),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows).sort_values(
        ["final_holdout_metric_percent", "model"], kind="stable"
    )
    metrics.to_csv(OUTPUT_DIR / "boosting_baseline_metrics.csv", index=False)
    for name, prediction in predictions.items():
        save_predictions(
            prediction.round(6),
            OUTPUT_DIR / f"validation_predictions_{name}.tsv",
        )

    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
