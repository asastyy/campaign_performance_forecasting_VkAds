from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.vk_ads_solution import (
    AuctionReplayForecaster,
    TARGET_COLUMNS,
    TargetBlendConfig,
    apply_log_bias,
    geometric_prediction_blend,
    load_dataset,
    median_log_bias,
    purged_three_way_split,
    save_predictions,
    smoothed_mean_log_accuracy_ratio,
    targetwise_geometric_prediction_blend,
)


DEFAULT_DATA_DIR = Path("/Users/anastasiasergeeva/Desktop/HSE/Сессия 2026/НИР Vk Ads/data")
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
PREDICTION_PATH = OUTPUT_DIR / "strict_locked_predictions.tsv"
LOCK_PATH = OUTPUT_DIR / "strict_model_lock.json"


def score(answers: pd.DataFrame, predictions: pd.DataFrame) -> float:
    return smoothed_mean_log_accuracy_ratio(
        answers.reset_index(drop=True), predictions.reset_index(drop=True)
    )


def target_log_error(
    answers: pd.DataFrame,
    predictions: pd.DataFrame,
    target: str,
    epsilon: float = 0.005,
) -> float:
    return float(
        np.abs(
            np.log(
                (predictions[target].to_numpy(dtype=float) + epsilon)
                / (answers[target].to_numpy(dtype=float) + epsilon)
            )
        ).mean()
    )


def config_to_dict(config: TargetBlendConfig) -> dict[str, float | int]:
    return {
        "daily_lags": int(config.daily_lags),
        "weekly_lags": int(config.weekly_lags),
        "monthly_lags": int(config.monthly_lags),
        "daily_weight": float(config.daily_weight),
        "weekly_weight": float(config.weekly_weight),
        "monthly_weight": float(config.monthly_weight),
    }


def config_from_dict(values: dict[str, float | int]) -> TargetBlendConfig:
    return TargetBlendConfig(
        daily_lags=int(values["daily_lags"]),
        weekly_lags=int(values["weekly_lags"]),
        monthly_lags=int(values.get("monthly_lags", 1)),
        daily_weight=float(values["daily_weight"]),
        weekly_weight=float(values["weekly_weight"]),
        monthly_weight=float(values["monthly_weight"]),
    )


def blend_single_target(
    monthly: pd.DataFrame,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    target: str,
    monthly_weight: float,
    daily_weight: float,
    weekly_weight: float,
    epsilon: float = 0.005,
) -> pd.DataFrame:
    weights = {
        "monthly": float(monthly_weight),
        "daily": float(daily_weight),
        "weekly": float(weekly_weight),
    }
    total_weight = sum(weights.values())
    if total_weight <= 0.0 or any(weight < 0.0 for weight in weights.values()):
        raise ValueError("Weights must be non-negative and sum to a positive value.")

    blended_log = (
        weights["monthly"] / total_weight * np.log(monthly[target].to_numpy(float) + epsilon)
        + weights["daily"] / total_weight * np.log(daily[target].to_numpy(float) + epsilon)
        + weights["weekly"] / total_weight * np.log(weekly[target].to_numpy(float) + epsilon)
    )
    values = np.clip(np.exp(blended_log) - epsilon, 0.0, 1.0)
    return pd.DataFrame({target: values})


def main() -> None:
    data_dir = Path(os.environ.get("VK_ADS_DATA_DIR", DEFAULT_DATA_DIR))
    _, history, campaigns, all_answers = load_dataset(data_dir)
    split = purged_three_way_split(
        campaigns, development_fraction=0.6, calibration_fraction=0.2
    )

    forecast_cutoff = int(campaigns["hour_start"].min())
    past_history = history.loc[history["hour"] < forecast_cutoff].copy()
    history_end = int(past_history["hour"].max())
    forecaster = AuctionReplayForecaster(session_gap_hours=6).fit(past_history)

    monthly = forecaster.predict_past_ensemble(
        campaigns, history_end, alignment_hours=24 * 31, max_lags=1
    )
    daily_bank = {
        lags: forecaster.predict_past_ensemble(
            campaigns, history_end, alignment_hours=24, max_lags=lags
        )
        for lags in range(1, 11)
    }
    weekly_bank = {
        lags: forecaster.predict_past_ensemble(
            campaigns, history_end, alignment_hours=24 * 7, max_lags=lags
        )
        for lags in range(1, 7)
    }
    component_bank = {"monthly_1": monthly}
    component_bank.update({f"daily_{lags}": prediction for lags, prediction in daily_bank.items()})
    component_bank.update({f"weekly_{lags}": prediction for lags, prediction in weekly_bank.items()})

    development_answers = all_answers.iloc[split.development_idx]
    calibration_answers = all_answers.iloc[split.calibration_idx]
    candidate_predictions: dict[str, pd.DataFrame] = {}
    selection_rows = []

    for daily_lags, daily_prediction in daily_bank.items():
        for weekly_lags, weekly_prediction in weekly_bank.items():
            for monthly_weight in (0.0, 0.05, 0.1, 0.15, 0.2):
                remaining_weight = 1.0 - monthly_weight
                for daily_weight in np.round(
                    np.arange(0.0, remaining_weight + 1e-9, 0.05), 2
                ):
                    weekly_weight = round(remaining_weight - float(daily_weight), 2)
                    if daily_weight == 0.0 and weekly_weight == 0.0:
                        continue
                    candidate_id = (
                        f"scalar_d{daily_lags}_w{weekly_lags}_"
                        f"weights_{monthly_weight:.2f}_{float(daily_weight):.2f}_{weekly_weight:.2f}"
                    )
                    blend = geometric_prediction_blend(
                        {
                            "monthly": monthly,
                            "daily": daily_prediction,
                            "weekly": weekly_prediction,
                        },
                        {
                            "monthly": monthly_weight,
                            "daily": float(daily_weight),
                            "weekly": weekly_weight,
                        },
                    )
                    candidate_predictions[candidate_id] = blend
                    development_prediction = blend.iloc[split.development_idx]
                    calibration_prediction = blend.iloc[split.calibration_idx]
                    development_bias = median_log_bias(
                        development_answers, development_prediction
                    )
                    calibrated = apply_log_bias(calibration_prediction, development_bias)
                    pretest_prediction = blend.iloc[split.pretest_idx]
                    pretest_calibrated = apply_log_bias(pretest_prediction, development_bias)

                    selection_rows.append(
                        {
                            "candidate_id": candidate_id,
                            "model_family": "scalar_past_only_blend",
                            "target_configs": "",
                            "daily_lags": daily_lags,
                            "weekly_lags": weekly_lags,
                            "monthly_weight": monthly_weight,
                            "daily_weight": float(daily_weight),
                            "weekly_weight": weekly_weight,
                            "use_bias": False,
                            "calibration_metric_percent": score(
                                calibration_answers, calibration_prediction
                            ),
                            "pretest_metric_percent": score(
                                all_answers.iloc[split.pretest_idx], pretest_prediction
                            ),
                        }
                    )
                    selection_rows.append(
                        {
                            "candidate_id": candidate_id,
                            "model_family": "scalar_past_only_blend",
                            "target_configs": "",
                            "daily_lags": daily_lags,
                            "weekly_lags": weekly_lags,
                            "monthly_weight": monthly_weight,
                            "daily_weight": float(daily_weight),
                            "weekly_weight": weekly_weight,
                            "use_bias": True,
                            "calibration_metric_percent": score(
                                calibration_answers, calibrated
                            ),
                            "pretest_metric_percent": score(
                                all_answers.iloc[split.pretest_idx], pretest_calibrated
                            ),
                        }
                    )

    weight_grid = []
    for monthly_weight in (0.0, 0.05, 0.1, 0.15, 0.2):
        remaining_weight = 1.0 - monthly_weight
        for daily_weight in np.round(np.arange(0.0, remaining_weight + 1e-9, 0.05), 2):
            weekly_weight = round(remaining_weight - float(daily_weight), 2)
            if daily_weight == 0.0 and weekly_weight == 0.0:
                continue
            weight_grid.append((monthly_weight, float(daily_weight), weekly_weight))

    per_target_top = {}
    for target in TARGET_COLUMNS:
        rows = []
        target_answers = calibration_answers[target].reset_index(drop=True)
        for daily_lags, daily_prediction in daily_bank.items():
            for weekly_lags, weekly_prediction in weekly_bank.items():
                for monthly_weight, daily_weight, weekly_weight in weight_grid:
                    prediction = blend_single_target(
                        monthly,
                        daily_prediction,
                        weekly_prediction,
                        target,
                        monthly_weight,
                        daily_weight,
                        weekly_weight,
                    ).rename(columns={"at_least_one": target})
                    rows.append(
                        {
                            "target": target,
                            "daily_lags": daily_lags,
                            "weekly_lags": weekly_lags,
                            "monthly_lags": 1,
                            "monthly_weight": monthly_weight,
                            "daily_weight": daily_weight,
                            "weekly_weight": weekly_weight,
                            "target_log_error": target_log_error(
                                target_answers.to_frame(),
                                prediction.iloc[split.calibration_idx][[target]].reset_index(
                                    drop=True
                                ),
                                target,
                            ),
                        }
                    )
        per_target_top[target] = (
            pd.DataFrame(rows)
            .sort_values(
                [
                    "target_log_error",
                    "daily_lags",
                    "weekly_lags",
                    "monthly_weight",
                    "daily_weight",
                    "weekly_weight",
                ],
                kind="stable",
            )
            .head(10)
            .reset_index(drop=True)
        )

    targetwise_rows = []
    candidate_number = 0
    for _, first in per_target_top["at_least_one"].iterrows():
        for _, second in per_target_top["at_least_two"].iterrows():
            for _, third in per_target_top["at_least_three"].iterrows():
                configs = {
                    "at_least_one": TargetBlendConfig(
                        daily_lags=int(first["daily_lags"]),
                        weekly_lags=int(first["weekly_lags"]),
                        monthly_lags=1,
                        monthly_weight=float(first["monthly_weight"]),
                        daily_weight=float(first["daily_weight"]),
                        weekly_weight=float(first["weekly_weight"]),
                    ),
                    "at_least_two": TargetBlendConfig(
                        daily_lags=int(second["daily_lags"]),
                        weekly_lags=int(second["weekly_lags"]),
                        monthly_lags=1,
                        monthly_weight=float(second["monthly_weight"]),
                        daily_weight=float(second["daily_weight"]),
                        weekly_weight=float(second["weekly_weight"]),
                    ),
                    "at_least_three": TargetBlendConfig(
                        daily_lags=int(third["daily_lags"]),
                        weekly_lags=int(third["weekly_lags"]),
                        monthly_lags=1,
                        monthly_weight=float(third["monthly_weight"]),
                        daily_weight=float(third["daily_weight"]),
                        weekly_weight=float(third["weekly_weight"]),
                    ),
                }
                prediction = targetwise_geometric_prediction_blend(component_bank, configs)
                calibration_prediction = prediction.iloc[split.calibration_idx]
                development_prediction = prediction.iloc[split.development_idx]
                development_bias = median_log_bias(development_answers, development_prediction)
                calibrated = apply_log_bias(calibration_prediction, development_bias)
                pretest_prediction = prediction.iloc[split.pretest_idx]
                pretest_calibrated = apply_log_bias(pretest_prediction, development_bias)
                candidate_number += 1
                candidate_id = f"targetwise_{candidate_number:05d}"
                target_configs_json = json.dumps(
                    {target: config_to_dict(config) for target, config in configs.items()},
                    sort_keys=True,
                )
                candidate_predictions[candidate_id] = prediction
                targetwise_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "model_family": "targetwise_past_only_blend",
                        "target_configs": target_configs_json,
                        "daily_lags": np.nan,
                        "weekly_lags": np.nan,
                        "monthly_weight": np.nan,
                        "daily_weight": np.nan,
                        "weekly_weight": np.nan,
                        "use_bias": False,
                        "calibration_metric_percent": score(
                            calibration_answers, calibration_prediction
                        ),
                        "pretest_metric_percent": score(
                            all_answers.iloc[split.pretest_idx], pretest_prediction
                        ),
                    }
                )
                targetwise_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "model_family": "targetwise_past_only_blend",
                        "target_configs": target_configs_json,
                        "daily_lags": np.nan,
                        "weekly_lags": np.nan,
                        "monthly_weight": np.nan,
                        "daily_weight": np.nan,
                        "weekly_weight": np.nan,
                        "use_bias": True,
                        "calibration_metric_percent": score(
                            calibration_answers, calibrated
                        ),
                        "pretest_metric_percent": score(
                            all_answers.iloc[split.pretest_idx], pretest_calibrated
                        ),
                    }
                )
    selection_rows.extend(targetwise_rows)

    selection = pd.DataFrame(selection_rows).sort_values(
        ["pretest_metric_percent", "calibration_metric_percent", "candidate_id", "use_bias"],
        kind="stable",
    ).reset_index(drop=True)
    selected = selection.iloc[0]
    selected_prediction = candidate_predictions[str(selected["candidate_id"])]

    final_bias = pd.Series(0.0, index=TARGET_COLUMNS)
    if bool(selected["use_bias"]):
        final_bias = median_log_bias(
            all_answers.iloc[split.pretest_idx],
            selected_prediction.iloc[split.pretest_idx],
        )
        locked_prediction = apply_log_bias(selected_prediction, final_bias)
    else:
        locked_prediction = selected_prediction

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_predictions(locked_prediction.round(6), PREDICTION_PATH)
    prediction_sha256 = hashlib.sha256(PREDICTION_PATH.read_bytes()).hexdigest()
    selection.to_csv(OUTPUT_DIR / "strict_calibration_selection.csv", index=False)
    pd.concat(per_target_top.values(), axis=0).to_csv(
        OUTPUT_DIR / "strict_targetwise_component_selection.csv", index=False
    )

    zones = pd.DataFrame(index=np.arange(len(campaigns)))
    zones["zone"] = "purged_or_unused"
    zones.loc[split.development_idx, "zone"] = "development"
    zones.loc[split.calibration_idx, "zone"] = "calibration"
    zones.loc[split.test_idx, "zone"] = "final_holdout"
    zones["hour_start"] = campaigns["hour_start"].to_numpy()
    zones["hour_end"] = campaigns["hour_end"].to_numpy()
    zones.to_csv(OUTPUT_DIR / "strict_temporal_zones.csv", index_label="row_position")

    manifest = {
        "protocol": "purged_development_calibration_locked_final_holdout",
        "model_family": str(selected["model_family"]),
        "forecast_cutoff": forecast_cutoff,
        "history_end": history_end,
        "calibration_start": split.calibration_start,
        "test_start": split.test_start,
        "development_rows": int(len(split.development_idx)),
        "calibration_rows": int(len(split.calibration_idx)),
        "pretest_rows": int(len(split.pretest_idx)),
        "final_holdout_rows": int(len(split.test_idx)),
        "selected_candidate": str(selected["candidate_id"]),
        "daily_lags": (
            None if pd.isna(selected["daily_lags"]) else int(selected["daily_lags"])
        ),
        "weekly_lags": (
            None if pd.isna(selected["weekly_lags"]) else int(selected["weekly_lags"])
        ),
        "monthly_weight": (
            None
            if pd.isna(selected["monthly_weight"])
            else float(selected["monthly_weight"])
        ),
        "daily_weight": (
            None if pd.isna(selected["daily_weight"]) else float(selected["daily_weight"])
        ),
        "weekly_weight": (
            None
            if pd.isna(selected["weekly_weight"])
            else float(selected["weekly_weight"])
        ),
        "target_configs": (
            None
            if not str(selected["target_configs"])
            else json.loads(str(selected["target_configs"]))
        ),
        "use_bias": bool(selected["use_bias"]),
        "pretest_bias": {column: float(final_bias[column]) for column in TARGET_COLUMNS},
        "selection_metric": "pretest_metric_percent",
        "calibration_metric_percent": float(selected["calibration_metric_percent"]),
        "pretest_metric_percent": float(selected["pretest_metric_percent"]),
        "prediction_file": PREDICTION_PATH.name,
        "prediction_sha256": prediction_sha256,
        "final_holdout_metric": None,
    }
    LOCK_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Model selection completed without evaluating final holdout answers.")
    print(json.dumps(manifest, indent=2))
    print("Top selection candidates:")
    print(selection.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
