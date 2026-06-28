from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.vk_ads_solution import (
    EPSILON,
    PastOnlyEnsembleConfig,
    TARGET_COLUMNS,
    apply_log_bias,
    enforce_target_constraints,
    geometric_prediction_blend,
    load_dataset,
    median_log_bias,
    parse_int_list,
    predict_past_only_ensemble,
    purged_three_way_split,
    save_predictions,
    smoothed_mean_log_accuracy_ratio,
)


DEFAULT_DATA_DIR = Path("/Users/anastasiasergeeva/Desktop/HSE/Сессия 2026/НИР Vk Ads/data")
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
REPORT_DIR = PROJECT_DIR / "reports"


@dataclass(frozen=True)
class ExposureSimulatorConfig:
    rate_smoothing_weeks: float
    publisher_smoothing: float
    hour_smoothing: float
    cpm_smoothing: float
    session_power: float
    min_user_rate: float = 1e-5


@dataclass
class FittedExposureSimulator:
    user_ids: np.ndarray
    user_to_row: dict[int, int]
    publisher_values: np.ndarray
    publisher_to_col: dict[int, int]
    history_weeks: float
    user_event_count: np.ndarray
    user_session_count: np.ndarray
    user_pub_count: np.ndarray
    user_how_count: np.ndarray
    global_pub_share: np.ndarray
    global_how_share: np.ndarray
    global_weekly_rate: float
    global_cpm_sorted: np.ndarray
    publisher_cpm_sorted: dict[int, np.ndarray]


def metric_raw(answers: pd.DataFrame, predictions: pd.DataFrame) -> float:
    values = []
    for target in TARGET_COLUMNS:
        values.append(
            np.abs(
                np.log(
                    (predictions[target].to_numpy(dtype=float) + EPSILON)
                    / (answers[target].to_numpy(dtype=float) + EPSILON)
                )
            ).mean()
        )
    return float(100.0 * (np.exp(np.mean(values)) - 1.0))


def score(answers: pd.DataFrame, predictions: pd.DataFrame, rows: np.ndarray) -> float:
    return smoothed_mean_log_accuracy_ratio(
        answers.iloc[rows].reset_index(drop=True),
        predictions.iloc[rows].reset_index(drop=True),
    )


def count_hour_of_week_occurrences(hour_start: int, hour_end: int) -> np.ndarray:
    hours = np.arange(int(hour_start), int(hour_end) + 1, dtype=np.int32)
    return np.bincount(hours % 168, minlength=168).astype(np.float32)


def at_least_poisson(lambda_values: np.ndarray) -> np.ndarray:
    lambda_values = np.clip(lambda_values.astype(np.float64), 0.0, 50.0)
    p0 = np.exp(-lambda_values)
    p1 = p0 * lambda_values
    p2 = p1 * lambda_values / 2.0
    return np.column_stack(
        [
            1.0 - p0,
            1.0 - p0 - p1,
            1.0 - p0 - p1 - p2,
        ]
    ).clip(0.0, 1.0)


def empirical_win_probability(sorted_cpm: np.ndarray, cpm: float) -> float:
    if len(sorted_cpm) == 0:
        return 0.0
    left = int(np.searchsorted(sorted_cpm, cpm, side="left"))
    right = int(np.searchsorted(sorted_cpm, cpm, side="right"))
    return float((left + 0.5 * (right - left)) / len(sorted_cpm))


def fit_exposure_simulator(users: pd.DataFrame, history: pd.DataFrame) -> FittedExposureSimulator:
    users = users.sort_values("user_id").reset_index(drop=True)
    user_ids = users["user_id"].to_numpy(dtype=np.int64)
    user_to_row = {int(user_id): i for i, user_id in enumerate(user_ids)}
    row_id = pd.Series(history["user_id"]).map(user_to_row).to_numpy()
    keep = pd.notna(row_id)
    row_id = row_id[keep].astype(np.int32)
    hist = history.loc[keep].copy()

    publisher_values = np.sort(hist["publisher"].unique()).astype(np.int16)
    publisher_to_col = {int(value): i for i, value in enumerate(publisher_values)}
    publisher_col = hist["publisher"].map(publisher_to_col).to_numpy(dtype=np.int16)
    how = (hist["hour"].to_numpy(dtype=np.int32) % 168).astype(np.int16)

    n_users = len(user_ids)
    n_publishers = len(publisher_values)
    user_event_count = np.bincount(row_id, minlength=n_users).astype(np.float32)
    user_pub_count = np.zeros((n_users, n_publishers), dtype=np.float32)
    user_how_count = np.zeros((n_users, 168), dtype=np.float32)
    np.add.at(user_pub_count, (row_id, publisher_col), 1.0)
    np.add.at(user_how_count, (row_id, how), 1.0)

    hist_sorted = hist.assign(_row_id=row_id).sort_values(["_row_id", "hour"])
    hour_gap = hist_sorted.groupby("_row_id")["hour"].diff().fillna(10**9)
    new_session = (hour_gap >= 6).to_numpy(dtype=bool)
    session_rows = hist_sorted["_row_id"].to_numpy(dtype=np.int32)[new_session]
    user_session_count = np.bincount(session_rows, minlength=n_users).astype(np.float32)

    total_events = max(float(len(hist)), 1.0)
    global_pub_share = np.bincount(
        publisher_col, minlength=n_publishers
    ).astype(np.float64) / total_events
    global_how_share = np.bincount(how, minlength=168).astype(np.float64) / total_events
    history_weeks = max(
        (float(history["hour"].max()) - float(history["hour"].min()) + 1.0) / 168.0,
        1.0,
    )
    global_weekly_rate = total_events / (n_users * history_weeks)

    global_cpm_sorted = np.sort(hist["cpm"].to_numpy(dtype=float))
    publisher_cpm_sorted = {
        int(publisher): np.sort(hist.loc[hist["publisher"] == publisher, "cpm"].to_numpy(dtype=float))
        for publisher in publisher_values
    }
    return FittedExposureSimulator(
        user_ids=user_ids,
        user_to_row=user_to_row,
        publisher_values=publisher_values,
        publisher_to_col=publisher_to_col,
        history_weeks=history_weeks,
        user_event_count=user_event_count,
        user_session_count=user_session_count,
        user_pub_count=user_pub_count,
        user_how_count=user_how_count,
        global_pub_share=global_pub_share,
        global_how_share=global_how_share,
        global_weekly_rate=float(global_weekly_rate),
        global_cpm_sorted=global_cpm_sorted,
        publisher_cpm_sorted=publisher_cpm_sorted,
    )


def predict_exposure_simulator(
    fitted: FittedExposureSimulator,
    campaigns: pd.DataFrame,
    config: ExposureSimulatorConfig,
) -> pd.DataFrame:
    predictions = []
    user_count = fitted.user_event_count
    weekly_rate = (
        user_count + config.rate_smoothing_weeks * fitted.global_weekly_rate
    ) / (fitted.history_weeks + config.rate_smoothing_weeks)
    weekly_rate = np.maximum(weekly_rate, config.min_user_rate)
    session_ratio = (fitted.user_session_count + 1.0) / (fitted.user_event_count + 1.0)
    session_ratio = np.clip(session_ratio, 0.05, 1.0).astype(np.float32)

    for campaign in campaigns.itertuples(index=False):
        audience = parse_int_list(campaign.user_ids)
        audience_rows = np.array(
            [fitted.user_to_row.get(int(user_id), -1) for user_id in audience], dtype=np.int32
        )
        valid = audience_rows >= 0
        if not bool(valid.any()):
            predictions.append((0.0, 0.0, 0.0))
            continue
        rows = audience_rows[valid]
        event_count = user_count[rows]

        publisher_cols = [
            fitted.publisher_to_col[int(publisher)]
            for publisher in parse_int_list(campaign.publishers)
            if int(publisher) in fitted.publisher_to_col
        ]
        if not publisher_cols:
            predictions.append((0.0, 0.0, 0.0))
            continue

        win = np.zeros(len(fitted.publisher_values), dtype=np.float64)
        global_win = empirical_win_probability(fitted.global_cpm_sorted, float(campaign.cpm))
        for col in publisher_cols:
            publisher = int(fitted.publisher_values[col])
            pub_cpm = fitted.publisher_cpm_sorted[publisher]
            pub_win = empirical_win_probability(pub_cpm, float(campaign.cpm))
            pub_weight = len(pub_cpm) / (len(pub_cpm) + config.cpm_smoothing)
            win[col] = pub_weight * pub_win + (1.0 - pub_weight) * global_win

        pub_numerator = fitted.user_pub_count[rows] @ win
        global_pub_numerator = config.publisher_smoothing * float(fitted.global_pub_share @ win)
        pub_component = (pub_numerator + global_pub_numerator) / (
            event_count + config.publisher_smoothing
        )

        how_occurrences = count_hour_of_week_occurrences(campaign.hour_start, campaign.hour_end)
        how_numerator = fitted.user_how_count[rows] @ how_occurrences
        global_how_numerator = config.hour_smoothing * float(
            fitted.global_how_share @ how_occurrences
        )
        how_component = (how_numerator + global_how_numerator) / (
            event_count + config.hour_smoothing
        )

        lambda_events = weekly_rate[rows] * pub_component * how_component
        lambda_sessions = lambda_events * np.power(session_ratio[rows], config.session_power)
        user_probs = at_least_poisson(lambda_sessions)
        summed = user_probs.sum(axis=0)
        denominator = max(int(campaign.audience_size), 1)
        predictions.append(tuple((summed / denominator).clip(0.0, 1.0).tolist()))

    return enforce_target_constraints(pd.DataFrame(predictions, columns=TARGET_COLUMNS))


def config_grid() -> list[ExposureSimulatorConfig]:
    configs = []
    for rate_smoothing_weeks in (0.5, 1.5, 3.0):
        for publisher_smoothing in (25.0, 100.0):
            for hour_smoothing in (25.0, 100.0):
                for cpm_smoothing in (0.0, 1000.0):
                    for session_power in (0.5, 1.0):
                        configs.append(
                            ExposureSimulatorConfig(
                                rate_smoothing_weeks=rate_smoothing_weeks,
                                publisher_smoothing=publisher_smoothing,
                                hour_smoothing=hour_smoothing,
                                cpm_smoothing=cpm_smoothing,
                                session_power=session_power,
                            )
                        )
    return configs


def main() -> None:
    data_dir = Path(os.environ.get("VK_ADS_DATA_DIR", DEFAULT_DATA_DIR))
    users, history, campaigns, answers = load_dataset(data_dir)
    if len(answers) != len(campaigns):
        raise RuntimeError("Exposure simulator experiment requires validate_answers.tsv.")

    split = purged_three_way_split(
        campaigns,
        development_fraction=0.6,
        calibration_fraction=0.2,
    )
    forecast_cutoff = int(campaigns["hour_start"].min())
    past_history = history.loc[history["hour"] < forecast_cutoff].copy()
    fitted = fit_exposure_simulator(users, past_history)

    base_predictions, _, _ = predict_past_only_ensemble(
        past_history,
        campaigns,
        config=PastOnlyEnsembleConfig(
            daily_lags=8,
            weekly_lags=5,
            monthly_lags=1,
            daily_weight=0.40,
            weekly_weight=0.55,
            monthly_weight=0.05,
        ),
    )
    base_predictions = base_predictions.reset_index(drop=True)

    train_idx = split.development_idx
    selection_idx = np.setdiff1d(split.pretest_idx, train_idx, assume_unique=False)
    final_train_idx = split.pretest_idx
    final_idx = split.test_idx

    rows = []
    best = None
    best_metric = float("inf")
    blend_weights = (0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.35, 0.50, 0.75, 1.0)
    configs = config_grid()
    for config_id, config in enumerate(configs, start=1):
        if config_id == 1 or config_id % 5 == 0 or config_id == len(configs):
            print(f"Evaluating simulator config {config_id}/{len(configs)}", flush=True)
        hazard = predict_exposure_simulator(fitted, campaigns, config)
        bias = median_log_bias(
            answers.iloc[train_idx].reset_index(drop=True),
            hazard.iloc[train_idx].reset_index(drop=True),
        )
        hazard_calibrated = apply_log_bias(hazard, bias)
        for hazard_weight in blend_weights:
            if hazard_weight == 1.0:
                prediction = hazard_calibrated
            elif hazard_weight == 0.0:
                prediction = base_predictions
            else:
                prediction = geometric_prediction_blend(
                    {"base": base_predictions, "hazard": hazard_calibrated},
                    {"base": 1.0 - hazard_weight, "hazard": hazard_weight},
                )
            selection_prediction = prediction.iloc[selection_idx].reset_index(drop=True)
            selection_answers = answers.iloc[selection_idx].reset_index(drop=True)
            raw_metric = metric_raw(selection_answers, selection_prediction)
            rounded_metric = smoothed_mean_log_accuracy_ratio(
                selection_answers,
                selection_prediction,
            )
            rows.append(
                {
                    "rate_smoothing_weeks": config.rate_smoothing_weeks,
                    "publisher_smoothing": config.publisher_smoothing,
                    "hour_smoothing": config.hour_smoothing,
                    "cpm_smoothing": config.cpm_smoothing,
                    "session_power": config.session_power,
                    "hazard_blend_weight": hazard_weight,
                    "selection_metric_percent": rounded_metric,
                    "selection_metric_raw_percent": raw_metric,
                }
            )
            if raw_metric < best_metric:
                best_metric = raw_metric
                best = (config, hazard_weight)

    if best is None:
        raise RuntimeError("No simulator config was selected.")

    best_config, best_hazard_weight = best
    best_hazard = predict_exposure_simulator(fitted, campaigns, best_config)
    final_bias = median_log_bias(
        answers.iloc[final_train_idx].reset_index(drop=True),
        best_hazard.iloc[final_train_idx].reset_index(drop=True),
    )
    best_hazard_calibrated = apply_log_bias(best_hazard, final_bias)
    if best_hazard_weight == 1.0:
        final_prediction_all = best_hazard_calibrated
    elif best_hazard_weight == 0.0:
        final_prediction_all = base_predictions
    else:
        final_prediction_all = geometric_prediction_blend(
            {"base": base_predictions, "hazard": best_hazard_calibrated},
            {"base": 1.0 - best_hazard_weight, "hazard": best_hazard_weight},
        )

    metrics = pd.DataFrame(
        [
            {
                "model": "base_decomposed_replay",
                "selection_metric_percent": score(answers, base_predictions, selection_idx),
                "final_holdout_metric_percent": score(answers, base_predictions, final_idx),
            },
            {
                "model": "probabilistic_exposure_simulator_blend",
                "selection_metric_percent": float(
                    smoothed_mean_log_accuracy_ratio(
                        answers.iloc[selection_idx].reset_index(drop=True),
                        final_prediction_all.iloc[selection_idx].reset_index(drop=True),
                    )
                ),
                "final_holdout_metric_percent": score(answers, final_prediction_all, final_idx),
            },
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.DataFrame(rows).sort_values(
        ["selection_metric_raw_percent", "selection_metric_percent"],
        kind="stable",
    )
    candidates.to_csv(OUTPUT_DIR / "probabilistic_exposure_candidate_selection.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "probabilistic_exposure_metrics.csv", index=False)
    save_predictions(
        final_prediction_all.round(6),
        OUTPUT_DIR / "validation_predictions_probabilistic_exposure.tsv",
    )
    manifest = {
        "protocol": "development_bias_selection_refit_pretest_locked_final_evaluation",
        "selected_config": {
            "rate_smoothing_weeks": best_config.rate_smoothing_weeks,
            "publisher_smoothing": best_config.publisher_smoothing,
            "hour_smoothing": best_config.hour_smoothing,
            "cpm_smoothing": best_config.cpm_smoothing,
            "session_power": best_config.session_power,
            "hazard_blend_weight": best_hazard_weight,
        },
        "metrics": metrics.to_dict(orient="records"),
        "interpretation": (
            "A user-level hazard simulator was fitted only on pre-forecast history. "
            "It models user activity by publisher/hour-of-week, empirical CPM win curves "
            "and a Poisson session-level exposure aggregation."
        ),
    }
    (OUTPUT_DIR / "probabilistic_exposure_config.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    report = [
        "# Probabilistic user-level exposure simulator",
        "",
        "This experiment tries a deeper mechanistic decomposition than replay:",
        "user activity rate, publisher preference, hour-of-week profile, empirical CPM",
        "win curve and Poisson aggregation to 1+/2+/3+.",
        "",
        "## Leak-free protocol",
        "",
        "- Fit simulator statistics only on history before the forecast cutoff.",
        "- Fit log-bias on development rows.",
        "- Select simulator/blend hyperparameters on later pretest rows.",
        "- Refit log-bias on all pretest rows before final holdout evaluation.",
        "",
        "## Selected config",
        "",
        json.dumps(manifest["selected_config"], indent=2),
        "",
        "## Metrics",
        "",
        metrics.to_markdown(index=False),
    ]
    (REPORT_DIR / "probabilistic_exposure_experiment_log.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(metrics.to_string(index=False))
    print("\nSelected config:")
    print(json.dumps(manifest["selected_config"], indent=2))
    print("\nTop candidates:")
    print(candidates.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
