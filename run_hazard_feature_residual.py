from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from run_probabilistic_exposure_simulator import (
    ExposureSimulatorConfig,
    fit_exposure_simulator,
    predict_exposure_simulator,
)
from run_segment_residual_calibration import (
    RidgeResidualConfig,
    build_residual_features,
    fit_ridge_residual,
    metric_raw,
    predict_ridge_residual,
    score,
    target_abs_log_error_raw,
)
from src.vk_ads_solution import (
    EPSILON,
    PastOnlyEnsembleConfig,
    TARGET_COLUMNS,
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
REPORT_DIR = PROJECT_DIR / "reports"


def hazard_configurations() -> dict[str, ExposureSimulatorConfig]:
    return {
        "tight_hour": ExposureSimulatorConfig(
            rate_smoothing_weeks=0.5,
            publisher_smoothing=25.0,
            hour_smoothing=100.0,
            cpm_smoothing=1000.0,
            session_power=0.5,
        ),
        "tight_pub": ExposureSimulatorConfig(
            rate_smoothing_weeks=0.5,
            publisher_smoothing=25.0,
            hour_smoothing=25.0,
            cpm_smoothing=0.0,
            session_power=0.5,
        ),
        "smooth": ExposureSimulatorConfig(
            rate_smoothing_weeks=1.5,
            publisher_smoothing=100.0,
            hour_smoothing=100.0,
            cpm_smoothing=1000.0,
            session_power=0.5,
        ),
        "session_strict": ExposureSimulatorConfig(
            rate_smoothing_weeks=3.0,
            publisher_smoothing=100.0,
            hour_smoothing=100.0,
            cpm_smoothing=1000.0,
            session_power=1.0,
        ),
    }


def ridge_grid() -> list[RidgeResidualConfig]:
    configs = []
    for feature_set in ("all", "no_decomp", "static_replay"):
        for alpha in (100.0, 200.0, 300.0, 500.0, 1000.0, 2000.0):
            for correction_weight in (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20):
                for uncertainty_weight in (0.0, 0.25, 0.5, 1.0, 2.0):
                    for correction_clip in (0.08, 0.15, 0.25, 0.40):
                        configs.append(
                            RidgeResidualConfig(
                                alpha=alpha,
                                correction_weight=correction_weight,
                                uncertainty_weight=uncertainty_weight,
                                correction_clip=correction_clip,
                                feature_set=feature_set,
                            )
                        )
    return configs


def add_hazard_features(
    features: pd.DataFrame,
    base_predictions: pd.DataFrame,
    hazard_predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    enriched = features.copy()
    for name, prediction in hazard_predictions.items():
        for target in TARGET_COLUMNS:
            hazard = prediction[target].to_numpy(dtype=float)
            base = base_predictions[target].to_numpy(dtype=float)
            enriched[f"hazard_{name}_{target}"] = hazard.astype(np.float32)
            enriched[f"log_hazard_{name}_{target}"] = np.log(hazard + EPSILON).astype(np.float32)
            enriched[f"log_hazard_gap_{name}_{target}"] = (
                np.log(hazard + EPSILON) - np.log(base + EPSILON)
            ).astype(np.float32)

    for target in TARGET_COLUMNS:
        logs = np.column_stack(
            [
                np.log(prediction[target].to_numpy(dtype=float) + EPSILON)
                for prediction in hazard_predictions.values()
            ]
        )
        enriched[f"hazard_log_mean_{target}"] = logs.mean(axis=1).astype(np.float32)
        enriched[f"hazard_log_std_{target}"] = logs.std(axis=1).astype(np.float32)
    return (
        enriched.replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .select_dtypes(include=[np.number])
        .astype(np.float32)
    )


def fit_predict_full(
    residual_features: pd.DataFrame,
    answers: pd.DataFrame,
    base_predictions: pd.DataFrame,
    components: dict[str, pd.DataFrame],
    train_idx: np.ndarray,
    predict_idx: np.ndarray,
    config: RidgeResidualConfig,
) -> pd.DataFrame:
    fitted = fit_ridge_residual(
        residual_features,
        answers,
        base_predictions,
        components,
        train_idx=train_idx,
        config=config,
    )
    return predict_ridge_residual(
        fitted,
        residual_features,
        base_predictions,
        components,
        predict_idx,
    )


def main() -> None:
    data_dir = Path(os.environ.get("VK_ADS_DATA_DIR", DEFAULT_DATA_DIR))
    users, history, campaigns, answers = load_dataset(data_dir)
    if len(answers) != len(campaigns):
        raise RuntimeError("Hazard feature residual experiment requires validate_answers.tsv.")

    split = purged_three_way_split(
        campaigns,
        development_fraction=0.6,
        calibration_fraction=0.2,
    )
    forecast_cutoff = int(campaigns["hour_start"].min())
    past_history = history.loc[history["hour"] < forecast_cutoff].copy()

    base_predictions, components, _ = predict_past_only_ensemble(
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
    components = {name: value.reset_index(drop=True) for name, value in components.items()}

    fitted_simulator = fit_exposure_simulator(users, past_history)
    hazard_predictions = {}
    for name, config in hazard_configurations().items():
        print(f"Computing hazard feature: {name}", flush=True)
        hazard_predictions[name] = predict_exposure_simulator(
            fitted_simulator,
            campaigns,
            config,
        ).reset_index(drop=True)

    residual_features = build_residual_features(
        campaigns,
        users,
        past_history,
        base_predictions,
        components,
    )
    residual_features = add_hazard_features(
        residual_features,
        base_predictions,
        hazard_predictions,
    )

    train_idx = split.development_idx
    selection_idx = np.setdiff1d(split.pretest_idx, train_idx, assume_unique=False)
    final_train_idx = split.pretest_idx
    final_idx = split.test_idx

    rows = []
    best_config: RidgeResidualConfig | None = None
    best_metric = float("inf")
    configs = ridge_grid()
    for config_id, config in enumerate(configs, start=1):
        if config_id == 1 or config_id % 250 == 0 or config_id == len(configs):
            print(f"Evaluating hazard-feature ridge config {config_id}/{len(configs)}", flush=True)
        prediction = fit_predict_full(
            residual_features,
            answers,
            base_predictions,
            components,
            train_idx,
            selection_idx,
            config,
        )
        selection_answers = answers.iloc[selection_idx].reset_index(drop=True)
        selection_metric_raw = metric_raw(selection_answers, prediction.reset_index(drop=True))
        selection_metric = smoothed_mean_log_accuracy_ratio(
            selection_answers,
            prediction.reset_index(drop=True),
        )
        target_selection_errors = {
            f"selection_{target}_abs_log_error": target_abs_log_error_raw(
                selection_answers,
                prediction.reset_index(drop=True),
                target,
            )
            for target in TARGET_COLUMNS
        }
        rows.append(
            {
                "feature_set": config.feature_set,
                "alpha": config.alpha,
                "correction_weight": config.correction_weight,
                "uncertainty_weight": config.uncertainty_weight,
                "correction_clip": config.correction_clip,
                "train_rows": len(train_idx),
                "selection_rows": len(selection_idx),
                "selection_metric_percent": selection_metric,
                "selection_metric_raw_percent": selection_metric_raw,
                **target_selection_errors,
            }
        )
        if selection_metric_raw < best_metric:
            best_metric = selection_metric_raw
            best_config = config

    if best_config is None:
        raise RuntimeError("No hazard feature ridge config was selected.")

    final_prediction = fit_predict_full(
        residual_features,
        answers,
        base_predictions,
        components,
        final_train_idx,
        final_idx,
        best_config,
    )
    full_prediction = base_predictions.copy()
    full_prediction.loc[final_idx, TARGET_COLUMNS] = final_prediction[TARGET_COLUMNS].to_numpy()
    full_prediction = enforce_target_constraints(full_prediction)

    candidate_table = pd.DataFrame(rows).sort_values(
        ["selection_metric_raw_percent", "selection_metric_percent", "alpha"],
        kind="stable",
    )

    targetwise_configs: dict[str, RidgeResidualConfig] = {}
    for target in TARGET_COLUMNS:
        best_target_row = candidate_table.sort_values(
            [
                f"selection_{target}_abs_log_error",
                "selection_metric_raw_percent",
                "alpha",
            ],
            kind="stable",
        ).iloc[0]
        targetwise_configs[target] = RidgeResidualConfig(
            feature_set=str(best_target_row["feature_set"]),
            alpha=float(best_target_row["alpha"]),
            correction_weight=float(best_target_row["correction_weight"]),
            uncertainty_weight=float(best_target_row["uncertainty_weight"]),
            correction_clip=float(best_target_row["correction_clip"]),
        )

    targetwise_prediction = pd.DataFrame(index=np.arange(len(final_idx)), columns=TARGET_COLUMNS)
    for target, config in targetwise_configs.items():
        one_target_prediction = fit_predict_full(
            residual_features,
            answers,
            base_predictions,
            components,
            final_train_idx,
            final_idx,
            config,
        )
        targetwise_prediction[target] = one_target_prediction[target].to_numpy(dtype=float)
    targetwise_prediction = enforce_target_constraints(targetwise_prediction)

    full_targetwise_prediction = base_predictions.copy()
    full_targetwise_prediction.loc[final_idx, TARGET_COLUMNS] = targetwise_prediction[
        TARGET_COLUMNS
    ].to_numpy()
    full_targetwise_prediction = enforce_target_constraints(full_targetwise_prediction)

    metrics = pd.DataFrame(
        [
            {
                "model": "base_decomposed_replay",
                "selection_metric_percent": score(answers, base_predictions, selection_idx),
                "final_holdout_metric_percent": score(answers, base_predictions, final_idx),
            },
            {
                "model": "hazard_feature_ridge_residual",
                "selection_metric_percent": float(candidate_table.iloc[0]["selection_metric_percent"]),
                "final_holdout_metric_percent": smoothed_mean_log_accuracy_ratio(
                    answers.iloc[final_idx].reset_index(drop=True),
                    final_prediction.reset_index(drop=True),
                ),
            },
            {
                "model": "hazard_feature_targetwise_ridge_residual",
                "selection_metric_percent": np.nan,
                "final_holdout_metric_percent": smoothed_mean_log_accuracy_ratio(
                    answers.iloc[final_idx].reset_index(drop=True),
                    targetwise_prediction.reset_index(drop=True),
                ),
            },
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_table.to_csv(OUTPUT_DIR / "hazard_feature_ridge_candidate_selection.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "hazard_feature_residual_metrics.csv", index=False)
    save_predictions(
        full_prediction.round(6),
        OUTPUT_DIR / "validation_predictions_hazard_feature_ridge.tsv",
    )
    save_predictions(
        full_targetwise_prediction.round(6),
        OUTPUT_DIR / "validation_predictions_hazard_feature_targetwise_ridge.tsv",
    )

    manifest = {
        "protocol": "development_selected_pretest_refit_locked_final_evaluation",
        "hazard_configs": {
            name: {
                "rate_smoothing_weeks": config.rate_smoothing_weeks,
                "publisher_smoothing": config.publisher_smoothing,
                "hour_smoothing": config.hour_smoothing,
                "cpm_smoothing": config.cpm_smoothing,
                "session_power": config.session_power,
            }
            for name, config in hazard_configurations().items()
        },
        "selected_ridge_config": {
            "feature_set": best_config.feature_set,
            "alpha": best_config.alpha,
            "correction_weight": best_config.correction_weight,
            "uncertainty_weight": best_config.uncertainty_weight,
            "correction_clip": best_config.correction_clip,
        },
        "selected_targetwise_configs": {
            target: {
                "feature_set": config.feature_set,
                "alpha": config.alpha,
                "correction_weight": config.correction_weight,
                "uncertainty_weight": config.uncertainty_weight,
                "correction_clip": config.correction_clip,
            }
            for target, config in targetwise_configs.items()
        },
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "hazard_feature_residual_config.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    report = [
        "# Hazard-feature residual calibration",
        "",
        "This experiment uses the probabilistic exposure simulator only as an additional",
        "feature block for the regularized log-residual ridge model.",
        "",
        "## Leak-free protocol",
        "",
        "- Hazard features are fitted from history before the forecast cutoff only.",
        "- Ridge hyperparameters are selected on the later pretest split.",
        "- Final holdout is evaluated only after config selection.",
        "",
        "## Selected ridge config",
        "",
        json.dumps(manifest["selected_ridge_config"], indent=2),
        "",
        "## Selected target-wise configs",
        "",
        json.dumps(manifest["selected_targetwise_configs"], indent=2),
        "",
        "## Metrics",
        "",
        metrics.to_markdown(index=False),
    ]
    (REPORT_DIR / "hazard_feature_residual_log.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(metrics.to_string(index=False))
    print("\nSelected ridge config:")
    print(json.dumps(manifest["selected_ridge_config"], indent=2))
    print("\nSelected target-wise configs:")
    print(json.dumps(manifest["selected_targetwise_configs"], indent=2))
    print("\nTop candidates:")
    print(candidate_table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
