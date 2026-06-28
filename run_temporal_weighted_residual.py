from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from run_hazard_feature_residual import add_hazard_features, hazard_configurations
from run_probabilistic_exposure_simulator import fit_exposure_simulator, predict_exposure_simulator
from run_segment_residual_calibration import (
    build_residual_features,
    component_uncertainty,
    metric_raw,
    select_feature_columns,
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


@dataclass(frozen=True)
class TemporalRidgeConfig:
    feature_source: str
    feature_set: str
    alpha: float
    correction_weight: float
    uncertainty_weight: float
    correction_clip: float
    half_life_hours: float


@dataclass
class FittedTemporalRidge:
    config: TemporalRidgeConfig
    models: dict[str, Ridge]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    feature_columns: list[str]
    uncertainty_scale: pd.Series


def score(answers: pd.DataFrame, predictions: pd.DataFrame, rows: np.ndarray) -> float:
    return smoothed_mean_log_accuracy_ratio(
        answers.iloc[rows].reset_index(drop=True),
        predictions.iloc[rows].reset_index(drop=True),
    )


def temporal_sample_weights(
    campaigns: pd.DataFrame,
    rows: np.ndarray,
    reference_hour: int,
    half_life_hours: float,
) -> np.ndarray:
    if not math.isfinite(float(half_life_hours)) or float(half_life_hours) <= 0.0:
        return np.ones(len(rows), dtype=np.float64)
    hour_end = campaigns.iloc[rows]["hour_end"].to_numpy(dtype=float)
    age = np.maximum(float(reference_hour) - hour_end, 0.0)
    weights = np.exp(-math.log(2.0) * age / float(half_life_hours))
    weights = np.maximum(weights, 1e-4)
    return weights / max(float(weights.mean()), 1e-12)


def weighted_mean_scale(x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = weights.astype(np.float64)
    weights = weights / max(float(weights.sum()), 1e-12)
    mean = np.sum(x * weights.reshape(-1, 1), axis=0)
    variance = np.sum(((x - mean) ** 2) * weights.reshape(-1, 1), axis=0)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale = np.where(scale < 1e-6, 1.0, scale)
    return mean.astype(np.float32), scale.astype(np.float32)


def fit_temporal_ridge(
    features: pd.DataFrame,
    campaigns: pd.DataFrame,
    answers: pd.DataFrame,
    base_predictions: pd.DataFrame,
    components: dict[str, pd.DataFrame],
    train_idx: np.ndarray,
    reference_hour: int,
    config: TemporalRidgeConfig,
) -> FittedTemporalRidge:
    feature_columns = select_feature_columns(features, config.feature_set)
    x_train = features.iloc[train_idx][feature_columns].to_numpy(dtype=np.float32)
    weights = temporal_sample_weights(
        campaigns,
        train_idx,
        reference_hour=reference_hour,
        half_life_hours=config.half_life_hours,
    )
    mean, scale = weighted_mean_scale(x_train, weights)
    x_scaled = (x_train - mean) / scale

    models: dict[str, Ridge] = {}
    for target in TARGET_COLUMNS:
        y_train = np.log(answers.iloc[train_idx][target].to_numpy(dtype=float) + EPSILON) - np.log(
            base_predictions.iloc[train_idx][target].to_numpy(dtype=float) + EPSILON
        )
        model = Ridge(alpha=config.alpha, random_state=42)
        model.fit(x_scaled, y_train, sample_weight=weights)
        models[target] = model

    uncertainty = component_uncertainty(components)
    uncertainty_scale = uncertainty.iloc[train_idx].median(axis=0).replace(0.0, 1e-6)
    return FittedTemporalRidge(
        config=config,
        models=models,
        feature_mean=mean,
        feature_scale=scale,
        feature_columns=feature_columns,
        uncertainty_scale=uncertainty_scale,
    )


def predict_temporal_ridge(
    fitted: FittedTemporalRidge,
    features: pd.DataFrame,
    base_predictions: pd.DataFrame,
    components: dict[str, pd.DataFrame],
    rows: np.ndarray,
) -> pd.DataFrame:
    x = features.iloc[rows][fitted.feature_columns].to_numpy(dtype=np.float32)
    x = (x - fitted.feature_mean) / fitted.feature_scale
    uncertainty = component_uncertainty(components)
    adjusted = pd.DataFrame(index=np.arange(len(rows)), columns=TARGET_COLUMNS, dtype=float)
    for target in TARGET_COLUMNS:
        base_log = np.log(base_predictions.iloc[rows][target].to_numpy(dtype=float) + EPSILON)
        correction = fitted.models[target].predict(x)
        correction = np.clip(
            correction,
            -float(fitted.config.correction_clip),
            float(fitted.config.correction_clip),
        )
        scale = max(float(fitted.uncertainty_scale[target]), 1e-6)
        unc = uncertainty.iloc[rows][target].to_numpy(dtype=float)
        confidence = 1.0 / (1.0 + fitted.config.uncertainty_weight * (unc / scale))
        correction *= fitted.config.correction_weight * confidence
        adjusted[target] = np.exp(base_log + correction) - EPSILON
    return enforce_target_constraints(adjusted)


def config_grid() -> list[TemporalRidgeConfig]:
    configs: set[tuple[str, str, float, float, float, float, float]] = set()

    def add_family(
        feature_source: str,
        feature_set: str,
        alphas: tuple[float, ...],
        correction_weights: tuple[float, ...],
        uncertainty_weights: tuple[float, ...],
        correction_clips: tuple[float, ...],
        half_lives: tuple[float, ...],
    ) -> None:
        for alpha in alphas:
            for correction_weight in correction_weights:
                for uncertainty_weight in uncertainty_weights:
                    for correction_clip in correction_clips:
                        for half_life_hours in half_lives:
                            configs.add(
                                (
                                    feature_source,
                                    feature_set,
                                    alpha,
                                    correction_weight,
                                    uncertainty_weight,
                                    correction_clip,
                                    half_life_hours,
                                )
                            )

    # Refined winner family for 1+: standard/no_decomp with short recency.
    add_family(
        "standard",
        "no_decomp",
        (100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 750.0),
        (0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35),
        (0.0, 0.25, 0.5, 0.75, 1.0, 2.0),
        (0.15, 0.25, 0.40),
        (72.0, 96.0, 120.0, 168.0, 240.0, 336.0),
    )

    # Refined winner family for the overall model and 2+: standard/all.
    add_family(
        "standard",
        "all",
        (50.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0, 1500.0),
        (0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30),
        (0.0, 0.25, 0.5, 1.0),
        (0.08, 0.15, 0.25, 0.40),
        (float("inf"), 72.0, 120.0, 168.0, 240.0, 336.0),
    )

    # Refined winner family for 3+: hazard/all without recency weighting.
    add_family(
        "hazard",
        "all",
        (50.0, 100.0, 150.0, 200.0, 300.0, 500.0),
        (0.08, 0.10, 0.12, 0.15, 0.18, 0.20),
        (0.0, 0.25, 0.5, 1.0),
        (0.25, 0.40, 0.60),
        (float("inf"), 168.0, 336.0),
    )

    # Keep the previous selected points explicitly in the grid.
    add_family(
        "standard",
        "static_replay",
        (100.0, 200.0, 300.0, 500.0),
        (0.08, 0.12, 0.20),
        (0.0, 0.5, 2.0),
        (0.08, 0.25, 0.40),
        (float("inf"), 168.0, 336.0),
    )

    return [
        TemporalRidgeConfig(
            feature_source=feature_source,
            feature_set=feature_set,
            alpha=alpha,
            correction_weight=correction_weight,
            uncertainty_weight=uncertainty_weight,
            correction_clip=correction_clip,
            half_life_hours=half_life_hours,
        )
        for (
            feature_source,
            feature_set,
            alpha,
            correction_weight,
            uncertainty_weight,
            correction_clip,
            half_life_hours,
        ) in sorted(configs, key=lambda item: (item[0], item[1], item[2:]))
    ]


def config_to_dict(config: TemporalRidgeConfig) -> dict[str, float | str]:
    return {
        "feature_source": config.feature_source,
        "feature_set": config.feature_set,
        "alpha": config.alpha,
        "correction_weight": config.correction_weight,
        "uncertainty_weight": config.uncertainty_weight,
        "correction_clip": config.correction_clip,
        "half_life_hours": (
            "inf" if not math.isfinite(float(config.half_life_hours)) else config.half_life_hours
        ),
    }


def get_features(
    feature_tables: dict[str, pd.DataFrame],
    config: TemporalRidgeConfig,
) -> pd.DataFrame:
    return feature_tables[config.feature_source]


def fit_predict(
    feature_tables: dict[str, pd.DataFrame],
    campaigns: pd.DataFrame,
    answers: pd.DataFrame,
    base_predictions: pd.DataFrame,
    components: dict[str, pd.DataFrame],
    train_idx: np.ndarray,
    reference_hour: int,
    predict_idx: np.ndarray,
    config: TemporalRidgeConfig,
) -> pd.DataFrame:
    features = get_features(feature_tables, config)
    fitted = fit_temporal_ridge(
        features,
        campaigns,
        answers,
        base_predictions,
        components,
        train_idx=train_idx,
        reference_hour=reference_hour,
        config=config,
    )
    return predict_temporal_ridge(
        fitted,
        features,
        base_predictions,
        components,
        predict_idx,
    )


def main() -> None:
    data_dir = Path(os.environ.get("VK_ADS_DATA_DIR", DEFAULT_DATA_DIR))
    users, history, campaigns, answers = load_dataset(data_dir)
    if len(answers) != len(campaigns):
        raise RuntimeError("Temporal weighted residual experiment requires validate_answers.tsv.")

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

    standard_features = build_residual_features(
        campaigns,
        users,
        past_history,
        base_predictions,
        components,
    )

    fitted_simulator = fit_exposure_simulator(users, past_history)
    hazard_predictions = {}
    for name, config in hazard_configurations().items():
        print(f"Computing hazard feature: {name}", flush=True)
        hazard_predictions[name] = predict_exposure_simulator(
            fitted_simulator,
            campaigns,
            config,
        ).reset_index(drop=True)
    hazard_features = add_hazard_features(
        standard_features,
        base_predictions,
        hazard_predictions,
    )
    feature_tables = {
        "standard": standard_features,
        "hazard": hazard_features,
    }

    train_idx = split.development_idx
    selection_idx = np.setdiff1d(split.pretest_idx, train_idx, assume_unique=False)
    final_train_idx = split.pretest_idx
    final_idx = split.test_idx

    rows = []
    best_config: TemporalRidgeConfig | None = None
    best_metric = float("inf")
    configs = config_grid()
    selection_answers = answers.iloc[selection_idx].reset_index(drop=True)
    for config_id, config in enumerate(configs, start=1):
        if config_id == 1 or config_id % 500 == 0 or config_id == len(configs):
            print(f"Evaluating temporal weighted config {config_id}/{len(configs)}", flush=True)
        prediction = fit_predict(
            feature_tables,
            campaigns,
            answers,
            base_predictions,
            components,
            train_idx=train_idx,
            reference_hour=split.calibration_start,
            predict_idx=selection_idx,
            config=config,
        )
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
                **config_to_dict(config),
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
        raise RuntimeError("No temporal weighted ridge config was selected.")

    candidate_table = pd.DataFrame(rows).sort_values(
        ["selection_metric_raw_percent", "selection_metric_percent", "alpha"],
        kind="stable",
    )

    targetwise_configs: dict[str, TemporalRidgeConfig] = {}
    for target in TARGET_COLUMNS:
        best_target_row = candidate_table.sort_values(
            [
                f"selection_{target}_abs_log_error",
                "selection_metric_raw_percent",
                "alpha",
            ],
            kind="stable",
        ).iloc[0]
        half_life_value = best_target_row["half_life_hours"]
        if str(half_life_value) == "inf":
            half_life_hours = float("inf")
        else:
            half_life_hours = float(half_life_value)
        targetwise_configs[target] = TemporalRidgeConfig(
            feature_source=str(best_target_row["feature_source"]),
            feature_set=str(best_target_row["feature_set"]),
            alpha=float(best_target_row["alpha"]),
            correction_weight=float(best_target_row["correction_weight"]),
            uncertainty_weight=float(best_target_row["uncertainty_weight"]),
            correction_clip=float(best_target_row["correction_clip"]),
            half_life_hours=half_life_hours,
        )

    final_prediction = fit_predict(
        feature_tables,
        campaigns,
        answers,
        base_predictions,
        components,
        train_idx=final_train_idx,
        reference_hour=split.test_start,
        predict_idx=final_idx,
        config=best_config,
    )
    full_prediction = base_predictions.copy()
    full_prediction.loc[final_idx, TARGET_COLUMNS] = final_prediction[TARGET_COLUMNS].to_numpy()
    full_prediction = enforce_target_constraints(full_prediction)

    targetwise_prediction = pd.DataFrame(index=np.arange(len(final_idx)), columns=TARGET_COLUMNS)
    for target, config in targetwise_configs.items():
        one_target_prediction = fit_predict(
            feature_tables,
            campaigns,
            answers,
            base_predictions,
            components,
            train_idx=final_train_idx,
            reference_hour=split.test_start,
            predict_idx=final_idx,
            config=config,
        )
        targetwise_prediction[target] = one_target_prediction[target].to_numpy(dtype=float)
    targetwise_prediction = enforce_target_constraints(targetwise_prediction)
    full_targetwise_prediction = base_predictions.copy()
    full_targetwise_prediction.loc[final_idx, TARGET_COLUMNS] = targetwise_prediction[
        TARGET_COLUMNS
    ].to_numpy()
    full_targetwise_prediction = enforce_target_constraints(full_targetwise_prediction)

    final_answers = answers.iloc[final_idx].reset_index(drop=True)
    metrics = pd.DataFrame(
        [
            {
                "model": "base_decomposed_replay",
                "selection_metric_percent": score(answers, base_predictions, selection_idx),
                "selection_metric_raw_percent": metric_raw(
                    selection_answers,
                    base_predictions.iloc[selection_idx].reset_index(drop=True),
                ),
                "final_holdout_metric_percent": score(answers, base_predictions, final_idx),
                "final_holdout_metric_raw_percent": metric_raw(
                    final_answers,
                    base_predictions.iloc[final_idx].reset_index(drop=True),
                ),
            },
            {
                "model": "temporal_weighted_ridge_residual",
                "selection_metric_percent": float(candidate_table.iloc[0]["selection_metric_percent"]),
                "selection_metric_raw_percent": float(
                    candidate_table.iloc[0]["selection_metric_raw_percent"]
                ),
                "final_holdout_metric_percent": smoothed_mean_log_accuracy_ratio(
                    final_answers,
                    final_prediction.reset_index(drop=True),
                ),
                "final_holdout_metric_raw_percent": metric_raw(
                    final_answers,
                    final_prediction.reset_index(drop=True),
                ),
            },
            {
                "model": "temporal_weighted_targetwise_ridge_residual",
                "selection_metric_percent": np.nan,
                "selection_metric_raw_percent": np.nan,
                "final_holdout_metric_percent": smoothed_mean_log_accuracy_ratio(
                    final_answers,
                    targetwise_prediction.reset_index(drop=True),
                ),
                "final_holdout_metric_raw_percent": metric_raw(
                    final_answers,
                    targetwise_prediction.reset_index(drop=True),
                ),
            },
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_table.to_csv(OUTPUT_DIR / "temporal_weighted_candidate_selection.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "temporal_weighted_residual_metrics.csv", index=False)
    save_predictions(
        full_prediction.round(6),
        OUTPUT_DIR / "validation_predictions_temporal_weighted_ridge.tsv",
    )
    save_predictions(
        full_targetwise_prediction.round(6),
        OUTPUT_DIR / "validation_predictions_temporal_weighted_targetwise_ridge.tsv",
    )

    manifest = {
        "protocol": "recency_weighted_residual_selection_then_locked_final_evaluation",
        "selection_reference_hour": int(split.calibration_start),
        "final_reference_hour": int(split.test_start),
        "selected_config": config_to_dict(best_config),
        "selected_targetwise_configs": {
            target: config_to_dict(config) for target, config in targetwise_configs.items()
        },
        "metrics": metrics.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "temporal_weighted_residual_config.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    report = [
        "# Temporal-weighted residual Ridge",
        "",
        "This experiment gives larger training weight to campaigns whose labels are closer",
        "to the prediction cutoff. The goal is to adapt the residual layer to temporal",
        "drift while keeping all features and labels past-only.",
        "",
        "## Leak-free protocol",
        "",
        "- Selection fit uses development rows only and recency is measured relative to calibration_start.",
        "- Final fit uses pretest rows only and recency is measured relative to test_start.",
        "- Final holdout is evaluated only after config and target-wise configs are selected.",
        "",
        "## Selected config",
        "",
        json.dumps(manifest["selected_config"], indent=2),
        "",
        "## Selected target-wise configs",
        "",
        json.dumps(manifest["selected_targetwise_configs"], indent=2),
        "",
        "## Metrics",
        "",
        metrics.to_markdown(index=False),
    ]
    (REPORT_DIR / "temporal_weighted_residual_log.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(metrics.to_string(index=False))
    print("\nSelected config:")
    print(json.dumps(manifest["selected_config"], indent=2))
    print("\nSelected target-wise configs:")
    print(json.dumps(manifest["selected_targetwise_configs"], indent=2))
    print("\nTop candidates:")
    print(candidate_table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
