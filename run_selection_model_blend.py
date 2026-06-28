from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from run_hazard_feature_residual import (
    add_hazard_features,
    fit_predict_full,
    hazard_configurations,
)
from run_probabilistic_exposure_simulator import fit_exposure_simulator, predict_exposure_simulator
from run_segment_residual_calibration import (
    RidgeResidualConfig,
    build_residual_features,
    metric_raw,
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


@dataclass(frozen=True)
class TargetOption:
    name: str
    source_a: str
    source_b: str | None = None
    weight_b: float = 0.0


def ridge_config_from_dict(data: dict[str, object]) -> RidgeResidualConfig:
    return RidgeResidualConfig(
        feature_set=str(data["feature_set"]),
        alpha=float(data["alpha"]),
        correction_weight=float(data["correction_weight"]),
        uncertainty_weight=float(data["uncertainty_weight"]),
        correction_clip=float(data["correction_clip"]),
    )


def load_ridge_configs(path: Path) -> tuple[RidgeResidualConfig, dict[str, RidgeResidualConfig]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = ridge_config_from_dict(data["selected_ridge_config"])
    targetwise = {
        target: ridge_config_from_dict(config)
        for target, config in data["selected_targetwise_ridge_configs"].items()
    }
    return selected, targetwise


def load_hazard_ridge_configs(path: Path) -> tuple[RidgeResidualConfig, dict[str, RidgeResidualConfig]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = ridge_config_from_dict(data["selected_ridge_config"])
    targetwise = {
        target: ridge_config_from_dict(config)
        for target, config in data["selected_targetwise_configs"].items()
    }
    return selected, targetwise


def predict_targetwise(
    residual_features: pd.DataFrame,
    answers: pd.DataFrame,
    base_predictions: pd.DataFrame,
    components: dict[str, pd.DataFrame],
    train_idx: np.ndarray,
    predict_idx: np.ndarray,
    configs: dict[str, RidgeResidualConfig],
) -> pd.DataFrame:
    prediction = pd.DataFrame(index=np.arange(len(predict_idx)), columns=TARGET_COLUMNS)
    for target, config in configs.items():
        one_target = fit_predict_full(
            residual_features,
            answers,
            base_predictions,
            components,
            train_idx,
            predict_idx,
            config,
        )
        prediction[target] = one_target[target].to_numpy(dtype=float)
    return enforce_target_constraints(prediction)


def build_prediction_sources(
    old_features: pd.DataFrame,
    hazard_features: pd.DataFrame,
    answers: pd.DataFrame,
    base_predictions: pd.DataFrame,
    components: dict[str, pd.DataFrame],
    train_idx: np.ndarray,
    predict_idx: np.ndarray,
    old_ridge_config: RidgeResidualConfig,
    old_targetwise_configs: dict[str, RidgeResidualConfig],
    hazard_ridge_config: RidgeResidualConfig,
    hazard_targetwise_configs: dict[str, RidgeResidualConfig],
) -> dict[str, pd.DataFrame]:
    sources = {
        "base": base_predictions.iloc[predict_idx].reset_index(drop=True),
        "ridge": fit_predict_full(
            old_features,
            answers,
            base_predictions,
            components,
            train_idx,
            predict_idx,
            old_ridge_config,
        ),
        "targetwise": predict_targetwise(
            old_features,
            answers,
            base_predictions,
            components,
            train_idx,
            predict_idx,
            old_targetwise_configs,
        ),
        "hazard_ridge": fit_predict_full(
            hazard_features,
            answers,
            base_predictions,
            components,
            train_idx,
            predict_idx,
            hazard_ridge_config,
        ),
        "hazard_targetwise": predict_targetwise(
            hazard_features,
            answers,
            base_predictions,
            components,
            train_idx,
            predict_idx,
            hazard_targetwise_configs,
        ),
    }
    return {name: enforce_target_constraints(value.reset_index(drop=True)) for name, value in sources.items()}


def option_values(
    option: TargetOption,
    sources: dict[str, pd.DataFrame],
    target: str,
) -> np.ndarray:
    a = sources[option.source_a][target].to_numpy(dtype=float)
    if option.source_b is None:
        return a
    b = sources[option.source_b][target].to_numpy(dtype=float)
    return np.exp(
        (1.0 - option.weight_b) * np.log(a + EPSILON)
        + option.weight_b * np.log(b + EPSILON)
    ) - EPSILON


def candidate_options() -> list[TargetOption]:
    options = [
        TargetOption(name=source, source_a=source)
        for source in ("base", "ridge", "targetwise", "hazard_ridge", "hazard_targetwise")
    ]
    pairs = (
        ("targetwise", "hazard_ridge"),
        ("targetwise", "hazard_targetwise"),
        ("ridge", "hazard_ridge"),
        ("base", "targetwise"),
        ("base", "hazard_ridge"),
    )
    for source_a, source_b in pairs:
        for weight_b in (0.25, 0.50, 0.75):
            options.append(
                TargetOption(
                    name=f"{source_a}_{1.0 - weight_b:.2f}+{source_b}_{weight_b:.2f}",
                    source_a=source_a,
                    source_b=source_b,
                    weight_b=weight_b,
                )
            )
    return options


def assemble_prediction(
    combination: dict[str, TargetOption],
    sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    prediction = pd.DataFrame(index=sources["base"].index, columns=TARGET_COLUMNS, dtype=float)
    for target, option in combination.items():
        prediction[target] = option_values(option, sources, target)
    return enforce_target_constraints(prediction.reset_index(drop=True))


def main() -> None:
    data_dir = Path(os.environ.get("VK_ADS_DATA_DIR", DEFAULT_DATA_DIR))
    users, history, campaigns, answers = load_dataset(data_dir)
    if len(answers) != len(campaigns):
        raise RuntimeError("Selection blend experiment requires validate_answers.tsv.")

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

    old_features = build_residual_features(
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
        old_features,
        base_predictions,
        hazard_predictions,
    )

    old_ridge_config, old_targetwise_configs = load_ridge_configs(
        OUTPUT_DIR / "segment_residual_config.json"
    )
    hazard_ridge_config, hazard_targetwise_configs = load_hazard_ridge_configs(
        OUTPUT_DIR / "hazard_feature_residual_config.json"
    )

    train_idx = split.development_idx
    selection_idx = np.setdiff1d(split.pretest_idx, train_idx, assume_unique=False)
    final_train_idx = split.pretest_idx
    final_idx = split.test_idx

    print("Building selection predictions", flush=True)
    selection_sources = build_prediction_sources(
        old_features,
        hazard_features,
        answers,
        base_predictions,
        components,
        train_idx,
        selection_idx,
        old_ridge_config,
        old_targetwise_configs,
        hazard_ridge_config,
        hazard_targetwise_configs,
    )

    selection_answers = answers.iloc[selection_idx].reset_index(drop=True)
    source_rows = []
    for name, prediction in selection_sources.items():
        source_rows.append(
            {
                "source": name,
                "selection_metric_raw_percent": metric_raw(selection_answers, prediction),
                "selection_metric_percent": smoothed_mean_log_accuracy_ratio(
                    selection_answers,
                    prediction,
                ),
                **{
                    f"{target}_abs_log_error": target_abs_log_error_raw(
                        selection_answers,
                        prediction,
                        target,
                    )
                    for target in TARGET_COLUMNS
                },
            }
        )

    options = candidate_options()
    top_options_by_target: dict[str, list[TargetOption]] = {}
    option_rows = []
    for target in TARGET_COLUMNS:
        scored = []
        for option in options:
            prediction = pd.DataFrame({target: option_values(option, selection_sources, target)})
            target_error = target_abs_log_error_raw(
                selection_answers[[target]],
                prediction[[target]],
                target,
            )
            scored.append((target_error, option))
            option_rows.append(
                {
                    "target": target,
                    "option": option.name,
                    "target_abs_log_error": target_error,
                }
            )
        scored.sort(key=lambda item: (item[0], item[1].name))
        top_options_by_target[target] = [option for _, option in scored[:6]]

    blend_rows = []
    best_combination: dict[str, TargetOption] | None = None
    best_metric = float("inf")
    for choices in product(*(top_options_by_target[target] for target in TARGET_COLUMNS)):
        combination = dict(zip(TARGET_COLUMNS, choices))
        prediction = assemble_prediction(combination, selection_sources)
        raw_metric = metric_raw(selection_answers, prediction)
        rounded_metric = smoothed_mean_log_accuracy_ratio(selection_answers, prediction)
        blend_rows.append(
            {
                "at_least_one_option": combination["at_least_one"].name,
                "at_least_two_option": combination["at_least_two"].name,
                "at_least_three_option": combination["at_least_three"].name,
                "selection_metric_percent": rounded_metric,
                "selection_metric_raw_percent": raw_metric,
            }
        )
        if raw_metric < best_metric:
            best_metric = raw_metric
            best_combination = combination

    if best_combination is None:
        raise RuntimeError("No model blend was selected.")

    print("Building final predictions", flush=True)
    final_sources = build_prediction_sources(
        old_features,
        hazard_features,
        answers,
        base_predictions,
        components,
        final_train_idx,
        final_idx,
        old_ridge_config,
        old_targetwise_configs,
        hazard_ridge_config,
        hazard_targetwise_configs,
    )
    final_prediction = assemble_prediction(best_combination, final_sources)
    full_prediction = base_predictions.copy()
    full_prediction.loc[final_idx, TARGET_COLUMNS] = final_prediction[TARGET_COLUMNS].to_numpy()
    full_prediction = enforce_target_constraints(full_prediction)

    base_final = base_predictions.iloc[final_idx].reset_index(drop=True)
    metrics = pd.DataFrame(
        [
            {
                "model": "base_decomposed_replay",
                "selection_metric_percent": score(answers, base_predictions, selection_idx),
                "selection_metric_raw_percent": metric_raw(
                    answers.iloc[selection_idx].reset_index(drop=True),
                    base_predictions.iloc[selection_idx].reset_index(drop=True),
                ),
                "final_holdout_metric_percent": smoothed_mean_log_accuracy_ratio(
                    answers.iloc[final_idx].reset_index(drop=True),
                    base_final,
                ),
                "final_holdout_metric_raw_percent": metric_raw(
                    answers.iloc[final_idx].reset_index(drop=True),
                    base_final,
                ),
            },
            {
                "model": "selection_model_blend",
                "selection_metric_percent": float(
                    min(row["selection_metric_percent"] for row in blend_rows)
                ),
                "selection_metric_raw_percent": float(
                    min(row["selection_metric_raw_percent"] for row in blend_rows)
                ),
                "final_holdout_metric_percent": smoothed_mean_log_accuracy_ratio(
                    answers.iloc[final_idx].reset_index(drop=True),
                    final_prediction.reset_index(drop=True),
                ),
                "final_holdout_metric_raw_percent": metric_raw(
                    answers.iloc[final_idx].reset_index(drop=True),
                    final_prediction.reset_index(drop=True),
                ),
            },
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    source_table = pd.DataFrame(source_rows).sort_values("selection_metric_raw_percent")
    option_table = pd.DataFrame(option_rows).sort_values(["target", "target_abs_log_error"])
    blend_table = pd.DataFrame(blend_rows).sort_values(
        ["selection_metric_raw_percent", "selection_metric_percent"],
        kind="stable",
    )
    source_table.to_csv(OUTPUT_DIR / "selection_blend_source_metrics.csv", index=False)
    option_table.to_csv(OUTPUT_DIR / "selection_blend_target_options.csv", index=False)
    blend_table.to_csv(OUTPUT_DIR / "selection_blend_candidate_selection.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "selection_model_blend_metrics.csv", index=False)
    save_predictions(
        full_prediction.round(6),
        OUTPUT_DIR / "validation_predictions_selection_model_blend.tsv",
    )

    manifest = {
        "protocol": "selection_only_model_combination_then_locked_final_evaluation",
        "selected_combination": {
            target: {
                "name": option.name,
                "source_a": option.source_a,
                "source_b": option.source_b,
                "weight_b": option.weight_b,
            }
            for target, option in best_combination.items()
        },
        "metrics": metrics.to_dict(orient="records"),
        "note": (
            "The final holdout is not used for choosing target sources or blend weights. "
            "The selector searches only over models already selected on pre-final splits."
        ),
    }
    (OUTPUT_DIR / "selection_model_blend_config.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    report = [
        "# Selection-only model blend",
        "",
        "This experiment checks whether already selected residual variants are complementary.",
        "Target options and blend weights are chosen on the selection split only, then refit",
        "on all pretest rows and evaluated once on the locked final holdout.",
        "",
        "## Selected combination",
        "",
        json.dumps(manifest["selected_combination"], indent=2),
        "",
        "## Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Source metrics on selection",
        "",
        source_table.to_markdown(index=False),
    ]
    (REPORT_DIR / "selection_model_blend_log.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(metrics.to_string(index=False))
    print("\nSelected combination:")
    print(json.dumps(manifest["selected_combination"], indent=2))
    print("\nTop blend candidates:")
    print(blend_table.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
