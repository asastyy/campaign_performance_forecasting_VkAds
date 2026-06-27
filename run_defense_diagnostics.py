from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


DEFAULT_DATA_DIR = Path("/Users/anastasiasergeeva/Desktop/HSE/Сессия 2026/НИР Vk Ads/data")
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"
REPORT_DIR = PROJECT_DIR / "reports"
TARGET_COLUMNS = ["at_least_one", "at_least_two", "at_least_three"]
EPSILON = 0.005


PREDICTION_FILES = {
    "Final decomposed replay": "strict_locked_predictions.tsv",
    "Monthly replay": "validation_predictions_past_monthly.tsv",
    "Boosting + replay features": "validation_predictions_boosting_with_replay_features_trained_on_pretest.tsv",
    "Static/history boosting": "validation_predictions_boosting_static_history_trained_on_pretest.tsv",
    "Zero baseline": "validation_predictions_zero.tsv",
}


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def dataframe_to_markdown(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(format(value, floatfmt))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def smoothed_mean_log_accuracy_ratio(actual: pd.DataFrame, pred: pd.DataFrame) -> float:
    actual_values = actual[TARGET_COLUMNS].to_numpy(dtype=float)
    pred_values = pred[TARGET_COLUMNS].to_numpy(dtype=float)
    log_error = np.abs(np.log((pred_values + EPSILON) / (actual_values + EPSILON)))
    return float(100.0 * (np.exp(log_error.mean()) - 1.0))


def mean_abs_log_error(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(np.log((pred + EPSILON) / (actual + EPSILON))).mean())


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(os.environ.get("VK_ADS_DATA_DIR", DEFAULT_DATA_DIR))
    validate = pd.read_csv(data_dir / "validate.tsv", sep="\t")
    answers = pd.read_csv(data_dir / "validate_answers.tsv", sep="\t")
    zones = pd.read_csv(OUTPUT_DIR / "strict_temporal_zones.csv")
    predictions = pd.read_csv(OUTPUT_DIR / "strict_locked_predictions.tsv", sep="\t")
    return validate, answers, zones, predictions


def enrich_campaigns(validate: pd.DataFrame) -> pd.DataFrame:
    enriched = validate.copy()
    enriched["duration_hours"] = enriched["hour_end"] - enriched["hour_start"] + 1
    enriched["start_hour_of_day"] = enriched["hour_start"] % 24
    enriched["publisher_count"] = enriched["publishers"].fillna("").map(
        lambda value: len([item for item in str(value).split(",") if item.strip()])
    )
    return enriched


def add_quantile_segment(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    quantiles = pd.qcut(frame[column], q=4, duplicates="drop")
    return quantiles.astype(str).rename(label)


def quantile_segment(frame: pd.DataFrame, column: str) -> tuple[pd.Series, dict[str, int]]:
    quantiles = pd.qcut(frame[column], q=4, duplicates="drop")
    labels = quantiles.astype(str)
    order = {str(category): idx for idx, category in enumerate(quantiles.cat.categories)}
    return labels, order


def compute_model_metrics(answers: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    rows = []
    final_idx = zones.loc[zones["zone"] == "final_holdout", "row_position"].to_numpy()
    for model_name, file_name in PREDICTION_FILES.items():
        path = OUTPUT_DIR / file_name
        if not path.exists():
            continue
        pred = pd.read_csv(path, sep="\t")
        rows.append(
            {
                "model": model_name,
                "rows": len(final_idx),
                "final_holdout_metric_percent": smoothed_mean_log_accuracy_ratio(
                    answers.iloc[final_idx].reset_index(drop=True),
                    pred.iloc[final_idx].reset_index(drop=True),
                ),
            }
        )
    result = pd.DataFrame(rows).sort_values("final_holdout_metric_percent")
    result.to_csv(OUTPUT_DIR / "defense_model_metrics.csv", index=False)
    return result


def compute_target_breakdown(
    actual: pd.DataFrame, predicted: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for target in TARGET_COLUMNS:
        actual_values = actual[target].to_numpy(dtype=float)
        pred_values = predicted[target].to_numpy(dtype=float)
        rows.append(
            {
                "target": target,
                "mean_actual": actual_values.mean(),
                "mean_predicted": pred_values.mean(),
                "mean_bias_pred_minus_actual": pred_values.mean() - actual_values.mean(),
                "mean_abs_log_error": mean_abs_log_error(actual_values, pred_values),
                "median_abs_error": float(np.median(np.abs(pred_values - actual_values))),
                "p90_abs_error": float(np.quantile(np.abs(pred_values - actual_values), 0.90)),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "defense_target_error_breakdown.csv", index=False)
    return result


def compute_segment_metrics(
    campaigns: pd.DataFrame, actual: pd.DataFrame, predicted: pd.DataFrame
) -> pd.DataFrame:
    frame = campaigns.reset_index(drop=True).copy()
    frame[TARGET_COLUMNS] = actual[TARGET_COLUMNS].reset_index(drop=True)
    for target in TARGET_COLUMNS:
        frame[f"pred_{target}"] = predicted[target].to_numpy(dtype=float)

    cpm_labels, cpm_order = quantile_segment(frame, "cpm")
    audience_labels, audience_order = quantile_segment(frame, "audience_size")
    duration_labels, duration_order = quantile_segment(frame, "duration_hours")
    publisher_labels = frame["publisher_count"].clip(upper=3).astype(int).map(
        {1: "1 publisher", 2: "2 publishers", 3: "3+ publishers"}
    )
    hour_labels = pd.cut(
        frame["start_hour_of_day"],
        bins=[-1, 5, 11, 17, 23],
        labels=["night", "morning", "day", "evening"],
    ).astype(str)

    segment_defs = [
        ("CPM quartile", cpm_labels, cpm_order),
        ("Audience quartile", audience_labels, audience_order),
        ("Duration quartile", duration_labels, duration_order),
        ("Publisher count", publisher_labels, {"1 publisher": 0, "2 publishers": 1, "3+ publishers": 2}),
        ("Start hour bucket", hour_labels, {"night": 0, "morning": 1, "day": 2, "evening": 3}),
    ]

    rows = []
    for segment_name, labels, order_map in segment_defs:
        for segment_value, idx in labels.groupby(labels).groups.items():
            idx = list(idx)
            if len(idx) < 5:
                continue
            segment_actual = frame.loc[idx, TARGET_COLUMNS].reset_index(drop=True)
            segment_pred = frame.loc[idx, [f"pred_{target}" for target in TARGET_COLUMNS]].copy()
            segment_pred.columns = TARGET_COLUMNS
            rows.append(
                {
                    "segment": segment_name,
                    "segment_value": str(segment_value),
                    "segment_order": int(order_map.get(str(segment_value), 999)),
                    "rows": len(idx),
                    "metric_percent": smoothed_mean_log_accuracy_ratio(
                        segment_actual, segment_pred
                    ),
                    "mean_actual_at_least_one": segment_actual["at_least_one"].mean(),
                    "mean_pred_at_least_one": segment_pred["at_least_one"].mean(),
                    "mean_actual_at_least_two": segment_actual["at_least_two"].mean(),
                    "mean_pred_at_least_two": segment_pred["at_least_two"].mean(),
                    "mean_actual_at_least_three": segment_actual["at_least_three"].mean(),
                    "mean_pred_at_least_three": segment_pred["at_least_three"].mean(),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "defense_segment_metrics.csv", index=False)
    return result


def compute_calibration_bins(
    actual: pd.DataFrame, predicted: pd.DataFrame, bins: int = 8
) -> pd.DataFrame:
    rows = []
    for target in TARGET_COLUMNS:
        temp = pd.DataFrame(
            {
                "actual": actual[target].to_numpy(dtype=float),
                "predicted": predicted[target].to_numpy(dtype=float),
            }
        ).sort_values("predicted")
        temp["bin"] = pd.qcut(temp["predicted"].rank(method="first"), q=bins, labels=False)
        grouped = temp.groupby("bin", as_index=False).agg(
            rows=("actual", "size"),
            mean_actual=("actual", "mean"),
            mean_predicted=("predicted", "mean"),
            median_predicted=("predicted", "median"),
        )
        grouped["target"] = target
        grouped["abs_calibration_gap"] = (
            grouped["mean_predicted"] - grouped["mean_actual"]
        ).abs()
        rows.append(grouped)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(OUTPUT_DIR / "defense_calibration_bins.csv", index=False)
    return result


def plot_segment_metrics(segment_metrics: pd.DataFrame) -> None:
    selected_segments = ["CPM quartile", "Audience quartile", "Duration quartile"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, segment in zip(axes, selected_segments):
        part = segment_metrics.loc[segment_metrics["segment"] == segment].copy()
        part = part.sort_values("segment_order")
        ax.bar(part["segment_value"], part["metric_percent"], color="#60A5FA")
        ax.set_title(segment)
        ax.set_xlabel("Segment")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(part["metric_percent"]):
            ax.text(idx, value + 0.25, f"{value:.1f}%", ha="center", fontsize=8)
    axes[0].set_ylabel("Metric, %")
    fig.suptitle("Final holdout error by campaign segment", fontsize=15, fontweight="bold")
    savefig(FIGURE_DIR / "08_segment_error_analysis.png")


def plot_calibration(calibration: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, target, title in zip(axes, TARGET_COLUMNS, ["1+ reach", "2+ reach", "3+ reach"]):
        part = calibration.loc[calibration["target"] == target].sort_values("mean_predicted")
        ax.plot(part["mean_predicted"], part["mean_actual"], marker="o", linewidth=2)
        max_value = max(part["mean_predicted"].max(), part["mean_actual"].max())
        ax.plot([0, max_value], [0, max_value], color="#DC2626", linestyle="--", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Mean predicted")
        ax.set_ylabel("Mean actual")
        ax.grid(alpha=0.25)
    fig.suptitle("Calibration by prediction bins on locked final holdout", fontsize=15, fontweight="bold")
    savefig(FIGURE_DIR / "09_calibration_bins.png")


def plot_target_errors(target_breakdown: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = target_breakdown["target"].replace(
        {
            "at_least_one": "1+",
            "at_least_two": "2+",
            "at_least_three": "3+",
        }
    )
    axes[0].bar(labels, target_breakdown["mean_abs_log_error"], color="#818CF8")
    axes[0].set_title("Mean absolute log error by target")
    axes[0].set_ylabel("MALE")
    axes[0].grid(axis="y", alpha=0.25)

    x = np.arange(len(labels))
    width = 0.35
    axes[1].bar(x - width / 2, target_breakdown["mean_actual"], width, label="Actual", color="#94A3B8")
    axes[1].bar(x + width / 2, target_breakdown["mean_predicted"], width, label="Predicted", color="#10B981")
    axes[1].set_title("Mean target values on final holdout")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Mean share")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    savefig(FIGURE_DIR / "10_target_error_breakdown.png")


def plot_candidate_selection_curve() -> None:
    selection = pd.read_csv(OUTPUT_DIR / "strict_calibration_selection.csv")
    selection = selection.sort_values("pretest_metric_percent").head(80).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(selection.index + 1, selection["calibration_metric_percent"], label="Calibration", linewidth=2)
    ax.plot(selection.index + 1, selection["pretest_metric_percent"], label="Pretest", linewidth=2)
    ax.scatter([1], [selection.loc[0, "pretest_metric_percent"]], color="#DC2626", zorder=5)
    ax.set_title("Candidate selection curve: top configurations")
    ax.set_xlabel("Candidate rank by pretest metric")
    ax.set_ylabel("Metric, %")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    savefig(FIGURE_DIR / "11_candidate_selection_curve.png")


def plot_error_distribution(actual: pd.DataFrame, predicted: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for target, label in zip(TARGET_COLUMNS, ["1+", "2+", "3+"]):
        error = np.abs(
            np.log(
                (predicted[target].to_numpy(dtype=float) + EPSILON)
                / (actual[target].to_numpy(dtype=float) + EPSILON)
            )
        )
        ax.hist(error, bins=35, alpha=0.45, label=label)
    ax.set_title("Distribution of absolute log-ratio errors")
    ax.set_xlabel("|log((pred + eps) / (actual + eps))|")
    ax.set_ylabel("Campaigns")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    savefig(FIGURE_DIR / "12_error_distribution.png")


def write_log(
    model_metrics: pd.DataFrame,
    target_breakdown: pd.DataFrame,
    segment_metrics: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    worst_segments = (
        segment_metrics.sort_values("metric_percent", ascending=False)
        .head(8)[["segment", "segment_value", "rows", "metric_percent"]]
        .copy()
    )
    best_model = model_metrics.iloc[0]
    final_model = model_metrics.loc[model_metrics["model"] == "Final decomposed replay"].iloc[0]

    text = [
        "# Defense Diagnostics Log",
        "",
        "Все diagnostics ниже являются post-hoc анализом зафиксированных predictions на locked final holdout.",
        "Они не используются для подбора финальной модели и поэтому не создают data leakage.",
        "",
        "## Model comparison on final holdout",
        "",
        dataframe_to_markdown(model_metrics, ".3f"),
        "",
        "## Target-level error breakdown",
        "",
        dataframe_to_markdown(target_breakdown, ".5f"),
        "",
        "## Worst final-holdout segments",
        "",
        dataframe_to_markdown(worst_segments, ".3f"),
        "",
        "## Short interpretation",
        "",
        f"- Best final-holdout model in this comparison: `{best_model['model']}` with {best_model['final_holdout_metric_percent']:.2f}%.",
        f"- Locked decomposed replay metric: {final_model['final_holdout_metric_percent']:.2f}%.",
        "- Segment diagnostics show where the model is less stable; this is useful as an honest limitation and future-work direction.",
        "- Calibration bins are diagnostic only: no post-final calibration was fitted on final holdout.",
        "",
        "## Generated files",
        "",
        "- `outputs/defense_model_metrics.csv`",
        "- `outputs/defense_target_error_breakdown.csv`",
        "- `outputs/defense_segment_metrics.csv`",
        "- `outputs/defense_calibration_bins.csv`",
        "- `figures/08_segment_error_analysis.png`",
        "- `figures/09_calibration_bins.png`",
        "- `figures/10_target_error_breakdown.png`",
        "- `figures/11_candidate_selection_curve.png`",
        "- `figures/12_error_distribution.png`",
    ]
    (REPORT_DIR / "defense_diagnostics_log.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    validate, answers, zones, predictions = load_inputs()
    campaigns = enrich_campaigns(validate)
    final_idx = zones.loc[zones["zone"] == "final_holdout", "row_position"].to_numpy()
    final_campaigns = campaigns.iloc[final_idx].reset_index(drop=True)
    final_actual = answers.iloc[final_idx].reset_index(drop=True)
    final_pred = predictions.iloc[final_idx].reset_index(drop=True)

    model_metrics = compute_model_metrics(answers, zones)
    target_breakdown = compute_target_breakdown(final_actual, final_pred)
    segment_metrics = compute_segment_metrics(final_campaigns, final_actual, final_pred)
    calibration = compute_calibration_bins(final_actual, final_pred)

    plot_segment_metrics(segment_metrics)
    plot_calibration(calibration)
    plot_target_errors(target_breakdown)
    plot_candidate_selection_curve()
    plot_error_distribution(final_actual, final_pred)
    write_log(model_metrics, target_breakdown, segment_metrics, calibration)

    print("Saved defense diagnostics:")
    print(f"- {OUTPUT_DIR / 'defense_model_metrics.csv'}")
    print(f"- {OUTPUT_DIR / 'defense_segment_metrics.csv'}")
    print(f"- {OUTPUT_DIR / 'defense_target_error_breakdown.csv'}")
    print(f"- {OUTPUT_DIR / 'defense_calibration_bins.csv'}")
    print(f"- {REPORT_DIR / 'defense_diagnostics_log.md'}")


if __name__ == "__main__":
    main()
