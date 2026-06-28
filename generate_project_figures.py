from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA_DIR = Path("/Users/anastasiasergeeva/Desktop/HSE/Сессия 2026/НИР Vk Ads/data")
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"
TARGET_COLUMNS = ["at_least_one", "at_least_two", "at_least_three"]


def percent_formatter(value: float, _: int) -> str:
    return f"{value:.0%}"


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(os.environ.get("VK_ADS_DATA_DIR", DEFAULT_DATA_DIR))
    users = pd.read_csv(data_dir / "users.tsv", sep="\t")
    history = pd.read_csv(data_dir / "history.tsv", sep="\t")
    validate = pd.read_csv(data_dir / "validate.tsv", sep="\t")
    answers = pd.read_csv(data_dir / "validate_answers.tsv", sep="\t")
    return users, history, validate, answers


def plot_data_overview(history: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("EDA исторических показов", fontsize=16, fontweight="bold")

    by_hour = history.assign(hour_of_day=history["hour"] % 24).groupby("hour_of_day").size()
    axes[0, 0].bar(by_hour.index, by_hour.values, color="#3B82F6")
    axes[0, 0].set_title("Активность по часу суток")
    axes[0, 0].set_xlabel("Час суток")
    axes[0, 0].set_ylabel("Показы")
    axes[0, 0].grid(axis="y", alpha=0.25)

    by_day = history.assign(day=history["hour"] // 24).groupby("day").size()
    axes[0, 1].plot(by_day.index, by_day.values, color="#2563EB", linewidth=1.8)
    axes[0, 1].set_title("Динамика рекламного инвентаря по дням")
    axes[0, 1].set_xlabel("День истории")
    axes[0, 1].set_ylabel("Показы")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].hist(np.log1p(history["cpm"]), bins=60, color="#10B981", alpha=0.9)
    axes[1, 0].set_title("Распределение CPM, log(1 + CPM)")
    axes[1, 0].set_xlabel("log(1 + CPM)")
    axes[1, 0].set_ylabel("Частота")
    axes[1, 0].grid(axis="y", alpha=0.25)

    top_publishers = history["publisher"].value_counts().head(12).sort_values()
    axes[1, 1].barh(top_publishers.index.astype(str), top_publishers.values, color="#6366F1")
    axes[1, 1].set_title("Топ площадок по числу показов")
    axes[1, 1].set_xlabel("Показы")
    axes[1, 1].set_ylabel("Publisher")
    axes[1, 1].grid(axis="x", alpha=0.25)

    savefig(FIGURE_DIR / "01_eda_history_overview.png")


def plot_campaign_targets(validate: pd.DataFrame, answers: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("EDA validate-кампаний и целевых долей", fontsize=16, fontweight="bold")

    axes[0, 0].hist(np.log10(validate["audience_size"]), bins=40, color="#F59E0B")
    axes[0, 0].set_title("Размер аудитории кампаний")
    axes[0, 0].set_xlabel("log10(audience_size)")
    axes[0, 0].set_ylabel("Число кампаний")
    axes[0, 0].grid(axis="y", alpha=0.25)

    duration = validate["hour_end"] - validate["hour_start"] + 1
    axes[0, 1].hist(duration, bins=40, color="#EF4444")
    axes[0, 1].set_title("Длительность кампаний")
    axes[0, 1].set_xlabel("Часы")
    axes[0, 1].set_ylabel("Число кампаний")
    axes[0, 1].grid(axis="y", alpha=0.25)

    axes[1, 0].boxplot(
        [answers[col] for col in TARGET_COLUMNS],
        labels=["1+", "2+", "3+"],
        patch_artist=True,
        boxprops={"facecolor": "#DBEAFE", "color": "#2563EB"},
        medianprops={"color": "#B91C1C", "linewidth": 2},
    )
    axes[1, 0].set_title("Распределение целевых долей")
    axes[1, 0].set_ylabel("Доля пользователей")
    axes[1, 0].yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    axes[1, 0].grid(axis="y", alpha=0.25)

    corr = answers[TARGET_COLUMNS].corr()
    im = axes[1, 1].imshow(corr, vmin=0, vmax=1, cmap="Blues")
    axes[1, 1].set_title("Корреляция таргетов")
    axes[1, 1].set_xticks(range(3), ["1+", "2+", "3+"])
    axes[1, 1].set_yticks(range(3), ["1+", "2+", "3+"])
    for i in range(3):
        for j in range(3):
            axes[1, 1].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    savefig(FIGURE_DIR / "02_validate_targets_eda.png")


def plot_temporal_split() -> None:
    zones = pd.read_csv(OUTPUT_DIR / "strict_temporal_zones.csv")
    palette = {
        "development": "#60A5FA",
        "calibration": "#FBBF24",
        "pretest": "#34D399",
        "final_holdout": "#F87171",
        "purged_or_unused": "#CBD5E1",
    }
    fig, ax = plt.subplots(figsize=(13, 5))
    for zone, part in zones.groupby("zone"):
        ax.scatter(
            part["row_position"],
            part["hour_start"],
            s=18,
            color=palette.get(zone, "#94A3B8"),
            label=zone,
            alpha=0.85,
        )
    ax.set_title("Temporal split validate-кампаний без утечек", fontsize=15, fontweight="bold")
    ax.set_xlabel("Позиция кампании в validate.tsv")
    ax.set_ylabel("hour_start")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    savefig(FIGURE_DIR / "03_temporal_split.png")


def plot_model_comparison() -> None:
    boosting = pd.read_csv(OUTPUT_DIR / "boosting_baseline_metrics.csv")
    temporal_metrics_path = OUTPUT_DIR / "temporal_weighted_residual_metrics.csv"
    temporal_metrics = pd.read_csv(temporal_metrics_path) if temporal_metrics_path.exists() else None
    final_rows = [
        ("Monthly baseline", 11.70),
        ("Boosting + replay features", 10.94),
        ("Base decomposed replay", 9.54),
        ("Target-wise Ridge residual", 9.46),
        ("Temporal-weighted target-wise Ridge", 9.29),
        ("Static/history boosting", 30.89),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels, values = zip(*final_rows)
    colors = ["#94A3B8", "#60A5FA", "#10B981", "#34D399", "#059669", "#F87171"]
    axes[0].barh(labels, values, color=colors)
    axes[0].set_title("Сравнение моделей на locked final holdout")
    axes[0].set_xlabel("Метрика, %")
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.25)
    for idx, value in enumerate(values):
        axes[0].text(value + 0.25, idx, f"{value:.2f}%", va="center", fontsize=9)

    selected = boosting[
        boosting["model"].isin(
            [
                "replay_blend_locked",
                "boosting_with_replay_features_trained_on_pretest",
                "boosting_static_history_trained_on_pretest",
            ]
        )
    ].copy()
    selected["model"] = selected["model"].replace(
        {
            "replay_blend_locked": "Replay blend",
            "boosting_with_replay_features_trained_on_pretest": "Boosting + replay",
            "boosting_static_history_trained_on_pretest": "Static boosting",
        }
    )
    residual_rows = [
        ("Base replay", 9.54),
        ("Segment residual", 9.51),
        ("Ridge residual", 9.47),
        ("Target-wise Ridge", 9.46),
    ]
    if temporal_metrics is not None:
        temporal_value = float(
            temporal_metrics.loc[
                temporal_metrics["model"] == "temporal_weighted_targetwise_ridge_residual",
                "final_holdout_metric_percent",
            ].iloc[0]
        )
        residual_rows.append(("Temporal-weighted target-wise", temporal_value))
    labels, values = zip(*residual_rows)
    axes[1].plot(labels, values, marker="o", linewidth=2.2, color="#2563EB")
    axes[1].scatter(labels[-1], values[-1], s=110, color="#059669", zorder=3)
    for idx, value in enumerate(values):
        axes[1].text(idx, value + 0.025, f"{value:.2f}%", ha="center", fontsize=9)
    axes[1].set_xticks(range(len(labels)), labels, rotation=15, ha="right")
    axes[1].set_title("Усиление replay через residual calibration")
    axes[1].set_ylabel("Метрика, %")
    axes[1].grid(axis="y", alpha=0.25)

    savefig(FIGURE_DIR / "04_model_comparison.png")


def plot_ablation() -> None:
    summary = pd.read_csv(OUTPUT_DIR / "leak_free_metrics_summary.csv")
    keep = summary[
        summary["model"].isin(
            [
                "past_monthly_single",
                "past_daily_8_geometric",
                "past_weekly_5_geometric",
                "past_only_geometric_blend",
                "past_only_blend_past_median_calibration",
            ]
        )
    ].copy()
    keep["model"] = keep["model"].replace(
        {
            "past_monthly_single": "Monthly",
            "past_daily_8_geometric": "Daily x8",
            "past_weekly_5_geometric": "Weekly x5",
            "past_only_geometric_blend": "Blend",
            "past_only_blend_past_median_calibration": "Blend + median calib.",
        }
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    for evaluation, part in keep.groupby("evaluation"):
        part = part.sort_values("metric_percent", ascending=False)
        ax.plot(part["model"], part["metric_percent"], marker="o", linewidth=2, label=evaluation)
    ax.set_title("Абляция past-only replay-компонент")
    ax.set_ylabel("Метрика, %")
    ax.set_xlabel("Компонента")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    plt.xticks(rotation=15)
    savefig(FIGURE_DIR / "05_replay_ablation.png")


def plot_prediction_quality(answers: pd.DataFrame) -> None:
    zones = pd.read_csv(OUTPUT_DIR / "strict_temporal_zones.csv")
    prediction_path = OUTPUT_DIR / "validation_predictions_temporal_weighted_targetwise_ridge.tsv"
    title_suffix = "temporal-weighted target-wise residual"
    if not prediction_path.exists():
        prediction_path = OUTPUT_DIR / "strict_locked_predictions.tsv"
        title_suffix = "base replay"
    prediction = pd.read_csv(prediction_path, sep="\t")
    final_idx = zones.loc[zones["zone"] == "final_holdout", "row_position"].to_numpy()
    actual = answers.iloc[final_idx].reset_index(drop=True)
    predicted = prediction.iloc[final_idx].reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, col, title in zip(axes, TARGET_COLUMNS, ["1+ показ", "2+ показа", "3+ показа"]):
        ax.scatter(actual[col], predicted[col], s=18, alpha=0.65, color="#2563EB")
        max_value = max(actual[col].max(), predicted[col].max())
        ax.plot([0, max_value], [0, max_value], color="#DC2626", linestyle="--", linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"Predicted vs actual на locked final holdout: {title_suffix}",
        fontsize=15,
        fontweight="bold",
    )
    savefig(FIGURE_DIR / "06_predicted_vs_actual_final_holdout.png")


def plot_user_representations(users: pd.DataFrame, history: pd.DataFrame) -> None:
    top_publishers = history["publisher"].value_counts().head(10).index.tolist()
    base = history.groupby("user_id").agg(
        impressions=("hour", "size"),
        unique_publishers=("publisher", "nunique"),
        mean_cpm=("cpm", "mean"),
        std_cpm=("cpm", "std"),
        active_hours=("hour", "nunique"),
    )
    base["std_cpm"] = base["std_cpm"].fillna(0.0)

    hour_pivot = (
        history.assign(hour_of_day=history["hour"] % 24)
        .pivot_table(index="user_id", columns="hour_of_day", values="hour", aggfunc="size", fill_value=0)
    )
    hour_pivot.columns = [f"hour_{col}" for col in hour_pivot.columns]

    publisher_pivot = (
        history.loc[history["publisher"].isin(top_publishers)]
        .pivot_table(index="user_id", columns="publisher", values="hour", aggfunc="size", fill_value=0)
    )
    publisher_pivot.columns = [f"pub_{col}" for col in publisher_pivot.columns]

    features = base.join(hour_pivot, how="left").join(publisher_pivot, how="left").fillna(0.0)
    features = users[["user_id", "sex", "age", "city_id"]].merge(
        features, left_on="user_id", right_index=True, how="left"
    ).fillna(0.0)

    feature_cols = [col for col in features.columns if col != "user_id"]
    x = StandardScaler().fit_transform(features[feature_cols].to_numpy(float))
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x)

    rng = np.random.default_rng(42)
    sample_size = min(6000, len(features))
    sample_idx = rng.choice(len(features), size=sample_size, replace=False)

    color = np.log1p(features["impressions"].to_numpy())[sample_idx]
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        coords[sample_idx, 0],
        coords[sample_idx, 1],
        c=color,
        cmap="viridis",
        s=10,
        alpha=0.7,
        linewidths=0,
    )
    ax.set_title("PCA-проекция поведенческих представлений пользователей")
    ax.set_xlabel(f"PC1, {pca.explained_variance_ratio_[0] * 100:.1f}% variance")
    ax.set_ylabel(f"PC2, {pca.explained_variance_ratio_[1] * 100:.1f}% variance")
    ax.grid(alpha=0.2)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("log(1 + число исторических показов)")
    ax.text(
        0.01,
        -0.12,
        "Важно: это не pretrained embeddings, а визуализация user representations по агрегатам истории.",
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
    )
    savefig(FIGURE_DIR / "07_user_representations_pca.png")


def write_figure_readme() -> None:
    text = """# Figures

Графики для презентации и пояснительной записки.

- `01_eda_history_overview.png` — активность показов, динамика инвентаря, CPM и площадки.
- `02_validate_targets_eda.png` — размер и длительность кампаний, распределение таргетов 1+/2+/3+.
- `03_temporal_split.png` — схема temporal split без использования будущих данных.
- `04_model_comparison.png` — сравнение финальной модели с ML-baselines.
- `05_replay_ablation.png` — абляция monthly/daily/weekly replay-компонент.
- `06_predicted_vs_actual_final_holdout.png` — качество финальной temporal-weighted модели на locked final holdout.
- `07_user_representations_pca.png` — PCA-проекция пользовательских поведенческих представлений.

Файл `07_user_representations_pca.png` корректно называть именно user representations, а не pretrained embeddings: представления построены из агрегатов истории показов.
"""
    (FIGURE_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    users, history, validate, answers = load_data()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_data_overview(history)
    plot_campaign_targets(validate, answers)
    plot_temporal_split()
    plot_model_comparison()
    plot_ablation()
    plot_prediction_quality(answers)
    plot_user_representations(users, history)
    write_figure_readme()
    print(f"Saved figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
