from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from vk_ads_solution import (  # noqa: E402
    AuctionReplayForecaster,
    TARGET_COLUMNS,
    TargetBlendConfig,
    enforce_target_constraints,
    geometric_prediction_blend,
    predict_past_only_ensemble,
    predict_targetwise_past_only_ensemble,
    purged_three_way_split,
    purged_temporal_folds,
    targetwise_geometric_prediction_blend,
)


def synthetic_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour": [0, 1, 2, 4, 6, 7, 8, 9],
            "cpm": [50, 70, 40, 60, 50, 80, 40, 60],
            "publisher": [1] * 8,
            "user_id": [1] * 8,
        }
    )


def synthetic_campaigns(start: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cpm": [65],
            "hour_start": [start],
            "hour_end": [start + 1],
            "publishers": ["1"],
            "audience_size": [1],
            "user_ids": ["1"],
        }
    )


def test_past_ensemble_rejects_history_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        predict_past_only_ensemble(synthetic_history(), synthetic_campaigns(start=9))


def test_decomposition_rejects_target_period_replay() -> None:
    model = AuctionReplayForecaster().fit(synthetic_history())
    with pytest.raises(ValueError, match="Unsafe source window"):
        model.build_decomposition_features(synthetic_campaigns(), source_shift=0)


def test_past_ensemble_uses_only_complete_lags() -> None:
    model = AuctionReplayForecaster().fit(synthetic_history())
    prediction, counts = model.predict_past_ensemble(
        synthetic_campaigns(),
        available_history_end=9,
        alignment_hours=2,
        max_lags=2,
        return_counts=True,
    )
    assert counts.tolist() == [2]
    assert prediction.shape == (1, 3)
    assert np.isfinite(prediction.to_numpy()).all()


def test_purged_folds_have_observed_train_labels_and_disjoint_start_times() -> None:
    campaigns = pd.DataFrame(
        {
            "hour_start": [10, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            "hour_end": [12, 25, 22, 45, 42, 55, 65, 75, 85, 95],
        }
    )
    campaigns.index = np.arange(100, 110)
    folds = purged_temporal_folds(campaigns, n_splits=3, min_train_fraction=0.3)
    assert folds
    for fold in folds:
        assert (campaigns.iloc[fold.train_idx]["hour_end"] < fold.cutoff_hour).all()
        train_starts = set(campaigns.iloc[fold.train_idx]["hour_start"])
        test_starts = set(campaigns.iloc[fold.test_idx]["hour_start"])
        assert train_starts.isdisjoint(test_starts)


def test_geometric_blend_preserves_probability_constraints() -> None:
    first = pd.DataFrame([[0.4, 0.3, 0.2]], columns=TARGET_COLUMNS)
    second = pd.DataFrame([[0.2, 0.1, 0.05]], columns=TARGET_COLUMNS)
    blend = geometric_prediction_blend(
        {"first": first, "second": second}, {"first": 0.6, "second": 0.4}
    )
    constrained = enforce_target_constraints(blend)
    assert constrained.at[0, "at_least_one"] >= constrained.at[0, "at_least_two"]
    assert constrained.at[0, "at_least_two"] >= constrained.at[0, "at_least_three"]


def test_targetwise_blend_preserves_constraints_and_overlap_guard() -> None:
    first = pd.DataFrame([[0.4, 0.2, 0.1]], columns=TARGET_COLUMNS)
    second = pd.DataFrame([[0.3, 0.25, 0.2]], columns=TARGET_COLUMNS)
    third = pd.DataFrame([[0.5, 0.1, 0.05]], columns=TARGET_COLUMNS)
    configs = {
        "at_least_one": TargetBlendConfig(1, 1, daily_weight=0.2, weekly_weight=0.8),
        "at_least_two": TargetBlendConfig(1, 1, monthly_weight=0.1, daily_weight=0.2, weekly_weight=0.7),
        "at_least_three": TargetBlendConfig(1, 1, daily_weight=1.0),
    }
    blend = targetwise_geometric_prediction_blend(
        {"monthly_1": first, "daily_1": second, "weekly_1": third}, configs
    )
    assert blend.at[0, "at_least_one"] >= blend.at[0, "at_least_two"]
    assert blend.at[0, "at_least_two"] >= blend.at[0, "at_least_three"]

    with pytest.raises(ValueError, match="overlap"):
        predict_targetwise_past_only_ensemble(
            synthetic_history(), synthetic_campaigns(start=9), configs
        )


def test_three_way_split_purges_both_boundaries() -> None:
    campaigns = pd.DataFrame(
        {
            "hour_start": np.arange(10, 130, 10),
            "hour_end": [12, 22, 42, 43, 52, 75, 72, 83, 92, 103, 112, 123],
        },
        index=np.arange(200, 212),
    )
    split = purged_three_way_split(
        campaigns, development_fraction=0.5, calibration_fraction=0.25
    )
    assert (campaigns.iloc[split.development_idx]["hour_end"] < split.calibration_start).all()
    assert (campaigns.iloc[split.calibration_idx]["hour_end"] < split.test_start).all()
    assert (campaigns.iloc[split.pretest_idx]["hour_end"] < split.test_start).all()
    assert (campaigns.iloc[split.test_idx]["hour_start"] >= split.test_start).all()
