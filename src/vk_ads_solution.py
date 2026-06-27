from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_COLUMNS = ["at_least_one", "at_least_two", "at_least_three"]
EPSILON = 0.005


def parse_int_list(value: object) -> np.ndarray:
    """Parse comma-separated integer ids from validate.tsv cells."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.int32)
    text = str(value)
    if not text:
        return np.array([], dtype=np.int32)
    return np.fromstring(text, sep=",", dtype=np.int32)


def smoothed_mean_log_accuracy_ratio(
    answers: pd.DataFrame,
    responses: pd.DataFrame,
    epsilon: float = 0.005,
) -> float:
    """Competition metric: lower is better."""
    values = []
    for column in TARGET_COLUMNS:
        values.append(
            np.abs(
                np.log(
                    (responses[column].to_numpy(dtype=float) + epsilon)
                    / (answers[column].to_numpy(dtype=float) + epsilon)
                )
            ).mean()
        )
    return float(np.round(100.0 * (np.exp(np.mean(values)) - 1.0), 2))


def load_dataset(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    users = pd.read_csv(data_dir / "users.tsv", sep="\t")
    history = pd.read_csv(data_dir / "history.tsv", sep="\t")
    validate = pd.read_csv(data_dir / "validate.tsv", sep="\t")
    answer_path = data_dir / "validate_answers.tsv"
    if answer_path.exists():
        answers = pd.read_csv(answer_path, sep="\t")
    else:
        answers = pd.DataFrame(columns=TARGET_COLUMNS)
    return users, history, validate, answers


def save_predictions(predictions: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions[TARGET_COLUMNS].to_csv(path, sep="\t", index=False)


def enforce_target_constraints(predictions: pd.DataFrame) -> pd.DataFrame:
    """Clip probabilities and enforce P(1+) >= P(2+) >= P(3+)."""

    result = predictions[TARGET_COLUMNS].astype(float).clip(0.0, 1.0).copy()
    result["at_least_two"] = np.minimum(result["at_least_one"], result["at_least_two"])
    result["at_least_three"] = np.minimum(result["at_least_two"], result["at_least_three"])
    return result


def geometric_prediction_blend(
    predictions: dict[str, pd.DataFrame],
    weights: dict[str, float],
    epsilon: float = EPSILON,
) -> pd.DataFrame:
    """Blend probability forecasts in the metric's logarithmic geometry."""

    if set(predictions) != set(weights):
        raise ValueError("Prediction and weight names must match.")
    total_weight = float(sum(weights.values()))
    if total_weight <= 0.0 or any(weight < 0.0 for weight in weights.values()):
        raise ValueError("Blend weights must be non-negative and sum to a positive value.")

    first = next(iter(predictions.values()))
    blended_log = pd.DataFrame(0.0, index=first.index, columns=TARGET_COLUMNS)
    for name, prediction in predictions.items():
        if len(prediction) != len(first):
            raise ValueError("All prediction frames must have equal length.")
        blended_log += (weights[name] / total_weight) * np.log(
            prediction[TARGET_COLUMNS].to_numpy(dtype=float) + epsilon
        )
    blended = np.exp(blended_log) - epsilon
    return enforce_target_constraints(blended)


def median_log_bias(
    answers: pd.DataFrame,
    predictions: pd.DataFrame,
    epsilon: float = EPSILON,
) -> pd.Series:
    """Return the metric-aligned median log residual for each target."""

    residual = np.log(
        (answers[TARGET_COLUMNS].to_numpy(dtype=float) + epsilon)
        / (predictions[TARGET_COLUMNS].to_numpy(dtype=float) + epsilon)
    )
    return pd.Series(np.median(residual, axis=0), index=TARGET_COLUMNS, dtype=float)


def apply_log_bias(
    predictions: pd.DataFrame,
    bias: pd.Series,
    epsilon: float = EPSILON,
) -> pd.DataFrame:
    """Apply a pre-fitted log bias without accessing evaluation targets."""

    adjusted = pd.DataFrame(index=predictions.index, columns=TARGET_COLUMNS, dtype=float)
    for column in TARGET_COLUMNS:
        adjusted[column] = (
            np.exp(np.log(predictions[column].to_numpy(dtype=float) + epsilon) + float(bias[column]))
            - epsilon
        )
    return enforce_target_constraints(adjusted)


def build_user_behavior_embeddings(
    users: pd.DataFrame,
    history: pd.DataFrame,
    n_components: int = 8,
) -> pd.DataFrame:
    """Build lightweight user embeddings from historical ad contacts.

    The source data has no ad/item identifier, so true ad-item embeddings are
    not identifiable. The applicable substitute is a behavioural user embedding
    from publisher, hour-of-day and day-of-week activity patterns.
    """

    user_index = pd.Index(users["user_id"].to_numpy(dtype=np.int64))
    row_id = user_index.get_indexer(history["user_id"].to_numpy(dtype=np.int64))
    keep = row_id >= 0
    row_id = row_id[keep]
    hist = history.loc[keep, ["hour", "publisher"]].copy()

    publishers = np.sort(history["publisher"].unique())
    publisher_to_col = {int(publisher): i for i, publisher in enumerate(publishers)}
    n_publishers = len(publishers)
    n_features = n_publishers + 24 + 7
    matrix = np.zeros((len(user_index), n_features), dtype=np.float32)

    publisher_cols = hist["publisher"].map(publisher_to_col).to_numpy(dtype=np.int32)
    np.add.at(matrix, (row_id, publisher_cols), 1.0)

    hours = hist["hour"].to_numpy(dtype=np.int32)
    np.add.at(matrix, (row_id, n_publishers + (hours % 24)), 1.0)
    np.add.at(matrix, (row_id, n_publishers + 24 + ((hours // 24) % 7)), 1.0)

    matrix = np.log1p(matrix)
    matrix -= matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0, keepdims=True)
    matrix /= np.where(scale < 1e-6, 1.0, scale)

    rank = min(int(n_components), matrix.shape[1])
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    embedding = u[:, :rank] * singular_values[:rank]
    columns = [f"user_emb_{i:02d}" for i in range(rank)]
    return pd.DataFrame(embedding, index=user_index, columns=columns)


def temporal_expanding_folds(
    validate: pd.DataFrame,
    n_splits: int = 5,
    min_train_fraction: float = 0.4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return expanding temporal folds sorted by campaign start hour."""

    order = (
        validate.assign(_row=np.arange(len(validate)))
        .sort_values(["hour_start", "hour_end", "_row"])
        .index.to_numpy()
    )
    first_test = int(np.floor(len(order) * min_train_fraction))
    test_blocks = np.array_split(order[first_test:], n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for block in test_blocks:
        if len(block) == 0:
            continue
        first_pos = int(np.where(order == block[0])[0][0])
        train_idx = order[:first_pos]
        if len(train_idx) > 0:
            folds.append((train_idx, block))
    return folds


@dataclass(frozen=True)
class PurgedTemporalFold:
    """A temporal fold whose training labels are fully observed at cutoff."""

    cutoff_hour: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class TemporalThreeWaySplit:
    """Development/calibration/final-test split with label-window purging."""

    calibration_start: int
    test_start: int
    development_idx: np.ndarray
    calibration_idx: np.ndarray
    pretest_idx: np.ndarray
    test_idx: np.ndarray


def purged_three_way_split(
    campaigns: pd.DataFrame,
    development_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
) -> TemporalThreeWaySplit:
    """Split complete start-time groups and purge labels crossing cutoffs."""

    if development_fraction <= 0.0 or calibration_fraction <= 0.0:
        raise ValueError("Split fractions must be positive.")
    if development_fraction + calibration_fraction >= 1.0:
        raise ValueError("Development and calibration must leave a final test block.")

    starts = np.sort(campaigns["hour_start"].unique())
    cumulative = np.array(
        [int((campaigns["hour_start"] <= start).sum()) for start in starts], dtype=int
    )
    development_target = int(np.ceil(len(campaigns) * development_fraction))
    calibration_target = int(
        np.ceil(len(campaigns) * (development_fraction + calibration_fraction))
    )
    calibration_position = int(np.searchsorted(cumulative, development_target, side="left") + 1)
    test_position = int(np.searchsorted(cumulative, calibration_target, side="left") + 1)
    if calibration_position >= len(starts) or test_position >= len(starts):
        raise ValueError("Not enough distinct start times for the requested split.")

    calibration_start = int(starts[calibration_position])
    test_start = int(starts[test_position])
    hour_start = campaigns["hour_start"].to_numpy(dtype=int)
    hour_end = campaigns["hour_end"].to_numpy(dtype=int)

    development_idx = np.flatnonzero(hour_end < calibration_start)
    calibration_idx = np.flatnonzero(
        (hour_start >= calibration_start)
        & (hour_start < test_start)
        & (hour_end < test_start)
    )
    pretest_idx = np.flatnonzero(hour_end < test_start)
    test_idx = np.flatnonzero(hour_start >= test_start)
    if not all(len(index) for index in (development_idx, calibration_idx, pretest_idx, test_idx)):
        raise ValueError("At least one temporal split block is empty after purging.")

    return TemporalThreeWaySplit(
        calibration_start=calibration_start,
        test_start=test_start,
        development_idx=development_idx,
        calibration_idx=calibration_idx,
        pretest_idx=pretest_idx,
        test_idx=test_idx,
    )


def purged_temporal_folds(
    campaigns: pd.DataFrame,
    n_splits: int = 5,
    min_train_fraction: float = 0.4,
) -> list[PurgedTemporalFold]:
    """Create expanding folds without equal-time or label-window leakage.

    Test blocks are built from complete groups of ``hour_start``. A campaign is
    eligible for training only when its full target window ended before the
    first test hour, so its answer would genuinely be known at prediction time.
    """

    if not 0.0 < min_train_fraction < 1.0:
        raise ValueError("min_train_fraction must be between 0 and 1.")

    starts = np.sort(campaigns["hour_start"].unique())
    row_target = int(np.ceil(len(campaigns) * min_train_fraction))
    cumulative = 0
    first_test_group = 0
    for position, start in enumerate(starts):
        cumulative += int((campaigns["hour_start"] == start).sum())
        if cumulative >= row_target:
            first_test_group = position + 1
            break

    test_start_groups = starts[first_test_group:]
    folds: list[PurgedTemporalFold] = []
    for block in np.array_split(test_start_groups, n_splits):
        if len(block) == 0:
            continue
        cutoff = int(block[0])
        train_idx = np.flatnonzero((campaigns["hour_end"] < cutoff).to_numpy())
        test_idx = np.flatnonzero(campaigns["hour_start"].isin(block).to_numpy())
        if len(train_idx) and len(test_idx):
            folds.append(PurgedTemporalFold(cutoff, train_idx, test_idx))
    return folds


def log_bias_calibrate_per_target(
    predictions: pd.DataFrame,
    calibration_answers: pd.DataFrame,
    calibration_predictions: pd.DataFrame | None = None,
    epsilon: float = 0.005,
) -> pd.DataFrame:
    """Apply per-target multiplicative calibration in log space."""

    if calibration_predictions is None:
        calibration_predictions = predictions

    result = predictions.copy()
    for column in TARGET_COLUMNS:
        bias = np.mean(
            np.log(
                (calibration_answers[column].to_numpy(dtype=float) + epsilon)
                / (calibration_predictions[column].to_numpy(dtype=float) + epsilon)
            )
        )
        result[column] = np.clip((result[column] + epsilon) * np.exp(bias) - epsilon, 0.0, 1.0)

    result["at_least_two"] = np.minimum(result["at_least_one"], result["at_least_two"])
    result["at_least_three"] = np.minimum(result["at_least_two"], result["at_least_three"])
    return result[TARGET_COLUMNS]


@dataclass
class AuctionReplayForecaster:
    """Supply x auction x aggregation forecaster based on historical auctions.

    The model reuses an observed time interval as the supply template. A
    campaign wins a session with probability:
    - 1.0 if its CPM is greater than the observed winning CPM in that session;
    - tie_probability if it only ties the observed winning CPM in that session;
    - 0 otherwise.

    Per-user session probabilities are aggregated into P(1+), P(2+), P(3+)
    via a small Poisson-binomial dynamic program.
    """

    session_gap_hours: int = 6

    def fit(self, history: pd.DataFrame) -> "AuctionReplayForecaster":
        history = history.sort_values(["user_id", "hour"]).reset_index(drop=True).copy()
        hour_gap = history.groupby("user_id")["hour"].diff().fillna(10**9)
        new_session = hour_gap >= self.session_gap_hours
        history["session_num"] = new_session.groupby(history["user_id"]).cumsum().astype(np.int32)
        history["session_key"] = (
            (history["user_id"].to_numpy(dtype=np.int64) << 32)
            + history["session_num"].to_numpy(dtype=np.int64)
        )

        history = history.sort_values("hour").reset_index(drop=True)
        self.hours_ = history["hour"].to_numpy(dtype=np.int32)
        self.users_ = history["user_id"].to_numpy(dtype=np.int32)
        self.publishers_ = history["publisher"].to_numpy(dtype=np.int16)
        self.cpm_ = history["cpm"].to_numpy(dtype=float)
        self.session_keys_ = history["session_key"].to_numpy(dtype=np.int64)
        self.min_hour_ = int(self.hours_[0])
        self.max_hour_ = int(self.hours_[-1])
        return self

    def predict(
        self,
        validate: pd.DataFrame,
        source_shift: int = 0,
        tie_policy: str = "single",
        tie_probability: float = 0.5,
        round_digits: int | None = None,
    ) -> pd.DataFrame:
        if not hasattr(self, "hours_"):
            raise RuntimeError("Call fit(history) before predict().")

        rows = []
        for campaign in validate.itertuples(index=False):
            source_start = int(campaign.hour_start) - int(source_shift)
            source_end = int(campaign.hour_end) - int(source_shift)
            rows.append(
                self._predict_one(
                    campaign,
                    source_start,
                    source_end,
                    tie_policy,
                    tie_probability,
                )
            )

        predictions = pd.DataFrame(rows, columns=TARGET_COLUMNS).clip(0.0, 1.0)
        predictions["at_least_two"] = np.minimum(
            predictions["at_least_one"], predictions["at_least_two"]
        )
        predictions["at_least_three"] = np.minimum(
            predictions["at_least_two"], predictions["at_least_three"]
        )
        if round_digits is not None:
            predictions = predictions.round(round_digits)
        return predictions

    def predict_past_ensemble(
        self,
        campaigns: pd.DataFrame,
        available_history_end: int,
        alignment_hours: int,
        max_lags: int,
        geometric: bool = True,
        tie_probability: float = 0.5,
        return_counts: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
        """Average several aligned windows lying fully before a forecast cutoff.

        For example, ``alignment_hours=168`` preserves hour-of-week. The first
        lag is the most recent complete aligned window available at cutoff;
        older lags are added while their full windows remain in history.
        """

        if alignment_hours <= 0 or max_lags <= 0:
            raise ValueError("alignment_hours and max_lags must be positive.")
        if available_history_end > self.max_hour_:
            raise ValueError("available_history_end exceeds fitted history.")
        if int(campaigns["hour_start"].min()) <= available_history_end:
            raise ValueError("Campaigns must start after available_history_end.")

        rows: list[np.ndarray] = []
        counts: list[int] = []
        for campaign in campaigns.itertuples(index=False):
            required_gap = int(campaign.hour_end) - int(available_history_end)
            first_shift = max(
                int(alignment_hours),
                int(math.ceil(required_gap / alignment_hours) * alignment_hours),
            )
            lag_predictions = []
            for lag in range(int(max_lags)):
                shift = first_shift + lag * int(alignment_hours)
                source_start = int(campaign.hour_start) - shift
                source_end = int(campaign.hour_end) - shift
                if source_end > available_history_end or source_end >= int(campaign.hour_start):
                    raise RuntimeError("Internal error: an ensemble source window is not past-only.")
                if source_start < self.min_hour_ or source_end > self.max_hour_:
                    continue
                lag_predictions.append(
                    self._predict_one(
                        campaign,
                        source_start,
                        source_end,
                        "single",
                        float(tie_probability),
                    )
                )

            if lag_predictions:
                values = np.asarray(lag_predictions, dtype=float)
                if geometric:
                    prediction = np.exp(np.log(values + EPSILON).mean(axis=0)) - EPSILON
                else:
                    prediction = values.mean(axis=0)
            else:
                prediction = np.zeros(len(TARGET_COLUMNS), dtype=float)
            rows.append(np.clip(prediction, 0.0, 1.0))
            counts.append(len(lag_predictions))

        predictions = enforce_target_constraints(pd.DataFrame(rows, columns=TARGET_COLUMNS))
        if return_counts:
            return predictions, np.asarray(counts, dtype=np.int16)
        return predictions

    def build_decomposition_features(
        self,
        campaigns: pd.DataFrame,
        source_shift: int,
    ) -> pd.DataFrame:
        """Build leak-free supply/auction/session aggregates from a past window.

        This method deliberately rejects replay of the target interval. Every
        source window must end before the corresponding campaign starts.
        """

        rows = []
        for campaign in campaigns.itertuples(index=False):
            source_start = int(campaign.hour_start) - int(source_shift)
            source_end = int(campaign.hour_end) - int(source_shift)
            if source_end >= int(campaign.hour_start):
                raise ValueError(
                    "Unsafe source window: source_end must be earlier than target hour_start."
                )
            rows.append(self._decomposition_features_one(campaign, source_start, source_end))
        return pd.DataFrame(rows).fillna(0.0)

    def _decomposition_features_one(
        self,
        campaign: object,
        source_start: int,
        source_end: int,
    ) -> dict[str, float]:
        requested_hours = max(source_end - source_start + 1, 1)
        observed_start = max(source_start, self.min_hour_)
        observed_end = min(source_end, self.max_hour_)
        observed_hours = max(observed_end - observed_start + 1, 0)
        row: dict[str, float] = {
            "source_window_available_share": observed_hours / requested_hours,
            "source_window_missing": float(observed_hours < requested_hours),
        }
        if observed_hours == 0:
            return self._empty_decomposition_features(row)

        left = np.searchsorted(self.hours_, observed_start, side="left")
        right = np.searchsorted(self.hours_, observed_end, side="right")
        window = slice(left, right)
        publishers = parse_int_list(campaign.publishers)
        audience = parse_int_list(campaign.user_ids)
        audience_size = max(int(campaign.audience_size), 1)
        cpm = float(campaign.cpm)

        context_mask = np.isin(self.publishers_[window], publishers)
        context_mask &= np.isin(self.users_[window], audience)
        context_users = self.users_[window][context_mask]
        context_publishers = self.publishers_[window][context_mask]
        context_cpm = self.cpm_[window][context_mask]
        context_sessions = self.session_keys_[window][context_mask]

        row["context_missing"] = float(len(context_cpm) == 0)
        row["context_opportunities_per_user"] = len(context_cpm) / audience_size
        row["context_active_user_share"] = (
            len(np.unique(context_users)) / audience_size if len(context_users) else 0.0
        )
        row["context_active_publisher_share"] = (
            len(np.unique(context_publishers)) / max(len(publishers), 1)
            if len(context_publishers)
            else 0.0
        )
        row["context_sessions_per_user"] = (
            len(np.unique(context_sessions)) / audience_size if len(context_sessions) else 0.0
        )

        if len(context_cpm):
            quantiles = np.quantile(context_cpm, [0.25, 0.5, 0.75, 0.9])
            for name, value in zip(("p25", "p50", "p75", "p90"), quantiles):
                row[f"context_cpm_{name}"] = float(value)
                row[f"campaign_to_cpm_{name}_ratio"] = cpm / max(float(value), 1e-6)
            row["context_cpm_mean"] = float(context_cpm.mean())
            row["context_cpm_std"] = float(context_cpm.std())
            row["strict_opportunity_share"] = float((context_cpm < cpm).mean())
            row["tie_opportunity_share"] = float((context_cpm == cpm).mean())

            session_users = (np.unique(context_sessions) >> 32).astype(np.int64)
            sessions_per_active = np.unique(session_users, return_counts=True)[1]
            row["sessions_per_active_user_mean"] = float(sessions_per_active.mean())
            row["sessions_per_active_user_p75"] = float(np.quantile(sessions_per_active, 0.75))
            row["sessions_per_active_user_p90"] = float(np.quantile(sessions_per_active, 0.9))
        else:
            for name in ("p25", "p50", "p75", "p90"):
                row[f"context_cpm_{name}"] = 0.0
                row[f"campaign_to_cpm_{name}_ratio"] = 0.0
            row.update(
                {
                    "context_cpm_mean": 0.0,
                    "context_cpm_std": 0.0,
                    "strict_opportunity_share": 0.0,
                    "tie_opportunity_share": 0.0,
                    "sessions_per_active_user_mean": 0.0,
                    "sessions_per_active_user_p75": 0.0,
                    "sessions_per_active_user_p90": 0.0,
                }
            )

        eligible = context_cpm <= cpm
        eligible_sessions = context_sessions[eligible]
        strict_wins = context_cpm[eligible] < cpm
        row["eligible_missing"] = float(len(eligible_sessions) == 0)
        row["eligible_opportunities_per_user"] = len(eligible_sessions) / audience_size
        row["eligible_sessions_per_user"] = (
            len(np.unique(eligible_sessions)) / audience_size if len(eligible_sessions) else 0.0
        )

        prediction_variants = {
            "strict": self._aggregate_sessions(
                eligible_sessions, strict_wins, audience_size, "ignore", 0.0
            ),
            "tie_half": self._aggregate_sessions(
                eligible_sessions, strict_wins, audience_size, "single", 0.5
            ),
            "tie_full": self._aggregate_sessions(
                eligible_sessions, strict_wins, audience_size, "always", 1.0
            ),
            "tie_independent": self._aggregate_sessions(
                eligible_sessions, strict_wins, audience_size, "independent", 0.5
            ),
            "supply_ceiling": self._aggregate_sessions(
                context_sessions,
                np.ones(len(context_sessions), dtype=bool),
                audience_size,
                "single",
                0.5,
            ),
        }
        for variant, values in prediction_variants.items():
            for target, value in zip(TARGET_COLUMNS, values):
                row[f"{variant}_{target}"] = float(value)
                row[f"{variant}_{target}_zero"] = float(value <= 0.0)
                row[f"log_{variant}_{target}"] = float(np.log(value + EPSILON))
        return row

    @staticmethod
    def _empty_decomposition_features(row: dict[str, float]) -> dict[str, float]:
        row.update(
            {
                "context_missing": 1.0,
                "context_opportunities_per_user": 0.0,
                "context_active_user_share": 0.0,
                "context_active_publisher_share": 0.0,
                "context_sessions_per_user": 0.0,
                "context_cpm_mean": 0.0,
                "context_cpm_std": 0.0,
                "strict_opportunity_share": 0.0,
                "tie_opportunity_share": 0.0,
                "sessions_per_active_user_mean": 0.0,
                "sessions_per_active_user_p75": 0.0,
                "sessions_per_active_user_p90": 0.0,
                "eligible_missing": 1.0,
                "eligible_opportunities_per_user": 0.0,
                "eligible_sessions_per_user": 0.0,
            }
        )
        for name in ("p25", "p50", "p75", "p90"):
            row[f"context_cpm_{name}"] = 0.0
            row[f"campaign_to_cpm_{name}_ratio"] = 0.0
        for variant in ("strict", "tie_half", "tie_full", "tie_independent", "supply_ceiling"):
            for target in TARGET_COLUMNS:
                row[f"{variant}_{target}"] = 0.0
                row[f"{variant}_{target}_zero"] = 1.0
                row[f"log_{variant}_{target}"] = float(np.log(EPSILON))
        return row

    def _predict_one(
        self,
        campaign: object,
        source_start: int,
        source_end: int,
        tie_policy: str,
        tie_probability: float,
    ) -> tuple[float, float, float]:
        if source_end < self.min_hour_ or source_start > self.max_hour_:
            return 0.0, 0.0, 0.0

        left = np.searchsorted(self.hours_, source_start, side="left")
        right = np.searchsorted(self.hours_, source_end, side="right")
        if left >= right:
            return 0.0, 0.0, 0.0

        window = slice(left, right)
        campaign_publishers = parse_int_list(campaign.publishers)
        campaign_users = parse_int_list(campaign.user_ids)
        campaign_cpm = float(campaign.cpm)

        mask = np.isin(self.publishers_[window], campaign_publishers)
        mask &= np.isin(self.users_[window], campaign_users)
        mask &= self.cpm_[window] <= campaign_cpm
        if not bool(mask.any()):
            return 0.0, 0.0, 0.0

        session_keys = self.session_keys_[window][mask]
        strict_wins = self.cpm_[window][mask] < campaign_cpm
        return self._aggregate_sessions(
            session_keys,
            strict_wins,
            int(campaign.audience_size),
            tie_policy,
            tie_probability,
        )

    def _aggregate_sessions(
        self,
        session_keys: np.ndarray,
        strict_wins: np.ndarray,
        audience_size: int,
        tie_policy: str,
        tie_probability: float,
    ) -> tuple[float, float, float]:
        unique_sessions, inverse = np.unique(session_keys, return_inverse=True)
        strict_by_session = np.zeros(len(unique_sessions), dtype=bool)
        np.logical_or.at(strict_by_session, inverse, strict_wins)

        if tie_policy == "ignore":
            session_prob = strict_by_session.astype(float)
        elif tie_policy == "single":
            session_prob = np.where(strict_by_session, 1.0, tie_probability)
        elif tie_policy == "independent":
            counts = np.bincount(inverse, minlength=len(unique_sessions))
            session_prob = np.where(
                strict_by_session,
                1.0,
                1.0 - np.power(1.0 - tie_probability, counts),
            )
        elif tie_policy == "always":
            session_prob = np.ones(len(unique_sessions), dtype=float)
        else:
            raise ValueError(f"Unknown tie_policy: {tie_policy}")

        keep = session_prob > 0
        if not bool(keep.any()):
            return 0.0, 0.0, 0.0

        session_users = (unique_sessions[keep] >> 32).astype(np.int64)
        session_prob = session_prob[keep]
        order = np.argsort(session_users)
        session_users = session_users[order]
        session_prob = session_prob[order]

        sums = np.zeros(3, dtype=float)
        start = 0
        while start < len(session_users):
            end = start + 1
            while end < len(session_users) and session_users[end] == session_users[start]:
                end += 1
            sums += self._at_least_probabilities(session_prob[start:end])
            start = end

        return tuple((sums / max(audience_size, 1)).tolist())

    @staticmethod
    def _at_least_probabilities(probabilities: Iterable[float]) -> np.ndarray:
        p0 = 1.0
        p1 = 0.0
        p2 = 0.0
        for p in probabilities:
            q = 1.0 - float(p)
            p2 = p2 * q + p1 * p
            p1 = p1 * q + p0 * p
            p0 = p0 * q
        result = np.array([1.0 - p0, 1.0 - p0 - p1, 1.0 - p0 - p1 - p2], dtype=float)
        return np.clip(result, 0.0, 1.0)


@dataclass(frozen=True)
class PastOnlyEnsembleConfig:
    """Configuration chosen without target-period replay."""

    daily_lags: int = 8
    weekly_lags: int = 5
    monthly_lags: int = 1
    daily_weight: float = 0.4
    weekly_weight: float = 0.55
    monthly_weight: float = 0.05


@dataclass(frozen=True)
class TargetBlendConfig:
    """Per-target component mix for threshold reach forecasts."""

    daily_lags: int
    weekly_lags: int
    monthly_lags: int = 1
    daily_weight: float = 0.0
    weekly_weight: float = 0.0
    monthly_weight: float = 0.0


def predict_past_only_ensemble(
    history: pd.DataFrame,
    campaigns: pd.DataFrame,
    config: PastOnlyEnsembleConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Forecast future campaigns using only fully observed historical windows.

    ``history`` must end before every campaign starts. The returned diagnostics
    contain the number of complete lag windows used by each component.
    """

    config = config or PastOnlyEnsembleConfig()
    history_end = int(history["hour"].max())
    if int(campaigns["hour_start"].min()) <= history_end:
        raise ValueError("Future campaigns overlap supplied history; trim history at forecast cutoff.")

    forecaster = AuctionReplayForecaster(session_gap_hours=6).fit(history)
    monthly, monthly_count = forecaster.predict_past_ensemble(
        campaigns,
        available_history_end=history_end,
        alignment_hours=24 * 31,
        max_lags=config.monthly_lags,
        return_counts=True,
    )
    daily, daily_count = forecaster.predict_past_ensemble(
        campaigns,
        available_history_end=history_end,
        alignment_hours=24,
        max_lags=config.daily_lags,
        return_counts=True,
    )
    weekly, weekly_count = forecaster.predict_past_ensemble(
        campaigns,
        available_history_end=history_end,
        alignment_hours=24 * 7,
        max_lags=config.weekly_lags,
        return_counts=True,
    )
    components = {"monthly": monthly, "daily": daily, "weekly": weekly}
    blended = geometric_prediction_blend(
        components,
        {
            "monthly": config.monthly_weight,
            "daily": config.daily_weight,
            "weekly": config.weekly_weight,
        },
    )
    diagnostics = pd.DataFrame(
        {
            "monthly_windows_used": monthly_count,
            "daily_windows_used": daily_count,
            "weekly_windows_used": weekly_count,
            "monthly_missing": (monthly_count == 0).astype(np.int8),
            "daily_missing": (daily_count == 0).astype(np.int8),
            "weekly_missing": (weekly_count == 0).astype(np.int8),
        }
    )
    return blended, components, diagnostics


def targetwise_geometric_prediction_blend(
    component_bank: dict[str, pd.DataFrame],
    target_configs: dict[str, TargetBlendConfig],
    epsilon: float = EPSILON,
) -> pd.DataFrame:
    """Blend monthly/daily/weekly components with separate weights per target."""

    missing_targets = set(TARGET_COLUMNS) - set(target_configs)
    if missing_targets:
        raise ValueError(f"Missing target configs: {sorted(missing_targets)}")

    first = next(iter(component_bank.values()))
    blended = pd.DataFrame(index=first.index, columns=TARGET_COLUMNS, dtype=float)
    for target in TARGET_COLUMNS:
        config = target_configs[target]
        keys_and_weights = {
            f"monthly_{config.monthly_lags}": config.monthly_weight,
            f"daily_{config.daily_lags}": config.daily_weight,
            f"weekly_{config.weekly_lags}": config.weekly_weight,
        }
        total_weight = float(sum(keys_and_weights.values()))
        if total_weight <= 0.0 or any(weight < 0.0 for weight in keys_and_weights.values()):
            raise ValueError("Target blend weights must be non-negative and sum to a positive value.")

        blended_log = np.zeros(len(first), dtype=float)
        for key, weight in keys_and_weights.items():
            if weight == 0.0:
                continue
            if key not in component_bank:
                raise ValueError(f"Component {key!r} is missing from component_bank.")
            component = component_bank[key]
            if len(component) != len(first):
                raise ValueError("All component frames must have equal length.")
            blended_log += (weight / total_weight) * np.log(
                component[target].to_numpy(dtype=float) + epsilon
            )
        blended[target] = np.exp(blended_log) - epsilon

    return enforce_target_constraints(blended)


def predict_targetwise_past_only_ensemble(
    history: pd.DataFrame,
    campaigns: pd.DataFrame,
    target_configs: dict[str, TargetBlendConfig],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Forecast with target-specific lag counts and component weights."""

    history_end = int(history["hour"].max())
    if int(campaigns["hour_start"].min()) <= history_end:
        raise ValueError("Future campaigns overlap supplied history; trim history at forecast cutoff.")

    forecaster = AuctionReplayForecaster(session_gap_hours=6).fit(history)
    component_bank: dict[str, pd.DataFrame] = {}
    diagnostics = pd.DataFrame(index=np.arange(len(campaigns)))

    monthly_lags = sorted({config.monthly_lags for config in target_configs.values()})
    daily_lags = sorted({config.daily_lags for config in target_configs.values()})
    weekly_lags = sorted({config.weekly_lags for config in target_configs.values()})

    for lags in monthly_lags:
        prediction, counts = forecaster.predict_past_ensemble(
            campaigns,
            available_history_end=history_end,
            alignment_hours=24 * 31,
            max_lags=lags,
            return_counts=True,
        )
        component_bank[f"monthly_{lags}"] = prediction
        diagnostics[f"monthly_{lags}_windows_used"] = counts
        diagnostics[f"monthly_{lags}_missing"] = (counts == 0).astype(np.int8)

    for lags in daily_lags:
        prediction, counts = forecaster.predict_past_ensemble(
            campaigns,
            available_history_end=history_end,
            alignment_hours=24,
            max_lags=lags,
            return_counts=True,
        )
        component_bank[f"daily_{lags}"] = prediction
        diagnostics[f"daily_{lags}_windows_used"] = counts
        diagnostics[f"daily_{lags}_missing"] = (counts == 0).astype(np.int8)

    for lags in weekly_lags:
        prediction, counts = forecaster.predict_past_ensemble(
            campaigns,
            available_history_end=history_end,
            alignment_hours=24 * 7,
            max_lags=lags,
            return_counts=True,
        )
        component_bank[f"weekly_{lags}"] = prediction
        diagnostics[f"weekly_{lags}_windows_used"] = counts
        diagnostics[f"weekly_{lags}_missing"] = (counts == 0).astype(np.int8)

    blended = targetwise_geometric_prediction_blend(component_bank, target_configs)
    return blended, component_bank, diagnostics


def build_campaign_features(
    validate: pd.DataFrame,
    users: pd.DataFrame | None = None,
    history: pd.DataFrame | None = None,
    extra_predictions: dict[str, pd.DataFrame] | None = None,
    include_user_embeddings: bool = True,
) -> pd.DataFrame:
    """Build compact campaign-level features for tabular baselines."""
    rows = []
    user_activity = None
    user_embeddings = None
    if history is not None:
        user_activity = history.groupby("user_id").agg(
            hist_impressions=("hour", "size"),
            hist_publishers=("publisher", "nunique"),
            hist_cpm_mean=("cpm", "mean"),
            hist_cpm_p75=("cpm", lambda x: float(np.quantile(x, 0.75))),
        )
        if users is not None and include_user_embeddings:
            user_embeddings = build_user_behavior_embeddings(users, history)

    users_indexed = users.set_index("user_id") if users is not None else None
    for campaign in validate.itertuples(index=False):
        publishers = parse_int_list(campaign.publishers)
        audience = parse_int_list(campaign.user_ids)
        duration = int(campaign.hour_end) - int(campaign.hour_start) + 1
        row = {
            "cpm": float(campaign.cpm),
            "log_cpm": np.log1p(float(campaign.cpm)),
            "hour_start": int(campaign.hour_start),
            "hour_end": int(campaign.hour_end),
            "duration": duration,
            "log_duration": np.log1p(duration),
            "audience_size": int(campaign.audience_size),
            "log_audience_size": np.log1p(int(campaign.audience_size)),
            "n_publishers": int(len(publishers)),
            "start_hour_of_day": int(campaign.hour_start) % 24,
            "end_hour_of_day": int(campaign.hour_end) % 24,
            "start_day_of_week": (int(campaign.hour_start) // 24) % 7,
        }
        row["start_sin_24"] = np.sin(2 * np.pi * row["start_hour_of_day"] / 24)
        row["start_cos_24"] = np.cos(2 * np.pi * row["start_hour_of_day"] / 24)

        publisher_set = set(int(x) for x in publishers)
        for publisher in range(1, 22):
            row[f"publisher_{publisher}"] = int(publisher in publisher_set)

        if users_indexed is not None and len(audience) > 0:
            current_users = users_indexed.reindex(audience)
            age = current_users["age"].replace(0, np.nan)
            row["aud_age_mean"] = float(age.mean()) if not np.isnan(age.mean()) else 0.0
            row["aud_age_known_share"] = float(age.notna().mean())
            row["aud_city_known_share"] = float((current_users["city_id"] != 0).mean())
            row["aud_female_share"] = float((current_users["sex"] == 1).mean())
            row["aud_male_share"] = float((current_users["sex"] == 2).mean())

        if user_activity is not None and len(audience) > 0:
            current_activity = user_activity.reindex(audience).fillna(0)
            for column in current_activity.columns:
                row[f"aud_{column}_mean"] = float(current_activity[column].mean())
                row[f"aud_{column}_p75"] = float(current_activity[column].quantile(0.75))

        if user_embeddings is not None and len(audience) > 0:
            current_embeddings = user_embeddings.reindex(audience).fillna(0.0)
            for column in current_embeddings.columns:
                row[f"aud_{column}_mean"] = float(current_embeddings[column].mean())
                row[f"aud_{column}_std"] = float(current_embeddings[column].std(ddof=0))

        rows.append(row)

    features = pd.DataFrame(rows)
    if extra_predictions:
        for name, prediction in extra_predictions.items():
            for column in TARGET_COLUMNS:
                features[f"{name}_{column}"] = prediction[column].to_numpy(dtype=float)
    return features.fillna(0.0)


def build_leak_free_campaign_features(
    campaigns: pd.DataFrame,
    users: pd.DataFrame,
    forecaster: AuctionReplayForecaster,
    source_shift: int = 744,
) -> pd.DataFrame:
    """Combine static inputs with past-window decomposition features."""

    static = build_campaign_features(
        campaigns,
        users=users,
        history=None,
        extra_predictions=None,
        include_user_embeddings=False,
    )
    static = static.drop(columns=["hour_start", "hour_end"], errors="ignore")
    static["duration_short"] = (static["duration"] <= 12).astype(float)
    static["duration_long"] = (static["duration"] >= 168).astype(float)
    static["audience_small"] = (static["audience_size"] <= 500).astype(float)
    static["audience_large"] = (static["audience_size"] >= 1500).astype(float)
    static["cpm_low"] = (static["cpm"] <= 80).astype(float)
    static["cpm_high"] = (static["cpm"] >= 250).astype(float)

    decomposition = forecaster.build_decomposition_features(campaigns, source_shift=source_shift)
    return pd.concat(
        [static.reset_index(drop=True), decomposition.reset_index(drop=True)], axis=1
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
