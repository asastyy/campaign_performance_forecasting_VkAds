from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from src.vk_ads_solution import (
    load_dataset,
    purged_three_way_split,
    smoothed_mean_log_accuracy_ratio,
)


DEFAULT_DATA_DIR = Path("/Users/anastasiasergeeva/Desktop/HSE/Сессия 2026/НИР Vk Ads/data")
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
PREDICTION_PATH = OUTPUT_DIR / "strict_locked_predictions.tsv"
LOCK_PATH = OUTPUT_DIR / "strict_model_lock.json"


def main() -> None:
    data_dir = Path(os.environ.get("VK_ADS_DATA_DIR", DEFAULT_DATA_DIR))
    _, _, campaigns, answers = load_dataset(data_dir)
    manifest = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    current_sha256 = hashlib.sha256(PREDICTION_PATH.read_bytes()).hexdigest()
    if current_sha256 != manifest["prediction_sha256"]:
        raise RuntimeError("Locked prediction hash mismatch; rerun selection before evaluation.")

    split = purged_three_way_split(
        campaigns, development_fraction=0.6, calibration_fraction=0.2
    )
    if split.test_start != int(manifest["test_start"]):
        raise RuntimeError("Temporal split no longer matches the locked model.")

    prediction = pd.read_csv(PREDICTION_PATH, sep="\t")
    final_answers = answers.iloc[split.test_idx].reset_index(drop=True)
    final_prediction = prediction.iloc[split.test_idx].reset_index(drop=True)
    metric = smoothed_mean_log_accuracy_ratio(final_answers, final_prediction)

    result = pd.DataFrame(
        [
            {
                "protocol": manifest["protocol"],
                "selected_candidate": manifest["selected_candidate"],
                "test_start": split.test_start,
                "rows": len(split.test_idx),
                "metric_percent": metric,
                "prediction_sha256": current_sha256,
            }
        ]
    )
    result.to_csv(OUTPUT_DIR / "strict_final_holdout_metrics.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
