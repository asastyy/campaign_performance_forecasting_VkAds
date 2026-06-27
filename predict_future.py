from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from src.vk_ads_solution import (
    TargetBlendConfig,
    apply_log_bias,
    PastOnlyEnsembleConfig,
    predict_past_only_ensemble,
    predict_targetwise_past_only_ensemble,
    save_predictions,
)


DEFAULT_DATA_DIR = Path("/Users/anastasiasergeeva/Desktop/HSE/Сессия 2026/НИР Vk Ads/data")
DEFAULT_LOCK_PATH = Path(__file__).resolve().parent / "outputs" / "strict_model_lock.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast campaigns strictly after available history.")
    parser.add_argument("campaigns", type=Path, help="Future campaigns TSV in validate.tsv format.")
    parser.add_argument("output", type=Path, help="Output TSV with the three target columns.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("VK_ADS_DATA_DIR", DEFAULT_DATA_DIR)),
    )
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = pd.read_csv(args.data_dir / "history.tsv", sep="\t")
    future_campaigns = pd.read_csv(args.campaigns, sep="\t")

    manifest = json.loads(args.lock_file.read_text(encoding="utf-8"))
    model_family = manifest.get("model_family", "scalar_past_only_blend")
    if model_family == "targetwise_past_only_blend":
        target_configs = {
            target: TargetBlendConfig(
                daily_lags=int(values["daily_lags"]),
                weekly_lags=int(values["weekly_lags"]),
                monthly_lags=int(values.get("monthly_lags", 1)),
                daily_weight=float(values["daily_weight"]),
                weekly_weight=float(values["weekly_weight"]),
                monthly_weight=float(values["monthly_weight"]),
            )
            for target, values in manifest["target_configs"].items()
        }
        prediction, _, diagnostics = predict_targetwise_past_only_ensemble(
            history, future_campaigns, target_configs=target_configs
        )
    elif model_family == "scalar_past_only_blend":
        config = PastOnlyEnsembleConfig(
            daily_lags=int(manifest["daily_lags"]),
            weekly_lags=int(manifest["weekly_lags"]),
            monthly_lags=1,
            daily_weight=float(manifest["daily_weight"]),
            weekly_weight=float(manifest["weekly_weight"]),
            monthly_weight=float(manifest["monthly_weight"]),
        )
        prediction, _, diagnostics = predict_past_only_ensemble(
            history, future_campaigns, config=config
        )
    else:
        raise ValueError(f"Unknown model_family in lock file: {model_family}")

    if bool(manifest["use_bias"]):
        bias = pd.Series(manifest["pretest_bias"], dtype=float)
        prediction = apply_log_bias(prediction, bias)
        print("Applied locked historical bias:")
        print(bias.to_string())

    diagnostics["locked_candidate"] = str(manifest["selected_candidate"])
    diagnostics["model_prediction_sha256"] = str(manifest["prediction_sha256"])

    save_predictions(prediction.round(4), args.output)
    diagnostics_path = args.output.with_name(f"{args.output.stem}_diagnostics.csv")
    diagnostics.to_csv(diagnostics_path, index=False)
    print(f"Saved predictions: {args.output}")
    print(f"Saved diagnostics: {diagnostics_path}")


if __name__ == "__main__":
    main()
