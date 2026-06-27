from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_DIR / "VK_Ads_reach_forecasting_colab.ipynb"
SOURCE_PATH = PROJECT_DIR / "src" / "vk_ads_solution.py"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def build_notebook() -> None:
    source_code = SOURCE_PATH.read_text(encoding="utf-8")
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "colab": {"provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    notebook["cells"] = [
        markdown(
            """
            # VK Ads: leak-free reach/frequency forecasting

            Ноутбук самодостаточный: он содержит код модели и может запускаться в Google Colab
            после загрузки файлов `users.tsv`, `history.tsv`, `validate.tsv`.

            Если рядом есть `validate_answers.tsv`, ноутбук дополнительно посчитает метрики.
            Для настоящего future test ответы не нужны: будет сохранён файл `predictions.tsv`.
            """
        ),
        markdown(
            """
            ## 1. Загрузка данных

            Варианты:

            - загрузить файлы в `/content/data`;
            - загрузить zip-архив с tsv-файлами и распаковать его в `/content/data`;
            - положить файлы в Google Drive и указать путь в переменной `DATA_DIR`.
            """
        ),
        code(
            """
            from pathlib import Path
            import os
            import zipfile

            IN_COLAB = "google.colab" in str(get_ipython())
            DEFAULT_DATA_DIR = Path("/content/data") if IN_COLAB else Path("data")

            # Можно заменить вручную, например:
            # DATA_DIR = Path("/content/drive/MyDrive/vk_ads_data")
            DATA_DIR = Path(os.environ.get("VK_ADS_DATA_DIR", DEFAULT_DATA_DIR))

            def maybe_upload_files():
                if not IN_COLAB:
                    return
                required = {"users.tsv", "history.tsv", "validate.tsv"}
                if DATA_DIR.exists() and required.issubset({p.name for p in DATA_DIR.iterdir()}):
                    return
                print("Если данные ещё не загружены, выберите tsv-файлы или zip-архив с ними.")
                from google.colab import files
                uploaded = files.upload()
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                for name in uploaded:
                    path = Path(name)
                    if path.suffix.lower() == ".zip":
                        with zipfile.ZipFile(path) as archive:
                            archive.extractall(DATA_DIR)
                    else:
                        Path(name).replace(DATA_DIR / Path(name).name)

            def find_data_dir(start: Path) -> Path:
                required = {"users.tsv", "history.tsv", "validate.tsv"}
                candidates = [
                    start,
                    Path.cwd(),
                    Path("/content/data"),
                    Path("/content"),
                ]
                for candidate in candidates:
                    if candidate.exists() and required.issubset({p.name for p in candidate.iterdir()}):
                        return candidate
                search_roots = [Path("/content"), Path.cwd()] if IN_COLAB else [Path.cwd()]
                for root in search_roots:
                    if not root.exists():
                        continue
                    for history_file in root.rglob("history.tsv"):
                        candidate = history_file.parent
                        if required.issubset({p.name for p in candidate.iterdir()}):
                            return candidate
                raise FileNotFoundError(
                    "Не нашла users.tsv/history.tsv/validate.tsv. "
                    "Загрузите их в /content/data или укажите DATA_DIR."
                )

            maybe_upload_files()
            DATA_DIR = find_data_dir(DATA_DIR)
            print("DATA_DIR =", DATA_DIR)
            print(sorted(p.name for p in DATA_DIR.glob("*.tsv")))
            """
        ),
        markdown("## 2. Код модели"),
        code(source_code),
        markdown(
            """
            ## 3. Прогноз

            Зафиксированная строгая конфигурация:

            - `daily_lags = 8`;
            - `weekly_lags = 5`;
            - веса `monthly/daily/weekly = 0.05 / 0.40 / 0.55`;
            - bias отключён.

            Если `validate.tsv` пересекается с `history.tsv`, ноутбук считает это открытым validation
            и режет историю до `min(hour_start)`. Если кампании лежат после всей истории, используется
            вся история.
            """
        ),
        code(
            """
            users, history, campaigns, answers = load_dataset(DATA_DIR)

            config = PastOnlyEnsembleConfig(
                daily_lags=8,
                weekly_lags=5,
                monthly_lags=1,
                monthly_weight=0.05,
                daily_weight=0.40,
                weekly_weight=0.55,
            )

            forecast_cutoff = int(campaigns["hour_start"].min())
            history_end = int(history["hour"].max())
            if forecast_cutoff <= history_end:
                print(
                    "Open-validation режим: validate пересекается с history. "
                    f"Использую только history.hour < {forecast_cutoff}."
                )
                forecast_history = history.loc[history["hour"] < forecast_cutoff].copy()
            else:
                print("Future-test режим: использую всю доступную history.")
                forecast_history = history.copy()

            predictions, components, diagnostics = predict_past_only_ensemble(
                forecast_history, campaigns, config=config
            )

            output_path = Path("predictions.tsv")
            diagnostics_path = Path("prediction_diagnostics.csv")
            save_predictions(predictions.round(4), output_path)
            diagnostics.to_csv(diagnostics_path, index=False)

            print("Saved:", output_path.resolve())
            print("Saved:", diagnostics_path.resolve())
            display(predictions.head())
            """
        ),
        markdown("## 4. Метрики, если доступны ответы"),
        code(
            """
            if answers is not None and len(answers) == len(campaigns):
                metric_full = smoothed_mean_log_accuracy_ratio(
                    answers.reset_index(drop=True), predictions.reset_index(drop=True)
                )
                print(f"Metric on loaded campaigns: {metric_full:.2f}%")

                try:
                    split = purged_three_way_split(
                        campaigns, development_fraction=0.6, calibration_fraction=0.2
                    )
                    final_metric = smoothed_mean_log_accuracy_ratio(
                        answers.iloc[split.test_idx].reset_index(drop=True),
                        predictions.iloc[split.test_idx].reset_index(drop=True),
                    )
                    print(
                        "Strict final-holdout diagnostic:",
                        f"{final_metric:.2f}%",
                        f"rows={len(split.test_idx)}",
                    )
                except Exception as error:
                    print("Temporal holdout не посчитан:", repr(error))
            else:
                print("validate_answers.tsv не найден: это нормально для future test.")
            """
        ),
        markdown(
            """
            ## 5. Скачать результат из Colab
            """
        ),
        code(
            """
            if IN_COLAB:
                from google.colab import files
                files.download("predictions.tsv")
            """
        ),
    ]
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Written: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_notebook()
