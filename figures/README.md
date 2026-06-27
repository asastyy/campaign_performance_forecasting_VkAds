# Figures

Графики для презентации и пояснительной записки.

- `01_eda_history_overview.png` — активность показов, динамика инвентаря, CPM и площадки.
- `02_validate_targets_eda.png` — размер и длительность кампаний, распределение таргетов 1+/2+/3+.
- `03_temporal_split.png` — схема temporal split без использования будущих данных.
- `04_model_comparison.png` — сравнение финальной модели с ML-baselines.
- `05_replay_ablation.png` — абляция monthly/daily/weekly replay-компонент.
- `06_predicted_vs_actual_final_holdout.png` — качество прогноза на locked final holdout.
- `07_user_representations_pca.png` — PCA-проекция пользовательских поведенческих представлений.
- `08_segment_error_analysis.png` — ошибка финальной модели по сегментам кампаний.
- `09_calibration_bins.png` — post-hoc calibration diagnostics по бинам предсказаний.
- `10_target_error_breakdown.png` — разложение ошибки по таргетам 1+/2+/3+.
- `11_candidate_selection_curve.png` — устойчивость top-конфигураций на calibration/pretest.
- `12_error_distribution.png` — распределение absolute log-ratio errors.

Файл `07_user_representations_pca.png` корректно называть именно user representations, а не pretrained embeddings: представления построены из агрегатов истории показов.
