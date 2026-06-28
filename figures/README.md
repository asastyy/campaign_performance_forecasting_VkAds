# Figures

Графики для презентации и пояснительной записки.

- `01_eda_history_overview.png` — активность показов, динамика инвентаря, CPM и площадки.
- `02_validate_targets_eda.png` — размер и длительность кампаний, распределение таргетов 1+/2+/3+.
- `03_temporal_split.png` — схема temporal split без использования будущих данных.
- `04_model_comparison.png` — сравнение финальной модели с ML-baselines.
- `05_replay_ablation.png` — абляция monthly/daily/weekly replay-компонент.
- `06_predicted_vs_actual_final_holdout.png` — качество финальной temporal-weighted модели на locked final holdout.
- `07_user_representations_pca.png` — PCA-проекция пользовательских поведенческих представлений.

Файл `07_user_representations_pca.png` корректно называть именно user representations, а не pretrained embeddings: представления построены из агрегатов истории показов.
