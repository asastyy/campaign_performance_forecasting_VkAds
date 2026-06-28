# GitHub delta upload: VK Ads temporal experiments

Эта папка содержит файлы, которые нужно догрузить в репозиторий поверх предыдущей версии.

## Главное обновление

Финальная enhanced-модель после новых экспериментов:

- `temporal_weighted_targetwise_ridge_residual`
- locked final holdout: `9.29%`
- raw final holdout: `9.289324%`

Основная идея: decomposed replay остается базой, а поверх него добавляется
recency-weighted log-residual Ridge calibration. Для `1+`, `2+`, `3+` выбраны
отдельные конфигурации без доступа к final holdout.
