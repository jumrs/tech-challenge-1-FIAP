# Avaliacao final no conjunto de teste

- Conjunto de teste: 10209 clientes (20% reservados), churn de 28.8%.
- Corte de classificacao (linha `threshold`): o do campeao foi escolhido por CV no treino (maior F1); o do baseline e 0,5.
- `brier` e `mean_probability`: calibracao das probabilidades (taxa real no teste de 0.288).
- `topN_*`: abordando so os N% de clientes de maior risco.
- Metricas de custo com premissas ilustrativas: customer_value=1000.0, retention_success_rate=0.3, action_cost=50.0.
- Diferenca de AUC-ROC (campeao - baseline): +0.0366 (IC 95% bootstrap: +0.0288 a +0.0444).

| metrica | Regressao Logistica (baseline) | Campeao (LGBMClassifier) | ganho_campeao |
| --- | --- | --- | --- |
| threshold | 0.5 | 0.2698 | -0.2302 |
| accuracy | 0.5938 | 0.5884 | -0.0054 |
| precision | 0.3802 | 0.3844 | 0.0043 |
| recall | 0.6499 | 0.7124 | 0.0625 |
| f1 | 0.4797 | 0.4994 | 0.0197 |
| roc_auc | 0.6471 | 0.6838 | 0.0366 |
| pr_auc | 0.4118 | 0.4627 | 0.0509 |
| brier | 0.2338 | 0.1871 | -0.0467 |
| mean_probability | 0.4849 | 0.2897 | -0.1952 |
| top10_precision | 0.4858 | 0.5583 | 0.0725 |
| top10_recall | 0.1686 | 0.1937 | 0.0252 |
| top10_lift | 1.6858 | 1.9373 | 0.2515 |
| top20_precision | 0.4412 | 0.5024 | 0.0612 |
| top20_recall | 0.3063 | 0.3487 | 0.0425 |
| top20_lift | 1.5311 | 1.7435 | 0.2124 |
| true_negatives | 4150.0 | 3911.0 | -239.0 |
| false_positives | 3117.0 | 3356.0 | 239.0 |
| false_negatives | 1030.0 | 846.0 | -184.0 |
| true_positives | 1912.0 | 2096.0 | 184.0 |
| retention_action_cost | 251450.0 | 272600.0 | 21150.0 |
| missed_churn_cost | 309000.0 | 253800.0 | -55200.0 |
| avoided_churn_value | 573600.0 | 628800.0 | 55200.0 |
| net_avoided_churn_value | 322150.0 | 356200.0 | 34050.0 |
