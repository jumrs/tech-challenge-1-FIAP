# Modelos

Modelos treinados, salvos com `joblib`. Cada arquivo é um `Pipeline` do Scikit-Learn **completo**:
inclui o feature engineering e o pré-processamento, então recebe as colunas do CSV bruto (sem
`Churn`) e devolve a probabilidade de churn.

## Artefatos

| Arquivo | Gerado por | Conteúdo |
|---|---|---|
| `baseline_logreg.joblib` | `uv run python -m churn.main` ou `uv run python -m churn.baseline` | Regressão Logística (`class_weight='balanced'`), o baseline |
| `champion.joblib` | `uv run python -m churn.main` ou `uv run python -m churn.compare_models` | Modelo campeão da Etapa 2 (hoje o LightGBM sem seleção de features), treinado em todo o conjunto de treino |

O `champion.joblib` também pode ser gerado para um modelo específico:

```bash
uv run python -m churn.train --model LightGBM --no-selection
```

Modelos disponíveis: `Random Forest`, `LightGBM`, `XGBoost` e `MLP`.

## Como usar

```python
import joblib
import pandas as pd

model = joblib.load("models/champion.joblib")
clientes = pd.read_csv("data/raw/churn.csv").drop(columns=["Churn"]).head(5)
probabilidade_churn = model.predict_proba(clientes)[:, 1]
```

## Observações

- A pasta **não** está no `.gitignore`, então os `.joblib` são versionados junto com o código
  (o modelo final salvo é um entregável da Etapa 2). Para recriá-los, rode os comandos acima.
- Cada treino também registra o modelo no MLflow (`mlflow.db`), com métricas e parâmetros.
- A imagem Docker inclui só o `champion.joblib`, que precisa existir antes do `docker build`.
- Os modelos dependem das versões fixadas no `uv.lock` (scikit-learn, lightgbm, xgboost).
  Carregá-los com outras versões pode falhar.
