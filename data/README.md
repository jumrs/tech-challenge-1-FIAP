# Dados

## Estrutura

- `raw/churn.csv`: dataset bruto, sem nenhuma alteração. Está **versionado no Git**, pois os testes
  e os comandos de treino dependem dele.

Não há pasta de dados processados: limpeza, feature engineering e imputação acontecem em memória,
dentro do pipeline (`src/churn/data.py` e `src/churn/features.py`). Assim, o mesmo
pré-processamento é aplicado no treino e na inferência.

## Dataset original

**Cell2Cell**, base de churn de uma operadora de telefonia móvel dos EUA (Duke University /
Teradata Center for CRM).

```text
Fonte:            https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom
Arquivo original: cell2celltrain.csv
Salvo como:       data/raw/churn.csv
```

- 51.047 clientes e 58 colunas: `CustomerID`, o alvo `Churn` (Yes/No) e 56 variáveis de uso,
  faturamento, aparelho e perfil demográfico.
- Taxa de churn: 28,8%.
- A descrição de cada variável e a análise de qualidade estão no notebook
  `notebooks/tech_challenge_1_FIAP.ipynb`.

## Como baixar novamente

O comando abaixo baixa o dataset via KaggleHub e copia `cell2celltrain.csv` para
`data/raw/churn.csv`, sobrescrevendo o arquivo atual:

```bash
uv run python -m churn.download_data
```

Para baixar manualmente, faça o download no link acima e copie o arquivo:

```bash
mkdir -p data/raw
cp ~/Downloads/cell2celltrain.csv data/raw/churn.csv
```

## Validação e qualidade

Ao ser carregado (`load_churn_csv`), o arquivo é validado por `src/churn/schema.py` (Pydantic).
A validação confere tipos, categorias permitidas, formatos de `ServiceArea` e `HandsetPrice`,
regras entre colunas e `CustomerID` único. Um arquivo fora do esperado gera um erro que aponta a
linha e a coluna com problema.

Pontos de atenção encontrados na EDA e tratados no código:

| Situação | Tratamento |
|---|---|
| 156 clientes sem dados de uso e 367 sem variação percentual | Flags `Usage_Missing` e `PercChange_Missing` + imputação pela mediana |
| `AgeHH1 = 0` e `IncomeGroup = 0` significam informação ausente | Convertidos em NaN + flag `Demographics_Missing` |
| `HandsetPrice = "Unknown"` (56,8% dos clientes) | NaN + flag `HandsetPrice_Unknown` |
| `CurrentEquipmentDays` negativo (-5 a -1) | Substituído por 0 |
| 1 cliente sem dados de aparelho e 1 com `UniqueSubs = 196` | Registros removidos |
| Variáveis de retenção (`RetentionCalls` etc.) | Removidas por *target leakage* |
