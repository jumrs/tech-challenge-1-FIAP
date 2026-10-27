# Predição de Churn em Telecom — Tech Challenge Fase 1 (FIAP)

Pipeline preditivo de churn, da análise exploratória ao modelo campeão, construído com
Scikit-Learn em uma arquitetura modular e rastreado com MLflow.

## Problema de negócio

O problema tratado é a priorização de clientes para uma campanha de retenção. A equipe precisa
ordenar a carteira por propensão de cancelamento usando dados de uso, cobrança, aparelho e perfil.
O modelo apoia essa fila; não substitui a decisão da equipe nem prova que uma oferta causará retenção.

**Pergunta de negócio:** se a equipe puder abordar apenas uma fração da carteira, o modelo encontra
mais clientes que efetivamente cancelariam do que uma seleção aleatória?

- **Métrica técnica principal:** AUC-ROC, porque mede a capacidade de ordenar os clientes por
  risco, sem depender de um ponto de corte, e é adequada para bases desbalanceadas.
- **Métricas complementares:** PR-AUC, Recall, Precisão e F1. A acurácia aparece só como
  referência: um modelo que sempre prevê "não cancela" acerta 71% dos casos.
- **Métrica de negócio:** valor líquido estimado de churn evitado, calculado com premissas ilustrativas
  de valor do cliente, sucesso da retenção e custo da ação (`cost_tradeoff_metrics` em
  `src/churn/metrics.py`). Essas premissas ainda precisam de validação da operadora.

### Escopo e limites

O projeto cobre a seleção de uma base histórica, a comparação de classificadores binários, a
persistência do pipeline vencedor e uma API local para inferência. Não cobre campanha real,
experimento de retenção, calibragem operacional de probabilidades, monitoramento em produção ou
integração com CRM.

## Dataset

**Cell2Cell** (Duke University / Teradata), disponível no Kaggle em
[`jpacse/datasets-for-churn-telecom`](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom)
(arquivo `cell2celltrain.csv`).

- 51.047 clientes, 58 variáveis (uso, faturamento, aparelho, perfil demográfico).
- Taxa de churn de **28,8%**, uma base desbalanceada.
- Salvo em `data/raw/churn.csv`, versionado no repositório. Detalhes em
  [`data/README.md`](data/README.md).

### Formato esperado do CSV

O arquivo é validado por `src/churn/schema.py` antes de qualquer treino. Para usar outro CSV com o
mesmo layout:

| Item | Regra |
|---|---|
| Coluna alvo | `Churn` ou `Churn?` (renomeada internamente para `Churn`) |
| Valores do alvo | Apenas `Yes` e `No`. Valores como `1`/`0` ou `True`/`False` são **recusados** |
| Identificador | `CustomerID` inteiro e único (não entra no modelo) |
| Demais colunas | As 56 colunas do Cell2Cell, com tipos e categorias validados (ex.: `CreditRating` de `1-Highest` a `7-Lowest`, `HandsetPrice` numérico ou `Unknown`) |
| Valores ausentes | Aceitos só nas colunas que têm ausentes na base original (uso, `ServiceArea`, aparelho, idade) |

Se o arquivo não seguir essas regras, o treino é interrompido com uma mensagem indicando a linha,
a coluna e o valor com problema.

## Estrutura do projeto

```text
├── data/raw/churn.csv          # dataset bruto (versionado)
├── docs/reports/               # comparação dos modelos (CV) e avaliação final no teste (gerados no treino)
├── models/                     # modelos treinados (.joblib)
├── notebooks/                  # EDA, baseline e modelagem (experimentação)
├── frontend/                   # dashboard React/Vite para a demonstração
│   ├── src/main.jsx            # telas, gráficos e integração com a API
│   ├── src/styles.css          # identidade visual responsiva
│   ├── package.json            # dependências do frontend
│   └── vite.config.js          # proxy local para a API FastAPI
├── src/churn/                  # código produtivo
│   ├── main.py                 # fluxo completo: dados -> baseline -> comparação -> campeão -> teste
│   ├── config.py               # caminhos, seed, MLflow
│   ├── download_data.py        # download do dataset via KaggleHub
│   ├── schema.py               # validação do CSV bruto (Pydantic)
│   ├── data.py                 # leitura, limpeza e divisão treino/teste
│   ├── features.py             # feature engineering e pré-processamento (sklearn)
│   ├── metrics.py              # métricas técnicas e de negócio
│   ├── model.py                # modelos da Etapa 2 e seletor de features
│   ├── baseline.py             # Dummy + Regressão Logística
│   ├── compare_models.py       # comparação por validação cruzada e escolha do campeão
│   ├── train.py                # treino e registro do modelo campeão
│   ├── evaluate.py             # avaliação final no teste: campeão x baseline
│   ├── logger.py               # logs estruturados em JSON
│   └── api.py                  # API FastAPI (/health, /predict, /predict/batch)
├── tests/                      # testes unitários (pytest)
├── Dockerfile                  # imagem da API (multi-stage, usuário sem privilégios)
├── .dockerignore               # arquivos fora do contexto do build
├── .python-version             # versão do Python (3.13)
├── pyproject.toml              # dependências e configuração do pytest
└── uv.lock                     # versões exatas das dependências
```

## Setup

### 1. Pré-requisitos

- [Git](https://git-scm.com/downloads)
- [uv](https://docs.astral.sh/uv/), o gerenciador de pacotes e ambientes Python do projeto

Instale o uv, caso ainda não tenha:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Feche e reabra o terminal depois da instalação, para o comando `uv` ficar disponível.

### 2. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd tech-challenge-1-FIAP
```

### 3. Instalar o Python 3.13

O projeto usa **Python 3.13** (fixado em `.python-version`). O uv instala essa versão isolada, sem
mexer no Python do sistema:

```bash
uv python install 3.13
```

### 4. Criar o ambiente virtual

```bash
uv venv --python 3.13
```

### 5. Ativar o ambiente virtual

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

> **Windows:** se aparecer o erro "a execução de scripts foi desabilitada neste sistema", libere a
> execução para o seu usuário e ative de novo:
>
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

A ativação é opcional: todos os comandos deste README usam `uv run`, que já roda dentro do `.venv`.

### 6. Instalar as dependências

```bash
uv sync
```

Instala o pacote `churn` (de `src/`) e as versões exatas do `uv.lock`. O `pyproject.toml` é a única
fonte de dependências, divididas em três partes (o `uv sync` instala todas):

| Parte | Pacotes | Para quê |
|---|---|---|
| Principal | scikit-learn, LightGBM, XGBoost (`xgboost-cpu`, sem as bibliotecas de GPU), pandas, Pydantic, FastAPI, uvicorn, loguru | Pipeline do modelo e API (é o que vai para a imagem Docker) |
| Grupo `training` | MLflow, KaggleHub, matplotlib, seaborn | Treino, rastreamento dos experimentos, download dos dados e notebook |
| Grupo `dev` | pytest, httpx2 | Testes |

### 7. Verificar a instalação

```bash
uv --version
uv run python --version                       # Python 3.13.x
uv run python -c "import churn; print('ok')"  # ok
```

## Execução

Todos os comandos rodam a partir da raiz do projeto.

### Fluxo de execução

```bash
uv run python -m churn.main
```

Executa as etapas em sequência, com logs e a duração de cada uma:

1. **Dados:** usa o `data/raw/churn.csv` existente ou baixa o dataset se o arquivo não existir.
2. **Baseline:** Dummy e Regressão Logística.
3. **Comparação:** 4 modelos × com/sem seleção de features, por validação cruzada.
4. **Campeão:** treina o melhor modelo e salva em `models/champion.joblib`.
5. **Avaliação final:** compara campeão e baseline, uma única vez, no conjunto de teste.

Opções:

```bash
uv run python -m churn.main --download          # baixa o dataset de novo antes de treinar
uv run python -m churn.main --data outro.csv    # usa outro CSV com o mesmo layout
```

### Etapas individuais

```bash
uv run python -m churn.download_data                          # baixa o dataset
uv run python -m churn.baseline                               # baseline
uv run python -m churn.compare_models                         # comparação + campeão
uv run python -m churn.train --model LightGBM --no-selection  # treina um modelo específico
uv run python -m churn.evaluate                               # avaliação final no teste
```

Todos aceitam `--data <arquivo.csv>` para usar outro CSV com o mesmo layout.

| Comando | Gera |
|---|---|
| `churn.main` | Tudo o que está abaixo |
| `churn.baseline` | `models/baseline_logreg.joblib` e runs `baseline_*` no MLflow |
| `churn.compare_models` | `docs/reports/model_comparison.{csv,md}`, `docs/reports/feature_selection_gain.md`, `models/champion.joblib` e runs no MLflow |
| `churn.train` | `models/champion.joblib` e run `CAMPEAO \| <modelo>` no MLflow |
| `churn.evaluate` | `docs/reports/test_evaluation.{csv,md}` e run `AVALIACAO FINAL \| teste` no MLflow (exige os dois modelos salvos) |

### Do zero até o modelo campeão

```bash
git clone <url-do-repositorio>
cd tech-challenge-1-FIAP
uv python install 3.13
uv venv --python 3.13
uv sync
uv run pytest
uv run python -m churn.main
uv run uvicorn churn.api:app --reload
npm --prefix frontend install
npm --prefix frontend run dev
```

## API de inferência (FastAPI)

A API serve o modelo campeão (`models/champion.joblib`), carregado uma única vez na subida. Gere
o modelo antes com `uv run python -m churn.main`.

```bash
uv run uvicorn churn.api:app --reload
```

- Documentação interativa (Swagger): http://127.0.0.1:8000/docs, com um exemplo de cliente pronto
  para testar. A raiz `/` redireciona para ela.
- Cada resposta traz o header `X-Process-Time-ms`, e cada requisição é registrada no log com
  método, rota, status e latência.

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Status da API e do modelo: `ok` com o modelo carregado, `degraded` sem ele |
| `POST` | `/predict` | Propensão de churn de **um** cliente |
| `POST` | `/predict/batch` | Propensão de churn de até **1.000** clientes por chamada |

### Health check

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok", "model_loaded": true, "model": "LGBMClassifier"}
```

### Predição

O corpo é um cliente com as colunas do Cell2Cell, **sem** `Churn` e sem as variáveis de retenção
(`RetentionCalls`, `RetentionOffersAccepted`, `MadeCallToRetentionTeam`), que foram removidas do
modelo por *target leakage*. `CustomerID` é opcional e volta na resposta.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"CustomerID": 3000002, "MonthlyRevenue": 24.0, "MonthlyMinutes": 219, "TotalRecurringCharge": 22,
       "DirectorAssistedCalls": 0.25, "OverageMinutes": 0, "RoamingCalls": 0.0, "PercChangeMinutes": -157,
       "PercChangeRevenues": -19.0, "DroppedCalls": 0.7, "BlockedCalls": 0.7, "UnansweredCalls": 6.3,
       "CustomerCareCalls": 0.0, "ThreewayCalls": 0.0, "ReceivedCalls": 97.2, "OutboundCalls": 0.0,
       "InboundCalls": 0.0, "PeakCallsInOut": 58.0, "OffPeakCallsInOut": 24.0, "DroppedBlockedCalls": 1.3,
       "CallForwardingCalls": 0, "CallWaitingCalls": 0.3, "MonthsInService": 61, "UniqueSubs": 2,
       "ActiveSubs": 1, "ServiceArea": "SEAPOR503", "Handsets": 2, "HandsetModels": 2,
       "CurrentEquipmentDays": 361, "AgeHH1": 62, "AgeHH2": 0, "ChildrenInHH": "No",
       "HandsetRefurbished": "No", "HandsetWebCapable": "Yes", "TruckOwner": "No", "RVOwner": "No",
       "Homeownership": "Known", "BuysViaMailOrder": "Yes", "RespondsToMailOffers": "Yes",
       "OptOutMailings": "No", "NonUSTravel": "No", "OwnsComputer": "Yes", "HasCreditCard": "Yes",
       "NewCellphoneUser": "No", "NotNewCellphoneUser": "No", "ReferralsMadeBySubscriber": 0,
       "IncomeGroup": 4, "OwnsMotorcycle": "No", "AdjustmentsToCreditRating": 0, "HandsetPrice": "30",
       "CreditRating": "1-Highest", "PrizmCode": "Suburban", "Occupation": "Professional",
       "MaritalStatus": "No"}'
```

```json
{"CustomerID": 3000002, "churn_probability": 0.6964, "churn_prediction": 1, "threshold": 0.5}
```

- `churn_probability`: propensão de churn, de 0 a 1. Use-a para **ordenar** os clientes da
  campanha de retenção.
- `churn_prediction`: 1 quando a probabilidade é maior ou igual ao `threshold` (0,5, definido em
  `PREDICTION_THRESHOLD` no `config.py`).
- Campos que podem faltar na base (uso, `ServiceArea`, aparelho, idade) podem ser omitidos ou
  enviados como `null`: o pipeline os imputa.

Para várias predições: `POST /predict/batch` com `{"customers": [ {...}, {...} ]}`, que devolve
`{"predictions": [...]}` na mesma ordem.

### Erros

| Status | Quando |
|---|---|
| `422` | Entrada inválida, com o campo e o motivo no corpo: campo obrigatório ausente, categoria inexistente, valor negativo, regra entre colunas (ex.: `ActiveSubs > UniqueSubs`), campo desconhecido ou lote vazio ou maior que 1.000 |
| `503` | Modelo não carregado (`models/champion.joblib` ausente na subida da API) |

## Docker

A imagem empacota só a API e o modelo campeão. O modelo precisa existir antes do build:

```bash
uv run python -m churn.main          # gera models/champion.joblib
docker build -t churn-api .
docker run -d --name churn-api -p 8000:8000 churn-api
```

Depois, a API responde em http://127.0.0.1:8000/docs, e o `curl` da seção anterior funciona igual.

## Dashboard React para a apresentação

O projeto inclui uma interface React/Vite que organiza o produto em áreas operacionais e de
governança:

A descrição da contribuição do frontend, das telas, do roteiro de apresentação e dos critérios de
aceite está em [`docs/frontend-demonstracao.md`](docs/frontend-demonstracao.md).

```powershell
npm --prefix frontend install
npm --prefix frontend run dev
```

Abra http://localhost:5173. O dashboard contém:

- **Visão executiva:** problema, base, resultados no teste e fluxo de execução;
- **Fila de retenção:** prévia do contrato operacional, ordenação por probabilidade e limites da
  integração com CRM;
- **Dados e EDA:** qualidade, ausências, distribuições, correlações e decisões de tratamento;
- **Modelos e métricas:** comparação por validação cruzada e avaliação no teste reservado;
- **Inferência / API:** payload de exemplo, chamada ao `/predict`, resposta e contrato da API;
- **Controles de produto:** integrações, histórico de contato, monitoramento e teste incremental
  aparecem como pendências explícitas; não são apresentados como funcionalidades implementadas.

Para demonstrar a inferência pela API, mantenha a API em outro terminal:

```powershell
uv run uvicorn churn.api:app --reload
```

O Vite encaminha as chamadas `/api/*` para `http://127.0.0.1:8000`. Assim, o botão de inferência
consulta o endpoint real `/predict` sem configuração manual de CORS.
Antes da demonstração, gere os artefatos:

```powershell
uv run python -m churn.main
```

### O que falta para uma implantação comercial

O frontend demonstra o fluxo de priorização, mas a implantação exigiria serviços que não fazem parte
desta Fase 1:

- integração com a carteira de clientes e o CRM;
- autenticação, perfis e segregação por operadora;
- registro das abordagens e do resultado de cada campanha;
- monitoramento de latência, qualidade, drift e disponibilidade;
- validação temporal e experimento incremental de retenção.

Esses itens ficam como evolução do produto. A interface os exibe como pendências para não confundir
um protótipo demonstrável com um sistema operacional já integrado.

```bash
docker ps                     # coluna STATUS mostra "(healthy)" quando o /health responde
docker logs -f churn-api      # logs em JSON de cada requisição
docker rm -f churn-api        # para e remove o container
```

Como a imagem é construída:
- **Multi-stage:** um estágio instala as dependências com o uv (`uv.lock`, `--frozen`); a imagem
  final recebe só o Python, o `.venv` e o `champion.joblib`, sem uv, código-fonte, dados, testes
  ou notebooks.
- **Só dependências da API:** os grupos `training` e `dev` ficam de fora (`--no-default-groups`).
- **Cache de camadas:** as dependências são instaladas antes de copiar o código, então mudar o
  código não reinstala os pacotes.
- **Segurança:** a API roda com o usuário sem privilégios `appuser`.
- **Healthcheck:** o Docker consulta o `/health` a cada 30 s, usando o próprio Python (sem `curl`).
- `libgomp1` é instalada porque o LightGBM precisa dela em tempo de execução.

## Rastreamento dos experimentos (MLflow)

Todos os treinos são registrados no MLflow, em um banco SQLite local (`mlflow.db`, ignorado pelo
Git), no experimento **churn-cell2cell**.

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abra http://127.0.0.1:5000 e filtre os runs pela tag `etapa`:

| Tag `etapa` | Runs | Parâmetros | Métricas | Artefatos |
|---|---|---|---|---|
| `baseline` | `baseline_dummy_most_frequent`, `baseline_logistic_regression` | `model_type`, `cv_folds` | `cv_<métrica>_mean` e `cv_<métrica>_std` (5 folds); `test_<métrica>` | Modelo `.joblib`, matriz de confusão e `classification_report` da CV, coeficientes padronizados |
| `comparacao_cv` | `<modelo> \| selecao=sim/nao` (8 runs) + `Regressao Logistica (baseline) \| selecao=nao` como referência | `modelo`, `selecao_features` | `cv_<métrica>_mean`, `cv_<métrica>_std`, `fit_time_mean_s` | — |
| `campeao` | `CAMPEAO \| <modelo>` | `modelo`, `selecao_features`, `threshold` | `n_features` | Modelo MLflow (`mlflow.sklearn`), lista de features mantidas e removidas |
| `avaliacao_final` | `AVALIACAO FINAL \| teste` | `campeao`, `threshold`, premissas de negócio | `test_champion_*` e `test_baseline_*` (métricas técnicas, top 10%/20%, custo), `test_auc_difference` com `test_ci_low`/`test_ci_high` | — |

Onde `<métrica>` é `roc_auc`, `pr_auc`, `recall`, `precision`, `f1` ou `accuracy`. Para encontrar
o campeão, ordene os runs `comparacao_cv` por `cv_roc_auc_mean`.

## Testes

```bash
uv run pytest
```

Os 23 testes rodam em poucos segundos sobre uma amostra de `data/raw/churn.csv`:

| Arquivo | Cobre |
|---|---|
| `test_schema.py` | Validação do CSV bruto: valor inválido, regra entre colunas, `CustomerID` duplicado, nome alternativo do alvo |
| `test_data.py` | Arquivo inexistente, limpeza dos registros inconsistentes, alvo 0/1 e divisão 80/20 estratificada |
| `test_features.py` | Codificações, colunas removidas, fase do contrato, zeros que indicam ausência e pré-processamento sem NaN |
| `test_metrics.py` | Efeito do ponto de corte, métricas de custo, top X% e bootstrap da diferença de AUC |
| `test_model.py` | Os 4 modelos com seleção de features e a escolha do campeão (o baseline não concorre) |
| `test_evaluate.py` | Tabela da avaliação final (métricas, top X%, custo, matriz de confusão) |
| `test_api.py` | `/health`, `/predict` (sucesso e entrada inválida), `/predict/batch` e o modo sem modelo (503) |

## Metodologia

1. **Validação dos dados brutos** (`schema.py`): tipos, domínios das categorias, formatos
   (`ServiceArea`, `HandsetPrice`), regras entre colunas e `CustomerID` único.
2. **Limpeza** (`data.py`): remoção de 2 registros inconsistentes (cliente sem dados de aparelho
   e `UniqueSubs = 196`, um erro de carga).
3. **Feature engineering** (`features.py`, de acordo com o notebook):
   - Codificação binária e ordinal (`CreditRating`).
   - `HandsetPrice` numérico, com uma flag para preço desconhecido.
   - Região de serviço extraída do código de área, com as regiões raras agrupadas.
   - Fase da fidelidade contratual (`Contract_Locked` e `Contract_EndWindow`).
   - Flags de ausência e conversão em NaN dos zeros que indicam dado ausente.
   - Remoção das variáveis de retenção, por *target leakage*.
4. **Divisão treino/teste** 80/20, estratificada. O teste fica **reservado**: toda comparação usa
   validação cruzada estratificada de 5 folds no treino.
5. **Pré-processamento dentro do pipeline:** imputação pela mediana, padronização e one-hot.
   Assim, cada fold calcula as estatísticas só com os próprios dados de treino, sem vazamento.
6. **Modelos:**
   - Baseline: Dummy e Regressão Logística (`class_weight='balanced'`).
   - Etapa 2: Random Forest, LightGBM, XGBoost e MLP (`MLPClassifier`), cada um com e sem seleção
     de features (`SelectFromModel` com LightGBM).
7. **Campeão:** o maior AUC-ROC médio na validação cruzada.
8. **Avaliação final** (`evaluate.py`): o campeão e o baseline salvos são avaliados **uma única
   vez** no teste, com corte de 0,5, desempenho nos 10% e 20% de clientes de maior risco, métricas
   de custo e intervalo de confiança de 95% (bootstrap pareado) para a diferença de AUC.

## Boas práticas implementadas

**Reprodutibilidade**
- Seed única (`RANDOM_SEED = 42` em `config.py`), usada na divisão treino/teste, nos folds da
  validação cruzada, nos modelos e no seletor de features.
- Dependências com versões exatas no `uv.lock` e Python fixado em `.python-version`.
- Dataset versionado e fluxo inteiro reproduzível com um comando (`churn.main`).

**Rigor na modelagem**
- **Sem vazamento de dados:** imputação, padronização, one-hot e seleção de features ficam dentro
  do `Pipeline` e são reajustadas em cada fold.
- **Sem *target leakage*:** as variáveis de retenção, registradas quando o cliente já decidiu
  sair, foram removidas.
- **Teste reservado:** modelos comparados e campeão escolhido só por validação cruzada no treino;
  o teste é usado uma única vez, na avaliação final.
- **Desbalanceamento (28,8% de churn):** `class_weight='balanced'` (Regressão Logística e
  LightGBM), `balanced_subsample` (Random Forest) e `scale_pos_weight` (XGBoost). A divisão e os
  folds são estratificados.
- **Comparação justa:** os mesmos folds e o mesmo pré-processamento para todos os modelos, com
  baselines Dummy e Regressão Logística como referência.

**Engenharia de software**
- Experimentação (notebook) separada do código produtivo (`src/churn`), com um módulo por
  responsabilidade.
- O modelo salvo é um pipeline completo: recebe as colunas do CSV bruto, então treino e
  inferência usam o mesmo pré-processamento.
- Validação do dado de entrada com Pydantic, a mesma no CSV de treino e na API (`CustomerFeatures`
  em `schema.py`), com erros que indicam linha, coluna e valor.
- API que carrega o modelo uma única vez, recusa campos desconhecidos (inclusive os de vazamento)
  e mede a latência de cada requisição.
- Logs estruturados em JSON (loguru).
- Rastreamento de parâmetros, métricas e artefatos no MLflow.
- Testes automatizados com pytest.

## Resultados

AUC-ROC médio na validação cruzada (5 folds, conjunto de treino):

| Modelo | Seleção de features | AUC-ROC |
|---|---|---|
| **LightGBM (campeão)** | não | **0,678 ± 0,003** |
| LightGBM | sim | 0,676 ± 0,003 |
| Random Forest | sim | 0,670 ± 0,002 |
| Random Forest | não | 0,668 ± 0,002 |
| XGBoost | sim | 0,657 ± 0,007 |
| XGBoost | não | 0,656 ± 0,006 |
| Regressão Logística (baseline) | — | 0,652 ± 0,004 |
| MLP | sim | 0,650 ± 0,004 |
| MLP | não | 0,646 ± 0,004 |
| Dummy | — | 0,500 |

A tabela completa, com todas as métricas, é gerada em `docs/reports/model_comparison.md`. As
variáveis que mais aumentam o risco na Regressão Logística são `Contract_EndWindow` (fim da
fidelidade), `UniqueSubs` e `CurrentEquipmentDays` (aparelho antigo).

### Avaliação final no teste (campeão × baseline)

10.209 clientes nunca vistos no treino (churn de 28,8%), com corte de 0,5:

| Métrica | Regressão Logística (baseline) | LightGBM (campeão) | Ganho |
|---|---|---|---|
| **AUC-ROC** | 0,647 | **0,677** | +0,030 |
| PR-AUC | 0,412 | **0,454** | +0,043 |
| Precisão | 0,380 | **0,404** | +0,023 |
| Recall | **0,650** | 0,631 | −0,019 |
| F1 | 0,480 | **0,492** | +0,012 |
| Precisão nos 10% de maior risco | 0,486 | **0,545** | +0,059 |
| Lift nos 10% de maior risco | 1,69 | **1,89** | +0,20 |
| Churn alcançado abordando 20% da base | 30,6% | **33,7%** | +3,1 p.p. |

- **Comparação com o baseline:** a diferença de AUC no teste foi de +0,030, com intervalo de
  confiança de 95% entre **+0,022 e +0,038** no bootstrap pareado. Nesta divisão, o AUC do teste
  (0,677) ficou próximo do AUC médio da validação cruzada (0,678); isso não substitui validação
  temporal ou acompanhamento em produção.
- **Na campanha de retenção:** abordando os 10% de clientes de maior risco, 54,5% deles realmente
  cancelariam, contra 28,8% na seleção aleatória. O lift observado foi de 1,89x.
- **No corte de 0,5:** o campeão gera 376 falsos alarmes a menos, ao custo de deixar escapar 57
  clientes que cancelariam a mais. Com as premissas ilustrativas de custo (valor do cliente 1.000,
  30% de sucesso na retenção, custo da ação 50), o valor líquido evitado é ligeiramente maior
  (+4.550). As premissas precisam ser validadas com a área de negócio.
- **Limite:** um AUC de 0,68 indica poder de ordenação moderado. O modelo serve para priorizar a
  campanha, não para decidir cliente a cliente com alta certeza.

A tabela completa está em `docs/reports/test_evaluation.md`.
