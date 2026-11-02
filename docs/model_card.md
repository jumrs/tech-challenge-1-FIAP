# Model Card — Classificador de Churn (LightGBM)

Formato baseado em *Model Cards for Model Reporting* (Mitchell et al., 2019). Os números vêm do
conjunto de teste reservado (`docs/reports/test_evaluation.md`) e podem ser reproduzidos com
`uv run python -m churn.main`. O notebook chega ao mesmo resultado (AUC-ROC de 0,684 no teste).

## 1. Detalhes do modelo

| Item | Valor |
|---|---|
| Nome / versão | `champion.joblib`, versão 2.0 (02/11/2026) |
| Tipo | Classificação binária: probabilidade de o cliente cancelar o serviço (`Churn = 1`) |
| Algoritmo | `LGBMClassifier` (LightGBM 4.7.0) com `class_weight = 'balanced'`, sem seleção de features. Hiperparâmetros do tuning com Optuna (25 tentativas, ver notebook): 506 árvores, `learning_rate = 0.0108`, `num_leaves = 113`, `min_child_samples = 23`, `subsample = 0.60`, `colsample_bytree = 0.92`, `reg_lambda = 0.52` |
| Calibração | Sigmoide (`CalibratedClassifierCV`, 5 folds) |
| Corte de classificação | **0,270**, escolhido por validação cruzada no treino, maximizando o F1 (`TunedThresholdClassifierCV`). Fica salvo no artefato e é usado pela API |
| Artefato | Um único objeto do Scikit-Learn 1.9.1: feature engineering → imputação, padronização e one-hot (101 variáveis) → LightGBM calibrado → corte. Recebe as colunas brutas do cliente |
| Baseline | Regressão Logística (`class_weight = 'balanced'`, corte de 0,5), `models/baseline_logreg.joblib` |
| Seed | `RANDOM_SEED = 42` (divisão, folds, modelos, Optuna) |
| Servido por | API FastAPI (`/predict`, `/predict/batch`) |
| Autores | Grupo Tech Challenge Fase 1 (FIAP) |

## 2. Uso pretendido

- **Uso principal:** **ordenar** a carteira de clientes ativos por risco de churn, a cada 15 dias,
  para que o time de retenção aborde primeiro os de maior risco (ver `docs/ml_canvas.md`).
- **Usuários:** time de retenção (lista em lote) e atendentes (consulta individual pela API).
- **O modelo é apoio à decisão.** A ação (oferta de retenção) é decidida pela equipe.
- **`churn_probability`** é uma probabilidade calibrada: um grupo de clientes com 0,30 de média
  tem cerca de 30% de churn real. Pode ser usada em estimativas financeiras.
- **`churn_prediction`** (corte de 0,270) marca **53% da base**, porque o corte favorece o recall
  (um falso negativo custa ~20 vezes um falso positivo). Para a campanha, use a **ordem** e aborde
  até o limite da capacidade (10% a 20% da base).

### Fora de escopo

- **Clientes com menos de 6 meses de serviço.** O treino só tem clientes de 6 a 61 meses.
- **Decisões que prejudiquem o cliente**, como negar crédito, piorar condições ou cobrar multa
  com base no risco. O modelo foi feito para **oferecer** um benefício, não para restringir.
- **Inferência causal.** O modelo não mostra que uma oferta reduz o churn. Isso exige o grupo de
  controle descrito no canvas.
- **Outras operadoras ou outros mercados** sem nova validação. Os dados são de uma operadora dos
  EUA, em um período histórico.

## 3. Dados

- **Fonte:** Cell2Cell (Duke University / Teradata), via Kaggle
  ([`jpacse/datasets-for-churn-telecom`](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom)).
- **Volume:** 51.047 clientes e 58 variáveis. Depois da limpeza ficam **51.045** (removidos: 1
  cliente sem dados de aparelho e 1 com `UniqueSubs = 196`, um erro de carga).
- **Alvo:** taxa de churn de **28,8%** (base desbalanceada).
- **Divisão:** 80% treino (40.836) e 20% teste (10.209), estratificada. O teste só foi usado uma
  vez, na avaliação final.
- **Variáveis removidas por *target leakage*:** `RetentionCalls`, `RetentionOffersAccepted` e
  `MadeCallToRetentionTeam`, registradas quando o cliente já está saindo. A API recusa esses
  campos.
- **Pré-processamento:** descrito em `docs/ml_canvas.md` (seção Features) e implementado em
  `src/churn/features.py`.

## 4. Métricas

| Métrica | Por quê |
|---|---|
| **AUC-ROC** (principal) | Mede a capacidade de ordenar os clientes por risco, sem depender de um corte. É o uso real do modelo |
| PR-AUC | Mais sensível à classe minoritária (churn) |
| Precisão / lift nos 10% e 20% de maior risco | É o que a campanha de retenção vê: quantos dos abordados realmente cancelariam |
| Precisão, Recall, F1 no corte | Visão da classe prevista pela API |
| Brier e probabilidade média | Qualidade da calibração |
| Intervalo de confiança (bootstrap pareado, 1.000 reamostragens) | Mostra se o ganho sobre o baseline é robusto |

## 5. Desempenho

### Escolha do modelo (validação cruzada, 5 folds, treino)

| Modelo | AUC-ROC |
|---|---|
| **LightGBM tunado + calibrado (final)** | **0,686 ± 0,004** |
| LightGBM (parâmetros padrão) | 0,678 ± 0,003 |
| Random Forest | 0,670 ± 0,002 |
| XGBoost | 0,657 ± 0,007 |
| Regressão Logística (baseline) | 0,652 ± 0,004 |
| MLP | 0,650 ± 0,004 |
| Dummy | 0,500 |

O campeão é escolhido com os parâmetros padrão. Depois disso, só ele recebe o tuning e a
calibração. Tabela completa em `docs/reports/model_comparison.md`.

### Avaliação final (teste, 10.209 clientes)

| Métrica | Baseline (Reg. Logística) | **LightGBM** |
|---|---|---|
| AUC-ROC | 0,647 | **0,684** |
| PR-AUC | 0,412 | **0,463** |
| Corte | 0,5 | 0,270 |
| Precisão / Recall / F1 | 0,380 / 0,650 / 0,480 | **0,384 / 0,712 / 0,499** |
| Precisão nos 10% de maior risco (lift) | 48,6% (1,69×) | **55,8% (1,94×)** |
| Precisão nos 20% de maior risco (lift) | 44,1% (1,53×) | **50,2% (1,74×)** |
| Churn alcançado abordando 20% da base | 30,6% | **34,9%** |
| Brier (menor é melhor) | 0,234 | **0,187** |
| Probabilidade média (churn real: 0,288) | 0,485 | **0,290** |

- **Diferença de AUC-ROC:** +0,037, com IC 95% de **+0,029 a +0,044**. O ganho sobre o baseline é
  consistente.
- **CV × teste:** AUC de 0,686 na CV e 0,684 no teste, sem sinal de overfitting nesta divisão.

### Calibração (teste, quintis de probabilidade)

| Faixa prevista | Média prevista | Churn real |
|---|---|---|
| 0,04 – 0,17 | 0,12 | 0,12 |
| 0,17 – 0,25 | 0,21 | 0,21 |
| 0,25 – 0,32 | 0,28 | 0,27 |
| 0,32 – 0,40 | 0,36 | 0,35 |
| 0,40 – 0,78 | 0,48 | 0,50 |

Sem a calibração, a média prevista era 0,44 para um churn real de 0,29.

## 6. Limitações

1. **Poder de ordenação moderado.** Um AUC de 0,68 serve para **priorizar** uma campanha, não para
   decidir cliente a cliente com segurança. Nos 10% de maior risco, 44% dos abordados não
   cancelariam.
2. **Probabilidades concentradas.** A maior probabilidade prevista é 0,78. O modelo não chega a
   identificar clientes com certeza de churn.
3. **Dependência forte do tempo de contrato.** Os clientes nos meses 11–12 (fim da fidelidade)
   são 12,6% do teste, mas **40% dos 10% de maior risco**: 32% deles entram no top 10%, contra
   menos de 1% dos clientes com até 10 meses. No corte de 0,270, o modelo marca **89% dos que não
   cancelam** nesse grupo (taxa de falso positivo). Dentro de cada faixa de tempo de serviço, o
   AUC cai para 0,62–0,70. Boa parte da ordenação vem da fase do contrato, e não de sinais
   individuais. Se a operadora mudar o prazo de fidelidade, o modelo precisa ser retreinado.
4. **Validação apenas aleatória, não temporal.** O dataset não tem data de referência. O
   desempenho em clientes de períodos futuros não foi medido e pode ser menor.
5. **Dados históricos e de outro mercado.** O Cell2Cell é uma operadora dos EUA. Planos, aparelhos
   e comportamento de uso mudaram desde então.
6. **Muitos dados demográficos ausentes.** `AgeHH1 = 0` em 27% da base e `IncomeGroup = 0` em
   25%. São imputados pela mediana e sinalizados por `Demographics_Missing`.
7. **Efeito da própria campanha.** Em produção, clientes abordados e retidos deixam de aparecer
   como churn, o que distorce retreinos futuros. Só o grupo de controle dá rótulos limpos.
8. **Premissas de negócio não validadas.** Valor do cliente, custo da oferta e taxa de sucesso de
   35% são ilustrativos.

## 7. Análise de vieses (desempenho por subgrupo, teste)

O modelo **usa** variáveis demográficas e geográficas (idade, renda, estado civil, ocupação,
região, `PrizmCode`). Avaliamos o comportamento por grupo:

| Grupo | n | Churn real | AUC-ROC | Recall (corte 0,270) | % do grupo no top 10% |
|---|---|---|---|---|---|
| Idade até 30 | 1.341 | 30,8% | 0,666 | 75,3% | 12,5% |
| Idade 31–45 | 2.685 | 27,9% | 0,697 | 75,1% | 9,7% |
| Idade 46–60 | 2.709 | 27,7% | 0,697 | 69,0% | 7,7% |
| **Idade acima de 60** | 544 | 22,8% | 0,664 | **55,6%** | **4,8%** |
| Idade desconhecida | 2.930 | 30,9% | 0,667 | 70,2% | 12,2% |
| Renda 1–3 (mais baixa) | 1.260 | 27,8% | 0,707 | 69,4% | 8,3% |
| Renda 7–9 (mais alta) | 2.771 | 28,5% | 0,688 | 72,6% | 8,4% |
| Crédito `5-Low` | 1.305 | 21,8% | 0,674 | **49,5%** | 4,4% |
| Crédito `2-High` | 3.779 | 31,0% | 0,698 | 76,8% | 11,3% |
| **Ocupação `Retired`** | 153 | 19,6% | **0,563** | 53,3% | 5,9% |
| Região (25 áreas) | 153–1.162 | 18%–37% | **0,603 a 0,754** | 61%–85% | 5%–18% |

**O que os números mostram:**

- **Idosos e aposentados:** o modelo encontra menos churn entre clientes acima de 60 anos (recall
  de 56% contra 75% entre os mais jovens) e quase não separa os aposentados (AUC de 0,56, n = 153,
  amostra pequena). Parte disso reflete a taxa de churn menor desses grupos. Mesmo assim, na
  prática **eles recebem menos ofertas de retenção**.
- **Renda:** não há diferença relevante de desempenho entre as faixas (AUC de 0,69 a 0,71).
- **Crédito:** só 4% dos clientes com crédito `5-Low` entram no topo da lista, coerente com o
  churn menor desse grupo (21,8%).
- **Região:** o AUC varia de 0,60 (NEV) a 0,75 (OHI), mas as amostras por região são pequenas
  (150 a 1.200 clientes), e parte da variação é ruído.
- **Dados demográficos ausentes:** clientes sem idade ou renda informada têm churn maior e
  aparecem mais no topo. A ausência funciona como sinal de risco, o que é legítimo, mas depende da
  qualidade do cadastro.

**Por que é aceitável no uso pretendido:** a ação é **oferecer** um desconto. Errar contra um grupo
significa que ele recebe menos benefício, não que é prejudicado. Por isso o modelo **não deve**
ser usado para ações restritivas (seção 2).

**Recomendações:**

- Acompanhar, a cada ciclo, a cobertura das ofertas por faixa etária e por região.
- Testar o modelo **sem** as variáveis demográficas sensíveis (idade, estado civil, ocupação) e
  medir quanto AUC se perde. Se a perda for pequena, preferir a versão sem elas.
- Validar com o jurídico o uso de dados demográficos à luz da LGPD (finalidade e base legal).

## 8. Monitoramento e retreino

Resumo do `docs/ml_canvas.md` (seções Building Models e Monitoring):

- **Retreino:** trimestral, ou antes, se ocorrer qualquer um dos casos abaixo:
  - PSI > 0,2 em uma variável principal;
  - AUC-ROC menor que 0,65 ou precisão no top 10% menor que 45% no grupo de controle;
  - mudança no plano de fidelidade.
- **Calibração:** comparar a probabilidade média prevista com o churn real do grupo de controle a
  cada ciclo. Uma diferença acima de 5 p.p. indica recalibrar.
- **Monitorar também:** cobertura por subgrupo, drift das variáveis, taxa de erros 422 e latência
  da API.

## 9. Como reproduzir

```bash
uv sync
uv run python -m churn.main      # treina, compara, salva o campeão e gera docs/reports/
uv run pytest                    # 25 testes
```
