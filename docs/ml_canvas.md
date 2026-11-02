# Machine Learning Canvas — Churn em Telecom

| Projetado para | Projetado por | Data | Iteração |
|---|---|---|---|
| Diretoria e time de retenção da operadora | Grupo Tech Challenge Fase 1 (FIAP) | 02/11/2026 | 3 |

> Baseado no [Machine Learning Canvas](https://www.ownml.co/) v1.2, de Louis Dorard (OWNML),
> licenciado sob CC BY-SA 4.0. Os números vêm do dataset Cell2Cell e da avaliação no conjunto de
> teste (`docs/reports/test_evaluation.md`). Os valores de negócio são premissas que precisam de
> validação da operadora. Valores em **US$**: o Cell2Cell é de uma operadora dos EUA.
>
> **Implementado** nesta Fase 1: modelo, API (`/predict` e `/predict/batch`), validação dos dados,
> recusa de clientes com menos de 6 meses e logs de latência. **Proposta** (desenho para a
> operação, ainda não implementado): job quinzenal, grupo de controle, cálculo de PSI, gatilhos de
> retreino e painéis de monitoramento. Os itens propostos estão marcados com *(proposta)*.

---

## 1. Prediction Task (tarefa de predição)

- **Tarefa:** identificar clientes com risco de cancelar o serviço de telecom.
- **Tipo:** classificação binária (`Churn` = 1 cancela, 0 permanece). O modelo devolve a
  **probabilidade calibrada** de churn, usada para **ordenar** a carteira; a classe (corte de
  0,270, escolhido por validação cruzada) é secundária.
- **Entidade:** cliente ativo com pelo menos 6 meses de serviço.
- **Frequência:** a cada 15 dias, sobre a base ativa.
- **Resultado observado:** no período seguinte à extração dos dados, quando se confirma se o
  cliente cancelou.

## 2. Decisions (decisões)

- O time de retenção recebe, a cada 15 dias, a lista de clientes ordenada por
  `churn_probability`.
- A equipe aborda os clientes do topo da lista até o limite da sua capacidade operacional. A
  referência é **10% a 20% da base** (ver Impact Simulation).
- **Ação:** oferta de retenção com 20% de desconto por 3 meses.
- **Prioridade:** clientes perto do fim da fidelidade (meses 11 e 12), com churn de **42,7%**
  contra 28,8% na base. A abordagem deve acontecer **antes do 11º mês**. O modelo já recebe essa
  informação pela variável `Contract_EndWindow`.
- **Grupo de controle** *(proposta)*: **10% dos clientes selecionados** ficam sem abordagem, sorteados a cada
  ciclo, para medir o efeito real da oferta. Sem esse grupo não dá para separar o efeito da
  campanha do efeito do modelo.
- **Atendimento individual:** o atendente consulta o risco do cliente pela API (`POST /predict`).

## 3. Value Proposition (proposta de valor)

- **Beneficiário final:** o time de retenção, que hoje aborda clientes sem priorização e perde
  quem cancela antes do contato. Indiretamente, a diretoria, que precisa reduzir a perda de
  receita.
- **Dor resolvida:** a capacidade de contato é limitada. Ordenar a carteira por risco concentra o
  esforço em quem tem mais chance de sair.
- **Integração ao fluxo de trabalho:**
  - lista quinzenal gerada em lote (`POST /predict/batch`), entregue ao time de retenção (hoje
    pelo dashboard de demonstração, depois pelo CRM);
  - consulta individual em tempo real pelo atendente (`POST /predict`).
- **Proposta:** identificar a cada 15 dias os clientes de maior risco, para abordá-los antes do
  fim da fidelidade e reduzir a perda de receita.
- **Meta:** reduzir a taxa de churn em **cerca de 12% em 6 meses** (de 28,8% para ~25%),
  abordando os 20% de clientes de maior risco.
  - Como chegamos nesse número: no teste, os 20% de maior risco concentram **34,9% de todos os
    clientes que cancelaram**. Com 35% de sucesso na retenção, 34,9% × 35% ≈ **12,2% do churn
    evitado**.
  - Abordando só 10% da base, a meta cai para cerca de 6,8%.
  - A meta depende da taxa de sucesso da oferta (35%), que só o grupo de controle vai confirmar.

## 4. Data Collection (coleta de dados)

- **Treinamento inicial:** base Cell2Cell (Duke University / Teradata), disponível no Kaggle, com
  51.047 clientes, 58 variáveis e o rótulo de churn já observado. O download é automatizado
  (`churn.download_data`) e o arquivo bruto é validado por `schema.py` antes do treino.
- **Limpeza:** 2 registros removidos (cliente sem dados de aparelho e `UniqueSubs = 196`, um erro
  de carga), ficando 51.045 clientes.
- **Em produção:** a cada ciclo, os clientes pontuados recebem o rótulo real quando o período de
  observação termina (cancelou ou não). Esses dados alimentam o próximo retreino.
- **Custo e atualidade:** a extração reaproveita as tabelas de faturamento, uso e CRM que já
  existem. Os clientes do **grupo de controle** são os únicos com rótulo livre do efeito da
  campanha. Por isso eles são a amostra preferencial para avaliar o modelo, já que clientes
  abordados e retidos deixam de parecer "churn".

## 5. Data Sources (fontes de dados)

| Fonte | Variáveis (exemplos) |
|---|---|
| Faturamento | `MonthlyRevenue`, `TotalRecurringCharge`, `OverageMinutes`, `PercChangeRevenues`, `AdjustmentsToCreditRating` |
| Uso da rede | `MonthlyMinutes`, `PercChangeMinutes`, chamadas realizadas, recebidas, interrompidas (`DroppedCalls`) e bloqueadas (`BlockedCalls`), `CustomerCareCalls`, `RoamingCalls` |
| CRM / cadastro | `MonthsInService`, aparelho (`CurrentEquipmentDays`, `HandsetPrice`, `HandsetModels`, `HandsetWebCapable`), `CreditRating`, `ServiceArea`, `UniqueSubs`, `ActiveSubs` |
| Dados demográficos | `AgeHH1`/`AgeHH2`, `IncomeGroup`, `MaritalStatus`, `Occupation`, `PrizmCode`, `Homeownership` |
| **Excluídas** | `RetentionCalls`, `RetentionOffersAccepted`, `MadeCallToRetentionTeam`. São registradas quando o cliente já decidiu sair (*target leakage*) e por isso não entram no modelo nem são aceitas pela API |

## 6. Impact Simulation (simulação de impacto)

**Custos por cliente** (mediana de `MonthlyRevenue` = US$ 48,46):

| Situação | Impacto |
|---|---|
| Falso negativo (cliente sai sem abordagem) | perda de **US$ 581,52** (12 meses de receita) |
| Falso positivo (desconto desnecessário) | custo de **US$ 29,08** (20% × 3 meses) |
| Verdadeiro positivo | custo do desconto (US$ 29,08), com o cliente retido em ~35% dos casos (ganho esperado de 0,35 × 581,52 = US$ 203,53) |
| Verdadeiro negativo | sem custo |

**Quando o modelo se paga:** abordar um cliente vale a pena se
`P(churn) × 0,35 × US$ 581,52 > US$ 29,08`, isto é, se a **precisão da lista for maior que 14,3%**.

**Resultado no teste** (10.209 clientes nunca vistos no treino):

| Clientes abordados | Precisão: LightGBM (campeão) | Precisão: seleção aleatória | Precisão: baseline (Reg. Logística) |
|---|---|---|---|
| 10% de maior risco | **55,8%** (lift 1,94×) | 28,8% | 48,6% |
| 20% de maior risco | **50,2%** (lift 1,74×) | 28,8% | 44,1% |

A precisão fica bem acima do ponto de equilíbrio de 14,3%. Como até a seleção aleatória passa
desse ponto (28,8%), **o ganho do modelo deve ser medido contra a seleção aleatória**, e não
contra "não fazer nada".

**Valor líquido no teste** (estas mesmas premissas, `evaluate.py`): abordando os clientes acima
do corte, o campeão evita **US$ 268.059** líquidos contra US$ 242.910 do baseline, um ganho de
**US$ 25.149** em 10.209 clientes. O valor bruto depende da taxa de sucesso de 35%, que só o grupo de
controle vai confirmar.

**Critério para colocar em produção:** AUC-ROC no teste maior que o do baseline de Regressão
Logística, com intervalo de confiança que não inclua zero. Foi atingido: +0,037, IC 95% de
+0,029 a +0,044.

## 7. Making Predictions (fazendo predições)

- **Lote:** processamento a cada 15 dias sobre a base ativa (`POST /predict/batch`, até 1.000
  clientes por chamada; a base é enviada em partes).
- **Tempo real:** a API (`POST /predict`) atende consultas individuais.
- **Elegibilidade:** clientes com **menos de 6 meses** de serviço não entram no processamento. No
  Cell2Cell, `MonthsInService` vai de 6 a 61, então o modelo **nunca viu** clientes mais novos e
  não deve pontuá-los. A API recusa esses clientes com erro 422.
- **Prazos:** o lote precisa terminar antes do início da terceira semana operacional. A pontuação
  leva poucos minutos. Na API, a resposta deve vir em menos de 1 segundo. O header
  `X-Process-Time-ms` mede a latência de cada chamada.
- **Recursos:** servidor com CPU, sem GPU. A API roda em um container Docker leve, com
  healthcheck no `/health`.

## 8. Building Models (construindo modelos)

- **Um modelo em produção:** o campeão da validação cruzada, salvo como um único artefato
  (`models/champion.joblib`) com feature engineering, pré-processamento, classificador calibrado
  e corte de classificação.
- **Como o campeão é escolhido:** Regressão Logística (baseline), Random Forest, LightGBM, XGBoost
  e MLP, cada um com e sem seleção de features. Todos passam pela validação cruzada estratificada
  de 5 folds no treino (80%). Vence o maior AUC-ROC médio; hoje é o **LightGBM**, com 0,678 ±
  0,003.
- **Ajuste do campeão:** tuning dos hiperparâmetros com Optuna (notebook), calibração sigmoide e
  corte escolhido por CV (maior F1). Na CV, o AUC-ROC sobe para 0,686. O teste (20%) é usado uma
  única vez, na avaliação final.
- **Reprodutibilidade:** seed fixa (`RANDOM_SEED = 42`), versões travadas no `uv.lock` e fluxo
  completo com um comando (`uv run python -m churn.main`).
- **Tempo de treino:** minutos em CPU.
- **Retreino** *(proposta)*:
  - **Agendado:** a cada trimestre, com os rótulos dos ciclos mais recentes.
  - **Por gatilho:** antes do trimestre, se ocorrer qualquer um dos casos abaixo:
    - **data drift:** PSI (Population Stability Index) > 0,2 em alguma variável principal
      (`MonthsInService`, `CurrentEquipmentDays`, `MonthlyRevenue`, `MonthlyMinutes`,
      `PercChangeMinutes`), comparando o lote atual com a base de treino;
    - **queda de desempenho:** AUC-ROC menor que 0,65 (nível do baseline) ou precisão nos 10%
      de maior risco menor que 45% no grupo de controle de um ciclo já rotulado;
    - **mudança de negócio:** mudança no plano de fidelidade ou na oferta, que altera o
      significado de `Contract_EndWindow`.
  - O novo modelo só substitui o atual se vencer na mesma comparação (CV + teste reservado).

## 9. Features (variáveis)

O modelo recebe as **colunas brutas** do cliente, e o pipeline (`src/churn/features.py`) aplica
as transformações. Assim, treino e inferência usam exatamente o mesmo tratamento.

- **Codificação binária:** 14 variáveis Sim/Não (ex.: `HandsetWebCapable`, `OwnsComputer`) e
  `Homeownership` (Known/Unknown) viram 1/0.
- **Codificação ordinal:** `CreditRating` (`1-Highest` a `7-Lowest`) vira um número de 1 a 7.
- **Preço do aparelho:** `HandsetPrice` vira numérico, com a flag `HandsetPrice_Unknown` quando o
  preço é desconhecido.
- **Região:** `ServiceArea` vira `ServiceRegion` (3 primeiras letras do código), e as regiões com
  poucos clientes são agrupadas numa única categoria.
- **Fase da fidelidade:** `Contract_Locked` (até 10 meses, churn de 17,1%) e `Contract_EndWindow`
  (meses 11 e 12, churn de 42,7%).
- **Flags de ausência:** `Usage_Missing`, `PercChange_Missing` e `Demographics_Missing`. A falta
  de informação também é sinal de risco.
- **Zeros que significam "ausente":** `AgeHH1 = 0`, `AgeHH2 = 0` e `IncomeGroup = 0` viram NaN;
  `CurrentEquipmentDays` negativo é truncado em 0.
- **Pré-processamento** (ajustado só no treino, dentro de cada fold):
  - numéricas: imputação pela mediana e padronização;
  - categóricas (`PrizmCode`, `Occupation`, `MaritalStatus`, região): imputação pela moda e
    one-hot.
- **Mais importantes** (coeficientes da Regressão Logística): `Contract_EndWindow`, `UniqueSubs` e
  `CurrentEquipmentDays` (aparelho antigo) são as que mais aumentam o risco.

## 10. Monitoring (monitoramento)

Hoje a API já registra latência (`X-Process-Time-ms`), status de cada requisição e erros 422 nos
logs. O restante da tabela é *(proposta)* para a operação.

| Dimensão | Frequência | O que acompanhar |
|---|---|---|
| Negócio | Mensal | Taxa de churn dos clientes abordados contra o **grupo de controle** (efeito real da oferta); receita retida menos o custo dos descontos |
| Modelo | A cada ciclo rotulado | AUC-ROC, precisão e recall nos 10%/20% de maior risco, lift contra a seleção aleatória (medidos no grupo de controle) |
| Operação | Quinzenal | Quantos clientes da lista foram contatados, quantos aceitaram a oferta, se o lote terminou no prazo |
| Dados | Quinzenal, a cada lote | PSI das variáveis principais, taxa de valores ausentes, taxa de requisições recusadas (422) pela API |
| Sistema | Contínuo | Disponibilidade (`/health`) e latência (`X-Process-Time-ms`) da API |

Os gatilhos de retreino estão em **Building Models**.
