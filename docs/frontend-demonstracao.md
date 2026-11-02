# Frontend de demonstração

Interface web usada na apresentação do projeto. Mostra a base, os modelos, as métricas e o uso do
modelo pela API, do ponto de vista de uma equipe de retenção que precisa decidir quem abordar
primeiro.

O frontend não treina modelo nem calcula previsões. A análise e o treinamento ficam no notebook e no
pipeline Python; as previsões vêm da API (`POST /predict` e `POST /predict/batch`).

Feito com React, Vite, Ant Design e Recharts.

## Telas

### Visão geral

Pergunta de negócio, principais números da base e do teste (churn histórico de 28,8%, AUC-ROC de
0,677 e lift de 1,89x no top 10%) e consulta do cliente de exemplo `3000002` na API.

A taxa de 28,8% é a proporção de churn na base, não a probabilidade média do modelo.

### Dados e EDA

Resumo da base Cell2Cell: 51.047 registros (2 removidos na limpeza), 58 colunas, nenhum
`CustomerID` duplicado e 56,8% de `HandsetPrice` como `Unknown`. Também lista as decisões de
tratamento: imputação pela mediana, one-hot encoding, flags de ausência e remoção das variáveis de
retenção, que vazavam o alvo.

Os gráficos completos estão no notebook.

### Modelos e métricas

AUC-ROC de cada modelo na validação cruzada e comparação entre LightGBM e baseline no teste
reservado (AUC-ROC, PR-AUC, precisão, recall e lift no top 10%).

O recall do LightGBM ficou abaixo do baseline no corte de 0,5. A tela mostra isso: o ganho do
modelo está na ordenação dos clientes.

### API e inferência

Envia o cliente `3000002` para `POST /predict` e mostra `churn_probability`, `churn_prediction` e
`threshold`. O corte de 0,5 é o valor configurado na API; ele não foi validado como regra de negócio.

O simulador de lote gera 5, 20 ou 50 clientes e envia para `POST /predict/batch`. Cada cliente usa o
cadastro do `3000002` com 12 variáveis sorteadas, como meses de contrato, minutos, chamadas ao
atendimento, dias com o aparelho e rating de crédito. As faixas do sorteio foram escolhidas para a
demonstração e não seguem a distribuição da base. Os IDs começam com 91 e não existem na Cell2Cell.

A tabela do lote mostra a probabilidade retornada pela API ao lado dos valores sorteados.

Se a API estiver fora do ar ou sem modelo, a tela mostra o erro e não registra o lote. O indicador
no topo mostra o estado da API e tem um botão para verificar de novo.

### Fila de retenção

Clientes do lote selecionado, ordenados por `churn_probability`. Cada linha tem a probabilidade, a
classificação em relação ao corte, a prioridade, o status e o próximo passo.

Prioridade usada na demonstração:

| Prioridade | Regra |
| --- | --- |
| Alta | top 10% do lote (mínimo de 1 cliente) |
| Média | acima do corte, fora do top 10% |
| Baixa | abaixo do corte |

O top 10% foi escolhido porque é a faixa avaliada no teste, onde 54,5% dos clientes cancelaram. Em
um lote pequeno ele vira uma posição relativa: num lote de 5, só o primeiro cliente.

O status é sempre "Não iniciada", porque o projeto não tem CRM nem histórico de contatos.

### Base de lotes

Consulta dos lotes já executados: lista de lotes e tabela com todos os clientes, com busca por
CustomerID ou lote, filtros, ordenação, detalhe de cada cliente e exportação em CSV.

Os 10 lotes mais recentes ficam no `localStorage` do navegador. Não há banco compartilhado, então os
lotes só aparecem no navegador em que foram gerados.

## Origem dos dados

Cada painel tem uma etiqueta de origem:

- Resultado documentado: valores do notebook e de `docs/reports`;
- Retorno da API: probabilidades devolvidas pela API;
- Dado sintético: clientes gerados pelo simulador.

Os resultados documentados estão escritos em `frontend/src/main.jsx`. Se o modelo for treinado de
novo e as métricas mudarem, é preciso atualizar esses valores.

## Fora do escopo

Autenticação, banco compartilhado, integração com CRM, carteira real de clientes, histórico de
contatos e ofertas, disparo de campanhas, monitoramento em produção, validação temporal e teste
incremental para medir o efeito da campanha. Esses itens aparecem como pendentes na fila de
retenção.

## Como executar

Na raiz do repositório, gere o modelo (se ainda não existir) e suba a API:

```powershell
uv run python -m churn.main
uv run uvicorn churn.api:app --reload
```

Em outro terminal, suba o frontend:

```powershell
npm --prefix .\frontend install
npm --prefix .\frontend run dev
```

Abra http://localhost:5173. Cada tela também tem endereço próprio, como `/#queue` e `/#runs`.

O Vite redireciona `/api/*` para `http://127.0.0.1:8000`, então não é preciso configurar CORS.

## Roteiro de apresentação

1. Visão geral: a pergunta de negócio.
2. Dados e EDA: desbalanceamento e tratamento.
3. Modelos e métricas: escolha do LightGBM e o recall menor.
4. API e inferência: cliente de exemplo e um lote de 20 clientes.
5. Fila de retenção: ordenação e prioridade.
6. Base de lotes: consulta de um cliente.
7. O que faltaria para usar em produção.

## Critérios de aceite

- `npm --prefix .\frontend run dev` sobe a aplicação em http://localhost:5173.
- `npm --prefix .\frontend run build` termina sem erro.
- As seis telas abrem pelo menu.
- Com a API no ar, a consulta do cliente de exemplo e o lote retornam previsões.
- Sem a API, a tela mostra o erro e não exibe nem registra previsões.
- A base de lotes permite filtrar, ver o detalhe e exportar em CSV.
