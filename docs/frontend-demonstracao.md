# Frontend de demonstração

## O que é

O frontend é a interface visual do projeto de previsão de churn. Ele reúne, em uma única aplicação
web, as informações que seriam usadas por uma equipe de retenção para entender a base, consultar o
risco de um cliente e priorizar uma campanha.

Ele não treina o modelo e não substitui o notebook, o pipeline Python ou a API. Sua função é tornar
o trabalho compreensível durante a apresentação e mostrar como o resultado técnico poderia ser
consumido por uma área de negócio.

A aplicação foi construída com React e Vite, usando os componentes do Ant Design (antd) para layout,
tabelas, filtros e formulários. Os gráficos usam Recharts e os resultados documentados no projeto.
Quando a API está em execução, a tela faz chamadas reais aos endpoints `POST /predict` e
`POST /predict/batch`. Nenhuma probabilidade é calculada no navegador.

### Origem dos dados

Cada indicador ou painel mostra uma etiqueta com a origem do que está na tela:

- **Resultado documentado:** valores registrados no notebook e nos relatórios do pipeline
  (`docs/reports`), como AUC-ROC, lift e a taxa histórica de churn;
- **Retorno da API:** probabilidades e classificações devolvidas pela API durante a demonstração;
- **Dado sintético:** clientes gerados no navegador para a demonstração, que não são registros da
  base Cell2Cell.

Os resultados documentados estão fixos no código do frontend. Se o modelo for treinado novamente e
as métricas mudarem, esses valores precisam ser atualizados em `frontend/src/main.jsx`.

## O problema que a tela apresenta

O projeto usa a base Cell2Cell para estimar a propensão de cancelamento de clientes de uma operadora.
A pergunta de negócio é:

> Se a equipe puder abordar apenas uma parte da carteira, o modelo ajuda a encontrar mais clientes
> que efetivamente cancelariam do que uma seleção aleatória?

O frontend traduz essa pergunta para um fluxo simples:

1. entender a base e a taxa histórica de churn;
2. verificar como os dados foram tratados;
3. comparar os modelos avaliados;
4. consultar a propensão de um cliente e de um lote de clientes;
5. ordenar uma fila de retenção pela probabilidade;
6. consultar os lotes já executados;
7. explicar o que ainda seria necessário para uma operação real.

## O que cada tela demonstra

### Visão geral

Apresenta a taxa histórica de churn da base, os principais resultados do teste e o caminho percorrido
no projeto: definição do problema, análise dos dados, comparação dos modelos e disponibilização da
inferência pela API.

Os números exibidos nessa área são resultados do projeto. A taxa de 28,8% é uma proporção histórica
da base; não representa a probabilidade média produzida pelo modelo.

### Dados e EDA

Resume as características da base Cell2Cell:

- 51.047 registros;
- 58 colunas, incluindo identificador e alvo;
- distribuição de 28,8% de churn;
- presença de valores ausentes;
- validação de duplicidades pelo `CustomerID`.

Também apresenta as decisões usadas no tratamento dos dados, como imputação, codificação de
variáveis categóricas, criação de flags de ausência e remoção de variáveis que causariam vazamento
do alvo.

Essa tela é uma síntese visual da análise. Os detalhes e gráficos completos continuam no notebook e
nos módulos Python do projeto.

### Modelos e métricas

Compara o baseline e os classificadores avaliados no treinamento. A tela destaca o LightGBM como
modelo escolhido porque ele obteve o melhor resultado de AUC-ROC na comparação registrada no projeto.

Também mostra a avaliação no conjunto de teste reservado, com AUC-ROC, PR-AUC, precisão, recall e
lift no top 10%.

O recall não é escondido quando fica abaixo do baseline. Isso permite explicar o trade-off: o modelo
melhora a ordenação da fila, mas não é superior em todas as métricas.

### API e inferência

Mostra o uso do modelo persistido por meio da API FastAPI. O botão de consulta envia o payload do
`CustomerID 3000002`, registro real da base usado no Swagger, ao endpoint `/predict` e apresenta a
resposta quando o serviço está disponível.

O retorno contém:

- `CustomerID`;
- `churn_probability`;
- `churn_prediction`;
- `threshold`.

O threshold de 0,5 é exibido como a configuração atual da API, não como uma regra de negócio
validada.

Se a API estiver desligada, sem modelo carregado ou devolver uma resposta fora do contrato, a
interface mostra o motivo e não exibe nenhuma probabilidade, nem a de uma consulta anterior. O
indicador no topo da tela distingue API online, API sem modelo e API offline, e pode ser verificado
novamente a qualquer momento.

### Simulador de lote

Na mesma área, o simulador gera 5, 20 ou 50 clientes e envia o lote ao `POST /predict/batch`. Cada
cliente parte do cadastro do exemplo 3000002 e recebe valores aleatórios em 12 variáveis (por exemplo,
meses de contrato, minutos, variação de uso, chamadas ao atendimento, dias com o aparelho e rating de
crédito), dentro de faixas definidas para a demonstração. Essas faixas não reproduzem a distribuição
da base. Os IDs começam com 91 e não existem na Cell2Cell.

Depois da resposta, a tabela mostra lado a lado a entrada gerada e o retorno da API. O lote só é
registrado quando a API devolve uma previsão válida para cada cliente enviado.

### Fila de retenção

Monta a fila a partir do lote selecionado, ordenada por `churn_probability`, do maior para o menor.
Para cada cliente, a fila mostra:

- CustomerID e um resumo da entrada gerada;
- probabilidade devolvida pela API;
- classificação em relação ao corte da API (acima ou abaixo de 0,5);
- prioridade: **Alta** para o top 10% do lote (no mínimo um cliente), **Média** para clientes acima do
  corte fora do top 10% e **Baixa** para os demais;
- status da abordagem, sempre “Não iniciada”, porque não existe CRM nem histórico de contatos;
- próximo passo sugerido para a equipe.

O top 10% é usado como referência porque foi a faixa avaliada no teste (54,5% de churn). No lote, ele
é uma posição relativa: em um lote de 5 clientes, apenas o primeiro. A tela não dispara oferta, não
altera cadastro e não registra contato.

### Base de lotes

Atende à consulta dos lotes já executados. A área lista os lotes registrados (data, quantidade,
clientes acima do corte, maior probabilidade e tempo de resposta medido no navegador) e permite
consultar os clientes de todos os lotes com:

- busca por CustomerID ou identificador do lote;
- filtro por lote, classificação e prioridade;
- ordenação por probabilidade, posição no lote ou data;
- painel de detalhe do cliente, com o retorno da API e todas as variáveis geradas;
- exportação em CSV dos clientes filtrados;
- limpeza do histórico.

O histórico fica no `localStorage` do navegador e guarda os dez lotes mais recentes. Não é um banco
compartilhado, não substitui o CRM e desaparece se os dados do site forem limpos ou se o navegador
for trocado.

## O que está dentro e fora da entrega

### Incluído

- aplicação React/Vite com componentes do Ant Design para a apresentação;
- navegação entre as áreas do projeto;
- gráficos e indicadores baseados nos resultados documentados;
- integração local com `GET /health`, `POST /predict` e `POST /predict/batch`;
- geração de clientes sintéticos com valores aleatórios;
- fila de retenção montada a partir do lote selecionado;
- base de consulta dos lotes, com filtros, detalhe e exportação em CSV;
- histórico local das consultas em lote;
- etiquetas de origem dos dados (documentado, API ou sintético);
- layout responsivo;
- indicação visual de API offline ou sem modelo.

### Fora do escopo desta fase

- autenticação e controle de acesso;
- banco compartilhado para o histórico de lotes;
- integração com CRM;
- carregamento de carteira real;
- histórico de contatos e ofertas;
- disparo de campanhas;
- monitoramento de drift e qualidade em produção;
- validação temporal;
- experimento incremental para medir retenção causada pela campanha;
- garantia de que uma abordagem evitará o cancelamento.

Esses itens aparecem como pendências na interface. Eles não devem ser apresentados como recursos
implementados.

## Como executar

Na raiz do repositório, gere o modelo caso ele ainda não exista:

```powershell
uv run python -m churn.main
```

Inicie a API em um terminal:

```powershell
uv run uvicorn churn.api:app --reload
```

Inicie o frontend em outro terminal:

```powershell
npm --prefix .\frontend install
npm --prefix .\frontend run dev
```

Abra:

```text
http://localhost:5173
```

Cada área também pode ser aberta pelo endereço, por exemplo `http://localhost:5173/#queue` ou
`http://localhost:5173/#runs`.

O Vite encaminha as chamadas iniciadas em `/api` para `http://127.0.0.1:8000`. Portanto, o frontend
não precisa de uma configuração manual de CORS durante o desenvolvimento local.

## Roteiro de apresentação

Uma demonstração curta pode seguir esta ordem:

1. Apresentar a pergunta de negócio na **Visão geral**.
2. Abrir **Dados e EDA** e explicar o desbalanceamento e o tratamento dos dados.
3. Abrir **Modelos e métricas** e justificar a escolha do LightGBM, inclusive o recall menor.
4. Abrir **API e inferência**, consultar o cliente de exemplo e gerar um lote de 20 clientes.
5. Abrir **Fila de retenção** e explicar a ordenação e a prioridade relativa ao top 10%.
6. Abrir **Base de lotes** e consultar um cliente do lote.
7. Encerrar mostrando as limitações e o que seria necessário antes de uma implantação comercial.

O ponto central da demonstração é mostrar a ligação entre a métrica técnica e a decisão de negócio:
o modelo ajuda a ordenar a fila; ele não decide sozinho qual oferta será feita nem prova que a oferta
terá efeito.

## Critérios de aceite da contribuição

- A aplicação inicia com `npm --prefix .\frontend run dev`.
- A aplicação é acessível em `http://localhost:5173`.
- A navegação entre as seis áreas funciona.
- O build de produção termina com `npm --prefix .\frontend run build`.
- Com a API disponível, o botão de consulta retorna a inferência do cliente de exemplo.
- Com a API disponível, o simulador registra o lote e a fila passa a mostrar os clientes ordenados.
- Sem a API, a interface informa a indisponibilidade sem exibir uma previsão fabricada e sem
  registrar lote.
- A base de lotes permite filtrar, consultar o detalhe e exportar os clientes.
- As telas distinguem resultados do projeto de funcionalidades ainda pendentes.
