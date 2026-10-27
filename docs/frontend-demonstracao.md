# Frontend de demonstração

## O que é

O frontend é a interface visual do projeto de previsão de churn. Ele reúne, em uma única aplicação
web, as informações que seriam usadas por uma equipe de retenção para entender a base, consultar o
risco de um cliente e priorizar uma campanha.

Ele não treina o modelo e não substitui o notebook, o pipeline Python ou a API. Sua função é tornar
o trabalho compreensível durante a apresentação e mostrar como o resultado técnico poderia ser
consumido por uma área de negócio.

A aplicação foi construída com React e Vite. Os gráficos usam os resultados documentados no projeto.
Quando a API está em execução, a tela também faz uma chamada real ao endpoint `POST /predict`.

## O problema que a tela apresenta

O projeto usa a base Cell2Cell para estimar a propensão de cancelamento de clientes de uma operadora.
A pergunta de negócio é:

> Se a equipe puder abordar apenas uma parte da carteira, o modelo ajuda a encontrar mais clientes
> que efetivamente cancelariam do que uma seleção aleatória?

O frontend traduz essa pergunta para um fluxo simples:

1. entender a base e a taxa histórica de churn;
2. verificar como os dados foram tratados;
3. comparar os modelos avaliados;
4. consultar a propensão de um cliente;
5. ordenar uma fila de retenção por risco;
6. explicar o que ainda seria necessário para uma operação real.

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

### Fila de retenção

Representa como o score poderia apoiar uma equipe que não consegue abordar toda a carteira. O cliente
de exemplo é ordenado pela probabilidade retornada pela API, e o top 10% é usado como referência de
priorização porque foi a faixa avaliada nos resultados do projeto.

A tela não dispara oferta, não altera cadastro e não registra contato. Ela demonstra o contrato
esperado entre o modelo e uma futura operação de retenção.

### API e inferência

Mostra o uso do modelo persistido por meio da API FastAPI. O botão de consulta envia o payload do
`CustomerID 3000002` ao endpoint `/predict` e apresenta a resposta quando o serviço está disponível.

O retorno contém:

- `CustomerID`;
- `churn_probability`;
- `churn_prediction`;
- `threshold`.

Se a API estiver desligada ou o modelo não tiver sido gerado, a interface indica que a consulta não
está disponível. Isso é diferente de inventar um resultado local.

## O que está dentro e fora da entrega

### Incluído

- aplicação React/Vite para a apresentação;
- navegação entre as áreas do projeto;
- gráficos e indicadores baseados nos resultados documentados;
- exemplo de fila de retenção;
- integração local com `GET /health` e `POST /predict`;
- layout responsivo;
- indicação visual de indisponibilidade da API.

### Fora do escopo desta fase

- autenticação e controle de acesso;
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

O Vite encaminha as chamadas iniciadas em `/api` para `http://127.0.0.1:8000`. Portanto, o frontend
não precisa de uma configuração manual de CORS durante o desenvolvimento local.

## Roteiro de apresentação

Uma demonstração curta pode seguir esta ordem:

1. Apresentar a pergunta de negócio na **Visão geral**.
2. Abrir **Dados e EDA** e explicar o desbalanceamento e o tratamento dos dados.
3. Abrir **Modelos e métricas** e justificar a escolha do LightGBM.
4. Abrir **Fila de retenção** e explicar a priorização pelo top 10%.
5. Abrir **API e inferência** e consultar o cliente de exemplo.
6. Encerrar mostrando as limitações e o que seria necessário antes de uma implantação comercial.

O ponto central da demonstração é mostrar a ligação entre a métrica técnica e a decisão de negócio:
o modelo ajuda a ordenar a fila; ele não decide sozinho qual oferta será feita nem prova que a oferta
terá efeito.

## Critérios de aceite da contribuição

- A aplicação inicia com `npm --prefix .\frontend run dev`.
- A aplicação é acessível em `http://localhost:5173`.
- A navegação entre as cinco áreas funciona.
- O build de produção termina com `npm --prefix .\frontend run build`.
- Com a API disponível, o botão de consulta retorna a inferência do cliente de exemplo.
- Sem a API, a interface informa a indisponibilidade sem exibir uma previsão fabricada.
- As telas distinguem resultados do projeto de funcionalidades ainda pendentes.
