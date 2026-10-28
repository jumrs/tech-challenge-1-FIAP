import React, { useCallback, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Alert,
  App as AntApp,
  Badge,
  Button,
  Card,
  Col,
  ConfigProvider,
  Descriptions,
  Drawer,
  Empty,
  Flex,
  Grid,
  Input,
  Layout,
  List,
  Menu,
  Popconfirm,
  Progress,
  Row,
  Segmented,
  Select,
  Space,
  Spin,
  Statistic,
  Table,
  Tag,
  Tooltip,
  Typography,
} from "antd";
import ptBR from "antd/locale/pt_BR";
import {
  ApiOutlined,
  BarChartOutlined,
  CheckCircleOutlined,
  DashboardOutlined,
  DatabaseOutlined,
  DeleteOutlined,
  DownloadOutlined,
  ExperimentOutlined,
  FileTextOutlined,
  HistoryOutlined,
  MenuOutlined,
  OrderedListOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
} from "@ant-design/icons";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, Pie, PieChart, ResponsiveContainer, Tooltip as ChartTooltip, XAxis, YAxis } from "recharts";
import "./styles.css";

const { Title, Text, Paragraph } = Typography;

// Cliente real da base (CustomerID 3000002), o mesmo exemplo usado no Swagger da API.
const exampleCustomer = {
  CustomerID: 3000002,
  MonthlyRevenue: 24,
  MonthlyMinutes: 219,
  TotalRecurringCharge: 22,
  DirectorAssistedCalls: 0.25,
  OverageMinutes: 0,
  RoamingCalls: 0,
  PercChangeMinutes: -157,
  PercChangeRevenues: -19,
  DroppedCalls: 0.7,
  BlockedCalls: 0.7,
  UnansweredCalls: 6.3,
  CustomerCareCalls: 0,
  ThreewayCalls: 0,
  ReceivedCalls: 97.2,
  OutboundCalls: 0,
  InboundCalls: 0,
  PeakCallsInOut: 58,
  OffPeakCallsInOut: 24,
  DroppedBlockedCalls: 1.3,
  CallForwardingCalls: 0,
  CallWaitingCalls: 0.3,
  MonthsInService: 61,
  UniqueSubs: 2,
  ActiveSubs: 1,
  ServiceArea: "SEAPOR503",
  Handsets: 2,
  HandsetModels: 2,
  CurrentEquipmentDays: 361,
  AgeHH1: 62,
  AgeHH2: 0,
  ChildrenInHH: "No",
  HandsetRefurbished: "No",
  HandsetWebCapable: "Yes",
  TruckOwner: "No",
  RVOwner: "No",
  Homeownership: "Known",
  BuysViaMailOrder: "Yes",
  RespondsToMailOffers: "Yes",
  OptOutMailings: "No",
  NonUSTravel: "No",
  OwnsComputer: "Yes",
  HasCreditCard: "Yes",
  NewCellphoneUser: "No",
  NotNewCellphoneUser: "No",
  ReferralsMadeBySubscriber: 0,
  IncomeGroup: 4,
  OwnsMotorcycle: "No",
  AdjustmentsToCreditRating: 0,
  HandsetPrice: "30",
  CreditRating: "1-Highest",
  PrizmCode: "Suburban",
  Occupation: "Professional",
  MaritalStatus: "No",
};

// Resultados documentados em docs/reports (validação cruzada e teste reservado).
const modelData = [
  { name: "Dummy", score: 0.5 },
  { name: "MLP", score: 0.65 },
  { name: "Logística", score: 0.652 },
  { name: "XGBoost", score: 0.657 },
  { name: "Random Forest", score: 0.67 },
  { name: "LightGBM", score: 0.678 },
];
const concentrationData = [
  { name: "Base de teste", value: 28.8 },
  { name: "Top 20% do modelo", value: 48.6 },
  { name: "Top 10% do modelo", value: 54.5 },
];
const testMetrics = [
  { metric: "AUC-ROC", base: "0,647", winner: "0,677", gain: "+0,030" },
  { metric: "PR-AUC", base: "0,412", winner: "0,454", gain: "+0,043" },
  { metric: "Precisão", base: "0,380", winner: "0,404", gain: "+0,023" },
  { metric: "Recall", base: "0,650", winner: "0,631", gain: "−0,019", negative: true },
  { metric: "Lift no top 10%", base: "1,69x", winner: "1,89x", gain: "+0,20" },
];

const CHART_PRIMARY = "#1677ff";
const CHART_NEUTRAL = "#d9d9d9";

const STORAGE_KEY = "churn-batch-runs";
const MAX_RUNS = 10;
const BATCH_SIZES = [5, 20, 50];
const TOP_SHARE = 0.1;
const CREDIT_RATINGS = ["1-Highest", "2-High", "3-Good", "4-Medium", "5-Low", "6-VeryLow", "7-Lowest"];

const PAGES = [
  { key: "overview", label: "Visão geral", icon: <DashboardOutlined /> },
  { key: "data", label: "Dados & EDA", icon: <DatabaseOutlined /> },
  { key: "models", label: "Modelos & métricas", icon: <BarChartOutlined /> },
  { key: "api", label: "API & inferência", icon: <ApiOutlined /> },
  { key: "queue", label: "Fila de retenção", icon: <OrderedListOutlined /> },
  { key: "runs", label: "Base de lotes", icon: <HistoryOutlined /> },
];

const API_STATUS = {
  checking: { status: "processing", text: "Verificando API" },
  online: { status: "success", text: "API online" },
  degraded: { status: "warning", text: "API sem modelo" },
  offline: { status: "error", text: "API offline" },
};

const SOURCES = {
  documented: { label: "Resultado documentado", color: "default", icon: <FileTextOutlined />, title: "Valor registrado no notebook e nos relatórios do pipeline (docs/reports)." },
  api: { label: "Retorno da API", color: "blue", icon: <ApiOutlined />, title: "Valor devolvido pela API FastAPI nesta sessão ou registrado no histórico local." },
  synthetic: { label: "Dado sintético", color: "gold", icon: <ExperimentOutlined />, title: "Gerado no navegador apenas para a demonstração. Não é registro da base Cell2Cell." },
};

const PRIORITY = {
  high: { label: "Alta", color: "red", next: "Revisar com a equipe de retenção" },
  medium: { label: "Média", color: "orange", next: "Avaliar critério com o negócio" },
  low: { label: "Baixa", color: "default", next: "Manter em monitoramento" },
};

const integer = (value) => value.toLocaleString("pt-BR");
const decimal = (digits) => (value) => value.toLocaleString("pt-BR", { minimumFractionDigits: digits, maximumFractionDigits: digits });

// Variáveis sorteadas em cada cliente sintético; as demais vêm do cliente de exemplo.
const GENERATED_FIELDS = [
  { key: "MonthsInService", label: "Meses de contrato", short: "Meses", format: integer },
  { key: "MonthlyRevenue", label: "Receita mensal", short: "Receita", format: decimal(2) },
  { key: "MonthlyMinutes", label: "Minutos por mês", short: "Min/mês", format: integer },
  { key: "PercChangeMinutes", label: "Variação de minutos", short: "Var. min", format: integer },
  { key: "OverageMinutes", label: "Minutos excedentes", short: "Excedente", format: integer },
  { key: "TotalRecurringCharge", label: "Cobrança recorrente", short: "Recorrente", format: integer },
  { key: "CustomerCareCalls", label: "Chamadas ao atendimento", short: "Atend.", format: decimal(1) },
  { key: "DroppedCalls", label: "Chamadas derrubadas", short: "Derrub.", format: decimal(1) },
  { key: "CurrentEquipmentDays", label: "Dias com o aparelho atual", short: "Aparelho (dias)", format: integer },
  { key: "CreditRating", label: "Rating de crédito", short: "Crédito", format: (value) => value },
  { key: "HandsetWebCapable", label: "Aparelho com internet", short: "Internet", format: (value) => (value === "Yes" ? "Sim" : "Não") },
  { key: "IncomeGroup", label: "Faixa de renda", short: "Renda", format: integer },
];

const formatPercent = (value) => `${decimal(1)(value * 100)}%`;
const formatThreshold = (value) => value.toLocaleString("pt-BR");
const formatDate = (value) => new Date(value).toLocaleString("pt-BR", { dateStyle: "short", timeStyle: "short" });
const formatField = (field, input) => (input?.[field.key] == null ? "—" : field.format(input[field.key]));

class ApiError extends Error {
  constructor(message, kind) {
    super(message);
    this.kind = kind;
  }
}

// kind: "offline", "degraded" (API sem modelo) ou "request" (API no ar, payload recusado).
async function requestJson(path, body) {
  const options = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined;
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new ApiError("A API não respondeu. Verifique se ela está em execução em http://127.0.0.1:8000.", "offline");
  }
  const data = await response.json().catch(() => null);
  if (response.ok) return data;
  const detail = typeof data?.detail === "string" ? data.detail : null;
  if (response.status === 503) throw new ApiError(detail || "A API está no ar, mas o modelo não foi carregado.", "degraded");
  if (response.status === 422) throw new ApiError("A API recusou o payload (erro de validação 422).", "request");
  // Sem corpo da API, o erro veio do proxy do Vite: o backend não está respondendo.
  if (!data) throw new ApiError(`A API não respondeu (proxy retornou ${response.status}). Verifique se ela está em execução.`, "offline");
  throw new ApiError(detail || `A API retornou erro ${response.status}.`, "request");
}

const apiStateFromError = (error) => (error.kind === "request" ? "online" : error.kind || "offline");

function isValidPrediction(item) {
  return (
    item != null &&
    Number.isFinite(item.churn_probability) &&
    item.churn_probability >= 0 &&
    item.churn_probability <= 1 &&
    (item.churn_prediction === 0 || item.churn_prediction === 1) &&
    Number.isFinite(item.threshold)
  );
}

function loadRuns() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    if (!Array.isArray(stored)) return [];
    return stored
      .filter((run) => typeof run?.id === "string" && Array.isArray(run.predictions) && run.predictions.length > 0 && run.predictions.every(isValidPrediction))
      .slice(0, MAX_RUNS);
  } catch {
    return [];
  }
}

function saveRuns(runs) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(runs));
  } catch {
    // Sem acesso ao localStorage o histórico fica só na memória da aba.
  }
}

const randomBetween = (min, max) => min + Math.random() * (max - min);
const randomInt = (min, max) => Math.floor(randomBetween(min, max + 1));
const pick = (items) => items[randomInt(0, items.length - 1)];
const roundTo = (value, digits) => Number(value.toFixed(digits));

// Faixas escolhidas para a demonstração; não reproduzem a distribuição da base.
function createDemoCustomer(index, sequence) {
  const input = {
    MonthsInService: randomInt(6, 60),
    MonthlyRevenue: roundTo(randomBetween(10, 120), 2),
    MonthlyMinutes: randomInt(0, 1200),
    PercChangeMinutes: randomInt(-400, 300),
    OverageMinutes: Math.random() < 0.6 ? 0 : randomInt(1, 150),
    TotalRecurringCharge: randomInt(10, 80),
    CustomerCareCalls: Math.random() < 0.5 ? 0 : roundTo(randomBetween(0.3, 10), 1),
    DroppedCalls: roundTo(randomBetween(0, 20), 1),
    CurrentEquipmentDays: randomInt(10, 1200),
    CreditRating: pick(CREDIT_RATINGS),
    HandsetWebCapable: pick(["Yes", "No"]),
    IncomeGroup: randomInt(0, 9),
  };
  const payload = {
    ...exampleCustomer,
    ...input,
    CustomerID: 9100000 + sequence * 1000 + index,
    DroppedBlockedCalls: roundTo(input.DroppedCalls + exampleCustomer.BlockedCalls, 1),
  };
  return { payload, input };
}

const nextSequence = (runs) => (Math.max(0, ...runs.map((run) => run.sequence || 0)) + 1) % 900;

// Prioridade relativa ao lote: top 10% = alta.
function buildQueue(run) {
  const sorted = [...run.predictions].sort((a, b) => b.churn_probability - a.churn_probability);
  const topCount = Math.max(1, Math.ceil(sorted.length * TOP_SHARE));
  return sorted.map((item, index) => ({
    ...item,
    rank: index + 1,
    priority: index < topCount ? "high" : item.churn_prediction ? "medium" : "low",
  }));
}

function flattenRuns(runs) {
  return runs.flatMap((run) => buildQueue(run).map((item) => ({ ...item, runId: run.id, createdAt: run.createdAt, runSize: run.predictions.length })));
}

const rowKey = (row) => `${row.runId}:${row.CustomerID}`;

function exportCsv(rows) {
  const header = ["lote", "data", "CustomerID", ...GENERATED_FIELDS.map((field) => field.key), "churn_probability", "churn_prediction", "threshold", "posicao_no_lote", "prioridade", "origem"];
  const cell = (value) => {
    if (value == null) return "";
    const text = typeof value === "number" ? String(value).replace(".", ",") : String(value);
    return /[;"\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  const lines = rows.map((row) =>
    [row.runId, row.createdAt, row.CustomerID, ...GENERATED_FIELDS.map((field) => row.input?.[field.key]), row.churn_probability, row.churn_prediction, row.threshold, row.rank, PRIORITY[row.priority].label, "entrada sintética; probabilidade da API"]
      .map(cell)
      .join(";"),
  );
  const blob = new Blob([`﻿${[header.join(";"), ...lines].join("\n")}`], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "lotes-sinteticos-churn.csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const pageFromHash = () => {
  const key = window.location.hash.slice(1);
  return PAGES.some((item) => item.key === key) ? key : "overview";
};

function Dashboard() {
  const { message } = AntApp.useApp();
  const screens = Grid.useBreakpoint();
  const isMobile = screens.lg === false;
  const [active, setActive] = useState(pageFromHash);
  const [menuOpen, setMenuOpen] = useState(false);
  const [apiState, setApiState] = useState("checking");
  const [prediction, setPrediction] = useState(null);
  const [predictError, setPredictError] = useState("");
  const [loading, setLoading] = useState(false);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchError, setBatchError] = useState("");
  const [batchRuns, setBatchRuns] = useState(loadRuns);
  const [selectedRunId, setSelectedRunId] = useState(null);

  const selectedRun = batchRuns.find((run) => run.id === selectedRunId) || batchRuns[0] || null;

  const checkHealth = useCallback(async () => {
    setApiState("checking");
    try {
      const health = await requestJson("/api/health");
      setApiState(health?.model_loaded ? "online" : "degraded");
    } catch (error) {
      setApiState(apiStateFromError(error));
    }
  }, []);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  useEffect(() => {
    saveRuns(batchRuns);
  }, [batchRuns]);

  useEffect(() => {
    const onHashChange = () => setActive(pageFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const predict = async () => {
    setLoading(true);
    setPredictError("");
    try {
      const result = await requestJson("/api/predict", exampleCustomer);
      if (!isValidPrediction(result)) throw new ApiError("A resposta da API não segue o contrato esperado.", "request");
      setPrediction(result);
      setApiState("online");
    } catch (error) {
      setPrediction(null);
      setPredictError(error.message);
      setApiState(apiStateFromError(error));
    } finally {
      setLoading(false);
    }
  };

  const runBatch = async (size) => {
    setBatchLoading(true);
    setBatchError("");
    const sequence = nextSequence(batchRuns);
    const customers = Array.from({ length: size }, (_, index) => createDemoCustomer(index, sequence));
    const startedAt = performance.now();
    try {
      const result = await requestJson("/api/predict/batch", { customers: customers.map((customer) => customer.payload) });
      const predictions = result?.predictions;
      if (!Array.isArray(predictions) || predictions.length !== customers.length || !predictions.every(isValidPrediction)) {
        throw new ApiError("A resposta da API não segue o contrato esperado.", "request");
      }
      // A API devolve as previsões na mesma ordem dos clientes enviados.
      const run = {
        id: `LOTE-${Date.now()}`,
        sequence,
        createdAt: new Date().toISOString(),
        durationMs: Math.round(performance.now() - startedAt),
        predictions: predictions.map((item, index) => ({ ...item, input: customers[index].input })),
      };
      setBatchRuns((current) => [run, ...current].slice(0, MAX_RUNS));
      setSelectedRunId(run.id);
      setApiState("online");
      message.success(`Lote registrado com ${size} clientes.`);
    } catch (error) {
      setBatchError(error.message);
      setApiState(apiStateFromError(error));
    } finally {
      setBatchLoading(false);
    }
  };

  const clearRuns = () => {
    setBatchRuns([]);
    setSelectedRunId(null);
    message.info("Histórico local apagado.");
  };

  const navigate = (page) => {
    setActive(page);
    if (window.location.hash !== `#${page}`) window.location.hash = page;
    setMenuOpen(false);
    window.scrollTo({ top: 0 });
  };

  const openQueue = (runId) => {
    setSelectedRunId(runId);
    navigate("queue");
  };

  const batch = {
    runs: batchRuns,
    selectedRun,
    loading: batchLoading,
    error: batchError,
    onRun: runBatch,
    onSelect: setSelectedRunId,
    onOpenQueue: openQueue,
    onClear: clearRuns,
  };
  const predictionProps = { prediction, error: predictError, loading, onPredict: predict };

  const page = {
    overview: <Overview predictionProps={predictionProps} />,
    data: <DataView />,
    models: <ModelsView />,
    api: <ApiView apiState={apiState} predictionProps={predictionProps} batch={batch} onNavigate={navigate} />,
    queue: <QueueView apiState={apiState} batch={batch} onNavigate={navigate} />,
    runs: <RunsView batch={batch} onNavigate={navigate} />,
  }[active];

  const menu = (
    <Menu
      mode="inline"
      selectedKeys={[active]}
      onClick={({ key }) => navigate(key)}
      items={PAGES.map(({ key, label, icon }) => ({
        key,
        icon,
        label: key === "runs" && batchRuns.length ? <Flex justify="space-between" align="center">{label}<Badge count={batchRuns.length} color="#8c8c8c" size="small" /></Flex> : label,
      }))}
      style={{ borderInlineEnd: 0 }}
    />
  );
  const status = API_STATUS[apiState];

  return (
    <Layout className="app-layout">
      {!isMobile && (
        <Layout.Sider width={232} theme="light" className="app-sider">
          <div className="sider-title">
            <Text strong>Retenção de Clientes</Text>
            <Text type="secondary">Previsão de churn com ML</Text>
          </div>
          {menu}
        </Layout.Sider>
      )}
      <Drawer placement="left" open={isMobile && menuOpen} onClose={() => setMenuOpen(false)} width={260} title="Menu" styles={{ body: { padding: 0 } }}>
        {menu}
      </Drawer>
      <Layout>
        <Layout.Header className="app-header">
          <Space size="middle">
            {isMobile && <Button type="text" icon={<MenuOutlined />} onClick={() => setMenuOpen(true)} aria-label="Abrir menu" />}
            <Title level={4} style={{ margin: 0 }}>Tech Challenge FIAP · Fase 1</Title>
          </Space>
          <Space>
            <Badge status={status.status} text={status.text} />
            <Tooltip title="Verificar API novamente">
              <Button type="text" icon={<ReloadOutlined />} onClick={checkHealth} loading={apiState === "checking"} aria-label="Verificar API novamente" />
            </Tooltip>
          </Space>
        </Layout.Header>
        <Layout.Content className="app-content">{page}</Layout.Content>
        <Layout.Footer className="app-footer">
          Tech Challenge FIAP · Fase 1 · Previsão de churn (Cell2Cell)
        </Layout.Footer>
      </Layout>
    </Layout>
  );
}

function PageHeader({ title, description, extra }) {
  return (
    <Flex className="page-header" justify="space-between" align="flex-end" gap={16} wrap>
      <div>
        <Title level={3} style={{ margin: 0 }}>{title}</Title>
        <Paragraph type="secondary" style={{ margin: "4px 0 0", maxWidth: 720 }}>{description}</Paragraph>
      </div>
      {extra && <Space wrap>{extra}</Space>}
    </Flex>
  );
}

function SourceTag({ kind }) {
  const { label, color, icon, title } = SOURCES[kind];
  return (
    <Tooltip title={title}>
      <Tag color={color} icon={icon} style={{ marginInlineEnd: 0 }}>{label}</Tag>
    </Tooltip>
  );
}

const sources = (...kinds) => <Space size={4} wrap>{kinds.map((kind) => <SourceTag key={kind} kind={kind} />)}</Space>;

function StatCard({ title, value, detail, source }) {
  return (
    <Card size="small" style={{ height: "100%" }}>
      <Statistic title={title} value={value} />
      <Text type="secondary">{detail}</Text>
      {source && <div style={{ marginTop: 8 }}><SourceTag kind={source} /></div>}
    </Card>
  );
}

function StatRow({ items }) {
  return (
    <Row gutter={[16, 16]} className="section">
      {items.map((item) => (
        <Col key={item.title} xs={12} lg={6}>
          <StatCard {...item} />
        </Col>
      ))}
    </Row>
  );
}

function ApiNotice({ apiState }) {
  if (apiState === "offline") {
    return (
      <Alert
        className="section"
        type="error"
        showIcon
        message="API indisponível"
        description={<>Inicie com <Text code>uv run uvicorn churn.api:app --reload</Text>.</>}
      />
    );
  }
  if (apiState === "degraded") {
    return (
      <Alert
        className="section"
        type="warning"
        showIcon
        message="API no ar, mas sem modelo carregado"
        description={<>Gere o modelo com <Text code>uv run python -m churn.main</Text> e reinicie a API.</>}
      />
    );
  }
  return null;
}

function Overview({ predictionProps }) {
  return (
    <>
      <PageHeader
        title="Visão geral"
        description="Previsão de churn de clientes de telecomunicações com a base Cell2Cell. O modelo ordena os clientes pela probabilidade de cancelamento para apoiar uma campanha de retenção."
      />
      <StatRow
        items={[
          { title: "Clientes na base", value: "51.047", detail: "Base bruta; 51.045 após a limpeza", source: "documented" },
          { title: "Churn histórico", value: "28,8%", detail: "Proporção observada na base", source: "documented" },
          { title: "AUC-ROC no teste", value: "0,677", detail: "LightGBM · baseline 0,647", source: "documented" },
          { title: "Lift no top 10%", value: "1,89x", detail: "Baseline: 1,69x", source: "documented" },
        ]}
      />
      <Row gutter={[16, 16]} className="section">
        <Col xs={24} lg={14}>
          <Card title="Churn encontrado na fila ordenada pelo modelo" extra={<SourceTag kind="documented" />} style={{ height: "100%" }}>
            <Text type="secondary">Conjunto de teste reservado, 10.209 clientes</Text>
            <ResponsiveContainer width="100%" height={230}>
              <BarChart data={concentrationData} margin={{ top: 24, right: 8, left: 8, bottom: 0 }}>
                <CartesianGrid stroke="#f0f0f0" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: "#595959", fontSize: 12 }} axisLine={false} tickLine={false} />
                <YAxis hide domain={[0, 65]} />
                <ChartTooltip formatter={(value) => [`${decimal(1)(value)}%`, "Churn"]} cursor={{ fill: "#fafafa" }} />
                <Bar dataKey="value" isAnimationActive={false} maxBarSize={64}>
                  {concentrationData.map((entry, index) => (
                    <Cell key={entry.name} fill={index === 0 ? CHART_NEUTRAL : CHART_PRIMARY} />
                  ))}
                  <LabelList dataKey="value" position="top" formatter={(value) => `${decimal(1)(value)}%`} fill="#262626" fontSize={12} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <Paragraph type="secondary" style={{ marginBottom: 0 }}>
              Entre os 10% de clientes com maior probabilidade, 54,5% cancelaram, contra 28,8% na base de teste inteira. O modelo serve para ordenar a fila; a oferta e a decisão continuam com o negócio.
            </Paragraph>
          </Card>
        </Col>
        <Col xs={24} lg={10}>
          <Card title="Pergunta de negócio" style={{ height: "100%" }}>
            <Paragraph>
              Se a equipe só consegue abordar parte da carteira, o modelo encontra mais clientes que cancelariam do que uma escolha aleatória?
            </Paragraph>
            <Descriptions
              column={1}
              size="small"
              items={[
                { key: "dados", label: "Dados", children: "51.047 clientes e 58 colunas, com tratamento sem vazamento do alvo." },
                { key: "modelos", label: "Modelos", children: "Dummy, regressão logística (baseline), MLP, XGBoost, Random Forest e LightGBM." },
                { key: "avaliacao", label: "Avaliação", children: "Validação cruzada em 5 folds e teste reservado com 20% da base." },
                { key: "entrega", label: "Entrega", children: "Modelo campeão servido pela FastAPI, para um cliente ou em lote." },
              ]}
            />
          </Card>
        </Col>
      </Row>
      <Card
        title="Consultar o cliente de exemplo"
        extra={<Space wrap>{predictionProps.prediction && <SourceTag kind="api" />}<PredictButton {...predictionProps} /></Space>}
      >
        <Paragraph type="secondary">CustomerID 3000002, registro real da base, enviado para <Text code>POST /predict</Text>.</Paragraph>
        <PredictionResult {...predictionProps} />
      </Card>
    </>
  );
}

function PredictButton({ loading, onPredict }) {
  return (
    <Button type="primary" icon={<PlayCircleOutlined />} loading={loading} onClick={onPredict}>
      Consultar API
    </Button>
  );
}

function ProbabilityBar({ value, width = 120 }) {
  return <Progress percent={Math.round(value * 1000) / 10} showInfo={false} size={[width, 6]} strokeColor={CHART_PRIMARY} />;
}

function PredictionResult({ prediction, error, loading }) {
  if (loading) return <Flex justify="center" style={{ padding: 32 }}><Spin tip="Consultando a API..."><div style={{ width: 120, height: 40 }} /></Spin></Flex>;
  if (error) return <Alert type="error" showIcon message="Consulta não realizada" description={error} />;
  if (!prediction) return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="Nenhuma consulta feita nesta sessão." />;
  const above = prediction.churn_prediction === 1;
  return (
    <>
      <Row gutter={[24, 16]} align="middle">
        <Col xs={24} md={8}>
          <Statistic title="Probabilidade de churn" value={formatPercent(prediction.churn_probability)} />
          <ProbabilityBar value={prediction.churn_probability} width={180} />
        </Col>
        <Col xs={24} md={16}>
          <Descriptions
            bordered
            size="small"
            column={1}
            items={[
              { key: "id", label: "CustomerID", children: prediction.CustomerID ?? "—" },
              { key: "pred", label: "churn_prediction", children: <Space>{prediction.churn_prediction}<ClassTag value={prediction.churn_prediction} /></Space> },
              { key: "threshold", label: "threshold", children: formatThreshold(prediction.threshold) },
            ]}
          />
        </Col>
      </Row>
      <Paragraph type="secondary" style={{ margin: "12px 0 0" }}>
        {above ? "Candidato a avaliação pela equipe de retenção." : "Sugestão: manter em monitoramento."} Corte atual da API: {formatThreshold(prediction.threshold)}.
      </Paragraph>
    </>
  );
}

function DataView() {
  return (
    <>
      <PageHeader title="Dados & EDA" description="Qualidade da base, distribuição do alvo e decisões de tratamento que orientaram o pré-processamento." />
      <StatRow
        items={[
          { title: "Registros", value: "51.047", detail: "2 removidos na limpeza", source: "documented" },
          { title: "Colunas", value: "58", detail: "56 variáveis + ID + alvo", source: "documented" },
          { title: "Maior ausência", value: "56,8%", detail: "HandsetPrice como Unknown", source: "documented" },
          { title: "Duplicidades", value: "0", detail: "CustomerID validado", source: "documented" },
        ]}
      />
      <Row gutter={[16, 16]} className="section">
        <Col xs={24} lg={14}>
          <Card title="Distribuição do alvo" extra={<SourceTag kind="documented" />} style={{ height: "100%" }}>
            <Row gutter={16} align="middle">
              <Col xs={24} sm={12}>
                <ResponsiveContainer width="100%" height={210}>
                  <PieChart>
                    <Pie data={[{ name: "Não cancelou", value: 71.2 }, { name: "Cancelou", value: 28.8 }]} innerRadius={60} outerRadius={86} dataKey="value" stroke="#fff" startAngle={90} endAngle={-270} isAnimationActive={false}>
                      <Cell fill={CHART_NEUTRAL} />
                      <Cell fill={CHART_PRIMARY} />
                    </Pie>
                    <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" fill="#262626" fontSize="20" fontWeight="600">28,8%</text>
                  </PieChart>
                </ResponsiveContainer>
              </Col>
              <Col xs={24} sm={12}>
                <Descriptions
                  column={1}
                  size="small"
                  items={[
                    { key: "nao", label: <Badge color={CHART_NEUTRAL} text="Não cancelou" />, children: "71,2%" },
                    { key: "sim", label: <Badge color={CHART_PRIMARY} text="Cancelou" />, children: "28,8%" },
                  ]}
                />
                <Paragraph type="secondary" style={{ marginBottom: 0 }}>
                  Com o alvo desbalanceado, AUC-ROC e PR-AUC dizem mais sobre o modelo do que a acurácia isolada.
                </Paragraph>
              </Col>
            </Row>
          </Card>
        </Col>
        <Col xs={24} lg={10}>
          <Card title="Decisões de tratamento" style={{ height: "100%" }}>
            <List
              size="small"
              dataSource={[
                "Imputação pela mediana do treino",
                "One-hot encoding das categóricas",
                "Flags de ausência",
                "Variáveis de retenção removidas (vazamento do alvo)",
                "Teste reservado antes de qualquer ajuste",
              ]}
              renderItem={(item) => (
                <List.Item>
                  <Space><CheckCircleOutlined style={{ color: "#52c41a" }} />{item}</Space>
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>
      <Card title="Sinais encontrados na EDA" extra={<SourceTag kind="documented" />}>
        <Descriptions
          bordered
          size="small"
          column={{ xs: 1, md: 3 }}
          layout="vertical"
          items={[
            { key: "fid", label: "Fidelidade", children: <><Text strong>11–12 meses</Text><br /><Text type="secondary">Janela de maior risco perto do fim do contrato.</Text></> },
            { key: "eq", label: "Equipamento", children: <><Text strong>Aparelho antigo</Text><br /><Text type="secondary">Tempo com o aparelho atual aparece entre os sinais.</Text></> },
            { key: "camp", label: "Campanha", children: <><Text strong>Top 10%</Text><br /><Text type="secondary">54,5% dos clientes nessa faixa cancelaram no teste.</Text></> },
          ]}
        />
      </Card>
    </>
  );
}

function ModelsView() {
  return (
    <>
      <PageHeader title="Modelos & métricas" description="Comparação com validação estratificada e avaliação final em clientes que o modelo não viu no treinamento." />
      <Row gutter={[16, 16]} className="section">
        <Col xs={24} lg={14}>
          <Card title="AUC-ROC por modelo" extra={<SourceTag kind="documented" />} style={{ height: "100%" }}>
            <Text type="secondary">Validação cruzada, 5 folds</Text>
            <ResponsiveContainer width="100%" height={270}>
              <BarChart data={modelData} layout="vertical" margin={{ left: 8, right: 44, top: 8 }}>
                <CartesianGrid stroke="#f0f0f0" horizontal={false} />
                <XAxis type="number" domain={[0.45, 0.7]} hide />
                <YAxis dataKey="name" type="category" width={96} tick={{ fill: "#595959", fontSize: 12 }} axisLine={false} tickLine={false} />
                <ChartTooltip formatter={(value) => [decimal(3)(value), "AUC-ROC"]} cursor={{ fill: "#fafafa" }} />
                <Bar dataKey="score" isAnimationActive={false} maxBarSize={20}>
                  {modelData.map((entry) => (
                    <Cell key={entry.name} fill={entry.name === "LightGBM" ? CHART_PRIMARY : CHART_NEUTRAL} />
                  ))}
                  <LabelList dataKey="score" position="right" formatter={(value) => decimal(3)(value)} fill="#262626" fontSize={12} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        <Col xs={24} lg={10}>
          <Card title="Modelo escolhido" extra={<SourceTag kind="documented" />} style={{ height: "100%" }}>
            <Statistic title="LightGBM · AUC-ROC médio na validação cruzada" value="0,678" />
            <Descriptions
              bordered
              size="small"
              column={1}
              style={{ marginTop: 16 }}
              items={[
                { key: "base", label: "Baseline (regressão logística)", children: "0,652" },
                { key: "diff", label: "Diferença na validação", children: "+0,026" },
                { key: "test", label: "AUC-ROC no teste reservado", children: "0,677" },
                { key: "sel", label: "Seleção de variáveis", children: "Não" },
              ]}
            />
          </Card>
        </Col>
      </Row>
      <Card title="Avaliação final no teste" extra={<SourceTag kind="documented" />}>
        <Paragraph type="secondary">10.209 clientes reservados, corte de 0,5.</Paragraph>
        <Table
          size="middle"
          rowKey="metric"
          pagination={false}
          dataSource={testMetrics}
          scroll={{ x: 480 }}
          columns={[
            { title: "Métrica", dataIndex: "metric" },
            { title: "Baseline", dataIndex: "base", align: "right" },
            { title: "LightGBM", dataIndex: "winner", align: "right", render: (value) => <Text strong>{value}</Text> },
            { title: "Diferença", dataIndex: "gain", align: "right", render: (value, row) => <Text type={row.negative ? "danger" : "success"}>{value}</Text> },
          ]}
        />
        <Paragraph type="secondary" style={{ margin: "12px 0 0" }}>
          O recall do LightGBM ficou abaixo do baseline no corte de 0,5. O ganho está na ordenação (AUC, PR-AUC e lift), que é o uso proposto para a fila de retenção.
        </Paragraph>
      </Card>
    </>
  );
}

const PAYLOAD_PREVIEW = `{
  "CustomerID": 3000002,
  "MonthlyRevenue": 24.0,
  "MonthsInService": 61,
  "CurrentEquipmentDays": 361,
  "CreditRating": "1-Highest",
  "HandsetWebCapable": "Yes",
  ...
}`;

function ClassTag({ value }) {
  return value ? <Tag color="red">Acima do corte</Tag> : <Tag>Abaixo do corte</Tag>;
}

function PriorityTag({ priority }) {
  const { label, color } = PRIORITY[priority];
  return <Tag color={color}>{label}</Tag>;
}

const probabilityColumn = {
  title: "Probabilidade",
  dataIndex: "churn_probability",
  align: "right",
  render: (value) => <Text strong>{formatPercent(value)}</Text>,
};

function ApiView({ apiState, predictionProps, batch, onNavigate }) {
  const [size, setSize] = useState(5);
  const latest = batch.runs[0];
  return (
    <>
      <PageHeader title="API & inferência" description="Consultas à API FastAPI com o modelo campeão (LightGBM)." />
      <ApiNotice apiState={apiState} />
      <Row gutter={[16, 16]} className="section">
        <Col xs={24} lg={12}>
          <Card title="Requisição" extra={<PredictButton {...predictionProps} />} style={{ height: "100%" }}>
            <Paragraph type="secondary"><Text code>POST /predict</Text> com o cliente 3000002.</Paragraph>
            <pre className="code-block">{PAYLOAD_PREVIEW}</pre>
            <Text type="secondary">Trecho do payload. A requisição envia todas as variáveis do schema da API.</Text>
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="Resposta" extra={predictionProps.prediction && <SourceTag kind="api" />} style={{ height: "100%" }}>
            <PredictionResult {...predictionProps} />
          </Card>
        </Col>
      </Row>

      <Card
        className="section"
        title="Simulador de lote"
        extra={<SourceTag kind="synthetic" />}
      >
        <Flex gap={12} wrap align="center" style={{ marginBottom: 16 }}>
          <Text>Quantidade de clientes:</Text>
          <Segmented options={BATCH_SIZES} value={size} onChange={setSize} disabled={batch.loading} />
          <Button type="primary" icon={<PlayCircleOutlined />} loading={batch.loading} onClick={() => batch.onRun(size)}>
            Gerar e consultar {size} clientes
          </Button>
        </Flex>
        <Alert
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
          message="Como os clientes são gerados"
          description={`Cada cliente parte do cadastro do exemplo 3000002 e recebe valores aleatórios em ${GENERATED_FIELDS.length} variáveis, dentro de faixas definidas para a demonstração. Os IDs começam com 91 e não existem na base. A probabilidade de cada linha é a devolvida por POST /predict/batch.`}
        />
        {batch.error && (
          <Alert type="error" showIcon style={{ marginBottom: 16 }} message="Lote não registrado" description={batch.error} />
        )}
        {latest ? (
          <>
            <RunSummary run={latest} />
            <BatchTable run={latest} loading={batch.loading} />
            <Space wrap style={{ marginTop: 16 }}>
              <Button onClick={() => batch.onOpenQueue(latest.id)}>Ver na fila de retenção</Button>
              <Button onClick={() => onNavigate("runs")}>Consultar na base de lotes</Button>
            </Space>
          </>
        ) : batch.loading ? (
          <Flex justify="center" style={{ padding: 32 }}><Spin tip="Enviando o lote para a API..."><div style={{ width: 160, height: 40 }} /></Spin></Flex>
        ) : (
          <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="Nenhum lote enviado ainda." />
        )}
      </Card>

      <Card title="Endpoints da API">
        <Table
          size="small"
          rowKey="path"
          pagination={false}
          dataSource={[
            { method: "GET", path: "/health", text: "Status do serviço e do modelo" },
            { method: "POST", path: "/predict", text: "Probabilidade de churn de um cliente" },
            { method: "POST", path: "/predict/batch", text: "Até 1.000 clientes por chamada" },
            { method: "GET", path: "/docs", text: "Swagger interativo da API" },
          ]}
          columns={[
            { title: "Método", dataIndex: "method", width: 90, render: (value) => <Tag color={value === "GET" ? "blue" : "green"}>{value}</Tag> },
            { title: "Rota", dataIndex: "path", render: (value) => <Text code>{value}</Text> },
            { title: "Uso", dataIndex: "text" },
          ]}
        />
      </Card>
    </>
  );
}

function RunSummary({ run }) {
  const above = run.predictions.filter((item) => item.churn_prediction === 1).length;
  return (
    <Descriptions
      size="small"
      style={{ marginBottom: 12 }}
      column={{ xs: 1, sm: 2, lg: 5 }}
      items={[
        { key: "id", label: "Lote", children: <Text code>{run.id}</Text> },
        { key: "date", label: "Data", children: formatDate(run.createdAt) },
        { key: "n", label: "Clientes", children: run.predictions.length },
        { key: "above", label: "Acima do corte", children: above },
        { key: "time", label: "Tempo no navegador", children: run.durationMs != null ? `${run.durationMs} ms` : "—" },
      ]}
    />
  );
}

function BatchTable({ run, loading }) {
  const columns = [
    { title: "CustomerID", dataIndex: "CustomerID", fixed: "left", width: 110 },
    {
      title: "Retorno da API",
      children: [
        { ...probabilityColumn, width: 150, sorter: (a, b) => a.churn_probability - b.churn_probability },
        { title: "Classificação", dataIndex: "churn_prediction", width: 140, render: (value) => <ClassTag value={value} /> },
      ],
    },
    {
      title: "Entrada gerada (sintética)",
      children: GENERATED_FIELDS.map((field) => ({
        title: <Tooltip title={`${field.label} (${field.key})`}>{field.short}</Tooltip>,
        key: field.key,
        align: "right",
        width: 100,
        render: (_, row) => formatField(field, row.input),
      })),
    },
  ];
  return (
    <Table
      size="small"
      bordered
      rowKey="CustomerID"
      loading={loading}
      dataSource={run.predictions}
      columns={columns}
      scroll={{ x: 1550 }}
      pagination={run.predictions.length > 10 ? { pageSize: 10, showSizeChanger: false } : false}
    />
  );
}

function RunSelect({ batch }) {
  if (!batch.runs.length) return null;
  return (
    <Space>
      <Text type="secondary">Lote exibido:</Text>
      <Select
        value={batch.selectedRun.id}
        onChange={batch.onSelect}
        style={{ minWidth: 240 }}
        options={batch.runs.map((run) => ({ value: run.id, label: `${formatDate(run.createdAt)} · ${run.predictions.length} clientes` }))}
      />
    </Space>
  );
}

function describeInput(input) {
  if (!input) return "Entradas não registradas (lote anterior a esta versão)";
  return `${input.MonthsInService} meses · ${integer(input.MonthlyMinutes)} min/mês · aparelho com ${integer(input.CurrentEquipmentDays)} dias`;
}

const priorityFilters = Object.entries(PRIORITY).map(([value, { label }]) => ({ text: label, value }));

function QueueView({ apiState, batch, onNavigate }) {
  const run = batch.selectedRun;
  const queue = run ? buildQueue(run) : [];
  const above = queue.filter((item) => item.churn_prediction === 1).length;
  const threshold = queue[0] ? formatThreshold(queue[0].threshold) : "0,5";

  const columns = [
    { title: "#", dataIndex: "rank", width: 56 },
    {
      title: "Cliente",
      dataIndex: "CustomerID",
      render: (value, row) => (
        <>
          <Text strong>{value}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: 12 }}>{describeInput(row.input)}</Text>
        </>
      ),
    },
    {
      title: "Probabilidade",
      dataIndex: "churn_probability",
      width: 150,
      render: (value) => (
        <>
          <Text strong>{formatPercent(value)}</Text>
          <ProbabilityBar value={value} />
        </>
      ),
    },
    { title: "Classificação", dataIndex: "churn_prediction", width: 140, render: (value) => <ClassTag value={value} /> },
    { title: "Prioridade", dataIndex: "priority", width: 110, filters: priorityFilters, onFilter: (value, row) => row.priority === value, render: (value) => <PriorityTag priority={value} /> },
    { title: "Status", key: "status", width: 120, render: () => <Tooltip title="Sem CRM conectado: nenhum contato registrado"><Tag>Não iniciada</Tag></Tooltip> },
    { title: "Próximo passo", key: "next", render: (_, row) => PRIORITY[row.priority].next },
  ];

  return (
    <>
      <PageHeader
        title="Fila de retenção"
        description="Clientes do lote ordenados pela probabilidade de churn devolvida pela API, do maior para o menor."
        extra={<RunSelect batch={batch} />}
      />
      <ApiNotice apiState={apiState} />
      <Alert
        className="section"
        type="warning"
        showIcon
        message="Fila de demonstração"
        description="Clientes gerados pelo simulador e pontuados pela API. Não há integração com CRM, então nenhum contato ou oferta é registrado."
      />
      <StatRow
        items={[
          { title: "Clientes no lote", value: run ? run.predictions.length : "—", detail: run ? formatDate(run.createdAt) : "Nenhum lote gerado", source: "synthetic" },
          { title: `Acima do corte (${threshold})`, value: run ? above : "—", detail: "churn_prediction = 1", source: "api" },
          { title: "Maior probabilidade", value: run ? formatPercent(queue[0].churn_probability) : "—", detail: run ? `CustomerID ${queue[0].CustomerID}` : "—", source: "api" },
          { title: "Referência do top 10%", value: "54,5%", detail: "Precisão no teste reservado", source: "documented" },
        ]}
      />
      <Card className="section" title="Fila do lote">
        {run && <div style={{ marginBottom: 12 }}>{sources("api", "synthetic")}</div>}
        {run ? (
          <Table
            size="middle"
            rowKey="CustomerID"
            dataSource={queue}
            columns={columns}
            scroll={{ x: 960 }}
            rowClassName={(row) => (row.priority === "high" ? "row-high" : "")}
            pagination={queue.length > 20 ? { pageSize: 20, showSizeChanger: false } : false}
          />
        ) : (
          <Empty description="A fila é montada a partir de um lote enviado à API.">
            <Space wrap>
              <Button type="primary" icon={<PlayCircleOutlined />} loading={batch.loading} onClick={() => batch.onRun(5)}>
                Gerar e consultar 5 clientes
              </Button>
              <Button onClick={() => onNavigate("api")}>Abrir o simulador de lote</Button>
            </Space>
          </Empty>
        )}
        {batch.error && !run && <Alert type="error" showIcon style={{ marginTop: 16 }} message="Lote não registrado" description={batch.error} />}
      </Card>
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={12}>
          <Card title="Como a prioridade é definida" style={{ height: "100%" }}>
            <Descriptions
              bordered
              size="small"
              column={1}
              items={[
                { key: "high", label: <PriorityTag priority="high" />, children: "Top 10% do lote pela probabilidade, com no mínimo um cliente. É uma posição relativa: em um lote de 5, só o primeiro." },
                { key: "medium", label: <PriorityTag priority="medium" />, children: `Acima do corte da API (${threshold}), fora do top 10%. O corte ainda não foi validado com o negócio.` },
                { key: "low", label: <PriorityTag priority="low" />, children: "Abaixo do corte. Fica na lista apenas para monitoramento." },
              ]}
            />
            <Paragraph type="secondary" style={{ margin: "12px 0 0" }}>
              A probabilidade indica propensão, não certeza de cancelamento, e a fila não mede o efeito de uma campanha.
            </Paragraph>
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="Antes de uma implantação comercial" style={{ height: "100%" }}>
            <List
              size="small"
              dataSource={[
                ["Integração com CRM", "Receber a carteira real e devolver a campanha."],
                ["Autenticação", "Controlar quem consulta e quem altera a fila."],
                ["Histórico de contato", "Evitar abordar o mesmo cliente repetidas vezes."],
                ["Monitoramento", "Acompanhar drift, latência e qualidade do score."],
                ["Validação temporal", "Confirmar o desempenho em períodos posteriores ao treino."],
                ["Teste incremental", "Medir a retenção causada pela campanha com grupo de controle."],
              ]}
              renderItem={([title, text]) => (
                <List.Item extra={<Tag color="orange">Pendente</Tag>}>
                  <List.Item.Meta title={title} description={text} />
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>
    </>
  );
}

function RunsView({ batch, onNavigate }) {
  const [query, setQuery] = useState("");
  const [runFilter, setRunFilter] = useState("all");
  const [classFilter, setClassFilter] = useState("all");
  const [selected, setSelected] = useState(null);

  const rows = useMemo(() => flattenRuns(batch.runs), [batch.runs]);
  const term = query.trim().toLowerCase();
  const filtered = rows
    .filter((row) => runFilter === "all" || row.runId === runFilter)
    .filter((row) => classFilter === "all" || String(row.churn_prediction) === classFilter)
    .filter((row) => !term || String(row.CustomerID).includes(term) || row.runId.toLowerCase().includes(term));

  if (!batch.runs.length) {
    return (
      <>
        <PageHeader title="Base de lotes" description="Consulta dos lotes sintéticos enviados à API neste navegador." />
        <Card>
          <Empty description="Os lotes aparecem aqui depois de enviados pelo simulador.">
            <Button type="primary" onClick={() => onNavigate("api")}>Abrir o simulador de lote</Button>
          </Empty>
        </Card>
      </>
    );
  }

  const runColumns = [
    { title: "Lote", dataIndex: "id", render: (value) => <Text code>{value}</Text> },
    { title: "Data", dataIndex: "createdAt", render: formatDate },
    { title: "Clientes", key: "n", align: "right", render: (_, run) => run.predictions.length },
    { title: "Acima do corte", key: "above", align: "right", render: (_, run) => run.predictions.filter((item) => item.churn_prediction === 1).length },
    { title: "Maior prob.", key: "max", align: "right", render: (_, run) => formatPercent(Math.max(...run.predictions.map((item) => item.churn_probability))) },
    { title: "Tempo", dataIndex: "durationMs", align: "right", render: (value) => (value != null ? `${value} ms` : "—") },
    {
      title: "Ações",
      key: "actions",
      render: (_, run) => (
        <Space>
          <Button size="small" onClick={() => setRunFilter(runFilter === run.id ? "all" : run.id)}>{runFilter === run.id ? "Mostrar todos" : "Filtrar"}</Button>
          <Button size="small" onClick={() => batch.onOpenQueue(run.id)}>Ver fila</Button>
        </Space>
      ),
    },
  ];

  const customerColumns = [
    { title: "CustomerID", dataIndex: "CustomerID", render: (value, row) => <Button type="link" size="small" style={{ padding: 0 }} onClick={() => setSelected(row)}>{value}</Button> },
    { ...probabilityColumn, sorter: (a, b) => a.churn_probability - b.churn_probability, defaultSortOrder: "descend" },
    { title: "Prioridade", dataIndex: "priority", filters: priorityFilters, onFilter: (value, row) => row.priority === value, render: (value) => <PriorityTag priority={value} /> },
    { title: "Classificação", dataIndex: "churn_prediction", render: (value) => <ClassTag value={value} /> },
    { title: "Posição", dataIndex: "rank", align: "right", sorter: (a, b) => a.rank - b.rank, render: (value, row) => `${value} de ${row.runSize}` },
    { title: "Lote", dataIndex: "createdAt", sorter: (a, b) => a.createdAt.localeCompare(b.createdAt), render: formatDate },
  ];

  return (
    <>
      <PageHeader
        title="Base de lotes"
        description="Consulta dos lotes sintéticos enviados à API neste navegador, com a entrada gerada e o retorno de cada cliente."
        extra={
          <>
            <Button icon={<DownloadOutlined />} onClick={() => exportCsv(filtered)} disabled={!filtered.length}>Exportar CSV</Button>
            <Popconfirm title="Apagar o histórico local?" description="Os lotes salvos neste navegador serão removidos." okText="Apagar" cancelText="Cancelar" okButtonProps={{ danger: true }} onConfirm={batch.onClear}>
              <Button danger icon={<DeleteOutlined />}>Limpar histórico</Button>
            </Popconfirm>
          </>
        }
      />
      <Alert
        className="section"
        type="info"
        showIcon
        message="Histórico local"
        description={`Os ${MAX_RUNS} lotes mais recentes ficam no localStorage deste navegador. Não há banco compartilhado: os dados somem se o histórico do site for limpo ou se outro navegador for usado.`}
      />
      <Card className="section" title="Lotes registrados">
        <div style={{ marginBottom: 12 }}>{sources("synthetic", "api")}</div>
        <Table size="small" rowKey="id" dataSource={batch.runs} columns={runColumns} pagination={false} scroll={{ x: 820 }} rowClassName={(run) => (run.id === runFilter ? "ant-table-row-selected" : "")} />
      </Card>
      <Card title="Consulta de clientes" extra={<Text type="secondary">{filtered.length} de {rows.length} clientes</Text>}>
        <Row gutter={[12, 12]} style={{ marginBottom: 16 }}>
          <Col xs={24} md={10}>
            <Input.Search allowClear placeholder="Buscar por CustomerID ou lote" value={query} onChange={(event) => setQuery(event.target.value)} />
          </Col>
          <Col xs={12} md={7}>
            <Select
              style={{ width: "100%" }}
              value={runFilter}
              onChange={setRunFilter}
              options={[{ value: "all", label: "Todos os lotes" }, ...batch.runs.map((run) => ({ value: run.id, label: `${formatDate(run.createdAt)} · ${run.predictions.length}` }))]}
            />
          </Col>
          <Col xs={12} md={7}>
            <Select
              style={{ width: "100%" }}
              value={classFilter}
              onChange={setClassFilter}
              options={[{ value: "all", label: "Todas as classificações" }, { value: "1", label: "Acima do corte" }, { value: "0", label: "Abaixo do corte" }]}
            />
          </Col>
        </Row>
        <Table
          size="small"
          rowKey={rowKey}
          dataSource={filtered}
          columns={customerColumns}
          scroll={{ x: 720 }}
          onRow={(row) => ({ onClick: () => setSelected(row), style: { cursor: "pointer" } })}
          pagination={{ pageSize: 10, showSizeChanger: false, showTotal: (total) => `${total} clientes` }}
        />
      </Card>
      <CustomerDrawer row={selected} onClose={() => setSelected(null)} />
    </>
  );
}

function CustomerDrawer({ row, onClose }) {
  return (
    <Drawer open={Boolean(row)} onClose={onClose} width={420} title={row ? `CustomerID ${row.CustomerID}` : ""}>
      {row && (
        <>
          <Text type="secondary">{row.runId} · {formatDate(row.createdAt)}</Text>
          <Flex justify="space-between" align="center" style={{ margin: "20px 0 8px" }}>
            <Title level={5} style={{ margin: 0 }}>Retorno da API</Title>
            <SourceTag kind="api" />
          </Flex>
          <Statistic title="Probabilidade de churn" value={formatPercent(row.churn_probability)} />
          <ProbabilityBar value={row.churn_probability} width={200} />
          <Descriptions
            bordered
            size="small"
            column={1}
            style={{ marginTop: 12 }}
            items={[
              { key: "pred", label: "churn_prediction", children: <Space>{row.churn_prediction}<ClassTag value={row.churn_prediction} /></Space> },
              { key: "threshold", label: "threshold", children: formatThreshold(row.threshold) },
              { key: "rank", label: "Posição no lote", children: `${row.rank} de ${row.runSize}` },
              { key: "priority", label: "Prioridade", children: <PriorityTag priority={row.priority} /> },
            ]}
          />
          <Flex justify="space-between" align="center" style={{ margin: "24px 0 8px" }}>
            <Title level={5} style={{ margin: 0 }}>Entrada gerada</Title>
            <SourceTag kind="synthetic" />
          </Flex>
          {row.input ? (
            <Descriptions
              bordered
              size="small"
              column={1}
              items={GENERATED_FIELDS.map((field) => ({ key: field.key, label: <Tooltip title={field.key}>{field.label}</Tooltip>, children: formatField(field, row.input) }))}
            />
          ) : (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="Entradas não registradas: lote criado antes desta versão do simulador." />
          )}
          <Paragraph type="secondary" style={{ marginTop: 12 }}>As demais variáveis são iguais às do cliente de exemplo 3000002.</Paragraph>
        </>
      )}
    </Drawer>
  );
}

function Root() {
  return (
    <ConfigProvider locale={ptBR}>
      <AntApp>
        <Dashboard />
      </AntApp>
    </ConfigProvider>
  );
}

createRoot(document.getElementById("root")).render(<Root />);
