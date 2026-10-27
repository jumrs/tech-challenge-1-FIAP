import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertCircle,
  ArrowUpRight,
  BarChart3,
  ClipboardList,
  CheckCircle2,
  ChevronRight,
  CircleDollarSign,
  Database,
  Gauge,
  LayoutDashboard,
  Menu,
  Network,
  Play,
  Radio,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Target,
  Users,
  X,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./styles.css";

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

const riskData = [
  { name: "Base", churn: 28.8 },
  { name: "Teste", churn: 28.8 },
];
const modelData = [
  { name: "Dummy", score: 0.500 },
  { name: "MLP", score: 0.650 },
  { name: "XGBoost", score: 0.657 },
  { name: "Random Forest", score: 0.670 },
  { name: "Logística", score: 0.652 },
  { name: "LightGBM", score: 0.678 },
];

function App() {
  const [active, setActive] = useState("overview");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [apiState, setApiState] = useState("checking");
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch("/api/health")
      .then((response) => (response.ok ? response.json() : Promise.reject()))
      .then(() => setApiState("online"))
      .catch(() => setApiState("offline"));
  }, []);

  const predict = async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(exampleCustomer),
      });
      if (!response.ok) throw new Error("API indisponível");
      setPrediction(await response.json());
      setApiState("online");
    } catch {
      setApiState("offline");
    } finally {
      setLoading(false);
    }
  };

  const page = {
    overview: <Overview onPredict={predict} prediction={prediction} loading={loading} />,
    portfolio: <PortfolioView onPredict={predict} prediction={prediction} loading={loading} />,
    data: <DataView />,
    models: <ModelsView />,
    api: <ApiView apiState={apiState} onPredict={predict} prediction={prediction} loading={loading} />,
  }[active];

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileOpen ? "open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Radio size={21} /></div>
          <div><strong>Churn<span>IQ</span></strong><small>CELL2CELL / CHURN</small></div>
          <button className="mobile-close" onClick={() => setMobileOpen(false)}><X size={20} /></button>
        </div>
        <div className="workspace"><span className="live-dot" /> Base Cell2Cell <ChevronRight size={14} /></div>
        <nav>
          <NavItem icon={<LayoutDashboard size={18} />} label="Visão geral" active={active === "overview"} onClick={() => { setActive("overview"); setMobileOpen(false); }} />
          <NavItem icon={<ClipboardList size={18} />} label="Fila de retenção" active={active === "portfolio"} onClick={() => { setActive("portfolio"); setMobileOpen(false); }} />
          <NavItem icon={<Database size={18} />} label="Dados & EDA" active={active === "data"} onClick={() => { setActive("data"); setMobileOpen(false); }} />
          <NavItem icon={<BarChart3 size={18} />} label="Modelos & métricas" active={active === "models"} onClick={() => { setActive("models"); setMobileOpen(false); }} />
          <NavItem icon={<Network size={18} />} label="API & inferência" active={active === "api"} onClick={() => { setActive("api"); setMobileOpen(false); }} />
        </nav>
        <div className="sidebar-bottom">
          <div className="progress-label"><span>Escopo do protótipo</span><b>EDA · ML · API</b></div>
          <div className="sidebar-note"><Sparkles size={16} /><span>Dados, modelo e inferência no mesmo fluxo</span></div>
        </div>
      </aside>
      {mobileOpen && <div className="scrim" onClick={() => setMobileOpen(false)} />}
      <main className="main-content">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMobileOpen(true)}><Menu size={22} /></button>
          <div className="breadcrumb"><span>Tech Challenge</span><ChevronRight size={14} /><b>Fase 1</b></div>
          <div className="top-actions"><span className={`status-pill ${apiState}`}><span /> API {apiState === "online" ? "online" : apiState === "offline" ? "offline" : "verificando"}</span><div className="avatar">JV</div></div>
        </header>
        <div className="page-wrap">{page}</div>
      </main>
    </div>
  );
}

function NavItem({ icon, label, active, onClick }) {
  return <button className={`nav-item ${active ? "active" : ""}`} onClick={onClick}>{icon}<span>{label}</span>{active && <ChevronRight size={15} />}</button>;
}

function PageTitle({ eyebrow, title, description, action }) {
  return <div className="page-title"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action}</div>;
}

function Overview({ onPredict, prediction, loading }) {
  return <><PageTitle eyebrow="VISÃO GERAL / 01" title={<>Risco de cancelamento,<br /><em>com evidência.</em></>} description="Leitura executiva da base Cell2Cell, do modelo escolhido e da chamada de inferência." action={<button className="primary-btn" onClick={onPredict}><Play size={16} fill="currentColor" /> Consultar cliente exemplo</button>} />
    <div className="hero-grid">
      <div className="hero-card"><div className="hero-top"><div><span className="card-label">TAXA HISTÓRICA DE CHURN</span><div className="hero-number">28,8<span>%</span></div><div className="trend"><Database size={14} /> 51.047 clientes na base</div></div><div className="hero-icon"><Target size={24} /></div></div><div className="chart-wrap"><ResponsiveContainer width="100%" height={120}><AreaChart data={riskData}><defs><linearGradient id="risk" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#c7f36b" stopOpacity=".32" /><stop offset="100%" stopColor="#c7f36b" stopOpacity="0" /></linearGradient></defs><XAxis dataKey="name" hide /><YAxis hide /><Tooltip content={<ChartTip suffix="%" />} /><Area type="monotone" dataKey="churn" stroke="#c7f36b" strokeWidth={2.5} fill="url(#risk)" /></AreaChart></ResponsiveContainer></div><div className="chart-foot"><span>AMOSTRA HISTÓRICA</span><span>APENAS ILUSTRAÇÃO DA EDA</span></div></div>
      <div className="side-stat"><span className="card-label">RESULTADOS DO TESTE</span><div className="stat-main">AUC 0,677</div><p>LightGBM no conjunto reservado de 10.209 clientes.</p><div className="mini-divider" /><div className="stat-row"><span><CircleDollarSign size={15} /> AUC do baseline</span><b>0,647</b></div><div className="stat-row"><span><Users size={15} /> Precisão no top 10%</span><b>54,5%</b></div><div className="stat-row"><span><ShieldCheck size={15} /> Lift no top 10%</span><b className="green">1,89x</b></div></div>
    </div>
    <div className="section-heading"><div><h2>Fluxo do projeto</h2><p>Da pergunta de negócio à chamada de inferência.</p></div><span className="section-meta"><CheckCircle2 size={15} /> Escopo da Fase 1</span></div>
    <div className="pipeline"><PipelineStep number="01" icon={<Users size={19} />} title="Definir" text="Churn e retenção" done /><PipelineStep number="02" icon={<Database size={19} />} title="Explorar" text="51.047 clientes Cell2Cell" done /><PipelineStep number="03" icon={<Gauge size={19} />} title="Comparar" text="4 modelos + baseline" done /><PipelineStep number="04" icon={<Activity size={19} />} title="Servir" text="FastAPI /predict" current /></div>
    <div className="bottom-grid"><div className="insight-card"><div className="insight-icon"><Sparkles size={19} /></div><div><span className="card-label">RESULTADO NO TESTE</span><h3>O top 10% concentra 54,5% de churn</h3><p>O lift de 1,89x sustenta o uso do modelo para ordenar a fila da campanha, não para decidir sozinho.</p></div><ArrowUpRight className="insight-arrow" size={20} /></div><div className="demo-card"><div><span className="card-label">CLIENTE DE EXEMPLO</span><h3>Consultar o CustomerID 3000002</h3><p>Envie o payload da documentação para a API.</p></div><button className="round-btn" onClick={onPredict} disabled={loading}>{loading ? "..." : <Play size={18} fill="currentColor" />}</button>{prediction && <div className="demo-result">Probabilidade: <b>{(prediction.churn_probability * 100).toFixed(1)}%</b></div>}</div></div>
  </>;
}

function PortfolioView({ onPredict, prediction, loading }) {
  return <><PageTitle eyebrow="OPERAÇÃO / 02" title={<>Fila de retenção<br /><em>orientada por risco.</em></>} description="A área onde a equipe escolhe quem abordar, com qual prioridade e usando qual canal." action={<button className="secondary-btn"><SlidersHorizontal size={16} /> Filtros da campanha</button>} />
    <div className="notice-panel"><div className="notice-icon"><AlertCircle size={19} /></div><div><b>Integração de carteira pendente</b><p>A API atual recebe um cliente ou um lote para inferência. A fila abaixo mostra o contrato da operação; a conexão com CRM e histórico de ações fica fora do escopo desta fase.</p></div><span className="scope-tag">FORA DO ESCOPO ATUAL</span></div>
    <div className="kpi-row"><Kpi title="Regra de priorização" value="Probabilidade" detail="Ordenar do maior para o menor risco" icon={<Target />} /><Kpi title="Janela sugerida" value="Top 10%" detail="Precisão observada: 54,5%" icon={<Users />} /><Kpi title="Ação do time" value="Retenção" detail="Oferta definida pelo negócio" icon={<ClipboardList />} /><Kpi title="Origem do score" value="API" detail="POST /predict ou /predict/batch" icon={<Network />} /></div>
    <div className="content-grid"><div className="panel large"><PanelTitle title="Prévia da fila" subtitle="Cliente usado na documentação da API" /><div className="queue-table"><div className="queue-head"><span>CLIENTE</span><span>RISCO</span><span>STATUS</span><span>PRÓXIMO PASSO</span></div><div className="queue-row"><div><b>CustomerID 3000002</b><small>61 meses · SEAPOR503 · 2 aparelhos</small></div><strong className="risk-text">{prediction ? `${(prediction.churn_probability * 100).toFixed(1)}%` : "—"}</strong><span className="queue-status">Aguardando ação</span><button className="table-action" onClick={onPredict} disabled={loading}>{loading ? "Consultando" : prediction ? "Atualizar score" : "Calcular score"} <ArrowUpRight size={14} /></button></div></div><div className="empty-queue"><ClipboardList size={22} /><span>Para uma fila real, envie um lote de clientes pelo endpoint <code>/predict/batch</code>.</span></div></div><div className="panel"><PanelTitle title="Regra de decisão" subtitle="O que o score significa" /><div className="rule-block"><div className="rule-line"><span className="rule-mark high" /><div><b>Alta prioridade</b><p>Clientes no topo da ordenação da campanha.</p></div></div><div className="rule-line"><span className="rule-mark medium" /><div><b>Avaliar com o negócio</b><p>O threshold 0,5 é uma configuração inicial, não uma política de retenção validada.</p></div></div><div className="rule-line"><span className="rule-mark low" /><div><b>Sem decisão automática</b><p>O modelo não dispara ofertas nem altera o cadastro do cliente.</p></div></div></div></div></div>
    <div className="panel full-panel"><PanelTitle title="Controles necessários antes de vender" subtitle="Itens que transformam o protótipo em produto operacional" /><div className="control-grid"><Control title="Integração CRM" text="Receber carteira e devolver campanha." status="Pendente" /><Control title="Histórico de contato" text="Evitar abordar o mesmo cliente repetidamente." status="Pendente" /><Control title="Monitoramento" text="Acompanhar drift, latência e qualidade do score." status="Pendente" /><Control title="Teste incremental" text="Medir retenção causada pela campanha." status="Pendente" /></div></div>
  </>;
}

function PipelineStep({ number, icon, title, text, done, current }) { return <div className={`pipeline-step ${current ? "current" : ""}`}><div className="step-number">{done ? <CheckCircle2 size={17} /> : number}</div><div className="step-icon">{icon}</div><div><b>{title}</b><span>{text}</span></div>{!current && <ChevronRight className="step-arrow" size={17} />}</div>; }
function ChartTip({ active, payload, label, suffix = "" }) { return active && payload?.length ? <div className="chart-tip"><b>{label}</b><span>{Number(payload[0].value).toFixed(suffix ? 1 : 3)}{suffix}</span></div> : null; }

function DataView() { return <><PageTitle eyebrow="DATA LAYER / 02" title={<>Conheça a base<br /><em>antes de modelar.</em></>} description="Qualidade, comportamento e sinais que orientaram o pré-processamento." /><div className="kpi-row"><Kpi title="Registros" value="51.047" detail="2 removidos na limpeza" icon={<Database />} /><Kpi title="Variáveis" value="58" detail="56 features + ID + target" icon={<BarChart3 />} /><Kpi title="Missing crítico" value="56,8%" detail="HandsetPrice desconhecido" icon={<AlertCircle />} /><Kpi title="Duplicidades" value="0" detail="CustomerID validado" icon={<ShieldCheck />} /></div><div className="content-grid"><div className="panel large"><PanelTitle title="Perfil de churn" subtitle="Distribuição do target na base Cell2Cell" /><div className="donut-area"><ResponsiveContainer width="48%" height={230}><PieChart><Pie data={[{ name: "Não churn", value: 71.2 }, { name: "Churn", value: 28.8 }]} innerRadius={68} outerRadius={92} dataKey="value" stroke="none"><Cell fill="#c7f36b" /><Cell fill="#60708d" /></Pie><text x="50%" y="47%" textAnchor="middle" dominantBaseline="middle" fill="#f2f5f9" fontSize="25" fontWeight="700">28,8%</text><text x="50%" y="58%" textAnchor="middle" fill="#8290a8" fontSize="11">CHURN</text></PieChart></ResponsiveContainer><div className="legend-list"><div><i className="dot lime" /><span>Não churn</span><b>71,2%</b></div><div><i className="dot blue" /><span>Churn</span><b>28,8%</b></div><p>O desbalanceamento torna AUC-ROC e PR-AUC mais úteis que acurácia isolada.</p></div></div></div><div className="panel"><PanelTitle title="Decisões de tratamento" subtitle="Transparência do pipeline" /><Decision text="Imputação pela mediana" /><Decision text="One-hot encoding" /><Decision text="Flags de ausência" /><Decision text="Leakage removido" /><Decision text="Teste reservado" /></div></div><div className="panel full-panel"><PanelTitle title="Sinais encontrados na EDA" subtitle="Hipóteses que viraram features" /><div className="signal-grid"><Signal label="Fidelidade" value="11–12 meses" text="Janela de maior risco ao fim do contrato" /><Signal label="Equipamento" value="361 dias" text="Aparelho antigo aparece entre os sinais" /><Signal label="Campanha" value="Top 10%" text="54,5% realmente cancelariam" /></div></div></>; }
function ModelsView() { return <><PageTitle eyebrow="MODEL LAB / 03" title={<>Do baseline ao<br /><em>modelo campeão.</em></>} description="Comparação justa, validação estratificada e avaliação em clientes nunca vistos." action={<div className="champion-badge"><Sparkles size={16} /> CAMPEÃO: LIGHTGBM</div>} /><div className="model-layout"><div className="panel chart-panel"><PanelTitle title="AUC-ROC por modelo" subtitle="Validação cruzada · 5 folds" /><ResponsiveContainer width="100%" height={280}><BarChart data={modelData} layout="vertical" margin={{ left: 12, right: 20 }}><CartesianGrid stroke="#27344a" horizontal={false} /><XAxis type="number" domain={[0.45, 0.72]} hide /><YAxis dataKey="name" type="category" width={90} tick={{ fill: "#93a1b6", fontSize: 11 }} axisLine={false} tickLine={false} /><Tooltip content={<ChartTip />} /><Bar dataKey="score" radius={[0, 5, 5, 0]}>{modelData.map((entry) => <Cell key={entry.name} fill={entry.name === "LightGBM" ? "#c7f36b" : "#3a4c68"} />)}</Bar></BarChart></ResponsiveContainer></div><div className="panel champion-panel"><span className="card-label">MODELO ESCOLHIDO</span><div className="champion-logo"><Sparkles size={22} /></div><h2>LightGBM</h2><p>Sem seleção de features</p><div className="score-big">0,678 <small>AUC-ROC CV</small></div><div className="score-list"><div><span>Baseline</span><b>0,652</b></div><div><span>Ganho</span><b className="green">+0,026</b></div><div><span>Teste reservado</span><b>0,677</b></div></div></div></div><div className="panel full-panel"><PanelTitle title="Avaliação final no teste" subtitle="10.209 clientes nunca vistos no treinamento" /><div className="metrics-table"><div className="table-head"><span>MÉTRICA</span><span>BASELINE</span><span>CAMPEÃO</span><span>GANHO</span></div><MetricRow label="AUC-ROC" base="0,647" winner="0,677" gain="+0,030" /><MetricRow label="PR-AUC" base="0,412" winner="0,454" gain="+0,043" /><MetricRow label="Precisão" base="0,380" winner="0,404" gain="+0,023" /><MetricRow label="Recall" base="0,650" winner="0,631" gain="−0,019" negative /><MetricRow label="Lift no top 10%" base="1,69x" winner="1,89x" gain="+0,20" /></div></div></>; }
function ApiView({ apiState, onPredict, prediction, loading }) { return <><PageTitle eyebrow="DECISION API / 04" title={<>Uma previsão.<br /><em>Uma próxima ação.</em></>} description="O modelo persistido transforma o cadastro de um cliente em propensão de churn em milissegundos." action={<span className={`status-pill big ${apiState}`}><span /> {apiState === "online" ? "API conectada" : "API não conectada"}</span>} /><div className="api-layout"><div className="panel request-panel"><div className="code-head"><span><span className="method">POST</span> /predict</span><span className="live-code"><span /> LIVE</span></div><div className="code-body"><span className="code-muted">{"{"}</span><br />&nbsp;&nbsp;<CodeKey>CustomerID</CodeKey>: <CodeValue>3000002</CodeValue>,<br />&nbsp;&nbsp;<CodeKey>MonthlyRevenue</CodeKey>: <CodeValue>24.0</CodeValue>,<br />&nbsp;&nbsp;<CodeKey>MonthsInService</CodeKey>: <CodeValue>61</CodeValue>,<br />&nbsp;&nbsp;<CodeKey>CreditRating</CodeKey>: <CodeString>"1-Highest"</CodeString>,<br />&nbsp;&nbsp;<CodeKey>HandsetWebCapable</CodeKey>: <CodeString>"Yes"</CodeString>,<br />&nbsp;&nbsp;<span className="code-muted">...</span><br /><span className="code-muted">{"}"}</span></div><button className="primary-btn full" onClick={onPredict} disabled={loading}><Play size={16} fill="currentColor" /> {loading ? "Consultando..." : "Executar inferência"}</button></div><div className="panel response-panel"><div className="code-head"><span>Resposta do modelo</span><span className="response-code">200 OK</span></div>{prediction ? <><div className="risk-result"><div className="risk-circle"><strong>{(prediction.churn_probability * 100).toFixed(1)}<small>%</small></strong><span>probabilidade</span></div><div><span className="card-label">AÇÃO RECOMENDADA</span><h2>{prediction.churn_prediction ? "Priorizar retenção" : "Monitorar cliente"}</h2><p>Threshold configurado: {prediction.threshold}</p></div></div><div className="response-items"><span>Customer ID <b>{prediction.CustomerID}</b></span><span>Predição <b className={prediction.churn_prediction ? "red" : "green"}>{prediction.churn_prediction ? "CHURN" : "NÃO CHURN"}</b></span></div></> : <div className="empty-response"><Network size={28} /><p>Clique em executar para consultar o modelo campeão.</p></div>}</div></div><div className="endpoint-row"><Endpoint method="GET" path="/health" text="Status do serviço e do modelo" /><Endpoint method="POST" path="/predict/batch" text="Até 1.000 clientes por chamada" /><Endpoint method="GET" path="/docs" text="Swagger interativo" /></div></>; }
function CodeKey({ children }) { return <span className="code-key">{children}</span>; } function CodeValue({ children }) { return <span className="code-value">{children}</span>; } function CodeString({ children }) { return <span className="code-string">{children}</span>; }
function Kpi({ title, value, detail, icon }) { return <div className="kpi-card"><div className="kpi-icon">{icon}</div><span className="card-label">{title}</span><strong>{value}</strong><small>{detail}</small></div>; }
function PanelTitle({ title, subtitle }) { return <div className="panel-title"><div><h3>{title}</h3><p>{subtitle}</p></div><MoreDots /></div>; } function MoreDots() { return <span className="dots">•••</span>; }
function Decision({ text }) { return <div className="decision"><CheckCircle2 size={16} /><span>{text}</span></div>; } function Signal({ label, value, text }) { return <div className="signal"><span className="card-label">{label}</span><b>{value}</b><p>{text}</p></div>; } function MetricRow({ label, base, winner, gain, negative }) { return <div className="table-row"><span>{label}</span><span>{base}</span><strong>{winner}</strong><span className={negative ? "negative" : "green"}>{gain}</span></div>; } function Endpoint({ method, path, text }) { return <div className="endpoint"><span className={`method ${method === "GET" ? "get" : ""}`}>{method}</span><b>{path}</b><small>{text}</small></div>; }
function Control({ title, text, status }) { return <div className="control"><span className="control-icon"><SlidersHorizontal size={16} /></span><div><b>{title}</b><p>{text}</p></div><span className="scope-tag">{status}</span></div>; }

createRoot(document.getElementById("root")).render(<App />);
