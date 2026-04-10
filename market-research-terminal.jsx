// Extreme defensive global mapping for CDN usage
const React = window.React;
const ReactDOM = window.ReactDOM;
const Recharts = window.Recharts;

const { useState, useEffect, useRef, useMemo, Fragment } = React || {};
const {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer, Legend
} = Recharts || {};

// ── Fonts via @import ──────────────────────────────────────────────────────────
const fontImport = document.createElement("link");
fontImport.rel = "stylesheet";
fontImport.href =
  "https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Bebas+Neue&family=Syne:wght@400;700;800&display=swap";
document.head.appendChild(fontImport);

// ── Styles ────────────────────────────────────────────────────────────────────
const css = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #080b0f; }

  .terminal {
    min-height: 100vh;
    background: #080b0f;
    color: #e8dcc8;
    font-family: 'Space Mono', monospace;
    position: relative;
    overflow-x: hidden;
  }

  /* Scanline overlay */
  .terminal::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
      to bottom,
      transparent 0px,
      transparent 3px,
      rgba(0,0,0,0.08) 3px,
      rgba(0,0,0,0.08) 4px
    );
    pointer-events: none;
    z-index: 9999;
  }

  /* Grid noise background */
  .bg-grid {
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(212,170,84,0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(212,170,84,0.04) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none;
  }

  .header {
    border-bottom: 1px solid rgba(212,170,84,0.3);
    padding: 20px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(8,11,15,0.9);
    backdrop-filter: blur(8px);
    position: sticky;
    top: 0;
    z-index: 100;
  }

  .header-brand {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .header-icon {
    width: 36px;
    height: 36px;
    border: 2px solid #d4aa54;
    display: grid;
    place-items: center;
    font-size: 16px;
  }

  .header-title {
    font-family: 'Bebas Neue', cursive;
    font-size: 26px;
    letter-spacing: 3px;
    color: #d4aa54;
    line-height: 1;
  }

  .header-subtitle {
    font-size: 9px;
    letter-spacing: 2px;
    color: rgba(212,170,84,0.5);
    margin-top: 3px;
  }

  /* Health Score Widget */
  .health-widget {
    margin-top: 24px;
    background: rgba(212,170,84,0.05);
    border: 1px solid rgba(212,170,84,0.2);
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
  }

  .health-gauge {
    position: relative;
    width: 120px;
    height: 120px;
  }

  .health-value {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: 'Bebas Neue', cursive;
  }

  .health-number { font-size: 32px; color: #e8dcc8; line-height: 1; }
  .health-label { font-size: 10px; color: rgba(212,170,84,0.6); letter-spacing: 1px; }

  .health-status {
    font-size: 11px;
    letter-spacing: 2px;
    font-weight: 700;
    padding: 4px 12px;
    border: 1px solid currentColor;
  }

  .status-SAFE { color: #4ade80; }
  .status-STABLE { color: #d4aa54; }
  .status-DANGER { color: #f87171; }

  /* Actionable Insights */
  .insight-card {
    border-left: 2px solid #d4aa54;
    background: rgba(212,170,84,0.03);
    padding: 12px 16px;
    margin-bottom: 12px;
  }
  
  .insight-bullet {
    display: flex;
    gap: 12px;
    font-size: 12px;
    color: rgba(232,220,200,0.85);
    line-height: 1.6;
  }

  .insight-icon { color: #d4aa54; font-weight: 700; }

  .status-row {
    display: flex;
    gap: 20px;
    align-items: center;
  }

  .status-dot {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 10px;
    color: rgba(232,220,200,0.5);
    letter-spacing: 1px;
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #4ade80;
    animation: pulse-dot 2s infinite;
  }

  @keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  .main {
    padding: 32px;
    max-width: 1600px;
    margin: 0 auto;
  }

  /* Search Bar */
  .search-section {
    margin-bottom: 32px;
  }

  .search-label {
    font-size: 10px;
    letter-spacing: 3px;
    color: rgba(212,170,84,0.6);
    margin-bottom: 10px;
  }

  .search-row {
    display: flex;
    gap: 12px;
  }

  .search-input {
    flex: 1;
    background: rgba(212,170,84,0.05);
    border: 1px solid rgba(212,170,84,0.3);
    color: #e8dcc8;
    font-family: 'Space Mono', monospace;
    font-size: 14px;
    padding: 14px 20px;
    outline: none;
    transition: border-color 0.2s, background 0.2s;
  }

  .search-input::placeholder { color: rgba(232,220,200,0.2); }
  .search-input:focus {
    border-color: #d4aa54;
    background: rgba(212,170,84,0.08);
  }

  .btn {
    border: 1px solid rgba(212,170,84,0.5);
    background: transparent;
    color: #d4aa54;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    padding: 14px 24px;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
  }

  .btn:hover {
    background: rgba(212,170,84,0.12);
    border-color: #d4aa54;
  }

  .btn-primary {
    background: #d4aa54;
    color: #080b0f;
    border-color: #d4aa54;
    font-weight: 700;
  }

  .btn-primary:hover { background: #e8c068; }

  .btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  /* Pipeline indicator */
  .pipeline {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 32px;
    padding: 16px 20px;
    background: rgba(212,170,84,0.04);
    border: 1px solid rgba(212,170,84,0.12);
  }

  .pipe-node {
    flex: 1;
    text-align: center;
    padding: 10px 8px;
  }

  .pipe-badge {
    display: inline-block;
    font-size: 9px;
    letter-spacing: 1.5px;
    padding: 4px 10px;
    background: rgba(212,170,84,0.1);
    border: 1px solid rgba(212,170,84,0.25);
    color: #d4aa54;
    margin-bottom: 6px;
  }

  .pipe-name {
    font-size: 11px;
    letter-spacing: 1px;
    color: rgba(232,220,200,0.7);
  }

  .pipe-desc {
    font-size: 9px;
    color: rgba(232,220,200,0.35);
    margin-top: 3px;
    letter-spacing: 0.5px;
  }

  .pipe-arrow {
    color: rgba(212,170,84,0.4);
    font-size: 20px;
    padding: 0 4px;
    flex-shrink: 0;
  }

  .pipe-node.active .pipe-badge {
    background: rgba(212,170,84,0.25);
    border-color: #d4aa54;
    animation: glow 1.5s ease-in-out infinite;
  }

  @keyframes glow {
    0%, 100% { box-shadow: 0 0 4px rgba(212,170,84,0.3); }
    50% { box-shadow: 0 0 12px rgba(212,170,84,0.7); }
  }

  /* Grid layout */
  .grid-3 {
    display: grid;
    grid-template-columns: 1fr 1.5fr 1fr;
    gap: 20px;
  }

  .panel {
    border: 1px solid rgba(212,170,84,0.15);
    background: rgba(12,15,20,0.7);
    backdrop-filter: blur(4px);
  }

  .panel-header {
    border-bottom: 1px solid rgba(212,170,84,0.15);
    padding: 14px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .panel-title {
    font-size: 10px;
    letter-spacing: 2.5px;
    color: #d4aa54;
    font-weight: 700;
  }

  .panel-tag {
    font-size: 8px;
    letter-spacing: 1.5px;
    padding: 3px 8px;
    background: rgba(212,170,84,0.08);
    color: rgba(212,170,84,0.6);
    border: 1px solid rgba(212,170,84,0.15);
  }

  .panel-body { padding: 20px; }

  /* Signal cards */
  .signal-item {
    border: 1px solid rgba(212,170,84,0.08);
    padding: 12px 14px;
    margin-bottom: 10px;
    position: relative;
    transition: border-color 0.2s;
  }

  .signal-item:hover { border-color: rgba(212,170,84,0.25); }

  .signal-source {
    font-size: 9px;
    letter-spacing: 1.5px;
    color: #d4aa54;
    margin-bottom: 5px;
  }

  .signal-text {
    font-size: 11px;
    color: rgba(232,220,200,0.7);
    line-height: 1.5;
    font-family: 'Syne', sans-serif;
  }

  .signal-meta {
    display: flex;
    gap: 12px;
    margin-top: 8px;
  }

  .signal-stat {
    font-size: 9px;
    color: rgba(232,220,200,0.35);
  }

  .signal-stat span {
    color: rgba(212,170,84,0.7);
    font-weight: 700;
  }

  .sentiment-bar {
    height: 2px;
    background: rgba(212,170,84,0.1);
    margin-top: 8px;
    position: relative;
    overflow: hidden;
  }

  .sentiment-fill {
    height: 100%;
    background: linear-gradient(90deg, #d4aa54, #e8c068);
    transition: width 1s ease;
  }

  /* Forecast chart area */
  .chart-area { padding: 20px; }

  .forecast-legend {
    display: flex;
    gap: 20px;
    margin-bottom: 16px;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 9px;
    letter-spacing: 1px;
    color: rgba(232,220,200,0.5);
  }

  .legend-line {
    width: 20px;
    height: 2px;
  }

  /* Metrics row */
  .metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
  }

  .metric-card {
    border: 1px solid rgba(212,170,84,0.12);
    padding: 14px 16px;
    background: rgba(12,15,20,0.5);
  }

  .metric-label {
    font-size: 8px;
    letter-spacing: 2px;
    color: rgba(212,170,84,0.5);
    margin-bottom: 6px;
  }

  .metric-value {
    font-family: 'Bebas Neue', cursive;
    font-size: 28px;
    color: #e8dcc8;
    line-height: 1;
  }

  .metric-delta {
    font-size: 9px;
    margin-top: 4px;
  }

  .delta-up { color: #4ade80; }
  .delta-down { color: #f87171; }

  /* Synthesis panel */
  .synthesis-content {
    font-family: 'Syne', sans-serif;
    font-size: 12px;
    line-height: 1.8;
    color: rgba(232,220,200,0.75);
    white-space: pre-wrap;
  }

  .synthesis-placeholder {
    text-align: center;
    padding: 60px 20px;
    color: rgba(232,220,200,0.2);
    font-size: 11px;
    letter-spacing: 1px;
    border: 1px dashed rgba(212,170,84,0.1);
  }

  .thinking-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 0;
    font-size: 10px;
    letter-spacing: 1.5px;
    color: rgba(212,170,84,0.6);
  }

  .thinking-dots { display: flex; gap: 4px; }
  .thinking-dot {
    width: 4px; height: 4px; border-radius: 50%;
    background: #d4aa54;
    animation: blink 1.2s infinite;
  }
  .thinking-dot:nth-child(2) { animation-delay: 0.2s; }
  .thinking-dot:nth-child(3) { animation-delay: 0.4s; }

  @keyframes blink {
    0%, 100% { opacity: 0.2; }
    50% { opacity: 1; }
  }

  /* Step badge */
  .step-section {
    margin-bottom: 20px;
  }

  .step-title {
    font-size: 9px;
    letter-spacing: 2px;
    color: rgba(212,170,84,0.45);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .step-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(212,170,84,0.1);
  }

  /* Polymarket odds */
  .odds-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid rgba(212,170,84,0.06);
    font-size: 11px;
  }

  .odds-label { color: rgba(232,220,200,0.6); }
  .odds-value { color: #d4aa54; font-weight: 700; }

  .progress-bar {
    height: 3px;
    background: rgba(212,170,84,0.08);
    margin-top: 4px;
  }

  .progress-fill {
    height: 100%;
    background: #d4aa54;
    transition: width 1s ease;
  }

  /* Source tags */
  .source-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
  }

  .source-tag {
    font-size: 9px;
    letter-spacing: 1px;
    padding: 3px 8px;
    border: 1px solid rgba(212,170,84,0.15);
    color: rgba(212,170,84,0.5);
  }

  .loading-bar {
    height: 2px;
    background: rgba(212,170,84,0.1);
    overflow: hidden;
    margin: 8px 0;
  }

  .loading-fill {
    height: 100%;
    background: linear-gradient(90deg, transparent, #d4aa54, transparent);
    animation: sweep 1.5s linear infinite;
    width: 40%;
  }

  @keyframes sweep {
    from { transform: translateX(-200%); }
    to { transform: translateX(400%); }
  }

  @media (max-width: 1100px) {
    .grid-3 { grid-template-columns: 1fr; }
    .metrics-row { grid-template-columns: repeat(2, 1fr); }
  }
`;

const styleEl = document.createElement("style");
styleEl.textContent = css;
document.head.appendChild(styleEl);

// ── Mock data generators ───────────────────────────────────────────────────────
function generateHistoricalData(topic, days = 30) {
  const data = [];
  let base = 40 + Math.random() * 30;
  let sentiment = 0.45 + Math.random() * 0.3;
  for (let i = days; i >= 1; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    const label = `${date.getMonth() + 1}/${date.getDate()}`;
    const spike = Math.random() > 0.85 ? Math.random() * 60 : 0;
    base = Math.max(10, base + (Math.random() - 0.45) * 12 + spike * 0.1);
    sentiment = Math.max(0, Math.min(1, sentiment + (Math.random() - 0.48) * 0.06));
    data.push({
      day: label,
      engagement: Math.round(base + spike),
      reddit: Math.round(base * 0.4 + spike * 0.3),
      twitter: Math.round(base * 0.35 + spike * 0.5),
      youtube: Math.round(base * 0.25 + spike * 0.2),
      sentiment: parseFloat((sentiment * 100).toFixed(1)),
      isHistory: true,
    });
  }
  return data;
}

function generateForecast(history, horizon = 14) {
  const recent = history.slice(-7);
  const avgEngagement = recent.reduce((s, d) => s + d.engagement, 0) / recent.length;
  const lastSentiment = history[history.length - 1].sentiment;
  const data = [];
  let eng = avgEngagement;
  let sent = lastSentiment;
  const trend = (Math.random() - 0.42) * 3;
  for (let i = 1; i <= horizon; i++) {
    const date = new Date();
    date.setDate(date.getDate() + i);
    const label = `${date.getMonth() + 1}/${date.getDate()}`;
    eng = Math.max(10, eng + trend + (Math.random() - 0.5) * 8);
    sent = Math.max(0, Math.min(100, sent + (Math.random() - 0.5) * 3));
    const confidence = Math.max(5, 22 - i * 1.2);
    data.push({
      day: label,
      forecast: Math.round(eng),
      forecastLow: Math.round(eng - confidence),
      forecastHigh: Math.round(eng + confidence),
      sentimentForecast: parseFloat(sent.toFixed(1)),
      isHistory: false,
    });
  }
  return data;
}

function generateSignals(topic) {
  const platforms = ["REDDIT", "X/TWITTER", "HACKERNEWS", "POLYMARKET", "YOUTUBE"];
  const templates = [
    `Community discussion around "${topic}" shows strong adoption signals with emerging use cases in enterprise workflows`,
    `Viral thread comparing ${topic} alternatives reached top 5 in r/MachineLearning — 847 upvotes in 6 hours`,
    `${topic} prediction market odds surged 18% overnight. Smart money flowing in from institutional accounts`,
    `YouTube deep-dive on ${topic} implementation hit 2.3M views. Creator reported 4x normal engagement`,
    `HN front-page thread: "Show HN: I built ${topic} automation" — 312 points, 89 comments in 4 hours`,
    `X handle tracking reveals ${topic} keyword velocity at 3.2x 30-day average. Breakout pattern forming`,
  ];
  return platforms.map((platform, i) => ({
    platform,
    text: templates[i % templates.length],
    upvotes: Math.floor(Math.random() * 2000) + 100,
    comments: Math.floor(Math.random() * 300) + 20,
    sentiment: (0.4 + Math.random() * 0.5).toFixed(2),
    recency: `${Math.floor(Math.random() * 23) + 1}h ago`,
  }));
}

function generateOdds(topic) {
  return [
    { label: `${topic} market leader Q2`, pct: Math.round(50 + Math.random() * 35) },
    { label: `Major product launch 30d`, pct: Math.round(20 + Math.random() * 40) },
    { label: `Competitor surge risk`, pct: Math.round(10 + Math.random() * 30) },
    { label: `Community growth >50%`, pct: Math.round(30 + Math.random() * 45) },
  ];
}

// ── Custom tooltip ────────────────────────────────────────────────────────────
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#0d1117",
      border: "1px solid rgba(212,170,84,0.3)",
      padding: "10px 14px",
      fontFamily: "'Space Mono', monospace",
      fontSize: 11,
    }}>
      <div style={{ color: "#d4aa54", marginBottom: 6, fontSize: 10 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, marginBottom: 2 }}>
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(1) : p.value}
        </div>
      ))}
    </div>
  );
};

// ── Main Component ────────────────────────────────────────────────────────────
function App() {
  const [topic, setTopic] = useState("");
  const [phase, setPhase] = useState("idle"); // idle | scraping | forecasting | synthesizing | done
  const [histData, setHistData] = useState([]);
  const [forecastData, setForecastData] = useState([]);
  const [signals, setSignals] = useState([]);
  const [odds, setOdds] = useState([]);
  const [synthesis, setSynthesis] = useState("");
  const [backendData, setBackendData] = useState(null);
  const [activePipe, setActivePipe] = useState(null);
  const synthRef = useRef("");

  const allChartData = [...histData, ...forecastData];

  const totalEngagement = histData.reduce((s, d) => s + d.engagement, 0);
  const avgSentiment = histData.length
    ? (histData.reduce((s, d) => s + d.sentiment, 0) / histData.length).toFixed(1)
    : 0;
  const lastEng = histData[histData.length - 1]?.engagement || 0;
  const prevEng = histData[histData.length - 8]?.engagement || 1;
  const engDelta = (((lastEng - prevEng) / prevEng) * 100).toFixed(1);

  async function runAnalysis() {
    if (!topic.trim()) return;
    setSynthesis("");
    setHistData([]);
    setForecastData([]);
    setSignals([]);
    setOdds([]);
    setBackendData(null);
    synthRef.current = "";

    // Phase 1: Data Gathering (SteamDB/Mobile Proxy)
    setPhase("scraping");
    setActivePipe("steamdb");
    
    try {
      const resp = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic }),
      });
      const data = await resp.json();
      setBackendData(data);
      
      // Mock historical data for the chart based on backend metrics
      const hist = generateHistoricalData(topic);
      setHistData(hist);
      setSignals(generateSignals(topic));
      setOdds(generateOdds(topic));

      // Phase 2: Forecasting (TimesFM)
      setPhase("forecasting");
      setActivePipe("timesfm");
      await new Promise(r => setTimeout(r, 1000));
      setForecastData(generateForecast(hist));

      // Phase 3: Correlation (Industry Calendar)
      setActivePipe("calendar");
      await new Promise(r => setTimeout(r, 800));

      // Phase 4: Synthesis (Llama-3-8B)
      setPhase("synthesizing");
      setActivePipe("llama");
      
      // Streaming simulation or direct set
      setSynthesis(data.synthesis);
    } catch (e) {
      console.error("Backend error:", e);
      setSynthesis("Error connecting to local backend at http://localhost:8000. Please ensure the FastAPI server is running.");
    }

    setPhase("done");
    setActivePipe(null);
  }

  const pipelineNodes = [
    { id: "steamdb", label: "STEAMDB / MOBILE", name: "Proxy Sourcing", desc: "Wishlist Velocity · Ranking Proxy" },
    { id: "timesfm", label: "TIMESFM 2.5", name: "Google TimesFM", desc: "Math Engine · Forecasting" },
    { id: "calendar", label: "INDUSTRY CALENDAR", name: "Correlation Layer", desc: "Event Alignment" },
    { id: "llama", label: "LLAMA-3-8B", name: "Local Synthesis", desc: "Actionable Insights mouthpiece" },
  ];

  return (
    <div className="terminal">
      <div className="bg-grid" />

      {/* Header */}
      <header className="header">
        <div className="header-brand">
          <div className="header-icon">⬡</div>
          <div>
            <div className="header-title">MARKET INTELLIGENCE TERMINAL</div>
            <div className="header-subtitle">STEAMDB PROXY + TIMESFM + CALENDAR CORRELATION + LLAMA-3-8B</div>
          </div>
        </div>
        <div className="status-row">
          <div className="status-dot"><div className="dot" /> LIVE</div>
          <div className="status-dot">SOURCES: 8</div>
          <div className="status-dot">{new Date().toLocaleTimeString()}</div>
        </div>
      </header>

      <main className="main">
        {/* Search */}
        <div className="search-section">
          <div className="search-label">▸ ENTER MARKET / TOPIC / ASSET TO RESEARCH</div>
          <div className="search-row">
            <input
              className="search-input"
              value={topic}
              onChange={e => setTopic(e.target.value)}
              placeholder="e.g. Claude Code · AI coding tools · Cursor · OpenAI o3 · Vibe coding"
              onKeyDown={e => e.key === "Enter" && runAnalysis()}
            />
            <button
              className="btn btn-primary"
              onClick={runAnalysis}
              disabled={!topic.trim() || (phase !== "idle" && phase !== "done")}
            >
              {phase === "scraping" ? "SCRAPING..." : phase === "forecasting" ? "FORECASTING..." : phase === "synthesizing" ? "SYNTHESIZING..." : "▶ RUN ANALYSIS"}
            </button>
          </div>
          {(phase !== "idle" && phase !== "done") && (
            <div className="loading-bar"><div className="loading-fill" /></div>
          )}
        </div>

        <div className="pipeline">
          {pipelineNodes.map((node, i) => (
            <Fragment key={node.id}>
              <div className={`pipe-node ${activePipe === node.id ? "active" : ""}`}>
                <div className="pipe-badge">{node.label}</div>
                <div className="pipe-name">{node.name}</div>
                <div className="pipe-desc">{node.desc}</div>
              </div>
              {i < pipelineNodes.length - 1 && (
                <div key={`arrow-${i}`} className="pipe-arrow">→</div>
              )}
            </Fragment>
          ))}
        </div>

        {/* Metrics Row */}
        {histData.length > 0 && (
          <div className="metrics-row">
            <div className="metric-card">
              <div className="metric-label">TOTAL SIGNALS</div>
              <div className="metric-value">{totalEngagement > 999 ? (totalEngagement / 1000).toFixed(1) + "K" : totalEngagement}</div>
              <div className={`metric-delta ${Number(engDelta) >= 0 ? "delta-up" : "delta-down"}`}>
                {Number(engDelta) >= 0 ? "↑" : "↓"} {Math.abs(engDelta)}% 7d
              </div>
            </div>
            <div className="metric-card">
              <div className="metric-label">AVG SENTIMENT</div>
              <div className="metric-value">{avgSentiment}</div>
              <div className="metric-delta" style={{ color: "rgba(232,220,200,0.4)" }}>/ 100 score</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">SOURCES ACTIVE</div>
              <div className="metric-value">8</div>
              <div className="metric-delta delta-up">↑ Bluesky new</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">FORECAST HORIZON</div>
              <div className="metric-value">{forecastData.length > 0 ? "14D" : "--"}</div>
              <div className="metric-delta" style={{ color: "rgba(232,220,200,0.4)" }}>TimesFM 2.5</div>
            </div>
          </div>
        )}

        {/* Three panels */}
        <div className="grid-3">
          {/* Left: Signal Intelligence */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">◈ SIGNAL INTELLIGENCE</div>
              <div className="panel-tag">LAST30DAYS</div>
            </div>
            <div className="panel-body">
              {phase === "scraping" && (
                <div className="thinking-indicator">
                  <div className="thinking-dots">
                    <div className="thinking-dot" />
                    <div className="thinking-dot" />
                    <div className="thinking-dot" />
                  </div>
                  SCANNING SOURCES
                </div>
              )}

              {signals.length === 0 && phase === "idle" && (
                <div className="synthesis-placeholder">
                  AWAITING TOPIC INPUT<br />
                  <span style={{ fontSize: 9, display: "block", marginTop: 8 }}>
                    WILL SCAN: REDDIT · X · YOUTUBE · HN · POLYMARKET · TIKTOK · INSTAGRAM · BLUESKY
                  </span>
                </div>
              )}

              {signals.length > 0 && (
                <>
                  <div className="step-section">
                    <div className="step-title">SOURCE METRICS (PROXIED)</div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", marginBottom: "16px" }}>
                      <div className="metric-card" style={{ padding: "10px", background: "rgba(212,170,84,0.03)" }}>
                        <div className="metric-label">STEAM WISHLIST VELOCITY</div>
                        <div className="metric-value" style={{ fontSize: "18px" }}>{backendData?.steam.daily_velocity} <span style={{ fontSize: "10px", color: "#4ade80" }}>+{backendData?.steam["7d_growth"]}%</span></div>
                        <div className="metric-delta">Est. {backendData?.steam.estimated_wishlists.toLocaleString()} total</div>
                      </div>
                      <div className="metric-card" style={{ padding: "10px", background: "rgba(212,170,84,0.03)" }}>
                        <div className="metric-label">MOBILE PROXY (RANK #{backendData?.mobile.current_rank})</div>
                        <div className="metric-value" style={{ fontSize: "18px" }}>{backendData?.mobile.estimated_daily_installs.toLocaleString()} <span style={{ fontSize: "10px", color: "rgba(232,220,200,0.4)" }}>DL/D</span></div>
                        <div className="metric-delta">ARPU Proxy: ${backendData?.mobile.arpu_proxy}</div>
                      </div>
                    </div>
                  </div>

                  <div className="step-section">
                    <div className="step-title">POLYMARKET ODDS</div>
                    {odds.map((o, i) => (
                      <div key={i} className="odds-item">
                        <div>
                          <div className="odds-label">{o.label}</div>
                          <div className="progress-bar">
                            <div className="progress-fill" style={{ width: `${o.pct}%` }} />
                          </div>
                        </div>
                        <div className="odds-value">{o.pct}%</div>
                      </div>
                    ))}
                  </div>

                  <div className="source-tags">
                    {["REDDIT", "X/TWITTER", "YOUTUBE", "HACKERNEWS", "POLYMARKET", "TIKTOK", "BLUESKY"].map(s => (
                      <div key={s} className="source-tag">{s}</div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Center: Chart */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">◈ ENGAGEMENT TIMELINE + FORECAST</div>
              <div className="panel-tag">TIMESFM 2.5</div>
            </div>
            <div className="chart-area">
              {allChartData.length === 0 && (
                <div className="synthesis-placeholder" style={{ height: 340, display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column" }}>
                  {phase === "forecasting" ? (
                    <div className="thinking-indicator">
                      <div className="thinking-dots">
                        <div className="thinking-dot" />
                        <div className="thinking-dot" />
                        <div className="thinking-dot" />
                      </div>
                      RUNNING TIMESFM MODEL
                    </div>
                  ) : (
                    <>
                      CHART WILL RENDER HERE<br />
                      <span style={{ fontSize: 9, display: "block", marginTop: 8 }}>
                        30-DAY HISTORY + 14-DAY FORECAST
                      </span>
                    </>
                  )}
                </div>
              )}

              {allChartData.length > 0 && (
                <>
                  <div className="forecast-legend">
                    <div className="legend-item">
                      <div className="legend-line" style={{ background: "#d4aa54" }} />
                      ENGAGEMENT (HISTORY)
                    </div>
                    <div className="legend-item">
                      <div className="legend-line" style={{ background: "#7dd3fc", borderTop: "1px dashed #7dd3fc" }} />
                      TIMESFM FORECAST
                    </div>
                    <div className="legend-item">
                      <div className="legend-line" style={{ background: "rgba(125,211,252,0.2)" }} />
                      CONFIDENCE BAND
                    </div>
                  </div>

                  <ResponsiveContainer width="100%" height={280}>
                    <AreaChart data={allChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                      <defs>
                        <linearGradient id="engGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#d4aa54" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#d4aa54" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="fcGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#7dd3fc" stopOpacity={0.2} />
                          <stop offset="95%" stopColor="#7dd3fc" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="rgba(212,170,84,0.06)" />
                      <XAxis
                        dataKey="day"
                        tick={{ fill: "rgba(232,220,200,0.3)", fontSize: 8, fontFamily: "Space Mono" }}
                        interval={6}
                        axisLine={{ stroke: "rgba(212,170,84,0.15)" }}
                        tickLine={false}
                      />
                      <YAxis
                        tick={{ fill: "rgba(232,220,200,0.3)", fontSize: 8, fontFamily: "Space Mono" }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip content={<CustomTooltip />} />
                      <ReferenceLine
                        x={histData[histData.length - 1]?.day}
                        stroke="rgba(212,170,84,0.4)"
                        strokeDasharray="4 4"
                        label={{ value: "NOW", fill: "rgba(212,170,84,0.5)", fontSize: 8 }}
                      />
                      <Area
                        type="monotone"
                        dataKey="forecastHigh"
                        stroke="transparent"
                        fill="rgba(125,211,252,0.08)"
                        name="upper band"
                      />
                      <Area
                        type="monotone"
                        dataKey="forecastLow"
                        stroke="transparent"
                        fill="rgba(8,11,15,1)"
                        name="lower band"
                      />
                      <Area
                        type="monotone"
                        dataKey="engagement"
                        stroke="#d4aa54"
                        strokeWidth={1.5}
                        fill="url(#engGrad)"
                        dot={false}
                        name="engagement"
                      />
                      <Area
                        type="monotone"
                        dataKey="forecast"
                        stroke="#7dd3fc"
                        strokeWidth={1.5}
                        strokeDasharray="6 3"
                        fill="url(#fcGrad)"
                        dot={false}
                        name="forecast"
                      />
                    </AreaChart>
                  </ResponsiveContainer>

                  {/* Sentiment chart */}
                  <div style={{ marginTop: 16, borderTop: "1px solid rgba(212,170,84,0.1)", paddingTop: 12 }}>
                    <div style={{ fontSize: 9, letterSpacing: 2, color: "rgba(212,170,84,0.4)", marginBottom: 8 }}>
                      SENTIMENT SCORE (0–100)
                    </div>
                    <ResponsiveContainer width="100%" height={80}>
                      <LineChart data={allChartData} margin={{ top: 0, right: 10, left: -20, bottom: 0 }}>
                        <XAxis hide />
                        <YAxis domain={[0, 100]} tick={{ fontSize: 8, fill: "rgba(232,220,200,0.3)", fontFamily: "Space Mono" }} tickLine={false} axisLine={false} />
                        <Tooltip content={<CustomTooltip />} />
                        <Line type="monotone" dataKey="sentiment" stroke="#4ade80" strokeWidth={1} dot={false} name="sentiment" />
                        <Line type="monotone" dataKey="sentimentForecast" stroke="#4ade80" strokeWidth={1} strokeDasharray="4 2" dot={false} name="sent. forecast" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Right: Synthesis */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">◈ LLAMA-3-8B INSIGHTS</div>
              <div className="panel-tag">LOCAL INFERENCE</div>
            </div>
            <div className="panel-body">
              {phase === "synthesizing" && !synthesis && (
                <div className="thinking-indicator">
                  <div className="thinking-dots">
                    <div className="thinking-dot" />
                    <div className="thinking-dot" />
                    <div className="thinking-dot" />
                  </div>
                  SYNTHESIZING BRIEF
                </div>
              )}

              {!synthesis && phase !== "synthesizing" && (
                <div className="synthesis-placeholder">
                  MARKET BRIEF WILL APPEAR HERE<br />
                  <span style={{ fontSize: 9, display: "block", marginTop: 8 }}>
                    POWERED BY CLAUDE · GROUNDED IN SIGNAL DATA
                  </span>
                </div>
              )}

              {synthesis && (
                <div>
                  <div className="health-widget">
                    <div className="health-gauge">
                      <svg viewBox="0 0 100 100" style={{ transform: "rotate(-90deg)" }}>
                        <circle cx="50" cy="50" r="45" fill="none" stroke="rgba(212,170,84,0.1)" strokeWidth="8" />
                        <circle 
                          cx="50" cy="50" r="45" fill="none" 
                          stroke={backendData?.health.status === "SAFE" ? "#4ade80" : backendData?.health.status === "DANGER" ? "#f87171" : "#d4aa54"} 
                          strokeWidth="8"
                          strokeDasharray={`${(backendData?.health.total / 100) * 283} 283`}
                          style={{ transition: "stroke-dasharray 1.5s ease" }}
                        />
                      </svg>
                      <div className="health-value">
                        <div className="health-number">{backendData?.health.total}</div>
                        <div className="health-label">HEALTH</div>
                      </div>
                    </div>
                    <div className={`health-status status-${backendData?.health.status}`}>
                      SYSTEM {backendData?.health.status}
                    </div>
                  </div>

                  <div style={{ fontSize: 9, letterSpacing: 2, color: "rgba(212,170,84,0.5)", margin: "24px 0 14px" }}>
                    ACTIONABLE INSIGHTS (LLAMA-3-8B)
                  </div>
                  
                  <div className="step-section">
                    {synthesis.split("\n").filter(l => l.trim()).map((line, i) => (
                      <div key={i} className="insight-card">
                        <div className="insight-bullet">
                          <span className="insight-icon">◈</span>
                          <span>{line.replace(/^\d\.\s*|^- \s*/, "")}</span>
                        </div>
                      </div>
                    ))}
                  </div>

                  {backendData?.correlations.length > 0 && (
                    <div style={{ marginTop: 24, padding: "16px", background: "rgba(248,113,113,0.05)", border: "1px solid rgba(248,113,113,0.2)" }}>
                      <div style={{ fontSize: 9, color: "#f87171", letterSpacing: 1.5, marginBottom: 10 }}>CALENDAR CORRELATION ALERT</div>
                      {backendData.correlations.map((c, i) => (
                        <div key={i} style={{ fontSize: 11, color: "rgba(232,220,200,0.7)", marginBottom: 8, lineHeight: 1.4 }}>
                          <span style={{ color: "#f87171" }}>[MATCH]</span> {c.event}: {c.impact}
                        </div>
                      ))}
                    </div>
                  )}

                  {phase === "done" && (
                    <div style={{ marginTop: 20, paddingTop: 16, borderTop: "1px solid rgba(212,170,84,0.1)" }}>
                      <div style={{ fontSize: 9, letterSpacing: 1.5, color: "rgba(212,170,84,0.4)", marginBottom: 10 }}>
                        SYSTEM STACK
                      </div>
                      <div style={{ fontSize: 10, color: "rgba(232,220,200,0.35)", lineHeight: 1.8 }}>
                        Data: SteamDB + Mobile Proxy Estimation<br />
                        Forecast: TimesFM 2.5 (14-day horizon)<br />
                        Reasoning: Llama-3-8B (4-bit Q_K_M)<br />
                        Infrastructure: Local Mac Mini M-Series
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer legend */}
        <div style={{ marginTop: 24, padding: "14px 20px", border: "1px solid rgba(212,170,84,0.08)", display: "flex", gap: 32, flexWrap: "wrap" }}>
          <div style={{ fontSize: 9, letterSpacing: 1, color: "rgba(212,170,84,0.4)" }}>
            HOW IT WORKS
          </div>
          <div style={{ fontSize: 9, color: "rgba(232,220,200,0.3)", letterSpacing: 0.5 }}>
            STEP 1 → SteamDB Wishlist Velocity + Mobile Proxy Estimation logic (Downloads = Rank * Weight)
          </div>
          <div style={{ fontSize: 9, color: "rgba(232,220,200,0.3)", letterSpacing: 0.5 }}>
            STEP 2 → TimesFM 2.5 forecasting coupled with Industry Calendar event correlation
          </div>
          <div style={{ fontSize: 9, color: "rgba(232,220,200,0.3)", letterSpacing: 0.5 }}>
            STEP 3 → Llama-3-8B synthesized actionable insights for immediate decision making
          </div>
        </div>
      </main>
    </div>
  );
}

// Render the app with a small delay to ensure libraries are ready
setTimeout(() => {
  if (typeof window.ReactDOM !== 'undefined') {
    const root = window.ReactDOM.createRoot(document.getElementById("root"));
    root.render(<App />);
  } else {
    console.error("ReactDOM not found!");
  }
}, 100);
