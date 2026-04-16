import React, { useState, useMemo, Fragment } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer
} from 'recharts';
import { 
  Zap, Shield, Target, AlertTriangle, 
  Search, Lightbulb, Users, BarChart3, Globe 
} from 'lucide-react';
import './App.css';

// ── Baseline Data Generators ──────────────────────────────────────────────────
function generateHistoricalData(topic, days = 30) {
  const data = [];
  let base = 50 + Math.random() * 20;
  for (let i = days; i >= 1; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    const label = `${date.getMonth() + 1}/${date.getDate()}`;
    const spike = Math.random() > 0.85 ? Math.random() * 40 : 0;
    base = Math.max(10, base + (Math.random() - 0.45) * 6 + spike * 0.1);
    data.push({
      day: label,
      engagement: Math.round(base + spike),
      isHistory: true,
    });
  }
  return data;
}

function generateForecast(history, horizon = 14) {
  const recent = history.slice(-7);
  const avgEngagement = recent.reduce((s, d) => s + d.engagement, 0) / (recent.length || 1);
  const data = [];
  let eng = avgEngagement;
  for (let i = 1; i <= horizon; i++) {
    const date = new Date();
    date.setDate(date.getDate() + i);
    const label = `${date.getMonth() + 1}/${date.getDate()}`;
    eng = Math.max(10, eng + (Math.random() - 0.42) * 4);
    const confidence = Math.max(5, 10 - i * 0.5);
    data.push({
      day: label,
      forecast: Math.round(eng),
      forecastLow: Math.round(eng - confidence),
      forecastHigh: Math.round(eng + confidence),
      isHistory: false,
    });
  }
  return data;
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass-panel" style={{ padding: "15px", fontSize: 12, border: "1px solid var(--accent-cyan)" }}>
      <div style={{ color: "var(--accent-cyan)", marginBottom: 8, fontWeight: 800 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ marginBottom: 4 }}>
          {p.name === 'engagement' ? 'TƯƠNG TÁC' : 'DỰ BÁO'}: {p.value}
        </div>
      ))}
    </div>
  );
};

export default function App() {
  const [mode, setMode] = useState("search"); // "search" or "concept"
  const [topic, setTopic] = useState("");
  const [concept, setConcept] = useState({ genre: "", platform: "PC", description: "" });
  
  const [phase, setPhase] = useState("idle");
  const [histData, setHistData] = useState([]);
  const [forecastData, setForecastData] = useState([]);
  const [backendData, setBackendData] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [activePipe, setActivePipe] = useState(null);

  const allChartData = useMemo(() => [...histData, ...forecastData], [histData, forecastData]);

  async function runSearchAnalysis() {
    if (!topic.trim()) return;
    performAnalysis("/api/analyze", { topic });
  }

  async function runConceptAnalysis() {
    if (!concept.genre.trim()) return;
    performAnalysis("/api/analyze-concept", concept);
  }

  async function performAnalysis(url, body) {
    setPhase("scraping");
    setHistData([]);
    setForecastData([]);
    setBackendData(null);
    setErrorMessage("");

    try {
      setActivePipe("steamdb");
      const resp = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        throw new Error(`Backend returned status ${resp.status}`);
      }
      const data = await resp.json();
      setBackendData(data);
      
      const hist = generateHistoricalData(topic || concept.genre);
      setHistData(hist);

      setPhase("forecasting");
      setActivePipe("timesfm");
      await new Promise(r => setTimeout(r, 1200));
      setForecastData(generateForecast(hist));

      setActivePipe("calendar");
      await new Promise(r => setTimeout(r, 500));

      setPhase("synthesizing");
      setActivePipe("llama");
      await new Promise(r => setTimeout(r, 800));

    } catch (e) {
      console.error("Analysis Error:", e);
      setErrorMessage("Không thể phân tích dữ liệu lúc này. Hãy kiểm tra backend FastAPI và thử lại.");
    }

    setPhase("done");
    setActivePipe(null);
  }

  const pipelineNodes = [
    { id: "steamdb", label: "PHÂN TÍCH", name: mode === "search" ? "Dữ liệu Thật" : "Dữ liệu Đối soát", desc: "MARKET SCAN" },
    { id: "timesfm", label: "MÔ HÌNH", name: "Dự báo Xu hướng", desc: "TIMESFM 2.5" },
    { id: "calendar", label: "PHẢN BIỆN", name: "Mô hình Kịch bản", desc: "RED-TEAM LOGIC" },
    { id: "llama", label: "TỔNG HỢP", name: "Chiến lược 3-Card", desc: "LLAMA-3-8B" },
  ];

  return (
    <div className="terminal">
      <div className="bg-grid" />

      <header className="header">
        <div className="header-brand">
          <div className="header-icon">⬟</div>
          <div>
            <div className="header-title">PENTAANA PRO</div>
            <div className="header-subtitle">CONCEPT INTELLIGENCE // FAILURE ANALYSIS // GHOST-CYAN</div>
          </div>
        </div>
        <div className="status-row">
          <div className="mode-switcher">
            <button className={`mode-btn ${mode === "search" ? "active" : ""}`} onClick={() => setMode("search")}>
              <Search size={14} /> SEARCH
            </button>
            <button className={`mode-btn ${mode === "concept" ? "active" : ""}`} onClick={() => setMode("concept")}>
              <Lightbulb size={14} /> CONCEPT
            </button>
          </div>
          <div className="status-dot">
            <div className="dot" /> <span style={{color: "var(--accent-cyan)"}}>SYSTEM READY</span>
          </div>
        </div>
      </header>

      <main className="main">
        {mode === "search" ? (
          <div className="search-section">
            <div className="search-label">▸ TRUY VẤN DỮ LIỆU THẬT CỦA GAME HIỆN TẠI / TƯƠNG LAI</div>
            <div className="search-row">
              <input
                className="search-input"
                value={topic}
                onChange={e => setTopic(e.target.value)}
                placeholder="Ví dụ: Elden Ring · Hollow Knight Silksong · GTA VI"
                onKeyDown={e => e.key === "Enter" && runSearchAnalysis()}
              />
              <button className="btn-primary" onClick={runSearchAnalysis} disabled={phase !== "idle" && phase !== "done"}>
                {phase === "idle" || phase === "done" ? "▶ CHẠY PHÂN TÍCH" : "ĐANG XỬ LÝ..."}
              </button>
            </div>
          </div>
        ) : (
          <div className="concept-section glass-panel">
            <div className="search-label">▸ THẨM ĐỊNH Ý TƯỞNG GAME MỚI (PRO MODE)</div>
            <div className="concept-grid">
              <input 
                className="search-input" 
                placeholder="Thể loại (e.g. Soulslike Mobile, Co-op FPS)" 
                value={concept.genre}
                onChange={e => setConcept({...concept, genre: e.target.value})}
              />
              <select 
                className="search-input"
                value={concept.platform}
                onChange={e => setConcept({...concept, platform: e.target.value})}
              >
                <option value="PC">Nền tảng: PC / Console</option>
                <option value="Mobile">Nền tảng: Mobile</option>
                <option value="Cross-platform">Nền tảng: Cross-platform</option>
              </select>
              <textarea 
                className="search-input" 
                placeholder="Giải thích ý tưởng / Cơ chế cốt lõi..." 
                style={{ gridColumn: "span 2", minHeight: "80px" }}
                value={concept.description}
                onChange={e => setConcept({...concept, description: e.target.value})}
              />
              <button 
                className="btn-primary" 
                style={{ gridColumn: "span 2" }}
                onClick={runConceptAnalysis}
                disabled={phase !== "idle" && phase !== "done"}
              >
                {phase === "idle" || phase === "done" ? "⚡ THẨM ĐỊNH Ý TƯỞNG" : "ĐANG CHẠY RED-TEAM ANALYSIS..."}
              </button>
            </div>
          </div>
        )}

        <div className="pipeline">
          {pipelineNodes.map((node) => (
            <div key={node.id} className={`pipe-node ${activePipe === node.id ? "active" : ""}`}>
              <div className="pipe-badge">{node.label}</div>
              <div className="pipe-name">{node.name}</div>
              <div className="pipe-desc">{node.desc}</div>
            </div>
          ))}
        </div>

        {errorMessage && (
          <div className="glass-panel" style={{ padding: 16, marginBottom: 24, border: "1px solid #f87171", color: "#fca5a5" }}>
            {errorMessage}
          </div>
        )}

        {backendData && (
          <Fragment>
            <div className="metrics-row" style={{ gridTemplateColumns: "repeat(5, 1fr)" }}>
              <div className="metric-card">
                <div className="metric-label"><Users size={12} /> ƯỚC TÍNH FOLLOWERS</div>
                <div className="metric-value" style={{ fontSize: 24 }}>{(backendData.scenarios?.followers_proxy ?? 0).toLocaleString()}</div>
                <div className="metric-caption">Dự báo Wishlist: x12</div>
              </div>
              <div className="metric-card">
                <div className="metric-label"><Zap size={12} /> REDDIT ENGAGEMENT</div>
                <div className="metric-value" style={{ fontSize: 24, color: "var(--accent-cyan)" }}>{(backendData.reddit?.engagement ?? 0).toLocaleString()}</div>
                <div className="metric-caption">Mentions: {backendData.reddit?.mentions}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label"><BarChart3 size={12} /> TWITCH VIEWERS</div>
                <div className="metric-value" style={{ fontSize: 24 }}>{(backendData.twitch?.viewers ?? 0).toLocaleString()}</div>
                <div className="metric-caption">Channels: {backendData.twitch?.channels}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label"><Globe size={12} /> TRẦN THỊ TRƯỜNG</div>
                <div className="metric-value" style={{ fontSize: 24 }}>{(backendData.scenarios?.market_ceiling ?? 0).toLocaleString()}</div>
                <div className="metric-caption">Top Category Benchmark</div>
              </div>
              <div className="metric-card">
                <div className="metric-label"><Search size={12} /> CATEGORY</div>
                <div className="metric-value" style={{ fontSize: 20 }}>{backendData.category}</div>
                <div className="metric-caption">Market Fit Identified</div>
              </div>
            </div>

            <div className="grid-3">
              <div className="panel">
                <div className="panel-header"><div className="panel-title">◈ MÔ PHỎNG THỊ TRƯỜNG</div></div>
                <div className="chart-area" style={{ padding: '20px 0' }}>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={allChartData}>
                      <defs>
                        <linearGradient id="cyanGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="var(--accent-cyan)" stopOpacity={0.2} />
                          <stop offset="95%" stopColor="var(--accent-cyan)" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="rgba(0, 242, 255, 0.05)" vertical={false} />
                      <XAxis dataKey="day" hide />
                      <YAxis hide domain={['auto', 'auto']} />
                      <Tooltip content={<CustomTooltip />} />
                      <Area type="monotone" dataKey="engagement" stroke="var(--accent-cyan)" strokeWidth={3} fill="url(#cyanGrad)" dot={false} />
                      <Area type="monotone" dataKey="forecast" stroke="var(--accent-white)" strokeWidth={2} strokeDasharray="5 5" fill="rgba(255, 255, 255, 0.02)" dot={false} />
                      <ReferenceLine x={histData[histData.length - 1]?.day} stroke="var(--accent-cyan)" strokeDasharray="3 3" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                
                <div className="panel-header" style={{borderTop: "1px solid rgba(0, 242, 255, 0.1)"}}>
                  <div className="panel-title" style={{ color: "#f87171" }}><AlertTriangle size={14} inline /> CẢNH BÁO THẤT BẠI (FAILURE PATTERNS)</div>
                </div>
                <div className="panel-body">
                  {(backendData.scenarios?.failure_warnings ?? []).map((w, i) => (
                    <div key={i} className="failure-item">
                       <span style={{ color: "#f87171" }}>[!]</span> {w}
                    </div>
                  ))}
                </div>
              </div>

              <div className="panel" style={{ background: "rgba(0, 242, 255, 0.02)" }}>
                <div className="panel-header"><div className="panel-title">◈ CHIẾN LƯỢC 3-CARD (DETAILED)</div></div>
                <div className="panel-body">
                  <div className="strat-card growth">
                    <div className="strat-header"><Zap size={14} /> ĐỘT PHÁ (GROWTH)</div>
                    <div className="strat-content">{backendData.scenarios?.scenarios?.[0]?.detail || "Tối ưu hóa phễu người chơi mới thông qua Content Creator."}</div>
                  </div>
                  <div className="strat-card stable">
                    <div className="strat-header"><Shield size={14} /> AN TOÀN (STABLE)</div>
                    <div className="strat-content">{backendData.scenarios?.scenarios?.[1]?.detail || "Duy trì Live-ops và tối ưu hóa tỷ lệ giữ chân (Retention)."}</div>
                  </div>
                  <div className="strat-card defense">
                    <div className="strat-header"><Target size={14} /> PHÒNG THỦ (RISK)</div>
                    <div className="strat-content">{backendData.scenarios?.scenarios?.[2]?.detail || "Kiểm soát chi phí vận hành và rà soát lỗi tối ưu hóa."}</div>
                  </div>
                </div>
                <div className="health-widget">
                  <div className="health-number">{backendData.health?.total ?? 0}</div>
                  <div className="health-label">PENTA-SCORE index</div>
                </div>
              </div>

              <div className="panel">
                <div className="panel-header"><div className="panel-title">◈ RED-TEAM SYNTHESIS (LLAMA-3)</div></div>
                <div className="panel-body">
                  <div className="ai-synthesis-content">
                    {backendData.synthesis}
                  </div>
                </div>
              </div>
            </div>
          </Fragment>
        )}
      </main>
    </div>
  );
}
