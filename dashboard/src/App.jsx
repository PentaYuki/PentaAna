import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Cell,
} from 'recharts';
import {
  TrendingUp, TrendingDown, Minus, Activity, Zap,
  Shield, Eye, Globe, RefreshCw, Star,
} from 'lucide-react';
import './App.css';

// ─── Constants ────────────────────────────────────────────────────────────────
const TICKERS = ['VNM','VCB','FPT','HPG','TCB','MWG','ACB','BID','GAS','MBB','MSN','PNJ','VHM'];

const SIG = {
  BUY:  { color: '#00FF88', glow: '0 0 35px rgba(0,255,136,0.45)', label: '▲ MUA' },
  SELL: { color: '#FF3860', glow: '0 0 35px rgba(255,56,96,0.45)',  label: '▼ BÁN' },
  HOLD: { color: '#FFD600', glow: '0 0 35px rgba(255,214,0,0.35)',  label: '■ GIỮ' },
};

const AGENT_COLORS = {
  technical: '#00E5FF',
  sentiment: '#00FF88',
  macro:     '#A78BFA',
  risk:      '#FF3860',
};

const PIPE_NODES = [
  { id: 'data',    label: 'DỮ LIỆU',    name: 'Price / Parquet' },
  { id: 'kronos',  label: 'PENTAANA',   name: 'LoRA Forecast'   },
  { id: 'agents',  label: 'MULTI-AGENT',name: 'Signal Engine'   },
  { id: 'rlhf',    label: 'RLHF',       name: 'Weight Adapt'    },
];

// ─── Helpers ─────────────────────────────────────────────────────────────────
const fmt = (v, d = 2) => v == null ? '—' : Number(v).toFixed(d);
const fmtPrice = v => v == null ? '—' : Number(v) >= 1000
  ? Number(v).toLocaleString('vi-VN')
  : Number(v).toFixed(2);
const fmtPct = (v, plus = false) =>
  v == null ? '—' : `${plus && v > 0 ? '+' : ''}${fmt(v)}%`;

// ─── Custom Tooltip ───────────────────────────────────────────────────────────
const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-title">{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.stroke || '#fff' }}>
          {p.name}: {Number(p.value).toLocaleString('vi-VN')}
        </div>
      ))}
    </div>
  );
};

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [ticker,      setTicker]      = useState('VNM');
  const [phase,       setPhase]       = useState('idle');   // idle|loading|done|error
  const [analysis,    setAnalysis]    = useState(null);
  const [priceData,   setPriceData]   = useState([]);
  const [weights,     setWeights]     = useState(null);
  const [signals,     setSignals]     = useState([]);
  const [trainStatus, setTrainStatus] = useState({ stage: 'idle', progress: 0 });
  const [activePipe,  setActivePipe]  = useState(null);
  const [clock,       setClock]       = useState(new Date());
  const [ratingMap,   setRatingMap]   = useState({});   // id→ pending rating
  const [ftForm,      setFtForm]      = useState({ epochs: 3, lr: 2e-4, batch_size: 4, context_len: 64, sentiment_alpha: 0.3 });
  const [ftRunning,   setFtRunning]   = useState(false);
  const [ftMsg,       setFtMsg]       = useState('');
  const [mlopsLog,    setMlopsLog]    = useState([]);
  const wsRef = useRef(null);

  // Live clock
  useEffect(() => {
    const t = setInterval(() => setClock(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  // WebSocket training status
  useEffect(() => {
    function connect() {
      const ws = new WebSocket(`ws://${location.host}/ws/status`);
      ws.onmessage = e => { try { setTrainStatus(JSON.parse(e.data)); } catch {} };
      ws.onclose   = () => setTimeout(connect, 2500);
      wsRef.current = ws;
    }
    connect();
    return () => wsRef.current?.close();
  }, []);

  // Fine-tune trigger
  const runFinetune = useCallback(async () => {
    setFtRunning(true);
    setFtMsg('Đang gửi lệnh fine-tune...');
    try {
      const res = await fetch('/api/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(ftForm),
      });
      const j = await res.json();
      setFtMsg(j.message || (j.ok ? '✓ Đã bắt đầu' : '✗ Lỗi'));
    } catch (e) {
      setFtMsg('✗ Không kết nối được backend');
    }
    setFtRunning(false);
  }, [ftForm]);

  // Load MLOps log
  const loadMlops = useCallback(async () => {
    try {
      const j = await fetch('/api/mlops/status').then(r => r.json());
      setMlopsLog(j.entries || []);
    } catch {}
  }, []);

  // Poll MLOps every 30s
  useEffect(() => {
    loadMlops();
    const t = setInterval(loadMlops, 30000);
    return () => clearInterval(t);
  }, [loadMlops]);

  // Main analysis pipeline
  async function runAnalysis() {
    if (phase === 'loading') return;
    setPhase('loading');
    setAnalysis(null);
    setPriceData([]);
    setWeights(null);
    setSignals([]);

    try {
      setActivePipe('data');
      const [priceRes, analysisRes] = await Promise.all([
        fetch(`/api/price/${ticker}`),
        fetch('/api/phase3/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ticker }),
        }),
      ]);

      const priceJson = await priceRes.json();
      setPriceData(priceJson.data || []);

      setActivePipe('kronos');
      const aJson = await analysisRes.json();
      if (aJson.ok) setAnalysis(aJson.result);

      setActivePipe('rlhf');
      const [wRes, sRes] = await Promise.all([
        fetch('/api/rlhf/weights').then(r => r.json()).catch(() => ({})),
        fetch('/api/rlhf/signals').then(r => r.json()).catch(() => ({})),
      ]);
      if (wRes.ok) setWeights(wRes.data?.weights);
      if (sRes.ok) setSignals(sRes.signals || []);

      setPhase('done');
    } catch (err) {
      console.error(err);
      setPhase('error');
    }
    setActivePipe(null);
  }

  // Rate a signal (1–5)
  async function rateSignal(signalDate, rating) {
    // find a signal by date to get its id
    const sig = signals.find(s => s.signal_date === signalDate);
    if (!sig) return;
    setRatingMap(m => ({ ...m, [signalDate]: rating }));
    await fetch(`/api/rlhf/rate/0`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rating }),
    }).catch(() => {});
  }

  // Derived chart data: history + forecast point
  const chartData = useMemo(() => {
    if (!priceData.length) return [];
    const hist = priceData.map(d => ({
      time: d.time?.slice(5),
      close: d.close,
    }));
    if (analysis?.forecast_return_pct != null && priceData.length) {
      const last = priceData[priceData.length - 1].close;
      hist.push({
        time: 'F+30',
        close: null,
        forecast: +(last * (1 + analysis.forecast_return_pct / 100)).toFixed(2),
      });
    }
    return hist;
  }, [priceData, analysis]);

  const weightsData = useMemo(() => {
    if (!weights) return [];
    return Object.entries(weights).map(([k, v]) => ({
      name: k.toUpperCase(),
      value: Math.round(v * 100),
      fill: AGENT_COLORS[k] || '#888',
    }));
  }, [weights]);

  const sig = analysis?.final_signal;
  const sigStyle = sig ? SIG[sig] : null;
  const isLoading = phase === 'loading';

  return (
    <div className="terminal">
      <div className="bg-grid" />

      {/* ── Header ── */}
      <header className="header">
        <div className="header-brand">
          <div className="header-icon">⬠</div>
          <div>
            <div className="header-title">PentaAna</div>
            <div className="header-subtitle">
              MULTI-AGENT STOCK INTELLIGENCE // VN-INDEX // PHASE 3
            </div>
          </div>
        </div>
        <div className="status-row">
          <div className="clock">
            {clock.toLocaleTimeString('vi-VN', { hour12: false })}
          </div>
          <div className="status-dot">
            <div className={`dot ${isLoading ? 'dot-pulse' : ''}`} />
            <span style={{ color: 'var(--accent-cyan)', fontSize: 10, letterSpacing: 2 }}>
              {isLoading ? 'ANALYZING...' : phase === 'error' ? 'ERROR' : 'READY'}
            </span>
          </div>
        </div>
      </header>

      <main className="main">
        {/* ── Ticker Search ── */}
        <div className="search-section">
          <span className="search-label">▸ CHỌN MÃ CHỨNG KHOÁN VÀ CHẠY PHÂN TÍCH MULTI-AGENT</span>
          <div className="search-row">
            <select
              className="search-input"
              value={ticker}
              onChange={e => setTicker(e.target.value)}
            >
              {TICKERS.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
            <button className="btn-primary" onClick={runAnalysis} disabled={isLoading}>
              {isLoading
                ? <><RefreshCw size={16} className="spin" /> ĐANG PHÂN TÍCH...</>
                : '▶ PHÂN TÍCH'}
            </button>
          </div>
        </div>

        {/* ── Pipeline ── */}
        <div className="pipeline">
          {PIPE_NODES.map(n => (
            <div key={n.id} className={`pipe-node ${activePipe === n.id ? 'active' : ''}`}>
              <div className="pipe-badge">{n.label}</div>
              <div className="pipe-name">{n.name}</div>
              {activePipe === n.id && (
                <div className="loading-bar" style={{ marginTop: 8 }}>
                  <div className="loading-fill" />
                </div>
              )}
            </div>
          ))}
        </div>

        {/* ── Results (visible after analysis) ── */}
        {analysis && (
          <>
            {/* ── Metrics Row ── */}
            <div className="metrics-row metrics-6">

              {/* Signal */}
              <div
                className="metric-card metric-signal"
                style={{
                  borderColor: sigStyle ? sigStyle.color + '55' : undefined,
                  boxShadow:   sigStyle?.glow,
                }}
              >
                <div className="metric-label">TÍN HIỆU</div>
                <div
                  className="metric-value"
                  style={{ color: sigStyle?.color || '#fff', fontSize: 36, letterSpacing: 2 }}
                >
                  {sigStyle?.label ?? sig}
                </div>
                <div className="metric-caption">
                  score: {fmt(analysis.final_score, 4)}
                </div>
              </div>

              {/* Price */}
              <div className="metric-card">
                <div className="metric-label">GIÁ ĐÓNG CỬA</div>
                <div className="metric-value" style={{ fontSize: 28 }}>
                  {fmtPrice(analysis.current_price)}
                </div>
                <div className="metric-caption">
                  dự báo: {fmtPct(analysis.forecast_return_pct, true)}
                </div>
              </div>

              {/* Confidence */}
              <div className="metric-card">
                <div className="metric-label"><Zap size={10} /> CONFIDENCE</div>
                <div className="metric-value">
                  {fmt(analysis.forecast_confidence * 100, 1)}%
                </div>
                <div className="metric-caption">độ tin cậy dự báo PentaAna</div>
              </div>

              {/* RSI */}
              <div className="metric-card">
                <div className="metric-label"><Activity size={10} /> RSI (14)</div>
                <div
                  className="metric-value"
                  style={{
                    color: analysis.rsi > 70 ? '#FF3860'
                         : analysis.rsi < 30 ? '#00FF88'
                         : '#fff',
                  }}
                >
                  {fmt(analysis.rsi, 1)}
                </div>
                <div className="metric-caption">
                  {analysis.rsi > 70 ? 'OVERBOUGHT ⚠' : analysis.rsi < 30 ? 'OVERSOLD ✓' : 'NEUTRAL'}
                </div>
              </div>

              {/* Macro */}
              <div className="metric-card">
                <div className="metric-label"><Globe size={10} /> MACRO</div>
                <div className="metric-value" style={{ fontSize: 24 }}>
                  {fmt(analysis.macro_score, 3)}
                </div>
                <div className="metric-caption">{analysis.macro_source || '—'}</div>
              </div>

              {/* Sentiment */}
              <div className="metric-card">
                <div className="metric-label"><Eye size={10} /> SENTIMENT</div>
                <div
                  className="metric-value"
                  style={{
                    fontSize: 24,
                    color: analysis.sentiment_score > 0 ? '#00FF88'
                         : analysis.sentiment_score < 0 ? '#FF3860'
                         : '#fff',
                  }}
                >
                  {fmtPct(analysis.sentiment_score, true)}
                </div>
                <div className="metric-caption">{analysis.sentiment_count} bài tin tức</div>
              </div>
            </div>

            {/* ── 3-Column Grid ── */}
            <div className="grid-3">

              {/* ── Left: Price Chart + Agent Votes ── */}
              <div className="panel">
                <div className="panel-header">
                  <div className="panel-title">◈ BIỂU ĐỒ GIÁ — {ticker}</div>
                </div>
                <div style={{ padding: '16px 8px 8px' }}>
                  {chartData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={240}>
                      <AreaChart data={chartData} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
                        <defs>
                          <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%"  stopColor="var(--accent-cyan)" stopOpacity={0.25} />
                            <stop offset="95%" stopColor="var(--accent-cyan)" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid stroke="rgba(0,229,255,0.04)" vertical={false} />
                        <XAxis
                          dataKey="time"
                          tick={{ fontSize: 9, fill: 'rgba(255,255,255,0.25)' }}
                          interval={14}
                          tickLine={false}
                          axisLine={false}
                        />
                        <YAxis
                          domain={['auto', 'auto']}
                          tick={{ fontSize: 9, fill: 'rgba(255,255,255,0.25)' }}
                          tickFormatter={v => v >= 1000 ? (v / 1000).toFixed(0) + 'k' : v}
                          width={40}
                          tickLine={false}
                          axisLine={false}
                        />
                        <Tooltip content={<ChartTooltip />} />
                        <Area
                          type="monotone"
                          dataKey="close"
                          name="Đóng cửa"
                          stroke="var(--accent-cyan)"
                          strokeWidth={2}
                          fill="url(#priceGrad)"
                          dot={false}
                          connectNulls={false}
                        />
                        <Area
                          type="monotone"
                          dataKey="forecast"
                          name="Dự báo"
                          stroke="#FFD600"
                          strokeWidth={2}
                          strokeDasharray="5 4"
                          fill="none"
                          dot={{ r: 5, fill: '#FFD600', strokeWidth: 0 }}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="empty-state">Không có dữ liệu giá</div>
                  )}
                </div>

                {/* Agent votes */}
                <div className="panel-header" style={{ borderTop: '1px solid rgba(0,229,255,0.06)' }}>
                  <div className="panel-title" style={{ fontSize: 13 }}>PHIẾU BẦU AGENT</div>
                </div>
                <div className="panel-body" style={{ paddingTop: 16 }}>
                  {analysis.agent_votes &&
                    Object.entries(analysis.agent_votes).map(([agent, vote]) => (
                      <div key={agent} className="agent-vote-row">
                        <span
                          className="agent-dot"
                          style={{ background: AGENT_COLORS[agent] || '#888' }}
                        />
                        <span className="agent-name">{agent.toUpperCase()}</span>
                        <span
                          className="agent-vote"
                          style={{ color: SIG[vote]?.color || '#fff' }}
                        >
                          {vote}
                        </span>
                        <span className="agent-score">
                          {fmt(analysis.agent_scores?.[agent], 3)}
                        </span>
                      </div>
                    ))}
                </div>
              </div>

              {/* ── Middle: Agent Weights + Technical Data ── */}
              <div className="panel">
                <div className="panel-header">
                  <div className="panel-title">◈ TRỌNG SỐ AGENT (RLHF)</div>
                </div>
                <div className="panel-body">
                  {weightsData.length > 0 ? (
                    <>
                      <ResponsiveContainer width="100%" height={170}>
                        <BarChart
                          data={weightsData}
                          layout="vertical"
                          margin={{ left: 10, right: 32, top: 4, bottom: 4 }}
                        >
                          <XAxis
                            type="number"
                            domain={[0, 65]}
                            tick={{ fontSize: 9, fill: 'rgba(255,255,255,0.3)' }}
                            tickFormatter={v => `${v}%`}
                            tickLine={false}
                            axisLine={false}
                          />
                          <YAxis
                            type="category"
                            dataKey="name"
                            tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.7)', fontFamily: 'monospace' }}
                            width={82}
                            tickLine={false}
                            axisLine={false}
                          />
                          <Tooltip formatter={v => `${v}%`} />
                          <Bar dataKey="value" radius={[0, 3, 3, 0]} barSize={16}>
                            {weightsData.map((e, i) => (
                              <Cell key={i} fill={e.fill} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                      <div className="weights-caption">
                        Cập nhật tự động mỗi RLHF cycle · MIN 5% · MAX 60%
                      </div>
                    </>
                  ) : (
                    <div className="empty-state" style={{ paddingTop: 30 }}>
                      Chưa có RLHF weights<br />
                      <span style={{ fontSize: 11 }}>Cần ≥10 tín hiệu có kết quả</span>
                    </div>
                  )}
                </div>

                {/* Technical data */}
                <div className="panel-header" style={{ borderTop: '1px solid rgba(0,229,255,0.06)' }}>
                  <div className="panel-title" style={{ fontSize: 13 }}>
                    {analysis.llm_analysis ? '◈ PHÂN TÍCH LLM (PHI-3)' : '◈ CHỈ SỐ KỸ THUẬT'}
                  </div>
                </div>
                <div className="panel-body">
                  {analysis.llm_analysis ? (
                    <div className="ai-synthesis-content">{analysis.llm_analysis}</div>
                  ) : (
                    <div className="kv-grid">
                      <div className="kv-row">
                        <span>MACD</span>
                        <span style={{ color: (analysis.macd ?? 0) > 0 ? '#00FF88' : '#FF3860' }}>
                          {fmt(analysis.macd, 4)}
                        </span>
                      </div>
                      <div className="kv-row">
                        <span>MACD Hist</span>
                        <span>{fmt(analysis.macd_hist, 4)}</span>
                      </div>
                      <div className="kv-row">
                        <span>BB Width</span>
                        <span>{fmtPct(analysis.bb_width_pct)}</span>
                      </div>
                      <div className="kv-row">
                        <span>ATR</span>
                        <span>{fmtPct(analysis.atr_pct)}</span>
                      </div>
                      <div className="kv-row">
                        <span>Risk Score</span>
                        <span>{fmt(analysis.risk_score, 3)}</span>
                      </div>
                      <div className="kv-row">
                        <span>Timestamp</span>
                        <span style={{ fontSize: 10 }}>{analysis.timestamp?.slice(0, 16)}</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* ── Right: RLHF Signals History ── */}
              <div className="panel">
                <div className="panel-header">
                  <div className="panel-title">◈ LỊCH SỬ TÍN HIỆU RLHF</div>
                </div>
                <div className="panel-body" style={{ padding: 0, overflowY: 'auto', maxHeight: 480 }}>
                  {signals.length > 0 ? (
                    <table className="signals-table">
                      <thead>
                        <tr>
                          <th>Ngày</th>
                          <th>TH</th>
                          <th>Conf</th>
                          <th>Thực tế</th>
                          <th>Reward</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[...signals].reverse().slice(0, 20).map((s, i) => (
                          <tr key={i}>
                            <td style={{ color: 'var(--text-dim)', fontSize: 10 }}>
                              {s.signal_date?.slice(5)}
                            </td>
                            <td style={{ color: SIG[s.signal]?.color || '#fff', fontWeight: 700 }}>
                              {s.signal}
                            </td>
                            <td>{((s.confidence ?? 0) * 100).toFixed(0)}%</td>
                            <td style={{
                              color: s.actual_return_pct > 0 ? '#00FF88'
                                   : s.actual_return_pct < 0 ? '#FF3860'
                                   : 'var(--text-dim)',
                            }}>
                              {s.actual_return_pct != null
                                ? fmtPct(s.actual_return_pct, true)
                                : '—'}
                            </td>
                            <td style={{
                              color: s.reward > 0 ? '#00FF88'
                                   : s.reward < 0 ? '#FF3860'
                                   : 'var(--text-dim)',
                            }}>
                              {s.reward != null ? fmt(s.reward, 3) : '—'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <div className="empty-state" style={{ padding: '50px 20px' }}>
                      Chưa có lịch sử tín hiệu.<br />
                      <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                        Tín hiệu sẽ xuất hiện sau khi chạy live.
                      </span>
                    </div>
                  )}
                </div>
              </div>

            </div>
          </>
        )}

        {/* ── Training Monitor ── */}
        <div className="train-panel">
          <div className="panel-header">
            <div className="panel-title" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              ⬠ TRAINING MONITOR — PentaAna LoRA
              <span style={{
                fontSize: 10,
                color: trainStatus.stage !== 'idle' ? '#00FF88' : 'var(--text-dim)',
                letterSpacing: 2,
              }}>
                {trainStatus.stage?.toUpperCase() || 'IDLE'}
              </span>
            </div>
          </div>
          <div className="train-body">
            <div className="train-meta">
              {trainStatus.loss  != null && <span>loss: <b>{Number(trainStatus.loss).toFixed(5)}</b></span>}
              {trainStatus.epoch != null && <span>epoch: <b>{trainStatus.epoch}</b></span>}
              {trainStatus.ticker      && <span>ticker: <b>{trainStatus.ticker}</b></span>}
              {trainStatus.progress > 0 && <span>progress: <b>{trainStatus.progress}%</b></span>}
            </div>
            <div className="loading-bar" style={{ height: 3, background: 'rgba(0,229,255,0.07)', marginTop: 10 }}>
              <div
                className="loading-fill"
                style={{
                  width: `${trainStatus.progress || 0}%`,
                  animation: trainStatus.stage === 'idle' ? 'none' : undefined,
                  background: trainStatus.stage === 'idle' ? 'rgba(0,229,255,0.2)' : undefined,
                }}
              />
            </div>
          </div>
        </div>

        {/* ── Fine-tune Controls ── */}
        <div className="train-panel" style={{ marginTop: 16 }}>
          <div className="panel-header">
            <div className="panel-title">⬠ FINE-TUNE HYPERPARAMETERS — PentaAna LoRA</div>
          </div>
          <div className="train-body">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 12, marginBottom: 14 }}>
              {[
                { key: 'epochs',          label: 'Epochs',          min: 1, max: 20,  step: 1,    type: 'int'   },
                { key: 'lr',              label: 'Learning Rate',   min: 1e-5, max: 1e-3, step: 1e-5, type: 'float' },
                { key: 'batch_size',      label: 'Batch Size',      min: 1, max: 32,  step: 1,    type: 'int'   },
                { key: 'context_len',     label: 'Context Length',  min: 16, max: 256, step: 16,  type: 'int'   },
                { key: 'sentiment_alpha', label: 'Sentiment Alpha', min: 0, max: 1,   step: 0.05, type: 'float' },
              ].map(({ key, label, min, max, step, type }) => (
                <label key={key} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  <span style={{ fontSize: 10, color: 'var(--text-dim)', letterSpacing: 1 }}>{label}</span>
                  <input
                    type="number"
                    min={min} max={max} step={step}
                    value={ftForm[key]}
                    onChange={e => {
                      const raw = e.target.value;
                      setFtForm(f => ({ ...f, [key]: type === 'int' ? parseInt(raw) : parseFloat(raw) }));
                    }}
                    style={{
                      background: 'rgba(0,229,255,0.06)',
                      border: '1px solid rgba(0,229,255,0.25)',
                      borderRadius: 4,
                      color: 'var(--accent-cyan)',
                      padding: '6px 10px',
                      fontSize: 13,
                      fontFamily: 'inherit',
                      outline: 'none',
                    }}
                  />
                </label>
              ))}
            </div>
            <button
              className="btn-primary"
              onClick={runFinetune}
              disabled={ftRunning || trainStatus.stage !== 'idle'}
              style={{ width: '100%' }}
            >
              {ftRunning ? <><RefreshCw size={14} className="spin" /> Đang gửi...</> : '▶ BẮT ĐẦU FINE-TUNE'}
            </button>
            {ftMsg && (
              <div style={{ marginTop: 8, fontSize: 11, color: ftMsg.startsWith('✓') ? '#00FF88' : '#FF3860', letterSpacing: 1 }}>
                {ftMsg}
              </div>
            )}
          </div>
        </div>

        {/* ── MLOps / Drift Panel ── */}
        <div className="train-panel" style={{ marginTop: 16 }}>
          <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div className="panel-title">⬠ MLOPS — PSI DRIFT &amp; AUTO-RETRAIN LOG</div>
            <button
              onClick={loadMlops}
              style={{ background: 'none', border: 'none', color: 'var(--accent-cyan)', cursor: 'pointer', fontSize: 11, letterSpacing: 1 }}
            >
              ↻ REFRESH
            </button>
          </div>
          <div className="train-body">
            {mlopsLog.length === 0 ? (
              <div className="empty-state" style={{ padding: '20px 0' }}>
                Chưa có log MLOps.<br />
                <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                  Log được tạo khi mlops_pipeline chạy PSI drift check.
                </span>
              </div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table className="signals-table" style={{ fontSize: 11 }}>
                  <thead>
                    <tr>
                      <th>Thời gian</th>
                      <th>Ticker</th>
                      <th>PSI</th>
                      <th>Sharpe</th>
                      <th>Hành động</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mlopsLog.map((e, i) => (
                      <tr key={i}>
                        <td style={{ color: 'var(--text-dim)' }}>{e.timestamp?.slice(0, 16) || '—'}</td>
                        <td style={{ color: 'var(--accent-cyan)' }}>{e.ticker || '—'}</td>
                        <td style={{ color: (e.psi ?? 0) > 0.2 ? '#FF3860' : '#00FF88' }}>
                          {e.psi != null ? Number(e.psi).toFixed(4) : '—'}
                        </td>
                        <td style={{ color: (e.sharpe ?? 1) < 0.3 ? '#FF3860' : '#00FF88' }}>
                          {e.sharpe != null ? Number(e.sharpe).toFixed(3) : '—'}
                        </td>
                        <td style={{ color: e.action === 'retrain' ? '#FFD600' : 'var(--text-dim)' }}>
                          {e.action?.toUpperCase() || 'MONITOR'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

      </main>
    </div>
  );
}
