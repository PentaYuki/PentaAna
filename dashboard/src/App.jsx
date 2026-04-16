import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, Cell,
} from 'recharts';
import {
  TrendingUp, TrendingDown, Minus, Activity, Zap,
  Shield, Eye, Globe, RefreshCw, Star, Radio, AlertTriangle,
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
        <div key={i} style={{ color: p.stroke || p.fill || '#fff', fontSize: 11 }}>
          {p.name}: {Number(p.value).toLocaleString('vi-VN')}
        </div>
      ))}
    </div>
  );
};

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [ticker,      setTicker]      = useState('VNM');
  const [phase,       setPhase]       = useState('idle');
  const [analysis,    setAnalysis]    = useState(null);
  const [priceData,   setPriceData]   = useState([]);
  const [weights,     setWeights]     = useState(null);
  const [signals,     setSignals]     = useState([]);
  const [trainStatus, setTrainStatus] = useState({ stage: 'idle', progress: 0 });
  const [activePipe,  setActivePipe]  = useState(null);
  const [clock,       setClock]       = useState(new Date());
  const [ftForm,      setFtForm]      = useState({ epochs: 3, lr: 2e-4, batch_size: 4, context_len: 64, sentiment_alpha: 0.3 });
  const [ftRunning,   setFtRunning]   = useState(false);
  const [ftMsg,       setFtMsg]       = useState('');
  const [mlopsLog,    setMlopsLog]    = useState([]);
  const [rlhfSummary, setRlhfSummary] = useState(null);  // RLHF stats
  const [ratingState, setRatingState] = useState({});    // { [signalId]: { rating, pending } }

  // ── LIVE TRADING STATE (mới) ──────────────────────────────────────────────
  const [livePositions, setLivePositions] = useState([]);
  const [liveSession,   setLiveSession]   = useState(null);
  const [liveRunning,   setLiveRunning]   = useState(false);
  const [liveMsg,       setLiveMsg]       = useState('');
  const [stopScanMsg,   setStopScanMsg]   = useState('');
  const [backendOk,     setBackendOk]     = useState(null);  // null=chưa check, true=OK, false=offline

  // ── SIMULATION STATE ──────────────────────────────────────────────────────
  const [simForm, setSimForm] = useState({ initial: 6000000, target: 9000000, start: '2023-01-01', end: '2024-12-31' });
  const [simRunning, setSimRunning] = useState(false);
  const [simResult, setSimResult] = useState(null);
  const [simMsg, setSimMsg] = useState('');

  // ── DRL GYM STATE ─────────────────────────────────────────────────────────
  const [drlStatus, setDrlStatus] = useState(null);
  const [drlRunning, setDrlRunning] = useState(false);

  const wsRef = useRef(null);

  // Live clock
  useEffect(() => {
    const t = setInterval(() => setClock(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  // WebSocket training status + backend health check
  useEffect(() => {
    function connect() {
      const ws = new WebSocket(`ws://${location.host}/ws/status`);
      ws.onopen    = () => setBackendOk(true);
      ws.onmessage = e => { try { setTrainStatus(JSON.parse(e.data)); } catch {} };
      ws.onclose   = () => { setBackendOk(false); setTimeout(connect, 2500); };
      ws.onerror   = () => setBackendOk(false);
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
    } catch {
      setFtMsg('✗ Không kết nối được backend');
    }
    setFtRunning(false);
  }, [ftForm]);

  // Load MLOps log — parse cấu trúc đúng từ mlops_pipeline.py
  const loadMlops = useCallback(async () => {
    try {
      const j = await fetch('/api/mlops/status').then(r => r.json());
      setBackendOk(true);
      const flat = [];
      for (const entry of (j.entries || [])) {
        if (entry.event === 'scheduled_check' && Array.isArray(entry.results)) {
          for (const r of entry.results) {
            flat.push({
              timestamp: entry.checked_at,
              ticker:    r.ticker,
              psi:       r.psi,
              sharpe:    r.rolling_sharpe_30d,
              drift:     r.drift_detected,
              reason:    r.drift_reason,
              action:    r.drift_detected ? 'retrain' : 'monitor',
            });
          }
        } else if (entry.event === 'retrain_queued' || entry.event === 'retrain_queue_failed') {
          flat.push({
            timestamp: entry.triggered_at,
            ticker:    entry.ticker,
            psi:       entry.drift_info?.psi,
            sharpe:    entry.drift_info?.rolling_sharpe_30d,
            drift:     true,
            action:    entry.event === 'retrain_queued' ? 'queued' : 'queue_failed',
          });
        }
      }
      setMlopsLog(flat.reverse().slice(0, 30));
    } catch {
      setBackendOk(false);
    }
  }, []);

  // Load RLHF summary stats
  const loadRlhfSummary = useCallback(async () => {
    try {
      const j = await fetch('/api/rlhf/summary').then(r => r.json());
      if (j.ok) setRlhfSummary(j);
    } catch {}
  }, []);

  // Poll RLHF summary every 20s
  useEffect(() => {
    loadRlhfSummary();
    const t = setInterval(loadRlhfSummary, 20000);
    return () => clearInterval(t);
  }, [loadRlhfSummary]);

  // Poll MLOps every 30s
  useEffect(() => {
    loadMlops();
    const t = setInterval(loadMlops, 30000);
    return () => clearInterval(t);
  }, [loadMlops]);

  // ── Load Live Positions ──────────────────────────────────────────────────
  const loadLivePositions = useCallback(async () => {
    try {
      const j = await fetch('/api/live/positions').then(r => r.json());
      if (j.ok) setLivePositions(j.positions || []);
    } catch {}
  }, []);

  useEffect(() => {
    loadLivePositions();
    const t = setInterval(loadLivePositions, 15000);  // Refresh mỗi 15s
    return () => clearInterval(t);
  }, [loadLivePositions]);

  // ── Run Live Signal ───────────────────────────────────────────────────────
  const runLiveSignal = useCallback(async () => {
    setLiveRunning(true);
    setLiveMsg(`Đang chạy tín hiệu live cho ${ticker}...`);
    try {
      const res = await fetch(`/api/live/signal/${ticker}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ use_llm: false }),
      });
      const j = await res.json();
      if (j.ok) {
        setLiveSession(j.report);
        const action = j.report?.action || 'NONE';
        setLiveMsg(`✓ Hoàn thành — Hành động: ${action}`);
        loadLivePositions();
      } else {
        setLiveMsg(`✗ Lỗi: ${j.error || 'unknown'}`);
      }
    } catch {
      setLiveMsg('✗ Không kết nối được backend');
    }
    setLiveRunning(false);
  }, [ticker, loadLivePositions]);

  // ── Scan Stops ────────────────────────────────────────────────────────────
  const scanStops = useCallback(async () => {
    setStopScanMsg('Đang quét stop loss...');
    try {
      const res = await fetch('/api/live/scan_stops', { method: 'POST' });
      const j = await res.json();
      const n = j.stopped?.length || 0;
      setStopScanMsg(n > 0 ? `⚠ ${n} vị thế bị stop out` : '✓ Không có stop-out');
      loadLivePositions();
    } catch {
      setStopScanMsg('✗ Lỗi quét');
    }
  }, [loadLivePositions]);

  // ── Run Simulation ─────────────────────────────────────────────────────────
  const runSimulation = useCallback(async () => {
    setSimRunning(true);
    setSimMsg('Đang chạy backtest chiến lược...');
    setSimResult(null);
    try {
      const res = await fetch('/api/simulate_strategy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker,
          initial_capital: Number(simForm.initial),
          target_profit: Number(simForm.target) - Number(simForm.initial), // User inputs target NAV, we need profit
          start_date: simForm.start,
          end_date: simForm.end
        }),
      });
      const j = await res.json();
      if (j.ok) {
        setSimResult(j.result);
        setSimMsg('✓ Hoàn thành mô phỏng');
      } else {
        setSimMsg(`✗ Lỗi: ${j.error}`);
      }
    } catch {
      setSimMsg('✗ Không kết nối được backend');
    }
    setSimRunning(false);
  }, [ticker, simForm]);

  // ── DRL Training Logic ────────────────────────────────────────────────────
  const loadDrlStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/drl/status');
      const j = await res.json();
      setDrlStatus(j);
    } catch {}
  }, []);

  useEffect(() => {
    loadDrlStatus();
    const t = setInterval(loadDrlStatus, 5000);
    return () => clearInterval(t);
  }, [loadDrlStatus]);

  const startDrl = async () => {
    setDrlRunning(true);
    await fetch('/api/drl/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker, episodes: 100, initial_capital: 9000000 })
    });
    setDrlRunning(false);
    loadDrlStatus();
  };

  const stopDrl = async () => {
    await fetch('/api/drl/stop', { method: 'POST' });
    loadDrlStatus();
  };

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

      setActivePipe('agents');
      if (aJson.ok) setAnalysis(aJson.result);

      setActivePipe('rlhf');
      const [wRes, sRes, sumRes] = await Promise.all([
        fetch('/api/rlhf/weights').then(r => r.json()).catch(() => ({})),
        fetch('/api/rlhf/signals?limit=30&lookback_days=90').then(r => r.json()).catch(() => ({})),
        fetch('/api/rlhf/summary').then(r => r.json()).catch(() => ({})),
      ]);
      if (wRes.ok) setWeights(wRes.data?.weights);
      if (sRes.ok) setSignals(sRes.signals || []);
      if (sumRes.ok) setRlhfSummary(sumRes);

      setPhase('done');
    } catch (err) {
      console.error(err);
      setPhase('error');
    }
    setActivePipe(null);
  }

  // FIX: Rate signal — instant feedback + trigger weight update
  async function rateSignal(signalId, rating) {
    if (!signalId) return;
    // Optimistic UI update
    setRatingState(prev => ({ ...prev, [signalId]: { rating, pending: true } }));
    try {
      const res = await fetch(`/api/rlhf/rate/${signalId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rating }),
      });
      const j = await res.json();
      setRatingState(prev => ({
        ...prev,
        [signalId]: {
          rating,
          pending: false,
          ok: j.ok,
          weightsUpdated: j.weights_updated,
        },
      }));
      // Refresh weights in chart + summary
      if (j.ok && j.new_weights) setWeights(j.new_weights);
      loadRlhfSummary();
      // Reload signals so the table shows the newly computed reward and actual_return
      fetch('/api/rlhf/signals?limit=30&lookback_days=90')
        .then(r => r.json())
        .then(d => { if (d.ok) setSignals(d.signals || []); })
        .catch(()=>{});
    } catch {
      setRatingState(prev => ({ ...prev, [signalId]: { rating, pending: false, ok: false } }));
    }
  }

  // Derived chart data
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
              MULTI-AGENT STOCK INTELLIGENCE // VN-INDEX // PHASE 4
            </div>
          </div>
        </div>
        <div className="status-row">
          <div className="clock">
            {clock.toLocaleTimeString('vi-VN', { hour12: false })}
          </div>
          {/* Backend connection indicator */}
          <div style={{ display:'flex', alignItems:'center', gap:6, fontSize:10, letterSpacing:1 }}>
            <div style={{
              width:7, height:7, borderRadius:'50%',
              background: backendOk === null ? '#888' : backendOk ? '#00FF88' : '#FF3860',
              boxShadow: backendOk ? '0 0 6px #00FF88' : backendOk === false ? '0 0 6px #FF3860' : 'none',
              flexShrink: 0,
            }} />
            <span style={{ color: backendOk === null ? '#888' : backendOk ? '#00FF88' : '#FF3860' }}>
              {backendOk === null ? 'CONNECTING...' : backendOk ? 'BACKEND :8088' : 'BACKEND OFFLINE'}
            </span>
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
            <input
              type="text"
              className="search-input"
              value={ticker}
              onChange={e => setTicker(e.target.value.toUpperCase())}
              placeholder="VD: DGC, FPT..."
              style={{ width: 150 }}
            />
            <button className="btn-primary" onClick={runAnalysis} disabled={isLoading}>
              {isLoading
                ? <><RefreshCw size={16} className="spin" /> ĐANG PHÂN TÍCH...</>
                : '▶ PHÂN TÍCH'}
            </button>
            {/* FIX: Nút Live Signal tích hợp với live_broker.py */}
            <button
              className="btn-live"
              onClick={runLiveSignal}
              disabled={liveRunning}
              title="Chạy tín hiệu giao dịch thực (live_broker.py)"
            >
              {liveRunning
                ? <><RefreshCw size={14} className="spin" /> ĐANG CHẠY...</>
                : <><Radio size={14} /> LIVE SIGNAL</>}
            </button>
          </div>
          {liveMsg && (
            <div style={{
              marginTop: 8, fontSize: 11,
              color: liveMsg.startsWith('✓') ? '#00FF88' : liveMsg.startsWith('⚠') ? '#FFD600' : '#FF3860',
              letterSpacing: 1,
            }}>
              {liveMsg}
            </div>
          )}
        </div>

        {/* ── Simulation Section ── */}
        <div className="search-section" style={{ marginTop: 16 }}>
          <span className="search-label">▸ MÔ PHỎNG CHIẾN LƯỢC ĐẦU TƯ (FAST-MODE)</span>
          <div className="search-row" style={{ flexWrap: 'wrap', gap: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 10, color: 'var(--text-dim)' }}>VỐN BĐ (VND):</span>
              <input
                type="number"
                className="search-input"
                style={{ width: 120 }}
                value={simForm.initial}
                onChange={e => setSimForm({ ...simForm, initial: e.target.value })}
              />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 10, color: 'var(--text-dim)' }}>MỤC TIÊU (VND):</span>
              <input
                type="number"
                className="search-input"
                style={{ width: 120 }}
                value={simForm.target}
                onChange={e => setSimForm({ ...simForm, target: e.target.value })}
              />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 10, color: 'var(--text-dim)' }}>TỪ:</span>
              <input
                type="date"
                className="search-input"
                value={simForm.start}
                onChange={e => setSimForm({ ...simForm, start: e.target.value })}
              />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 10, color: 'var(--text-dim)' }}>ĐẾN:</span>
              <input
                type="date"
                className="search-input"
                value={simForm.end}
                onChange={e => setSimForm({ ...simForm, end: e.target.value })}
              />
            </div>
            <button className="btn-primary" onClick={runSimulation} disabled={simRunning}>
              {simRunning ? <><RefreshCw size={16} className="spin" /> ĐANG CHẠY...</> : '▶ CHẠY MÔ PHỎNG'}
            </button>
          </div>
          {simMsg && (
            <div style={{
              marginTop: 8, fontSize: 11,
              color: simMsg.startsWith('✓') ? '#00FF88' : simMsg.startsWith('⚠') ? '#FFD600' : '#FF3860',
              letterSpacing: 1,
            }}>
              {simMsg}
            </div>
          )}

          {/* Simulation Results Display */}
          {simResult && (
            <div style={{ marginTop: 16, borderTop: '1px solid rgba(0,229,255,0.1)', paddingTop: 16 }}>
              <div className="metrics-row metrics-6">
                <div className="metric-card">
                  <div className="metric-label">NAV CUỐI KỲ</div>
                  <div className="metric-value">{Number(simResult.final_nav).toLocaleString()}</div>
                  <div className="metric-caption">Lợi nhuận: {fmtPct(simResult.total_return_pct, true)}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">MỨC DRAWDOWN MAX</div>
                  <div className="metric-value" style={{ color: '#FF3860' }}>{fmt(simResult.max_drawdown_pct)}%</div>
                  <div className="metric-caption">Rủi ro trượt giá</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">TỔNG GIAO DỊCH</div>
                  <div className="metric-value">{simResult.total_trades}</div>
                  <div className="metric-caption">Số vòng quay vốn</div>
                </div>
                <div className="metric-card" style={{
                  borderColor: simResult.target_hit_date ? '#00FF8855' : '#FF386055' 
                }}>
                  <div className="metric-label">TRẠNG THÁI MỤC TIÊU</div>
                  <div className="metric-value" style={{ color: simResult.target_hit_date ? '#00FF88' : '#FF3860', fontSize: 20 }}>
                    {simResult.target_hit_date ? '✓ ĐẠT MỤC TIÊU' : '✗ KHÔNG ĐẠT'}
                  </div>
                  <div className="metric-caption">
                    {simResult.target_hit_date ? `Vào ngày: ${simResult.target_hit_date}` : 'Kết thúc kỳ vọng'}
                  </div>
                </div>
              </div>

              <div className="grid-2" style={{ marginTop: 16, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <div className="panel">
                  <div className="panel-header"><div className="panel-title">◈ MÔ PHỎNG NAV & DRAWDOWN</div></div>
                  <div className="panel-body">
                    <ResponsiveContainer width="100%" height={200}>
                      <AreaChart data={simResult.equity_curve} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
                        <CartesianGrid stroke="rgba(0,229,255,0.04)" vertical={false} />
                        <XAxis dataKey="date" tick={{ fontSize: 9, fill: 'var(--text-dim)' }} interval={Math.floor(simResult.equity_curve.length/5)} tickLine={false} axisLine={false} />
                        <YAxis yAxisId="left" domain={['auto', 'auto']} tick={{ fontSize: 9, fill: 'rgba(255,255,255,0.25)' }} tickFormatter={v => v >= 1000000 ? (v / 1000000).toFixed(1) + 'M' : v} width={35} tickLine={false} axisLine={false} />
                        <YAxis yAxisId="right" orientation="right" domain={[0, 100]} tick={{ fontSize: 9, fill: '#FF3860' }} tickFormatter={v => `${v}%`} width={35} tickLine={false} axisLine={false} />
                        <Tooltip content={<ChartTooltip />} />
                        <Area yAxisId="left" type="monotone" dataKey="nav" name="NAV" stroke="var(--accent-cyan)" fill="rgba(0,229,255,0.1)" strokeWidth={2} />
                        <Area yAxisId="left" type="monotone" dataKey={() => simResult.target_nav} name="Mục Tiêu" stroke="#FFD600" strokeDasharray="5 5" fill="none" />
                        <Area yAxisId="right" type="monotone" dataKey="drawdown" name="Drawdown" stroke="#FF3860" fill="rgba(255,56,96,0.2)" strokeWidth={1} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="panel">
                  <div className="panel-header"><div className="panel-title">◈ NHẬT KÝ GIAO DỊCH</div></div>
                  <div className="panel-body" style={{ padding: 0, overflowY: 'auto', maxHeight: 240 }}>
                    <table className="signals-table">
                      <thead>
                        <tr>
                          <th>Ngày</th>
                          <th>Lệnh</th>
                          <th>Giá</th>
                          <th>KL</th>
                          <th>Lời/Lỗ</th>
                          <th>NAV HT</th>
                        </tr>
                      </thead>
                      <tbody>
                        {simResult.trade_logs.map((t, i) => (
                          <tr key={i}>
                            <td style={{ color: 'var(--text-dim)', fontSize: 10 }}>{t.date.slice(5)}</td>
                            <td style={{ color: t.action.includes('BUY') ? '#00FF88' : '#FF3860', fontWeight: 700 }}>{t.action}</td>
                            <td>{Number(t.price).toLocaleString()}</td>
                            <td>{t.shares}</td>
                            <td style={{ color: t.profit > 0 ? '#00FF88' : t.profit < 0 ? '#FF3860' : 'var(--text-dim)' }}>
                              {t.profit != null ? fmtPct(t.profit_pct, true) : '—'}
                            </td>
                            <td>{Number(t.nav).toLocaleString()}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ── CƠ SỞ HUẤN LUYỆN ẢO (AI GYM) ── */}
        <div className="search-section" style={{ marginTop: 16, borderColor: '#be95ff' }}>
          <span className="search-label" style={{ color: '#be95ff' }}>▸ CƠ SỞ HUẤN LUYỆN ẢO (DEEP REINFORCEMENT LEARNING)</span>
          
          <div className="search-row" style={{ marginTop: 8, alignItems: 'center' }}>
             <button className="btn-primary" onClick={startDrl} disabled={drlRunning || drlStatus?.status?.includes('Đang')} style={{ background: '#be95ff', color: '#161616' }}>
                ▶ BẮT ĐẦU CÀY CUỐC
             </button>
             <button className="btn-live" onClick={stopDrl} style={{ background: '#FF3860' }}>
                NGỪNG HUẤN LUYỆN
             </button>
             <span style={{ fontSize: 13, color: 'var(--text-dim)', marginLeft: 16 }}>
                Trạng thái: <strong style={{ color: '#fff' }}>{drlStatus?.status || 'Chưa khởi động'}</strong> 
                {drlStatus?.details && <span style={{ marginLeft: 8, color: '#00FF88' }}>({drlStatus.details})</span>}
             </span>
          </div>

          {drlStatus && drlStatus.progress > 0 && (
            <div style={{ marginTop: 16 }}>
              <div className="loading-bar" style={{ height: 8, background: 'rgba(190,149,255,0.2)' }}>
                <div className="loading-fill" style={{ width: `${drlStatus.progress}%`, background: '#be95ff', transition: 'width 1s linear' }} />
              </div>
              <div style={{ textAlign: 'right', fontSize: 10, color: '#be95ff', marginTop: 4 }}>
                TIẾN ĐỘ: {drlStatus.progress.toFixed(1)}%
              </div>
            </div>
          )}
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

        {/* ── Results ── */}
        {analysis && (
          <>
            {/* ── Metrics Row ── */}
            <div className="metrics-row metrics-6">

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
                <div className="metric-caption">score: {fmt(analysis.final_score, 4)}</div>
              </div>

              <div className="metric-card">
                <div className="metric-label">GIÁ ĐÓNG CỬA</div>
                <div className="metric-value" style={{ fontSize: 28 }}>
                  {fmtPrice(analysis.current_price)}
                </div>
                <div className="metric-caption">dự báo: {fmtPct(analysis.forecast_return_pct, true)}</div>
              </div>

              <div className="metric-card">
                <div className="metric-label"><Zap size={10} /> CONFIDENCE</div>
                <div className="metric-value">
                  {fmt((analysis.forecast_confidence ?? 0) * 100, 1)}%
                </div>
                <div className="metric-caption">độ tin cậy dự báo PentaAna</div>
              </div>

              <div className="metric-card">
                <div className="metric-label"><Activity size={10} /> RSI (14)</div>
                <div
                  className="metric-value"
                  style={{
                    color: (analysis.rsi ?? 50) > 70 ? '#FF3860'
                         : (analysis.rsi ?? 50) < 30 ? '#00FF88'
                         : '#fff',
                  }}
                >
                  {fmt(analysis.rsi, 1)}
                </div>
                <div className="metric-caption">
                  {(analysis.rsi ?? 50) > 70 ? 'OVERBOUGHT ⚠' : (analysis.rsi ?? 50) < 30 ? 'OVERSOLD ✓' : 'NEUTRAL'}
                </div>
              </div>

              <div className="metric-card">
                <div className="metric-label"><Globe size={10} /> MACRO</div>
                <div className="metric-value" style={{ fontSize: 24 }}>
                  {fmt(analysis.macro_score, 3)}
                </div>
                <div className="metric-caption">{analysis.macro_source || '—'}</div>
              </div>

              <div className="metric-card">
                <div className="metric-label"><Eye size={10} /> SENTIMENT</div>
                <div
                  className="metric-value"
                  style={{
                    fontSize: 24,
                    color: (analysis.sentiment_score ?? 0) > 0 ? '#00FF88'
                         : (analysis.sentiment_score ?? 0) < 0 ? '#FF3860'
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

              {/* ── Middle: Agent Weights + Technical ── */}
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
                        <span>ATR %</span>
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

              {/* ── Right: RLHF Signal History ── */}
              <div className="panel">
                <div className="panel-header">
                  <div className="panel-title" style={{ display:'flex', alignItems:'center', gap:8 }}>
                    ◈ LỊCH SỬ TÍN HIỆU RLHF
                    {rlhfSummary?.stats?.pending_rating > 0 && (
                      <span style={{
                        background: '#FFD600', color: '#000',
                        fontSize: 9, padding: '2px 6px', fontWeight: 700, letterSpacing: 1,
                      }}>
                        {rlhfSummary.stats.pending_rating} CHƯA CHẤM
                      </span>
                    )}
                  </div>
                </div>
                {/* RLHF Stats Bar */}
                {rlhfSummary?.stats && (
                  <div style={{
                    display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)',
                    borderBottom: '1px solid rgba(0,229,255,0.08)',
                    padding: '8px 16px', gap: 6,
                  }}>
                    {[
                      { label: 'Tổng tín hiệu', value: rlhfSummary.stats.total_signals ?? '—', color: '#00E5FF' },
                      { label: 'Đã chấm', value: rlhfSummary.stats.rated_by_user ?? '—', color: '#00FF88' },
                      { label: 'Win Rate', value: rlhfSummary.stats.win_rate_pct != null ? `${rlhfSummary.stats.win_rate_pct}%` : '—', color: (rlhfSummary.stats.win_rate_pct ?? 0) >= 50 ? '#00FF88' : '#FF3860' },
                      { label: 'Avg Reward', value: rlhfSummary.stats.avg_reward != null ? rlhfSummary.stats.avg_reward.toFixed(3) : '—', color: (rlhfSummary.stats.avg_reward ?? 0) >= 0 ? '#00FF88' : '#FF3860' },
                    ].map(({ label, value, color }) => (
                      <div key={label} style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: 16, fontWeight: 700, color, fontFamily: 'monospace' }}>{value}</div>
                        <div style={{ fontSize: 9, color: 'var(--text-dim)', letterSpacing: 1, marginTop: 2 }}>{label}</div>
                      </div>
                    ))}
                  </div>
                )}
                <div className="panel-body" style={{ padding: 0, overflowY: 'auto', maxHeight: 420 }}>
                  {signals.length > 0 ? (
                    <table className="signals-table">
                      <thead>
                        <tr>
                          <th>Ngày</th>
                          <th>Mã</th>
                          <th>TH</th>
                          <th>Conf</th>
                          <th>Thực tế</th>
                          <th>Reward</th>
                          <th title="Nhấn sao để chấm điểm AI">★ Chấm</th>
                        </tr>
                      </thead>
                      <tbody>
                        {signals.map((s, i) => {
                          const rs = ratingState[s.id];
                          const currentRating = rs?.rating ?? s.user_rating ?? 0;
                          const isPending = rs?.pending;
                          const wasUpdated = rs?.ok && rs?.weightsUpdated;
                          return (
                          <tr key={s.id ?? i} style={{
                            background: wasUpdated ? 'rgba(0,255,136,0.04)' : undefined,
                            transition: 'background 0.5s',
                          }}>
                            <td style={{ color: 'var(--text-dim)', fontSize: 10 }}>
                              {s.signal_date?.slice(5)}
                            </td>
                            <td style={{ color: 'var(--accent-cyan)', fontSize: 10, fontWeight: 700 }}>
                              {s.ticker}
                            </td>
                            <td style={{ color: SIG[s.signal]?.color || '#fff', fontWeight: 700, lineHeight: 1.2 }}>
                              <div title="Tín hiệu phát ra">{s.signal}</div>
                              {s.agent_scores && Object.keys(s.agent_scores).length > 0 && (
                                <div style={{ fontSize: 9, color: 'var(--text-dim)', fontWeight: 400, marginTop: 4, whiteSpace: 'nowrap', letterSpacing: 0 }} title="Lý do quyết định (Agent Scores)">
                                  T:{Number(s.agent_scores.technical||0).toFixed(1)} S:{Number(s.agent_scores.sentiment||0).toFixed(1)}
                                </div>
                              )}
                            </td>
                            <td>{((s.confidence ?? 0) * 100).toFixed(0)}%</td>
                            <td style={{
                              color: s.actual_return_pct > 0 ? '#00FF88'
                                   : s.actual_return_pct < 0 ? '#FF3860'
                                   : 'var(--text-dim)',
                            }}>
                              {s.actual_return_pct != null ? fmtPct(s.actual_return_pct, true) : '—'}
                            </td>
                            <td style={{
                              color: s.reward > 0 ? '#00FF88' : s.reward < 0 ? '#FF3860' : 'var(--text-dim)',
                            }}>
                              {s.reward != null ? fmt(s.reward, 3) : '—'}
                            </td>
                            <td>
                              {isPending ? (
                                <span style={{ color: '#FFD600', fontSize: 10 }}>⏳</span>
                              ) : wasUpdated ? (
                                <span style={{ color: '#00FF88', fontSize: 10 }}>✓ {currentRating}★</span>
                              ) : (
                                <div style={{ display: 'flex', gap: 1 }}>
                                  {[1,2,3,4,5].map(star => (
                                    <span
                                      key={star}
                                      onClick={() => rateSignal(s.id, star)}
                                      title={[
                                        '', 'Sai nghiêm trọng', 'Không tốt', 'Trung tính', 'Tốt', 'Tuyệt vời'
                                      ][star]}
                                      style={{
                                        cursor: 'pointer',
                                        fontSize: 13,
                                        color: star <= currentRating ? '#FFD600' : 'rgba(255,214,0,0.25)',
                                        transition: 'color 0.15s, transform 0.1s',
                                        display: 'inline-block',
                                        lineHeight: 1,
                                      }}
                                      onMouseEnter={e => {
                                        e.target.style.transform = 'scale(1.4)';
                                        e.target.style.color = '#FFD600';
                                      }}
                                      onMouseLeave={e => {
                                        e.target.style.transform = 'scale(1)';
                                        e.target.style.color = star <= currentRating ? '#FFD600' : 'rgba(255,214,0,0.25)';
                                      }}
                                    >★</span>
                                  ))}
                                </div>
                              )}
                            </td>
                          </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  ) : (
                    <div className="empty-state" style={{ padding: '50px 20px' }}>
                      Chưa có lịch sử tín hiệu.<br />
                      <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                        Xuất hiện sau khi chạy live.
                      </span>
                    </div>
                  )}
                </div>

              </div>

            </div>
          </>
        )}

        {/* ═══════════════════════════════════════════════════════════════ */}
        {/* ── LIVE TRADING PANEL (mới — tích hợp live_broker.py) ────── */}
        {/* ═══════════════════════════════════════════════════════════════ */}
        <div className="train-panel" style={{ marginBottom: 16 }}>
          <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div className="panel-title" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <Radio size={13} style={{ color: '#00FF88' }} />
              LIVE TRADING — VỊ THẾ ĐANG MỞ
              {livePositions.length > 0 && (
                <span style={{
                  background: '#00FF88', color: '#000', fontSize: 9,
                  padding: '2px 6px', fontWeight: 700, letterSpacing: 1,
                }}>
                  {livePositions.length} MỞ
                </span>
              )}
            </div>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
              {stopScanMsg && (
                <span style={{ fontSize: 10, color: stopScanMsg.startsWith('✓') ? '#00FF88' : '#FFD600', letterSpacing: 1 }}>
                  {stopScanMsg}
                </span>
              )}
              <button
                onClick={scanStops}
                style={{ background: 'rgba(255,56,96,0.15)', border: '1px solid rgba(255,56,96,0.4)', color: '#FF3860', padding: '5px 12px', fontSize: 10, cursor: 'pointer', letterSpacing: 1, fontFamily: 'inherit' }}
              >
                <AlertTriangle size={10} style={{ marginRight: 4 }} />
                QUÉT STOP LOSS
              </button>
              <button
                onClick={loadLivePositions}
                style={{ background: 'none', border: 'none', color: 'var(--accent-cyan)', cursor: 'pointer', fontSize: 11, letterSpacing: 1 }}
              >
                ↻ REFRESH
              </button>
            </div>
          </div>
          <div className="train-body">
            {livePositions.length === 0 ? (
              <div className="empty-state" style={{ padding: '20px 0' }}>
                Không có vị thế đang mở.<br />
                <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                  Nhấn LIVE SIGNAL để thực hiện giao dịch khi có tín hiệu BUY.
                </span>
              </div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table className="signals-table">
                  <thead>
                    <tr>
                      <th>Mã CK</th>
                      <th>Số lượng</th>
                      <th>Giá mua</th>
                      <th>Stop Loss</th>
                      <th>Trailing</th>
                      <th>ATR %</th>
                      <th>Ngày mua</th>
                    </tr>
                  </thead>
                  <tbody>
                    {livePositions.map((p, i) => (
                      <tr key={i}>
                        <td style={{ color: 'var(--accent-cyan)', fontWeight: 700, fontSize: 13 }}>
                          {p.ticker}
                        </td>
                        <td>{p.quantity?.toLocaleString('vi-VN')}</td>
                        <td style={{ color: '#fff' }}>
                          {p.entry_price ? p.entry_price.toLocaleString('vi-VN') : '—'}
                        </td>
                        <td style={{ color: '#FF3860' }}>
                          {p.stop_loss ? p.stop_loss.toLocaleString('vi-VN') : '—'}
                        </td>
                        <td style={{ color: '#FFD600' }}>
                          {p.trailing_stop ? p.trailing_stop.toLocaleString('vi-VN') : '—'}
                        </td>
                        <td style={{ color: 'var(--text-dim)' }}>
                          {p.atr_pct ? `${p.atr_pct.toFixed(2)}%` : '—'}
                        </td>
                        <td style={{ color: 'var(--text-dim)', fontSize: 10 }}>
                          {p.entry_date}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* ── Last Live Session Report ── */}
            {liveSession && (
              <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid rgba(0,229,255,0.08)' }}>
                <div style={{ fontSize: 9, color: 'var(--accent-cyan)', letterSpacing: 2, marginBottom: 8 }}>
                  KẾT QUẢ PHIÊN CUỐI — {liveSession.ticker}
                </div>
                <div style={{ display: 'flex', gap: 28, flexWrap: 'wrap' }}>
                  <span style={{ fontSize: 11 }}>
                    Hành động: <b style={{ color: liveSession.action?.startsWith('BUY') ? '#00FF88' : liveSession.action?.startsWith('SELL') ? '#FF3860' : '#FFD600' }}>
                      {liveSession.action}
                    </b>
                  </span>
                  <span style={{ fontSize: 11 }}>
                    Tín hiệu: <b style={{ color: SIG[liveSession.signal]?.color || '#fff' }}>
                      {liveSession.signal}
                    </b>
                  </span>
                  {liveSession.confidence != null && (
                    <span style={{ fontSize: 11 }}>
                      Confidence: <b>{(liveSession.confidence * 100).toFixed(1)}%</b>
                    </span>
                  )}
                  {liveSession.unrealized_pct != null && (
                    <span style={{ fontSize: 11 }}>
                      Unrealized: <b style={{ color: liveSession.unrealized_pct >= 0 ? '#00FF88' : '#FF3860' }}>
                        {liveSession.unrealized_pct > 0 ? '+' : ''}{liveSession.unrealized_pct.toFixed(2)}%
                      </b>
                    </span>
                  )}
                  {liveSession.drift_info?.drift_detected && (
                    <span style={{ fontSize: 11, color: '#FFD600' }}>
                      ⚠ DRIFT: {liveSession.drift_info.drift_reason}
                    </span>
                  )}
                  {liveSession.error && (
                    <span style={{ fontSize: 11, color: '#FF3860' }}>
                      ✗ {liveSession.error}
                    </span>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

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
                { key: 'epochs',          label: 'Epochs',          min: 1,    max: 20,   step: 1,    type: 'int'   },
                { key: 'lr',              label: 'Learning Rate',   min: 1e-5, max: 1e-3, step: 1e-5, type: 'float' },
                { key: 'batch_size',      label: 'Batch Size',      min: 1,    max: 32,   step: 1,    type: 'int'   },
                { key: 'context_len',     label: 'Context Length',  min: 16,   max: 256,  step: 16,   type: 'int'   },
                { key: 'sentiment_alpha', label: 'Sentiment Alpha', min: 0,    max: 1,    step: 0.05, type: 'float' },
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

        {/* ── MLOps / Drift Panel (FIX: parse đúng cấu trúc) ── */}
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
                {backendOk === false ? (
                  <span style={{ color: '#FF3860' }}>
                    ⚠️ Backend :8088 offline<br />
                    <span style={{ fontSize:11, color:'var(--text-dim)' }}>Khởi động: <code>python src/web_dashboard.py</code></span>
                  </span>
                ) : (
                  <>
                    Chưa có log MLOps.<br />
                    <span style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                      Log được tạo khi mlops_pipeline chạy PSI drift check.
                    </span>
                  </>
                )}
              </div>
            ) : (
              <div style={{ overflowX: 'auto' }}>
                <table className="signals-table" style={{ fontSize: 11 }}>
                  <thead>
                    <tr>
                      <th>Thời gian</th>
                      <th>Ticker</th>
                      <th>PSI</th>
                      <th>Sharpe 30d</th>
                      <th>Drift</th>
                      <th>Hành động</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mlopsLog.map((e, i) => (
                      <tr key={i}>
                        <td style={{ color: 'var(--text-dim)', fontSize: 10 }}>
                          {e.timestamp?.slice(0, 16) || '—'}
                        </td>
                        <td style={{ color: 'var(--accent-cyan)' }}>{e.ticker || '—'}</td>
                        <td style={{ color: (e.psi ?? 0) > 0.2 ? '#FF3860' : '#00FF88' }}>
                          {e.psi != null ? Number(e.psi).toFixed(4) : '—'}
                        </td>
                        <td style={{ color: (e.sharpe ?? 1) < 0.3 ? '#FF3860' : '#00FF88' }}>
                          {e.sharpe != null ? Number(e.sharpe).toFixed(3) : '—'}
                        </td>
                        <td>
                          {e.drift
                            ? <span style={{ color: '#FFD600', fontSize: 10 }}>⚠ YES</span>
                            : <span style={{ color: '#00FF88', fontSize: 10 }}>✓ NO</span>
                          }
                        </td>
                        <td style={{
                          color: e.action === 'queued' ? '#FFD600'
                               : e.action === 'queue_failed' ? '#FF3860'
                               : 'var(--text-dim)',
                          fontSize: 10,
                          letterSpacing: 1,
                        }}>
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