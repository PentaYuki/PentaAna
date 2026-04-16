import json
import os
import sys
import asyncio
import threading
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import uvicorn

from kronos_trainer import finetune_kronos
from phase3_multi_agent import run_multi_agent_analysis
try:
  from phase3_checklist_test import run_phase3_checklist
except ImportError:
  def run_phase3_checklist(ticker: str = "VNM"):
    return {"error": "checklist not implemented", "ticker": ticker}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATUS_PATH = os.path.join(DATA_DIR, "reports", "json", "finetune_status.json")
METRICS_PATH = os.path.join(DATA_DIR, "reports", "json", "kronos_metrics.json")
BACKTEST_PATH = os.path.join(DATA_DIR, "reports", "json", "backtest_report.json")
TUNING_PATH = os.path.join(DATA_DIR, "reports", "json", "lora_tuning_results.json")
PHASE3_PATH = os.path.join(DATA_DIR, "reports", "json", "phase3_last_analysis.json")
PHASE3_TEST_PATH = os.path.join(DATA_DIR, "reports", "json", "phase3_test_report.json")
RLHF_WEIGHTS_PATH = os.path.join(DATA_DIR, "reports", "json", "rlhf_weights.json")
MLOPS_LOG_PATH = os.path.join(DATA_DIR, "reports", "json", "mlops_log.json")

try:
    from financial_data import get_financial_summary, score_fundamentals
    _FINANCIALS_ENABLED = True
except ImportError:
    _FINANCIALS_ENABLED = False
    def get_financial_summary(ticker): return {}
    def score_fundamentals(ticker): return {"fundamental_score": 0.0, "rating": "N/A", "signals": {}, "data": {}}


app = FastAPI(title="Stock-AI Dashboard", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== LIVE BROKER ENDPOINTS ====
from live_broker import create_engine, job_scan_stops
from fastapi.responses import JSONResponse

try:
    from goal_simulator import simulate_goal_oriented
except ImportError:
    simulate_goal_oriented = None

@app.post("/api/simulate_strategy")
def api_simulate_strategy(params: dict):
    """Giả lập chạy backtest chiến lược hướng mục tiêu (Fast-Mode)."""
    if simulate_goal_oriented is None:
        return JSONResponse({"ok": False, "error": "goal_simulator not found"}, status_code=500)
    
    ticker = str(params.get("ticker", "VNM")).upper()
    initial_capital = float(params.get("initial_capital", 6000000))
    target_profit = float(params.get("target_profit", 3000000))
    start_date = str(params.get("start_date", "2023-01-01"))
    end_date = str(params.get("end_date", "2024-12-31"))
    
    try:
        res = simulate_goal_oriented(
            ticker=ticker,
            initial_capital=initial_capital,
            target_profit=target_profit,
            start_date=start_date,
            end_date=end_date
        )
        return JSONResponse({"ok": True, "result": res})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

try:
    import drl_trainer
except ImportError:
    drl_trainer = None

@app.post("/api/drl/start")
def drl_start(params: dict):
    if drl_trainer is None:
        return JSONResponse({"ok": False, "error": "drl_trainer is not available"}, status_code=500)
    ticker = params.get("ticker", "VNM")
    episodes = int(params.get("episodes", 50))
    initial_capital = float(params.get("initial_capital", 9000000))
    res = drl_trainer.start_training(ticker=ticker, episodes=episodes, initial_capital=initial_capital)
    return JSONResponse(res, status_code=200 if res.get("ok") else 400)

@app.post("/api/drl/stop")
def drl_stop():
    if drl_trainer is None:
        return JSONResponse({"ok": False, "error": "drl_trainer is not available"})
    res = drl_trainer.stop_training()
    return JSONResponse(res)

@app.get("/api/drl/status")
def drl_status():
    if drl_trainer is None:
        return JSONResponse({"status": "Not Installed", "progress": 0.0, "details": "Cần pip install stable-baselines3"})
    return JSONResponse(drl_trainer.get_status())


@app.get("/api/live/positions")
def live_positions():
  """Danh sách vị thế đang mở."""
  from live_broker import PositionTracker
  tracker = PositionTracker()
  positions = tracker.get_all_open()
  return JSONResponse({
    "ok": True,
    "positions": [
      {
        "ticker":        p.ticker,
        "quantity":      p.quantity,
        "entry_price":   p.entry_price,
        "stop_loss":     p.stop_loss,
        "trailing_stop": p.trailing_stop,
        "entry_date":    p.entry_date,
      }
      for p in positions
    ]
  })

@app.post("/api/live/signal/{ticker}")
def live_signal(ticker: str, params: dict):
  """Chạy tín hiệu và giao dịch ngay lập tức cho 1 mã."""
  engine = create_engine()
  report = engine.run_signal(ticker.upper(), use_llm=bool(params.get("use_llm", False)))
  return JSONResponse({"ok": True, "report": report})

@app.post("/api/live/scan_stops")
def live_scan_stops():
  """Quét và thực hiện stop loss cho tất cả vị thế."""
  stopped = job_scan_stops()
  return JSONResponse({"ok": True, "stopped": stopped})

TRAIN_STATE: dict[str, Any] = {
    "running": False,
    "started_at": None,
    "last_error": None,
}


def _safe_read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _count_files(path: str, suffix: str) -> int:
    if not os.path.exists(path):
        return 0
    return len([f for f in os.listdir(path) if f.endswith(suffix)])


def _train_job(params: dict):
    TRAIN_STATE["running"] = True
    TRAIN_STATE["started_at"] = time.time()
    TRAIN_STATE["last_error"] = None
    try:
        finetune_kronos(
            epochs=int(params.get("epochs", 5)),
            context_len=int(params.get("context_len", 128)),
            batch_size=int(params.get("batch_size", 2)),
            learning_rate=float(params.get("learning_rate", 1e-4)),
            use_sentiment=bool(params.get("use_sentiment", True)),
            sentiment_alpha=float(params.get("sentiment_alpha", 0.15)),
            status_path=STATUS_PATH,
        )
    except Exception as e:
        TRAIN_STATE["last_error"] = str(e)
    finally:
        TRAIN_STATE["running"] = False


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(open(os.path.join(BASE_DIR, "dashboard", "index_fallback.html"), encoding="utf-8").read()
                        if os.path.exists(os.path.join(BASE_DIR, "dashboard", "index_fallback.html"))
                        else _DASHBOARD_HTML)


_DASHBOARD_HTML = """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock-AI Dashboard — Phân tích Tài chính</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0d1117; --card: #161b22; --border: #30363d;
      --ink: #e6edf3; --muted: #8b949e;
      --green: #3fb950; --red: #f85149; --yellow: #d29922;
      --blue: #58a6ff; --purple: #bc8cff; --accent: #1f6feb;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--ink); font-family: 'Inter', sans-serif; font-size: 14px; }
    .wrap { max-width: 1280px; margin: 0 auto; padding: 20px 16px; }
    h1 { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
    .subtitle { color: var(--muted); font-size: 13px; margin-bottom: 20px; }
    .search-bar { display: flex; gap: 8px; margin-bottom: 24px; }
    .search-bar input { flex: 1; background: var(--card); border: 1px solid var(--border);
      color: var(--ink); padding: 10px 16px; border-radius: 8px; font-size: 15px; outline: none; }
    .search-bar button { background: var(--accent); color: #fff; border: none;
      padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 14px; }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
    .grid4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
    @media(max-width:900px){ .grid2,.grid3,.grid4 { grid-template-columns: 1fr 1fr; } }
    @media(max-width:600px){ .grid2,.grid3,.grid4 { grid-template-columns: 1fr; } }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }
    .card-title { font-size: 11px; font-weight: 600; text-transform: uppercase;
      letter-spacing: .08em; color: var(--muted); margin-bottom: 14px; display: flex;
      align-items: center; gap: 6px; }
    .kv-row { display: flex; justify-content: space-between; align-items: center;
      padding: 7px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
    .kv-row:last-child { border-bottom: none; }
    .kv-label { color: var(--muted); }
    .kv-val { font-weight: 600; }
    .badge { padding: 3px 8px; border-radius: 20px; font-size: 11px; font-weight: 700; }
    .badge-green  { background: rgba(63,185,80,.15);  color: var(--green); }
    .badge-red    { background: rgba(248,81,73,.15);  color: var(--red); }
    .badge-yellow { background: rgba(210,153,34,.15); color: var(--yellow); }
    .badge-blue   { background: rgba(88,166,255,.15); color: var(--blue); }
    .signal-row { display: flex; align-items: center; gap: 10px; padding: 8px 0;
      border-bottom: 1px solid var(--border); }
    .signal-row:last-child { border-bottom: none; }
    .signal-label { flex: 1; color: var(--muted); font-size: 12px; }
    .signal-val { font-weight: 600; }
    .metric-big { font-size: 28px; font-weight: 700; line-height: 1.1; }
    .metric-unit { font-size: 13px; color: var(--muted); margin-top: 2px; }
    .section-title { font-size: 15px; font-weight: 700; margin: 24px 0 12px; display: flex;
      align-items: center; gap: 8px; }
    .pos-green { color: var(--green); }
    .pos-red   { color: var(--red); }
    .pos-muted { color: var(--muted); }
    .loading { color: var(--muted); font-size: 13px; padding: 20px; text-align: center; }
    .score-bar-wrap { height: 8px; background: var(--border); border-radius: 4px; overflow:hidden; margin-top: 8px; }
    .score-bar { height: 100%; border-radius: 4px; transition: width .5s ease; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { text-align: left; color: var(--muted); font-weight: 600; font-size: 11px;
      text-transform: uppercase; padding: 6px 8px; border-bottom: 1px solid var(--border); }
    td { padding: 8px 8px; border-bottom: 1px solid var(--border); }
    tr:last-child td { border-bottom: none; }
  </style>
</head>
<body>
<div class="wrap">
  <h1>📊 Stock-AI Dashboard</h1>
  <p class="subtitle">Hệ thống phân tích cổ phiếu AI — Tài chính cơ bản + Kỹ thuật + Đa tác tử</p>

  <div class="search-bar">
    <input id="tickerInput" value="VCB" placeholder="Nhập mã cổ phiếu (VD: VCB, FPT, HPG)" />
    <button onclick="analyze()">Phân tích AI ▶</button>
    <button onclick="loadFundamentals()" style="background:#1a3a5c">Cơ bản 📊</button>
  </div>

  <!-- Kỹ thuật & AI Signal -->
  <div id="signalSection" style="display:none">
    <div class="section-title">🤖 Kết quả Phân tích AI</div>
    <div class="grid4" id="signalCards"></div>
    <div class="grid2" style="margin-top:16px">
      <div class="card" id="agentVotesCard">
        <div class="card-title">🗳️ Đa Tác Tử — Phếu Bầu</div>
        <div id="agentVotes" class="loading">...</div>
      </div>
      <div class="card" id="explanationCard">
        <div class="card-title">💡 Lý giải / LLM</div>
        <div id="explanation" class="loading">...</div>
      </div>
    </div>
  </div>

  <!-- Tài chính cơ bản -->
  <div id="fundamentalSection" style="display:none">
    <div class="section-title">🏦 Chỉ Tiêu Tài Chính Cơ Bản</div>
    <div class="grid4" id="valuationCards"></div>
    <div class="grid2" style="margin-top:16px">
      <div class="card">
        <div class="card-title">📊 Hiệu quả Hoạt động (Ngân hàng)</div>
        <div id="bankMetrics"></div>
      </div>
      <div class="card">
        <div class="card-title">⚖️ Chất lượng Tài sản</div>
        <div id="assetQuality"></div>
      </div>
    </div>
    <div class="card" style="margin-top:16px">
      <div class="card-title">📈 Tăng trưởng YoY (So với cùng kỳ năm trước)</div>
      <div id="growthTable"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <div class="card-title">🤖 Đánh giá cơ bản AI</div>
      <div id="fundamentalAI"></div>
    </div>
  </div>

  <!-- Training & System -->
  <div class="section-title" style="margin-top:28px">⚙️ Hệ thống</div>
  <div class="grid3">
    <div class="card">
      <div class="card-title">🧠 Training Kronos</div>
      <div id="trainStatus" class="loading">Ang tải...</div>
      <div class="score-bar-wrap" style="margin-top:10px">
        <div id="trainBar" class="score-bar" style="background:var(--blue);width:0%"></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">📂 Dữ liệu hệ thống</div>
      <div id="snapshot" class="loading">...</div>
    </div>
    <div class="card">
      <div class="card-title">📝 MLOps Logs</div>
      <div id="mlops" class="loading">...</div>
    </div>
  </div>
</div>

<script>
const fmt = (v, dec=2, unit='') => v != null ? `${Number(v).toFixed(dec)}${unit}` : '—';
const fmtPct = (v) => v != null ? `${Number(v).toFixed(2)}%` : '—';
const fmtT = (v) => v != null ? `${(Number(v)/1e12).toFixed(1)}N` : '—';  // tỷ -> nghìn tỷ
const colorPct = (v) => v == null ? '' : v > 0 ? 'pos-green' : v < 0 ? 'pos-red' : '';
const badge = (v, good, bad) => {
  if (v == null) return '<span class="kv-val pos-muted">—</span>';
  const cls = v > good ? 'badge-green' : v < bad ? 'badge-red' : 'badge-yellow';
  return `<span class="badge ${cls}">${Number(v).toFixed(2)}%</span>`;
};

function ticker() { return document.getElementById('tickerInput').value.trim().toUpperCase() || 'VCB'; }

async function analyze() {
  const t = ticker();
  document.getElementById('signalSection').style.display = 'block';
  document.getElementById('signalCards').innerHTML = '<div class="loading">🔄 Đang phân tích AI...</div>';
  try {
    const r = await fetch('/api/phase3/analyze', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ticker: t})
    }).then(r=>r.json());
    if (!r.ok) { document.getElementById('signalCards').innerHTML = `<div class="loading">❌ ${r.error}</div>`; return; }
    renderSignal(r.result);
    loadFundamentals();
  } catch(e) {
    document.getElementById('signalCards').innerHTML = `<div class="loading">❌ ${e}</div>`;
  }
}

function renderSignal(d) {
  const sigColor = d.final_signal === 'BUY' ? 'pos-green' : d.final_signal === 'SELL' ? 'pos-red' : 'pos-muted';
  const sigEmoji = d.final_signal === 'BUY' ? '🟢' : d.final_signal === 'SELL' ? '🔴' : '🟡';
  document.getElementById('signalCards').innerHTML = `
    <div class="card"><div class="card-title">Tín hiệu AI</div>
      <div class="metric-big ${sigColor}">${sigEmoji} ${d.final_signal || '—'}</div>
      <div class="metric-unit">Score: ${fmt(d.final_score, 3)}</div></div>
    <div class="card"><div class="card-title">Giá hiện tại</div>
      <div class="metric-big">${d.current_price ? Number(d.current_price).toLocaleString() : '—'}</div>
      <div class="metric-unit">VND</div></div>
    <div class="card"><div class="card-title">Dự báo 30 phiên</div>
      <div class="metric-big ${colorPct(d.forecast_return_pct)}">${fmtPct(d.forecast_return_pct)}</div>
      <div class="metric-unit ">Return kỳ vọng</div></div>
    <div class="card"><div class="card-title">Mức độ tin cậy</div>
      <div class="metric-big">${d.forecast_confidence != null ? Math.round(d.forecast_confidence*100) : '—'}%</div>
      <div class="metric-unit">AI Confidence</div></div>
  `;
  // Agent votes
  const votes = d.agent_votes || {};
  const scores = d.agent_scores || {};
  const labels = {technical:'📊 Kỹ thuật', sentiment:'🗣️ Sentiment', macro:'🌐 Macro', risk:'⚖️ Rủi ro', fundamental:'🏦 Cơ bản'};
  let votesHTML = '';
  for (const [k,v] of Object.entries(votes)) {
    const sc = scores[k] != null ? `(${Number(scores[k]).toFixed(3)})` : '';
    const vCls = v==='BUY'?'pos-green':v==='SELL'?'pos-red':'pos-muted';
    votesHTML += `<div class="signal-row"><span class="signal-label">${labels[k]||k}</span><span class="signal-val ${vCls}">${v} ${sc}</span></div>`;
  }
  document.getElementById('agentVotes').innerHTML = votesHTML || '—';
  // Explanation / LLM
  let expHTML = '';
  if (d.rsi != null) expHTML += `<div class="signal-row"><span class="signal-label">RSI(14)</span><span class="signal-val">${fmt(d.rsi)}</span></div>`;
  if (d.macd != null) expHTML += `<div class="signal-row"><span class="signal-label">MACD</span><span class="signal-val ${d.macd>=0?'pos-green':'pos-red'}">${fmt(d.macd,4)}</span></div>`;
  if (d.atr_pct != null) expHTML += `<div class="signal-row"><span class="signal-label">ATR%</span><span class="signal-val">${fmtPct(d.atr_pct)}</span></div>`;
  if (d.nim != null) expHTML += `<div class="signal-row"><span class="signal-label">NIM</span><span class="signal-val">${fmtPct(d.nim)}</span></div>`;
  if (d.roe != null) expHTML += `<div class="signal-row"><span class="signal-label">ROE</span><span class="signal-val">${fmtPct(d.roe)}</span></div>`;
  if (d.npl_ratio != null) expHTML += `<div class="signal-row"><span class="signal-label">Tỷ lệ Nợ xấu</span><span class="signal-val ${d.npl_ratio<1?'pos-green':d.npl_ratio<2?'pos-muted':'pos-red'}">${fmtPct(d.npl_ratio)}</span></div>`;
  if (d.fundamental_rating) expHTML += `<div class="signal-row"><span class="signal-label">Cơ bản</span><span class="signal-val">${d.fundamental_rating}</span></div>`;
  if (d.llm_analysis) expHTML += `<p style="margin-top:8px;font-size:12px;color:var(--muted)">${d.llm_analysis}</p>`;
  document.getElementById('explanation').innerHTML = expHTML || d.explanation || '—';
}

async function loadFundamentals() {
  const t = ticker();
  document.getElementById('fundamentalSection').style.display = 'block';
  ['valuationCards','bankMetrics','assetQuality','growthTable','fundamentalAI'].forEach(id=>
    document.getElementById(id).innerHTML = '<div class="loading">🔄 Đang tải...</div>');
  try {
    const r = await fetch(`/api/financials/${t}`).then(r=>r.json());
    if (!r.ok) {
      document.getElementById('valuationCards').innerHTML = `<div class="card loading">❌ ${r.error||'Lỗi'}</div>`;
      return;
    }
    renderFundamentals(r.data, r.scoring);
  } catch(e) {
    document.getElementById('valuationCards').innerHTML = `<div class="card loading">❌ ${e}</div>`;
  }
}

function renderFundamentals(d, scoring) {
  if (!d) return;
  // Valuation cards
  document.getElementById('valuationCards').innerHTML = `
    <div class="card"><div class="card-title">P/E</div>
      <div class="metric-big">${d.pe != null ? Number(d.pe).toFixed(2) : '—'}</div>
      <div class="metric-unit">Giá/Thu nhập</div></div>
    <div class="card"><div class="card-title">P/B</div>
      <div class="metric-big">${d.pb != null ? Number(d.pb).toFixed(2) : '—'}</div>
      <div class="metric-unit">Giá/Sổ sách</div></div>
    <div class="card"><div class="card-title">EPS (VND)</div>
      <div class="metric-big">${d.eps != null ? Number(d.eps).toLocaleString() : '—'}</div>
      <div class="metric-unit">Lợi nhuận/CP</div></div>
    <div class="card"><div class="card-title">Vốn hóa</div>
      <div class="metric-big">${d.market_cap != null ? Number(d.market_cap).toLocaleString() : '—'}</div>
      <div class="metric-unit">Tỷ VND</div></div>
  `;
  // Bank metrics
  const kvRow = (label, val, cls='') =>
    `<div class="kv-row"><span class="kv-label">${label}</span><span class="kv-val ${cls}">${val}</span></div>`;
  document.getElementById('bankMetrics').innerHTML = [
    kvRow('NIM (%)', d.nim != null ? badge(d.nim, 3, 2) : '—'),
    kvRow('YEA - Tỷ suất TS sinh lãi (%)', fmtPct(d.yea)),
    kvRow('CoF - Chi phí sử dụng vốn (%)', fmtPct(d.cof)),
    kvRow('CIR - Tỷ lệ chi phí (%)', fmtPct(d.cir)),
    kvRow('ROE (%)', d.roe != null ? badge(d.roe, 15, 10) : '—'),
    kvRow('ROA (%)', d.roa != null ? badge(d.roa, 1.5, 0.8) : '—'),
    kvRow('Biên lợi nhuận ròng (%)', fmtPct(d.net_profit_margin)),
  ].join('');
  // Asset quality
  document.getElementById('assetQuality').innerHTML = [
    kvRow('Tỷ lệ CASA (%)', fmtPct(d.casa_ratio)),
    kvRow('Tỷ lệ Nợ xấu NPL (%)',
      d.npl_ratio != null ? badge(d.npl_ratio, d.npl_ratio<0.5?-1:d.npl_ratio<1?0:1, 2) : '—'),
    kvRow('Dự phòng/Nợ xấu (%)', fmtPct(d.coverage)),
    kvRow('LDR - Cho vay/Huy động (%)', fmtPct(d.ldr)),
    kvRow('Tổng TS / VCSH', d.leverage != null ? Number(d.leverage).toFixed(2) : '—'),
    kvRow('Số lượng CP lưu hành', d.shares_outstanding != null ? Number(d.shares_outstanding).toLocaleString() : '—'),
  ].join('');
  // Growth table
  const rows = [
    ['Thu nhập lãi thuần (NII)', d.yoy_nii],
    ['Lãi thuần dịch vụ (Fee)', d.yoy_fee],
    ['Lợi nhuận sau thuế', d.yoy_pat],
    ['Tổng tài sản', d.yoy_total_assets],
    ['Cho vay khách hàng', d.yoy_loans],
    ['Tiền gửi khách hàng', d.yoy_deposits],
  ];
  document.getElementById('growthTable').innerHTML =
    '<table><tr><th>Chỉ tiêu</th><th>Tăng trưởng YoY</th><th>Đánh giá</th></tr>' +
    rows.map(([lbl, v]) => {
      if (v == null) return `<tr><td>${lbl}</td><td class="pos-muted">—</td><td></td></tr>`;
      const cls = v > 0 ? 'pos-green' : 'pos-red';
      const rating = v > 15 ? '⬆️ Mạnh' : v > 5 ? '↗️ Tốt' : v > 0 ? '➡️ Ổn' : '⬇️ Giảm';
      return `<tr><td>${lbl}</td><td class="${cls}">${v > 0 ? '+' : ''}${v.toFixed(2)}%</td><td>${rating}</td></tr>`;
    }).join('') + '</table>';
  // AI scoring
  if (scoring) {
    const sc = scoring.fundamental_score || 0;
    const pct = Math.round((sc + 1) / 2 * 100);
    const barColor = sc > 0.2 ? 'var(--green)' : sc < -0.2 ? 'var(--red)' : 'var(--yellow)';
    let sigsHTML = Object.entries(scoring.signals || {}).map(([k,v])=>
      `<div class="signal-row"><span class="signal-label">${k}</span><span class="signal-val">${v}</span></div>`).join('');
    document.getElementById('fundamentalAI').innerHTML = `
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:10px">
        <div class="metric-big ${sc>0.2?'pos-green':sc<-0.2?'pos-red':'pos-muted'}">${(sc>0?'+':'')+sc.toFixed(3)}</div>
        <div><div style="font-weight:700">${scoring.rating}</div><div class="metric-unit">Score Fundamental</div></div>
      </div>
      <div class="score-bar-wrap">
        <div class="score-bar" style="background:${barColor};width:${pct}%"></div>
      </div>
      <div style="margin-top:12px">${sigsHTML}</div>
    `;
  }
}

async function refreshSystem() {
  try {
    const s = await fetch('/api/snapshot').then(r=>r.json());
    const ts = s.train_state || {};
    const dc = s.data_counts || {};
    document.getElementById('trainStatus').innerHTML =
      `<div class="kv-row"><span class="kv-label">Trạng thái</span><span class="kv-val ${ts.running?'pos-green':'pos-muted'}">${ts.running?'🔄 Đang chạy':'⏸ Nếp'}</span></div>`;
    document.getElementById('trainBar').style.width =
      (s.progress || 0) + '%';
    document.getElementById('snapshot').innerHTML =
      Object.entries(dc).map(([k,v])=>`<div class="kv-row"><span class="kv-label">${k.replace(/_/g,' ')}</span><span class="kv-val">${v}</span></div>`).join('');
  } catch(e) {}
  try {
    const ml = await fetch('/api/mlops/status').then(r=>r.json());
    const entries = ml.entries || [];
    document.getElementById('mlops').innerHTML =
      entries.length ? entries.slice(-5).reverse().map(e=>
        `<div class="kv-row"><span class="kv-label" style="font-size:11px">${(e.timestamp||'').slice(11,19)}</span><span class="signal-val" style="font-size:11px">${e.event||''}</span></div>
      `).join('') : '<span class="pos-muted">Chưa có log</span>';
  } catch(e) {}
}

// WebSocket training status
(function connectWS(){
  const ws = new WebSocket(`ws://${location.host}/ws/status`);
  ws.onmessage = (ev) => {
    try {
      const d = JSON.parse(ev.data);
      const p = d.progress || 0;
      document.getElementById('trainBar').style.width = `${p}%`;
    } catch(e) {}
  };
  ws.onclose = () => setTimeout(connectWS, 1500);
})();

setInterval(refreshSystem, 5000);
refreshSystem();
loadFundamentals();
</script>
</body>
</html>
"""


@app.get("/api/snapshot")
def snapshot():
    payload = {
        "train_state": TRAIN_STATE,
        "data_counts": {
            "raw_csv": _count_files(os.path.join(DATA_DIR, "raw", "csv"), ".csv"),
            "raw_parquet": _count_files(os.path.join(DATA_DIR, "raw", "parquet"), ".parquet"),
            "indicator_csv": _count_files(os.path.join(DATA_DIR, "analyzed", "indicators"), ".csv"),
            "charts_png": _count_files(os.path.join(DATA_DIR, "reports", "charts"), ".png"),
        },
    }
    return JSONResponse(payload)


@app.get("/api/metrics")
def metrics():
    return JSONResponse(
        {
            "kronos_metrics": _safe_read_json(METRICS_PATH),
            "backtest": _safe_read_json(BACKTEST_PATH),
            "tuning": _safe_read_json(TUNING_PATH),
            "phase3_last_analysis": _safe_read_json(PHASE3_PATH),
            "phase3_test": _safe_read_json(PHASE3_TEST_PATH),
        }
    )


@app.post("/api/train/start")
def start_train(params: dict):
    if TRAIN_STATE["running"]:
        return JSONResponse({"ok": False, "message": "Training dang chay"}, status_code=409)

    th = threading.Thread(target=_train_job, args=(params,), daemon=True)
    th.start()
    return JSONResponse({"ok": True, "message": "Da bat dau training", "params": params})


@app.post("/api/phase3/analyze")
def phase3_analyze(params: dict):
    ticker = str(params.get("ticker", "VNM")).upper()
    try:
        payload = run_multi_agent_analysis(ticker)
        return JSONResponse({"ok": True, "ticker": ticker, "result": payload})
    except Exception as e:
        return JSONResponse({"ok": False, "ticker": ticker, "error": str(e)}, status_code=500)


@app.post("/api/phase3/test")
def phase3_test(params: dict):
    ticker = str(params.get("ticker", "VNM")).upper()
    try:
        payload = run_phase3_checklist(ticker)
        return JSONResponse({"ok": True, "ticker": ticker, "result": payload})
    except Exception as e:
        return JSONResponse({"ok": False, "ticker": ticker, "error": str(e)}, status_code=500)


@app.post("/api/phase4/run")
def phase4_run(params: dict):
    """Chạy toàn bộ Phase 4 pipeline: macro → multi-agent → drift check → RLHF update."""
    ticker = str(params.get("ticker", "VNM")).upper()
    use_llm = bool(params.get("use_llm", False))
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from phase4_orchestrator import run_phase4
        report = run_phase4(ticker=ticker, use_llm=use_llm)
        return JSONResponse({"ok": True, "ticker": ticker, "report": report})
    except Exception as e:
        return JSONResponse({"ok": False, "ticker": ticker, "error": str(e)}, status_code=500)


@app.get("/api/rlhf/weights")
def rlhf_weights():
    """Trả về RLHF agent weights hiện tại từ rlhf_weights.json."""
    data = _safe_read_json(RLHF_WEIGHTS_PATH)
    if not data:
        return JSONResponse({"ok": False, "message": "rlhf_weights.json chưa tồn tại"}, status_code=404)
    return JSONResponse({"ok": True, "data": data})


@app.get("/api/mlops/status")
def mlops_status():
    """Trả về 10 log entries gần nhất từ mlops_log.json."""
    if not os.path.exists(MLOPS_LOG_PATH):
        return JSONResponse({"ok": False, "message": "mlops_log.json chưa tồn tại", "entries": []}, status_code=404)
    try:
        with open(MLOPS_LOG_PATH, encoding="utf-8") as f:
            entries = json.load(f)
        last_10 = entries[-10:] if len(entries) > 10 else entries
        return JSONResponse({"ok": True, "total": len(entries), "entries": last_10})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/rlhf/rate/{signal_id}")
def rlhf_rate_signal(signal_id: int, params: dict):
    """Nhận đánh giá từ người dùng (1-5 sao) cho một tín hiệu RLHF.
    Body JSON: {"rating": 4}  — 1=rất tệ, 3=trung tính, 5=rất tốt
    Sau khi rate, trigger cập nhật weights ngay lập tức (fast RLHF).
    """
    rating = float(params.get("rating", 3))
    if not (1.0 <= rating <= 5.0):
        return JSONResponse({"ok": False, "error": "rating phải trong khoảng 1-5"}, status_code=400)
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from rlhf_engine import FeedbackStore, WeightAdapter
        import sqlite3
        store = FeedbackStore()
        store.update_outcome(row_id=signal_id, actual_return_pct=0.0, user_rating=rating)
        # Fast weight update: lấy reward vừa tính và cập nhật weights ngay
        with sqlite3.connect(store.db_path) as conn:
            row = conn.execute(
                "SELECT ticker, signal, agent_scores, reward FROM rlhf_feedback WHERE id=?",
                (signal_id,)
            ).fetchone()
        if row:
            ticker, signal, agent_scores_json, reward = row
            if reward is not None:
                import json as _json
                agent_scores = _json.loads(agent_scores_json or "{}")
                adapter = WeightAdapter.load(ticker=ticker)
                adapter.update(reward=reward, agent_scores=agent_scores, signal=signal or "HOLD")
                adapter.save(ticker=ticker)
                # Cũng cập nhật global weights
                global_adapter = WeightAdapter.load()
                global_adapter.update(reward=reward, agent_scores=agent_scores, signal=signal or "HOLD")
                global_adapter.save()
                new_weights = adapter.weights
            else:
                new_weights = None
        else:
            new_weights = None
        return JSONResponse({
            "ok": True,
            "signal_id": signal_id,
            "rating": rating,
            "weights_updated": new_weights is not None,
            "new_weights": new_weights,
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/rlhf/signals")
def rlhf_signals_list(limit: int = 30, lookback_days: int = 90):
    """Trả về N tín hiệu gần nhất — bao gồm row ID để frontend dùng cho rating."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from rlhf_engine import FeedbackStore
        import sqlite3
        store = FeedbackStore()
        # Lấy trực tiếp từ DB để có row ID
        since = (pd.Timestamp.utcnow() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        with sqlite3.connect(store.db_path) as conn:
            rows = conn.execute("""
                SELECT id, ticker, signal_date, signal, forecast_return_pct,
                       confidence, actual_return_pct, user_rating, reward, agent_scores
                FROM rlhf_feedback
                WHERE signal_date >= ?
                ORDER BY signal_date DESC
                LIMIT ?
            """, (since, limit)).fetchall()
        import json
        signals = [
            {
                "id":                 r[0],
                "ticker":             r[1],
                "signal_date":        r[2],
                "signal":             r[3],
                "forecast_return_pct": r[4],
                "confidence":         r[5],
                "actual_return_pct":  r[6],
                "user_rating":        r[7],
                "reward":             r[8],
                "agent_scores":       json.loads(r[9]) if r[9] else {}
            }
            for r in rows
        ]
        return JSONResponse({"ok": True, "signals": signals, "total": len(signals)})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/rlhf/rate/batch")
def rlhf_rate_batch(params: dict):
    """Chấm điểm nhiều tín hiệu cùng lúc.
    Body JSON: {"ratings": [{"signal_id": 1, "rating": 4}, ...]}
    """
    ratings = params.get("ratings", [])
    if not ratings:
        return JSONResponse({"ok": False, "error": "ratings array is empty"}, status_code=400)
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from rlhf_engine import FeedbackStore
        store = FeedbackStore()
        results = []
        for item in ratings:
            signal_id = item.get("signal_id")
            rating = float(item.get("rating", 3))
            if signal_id is None:
                results.append({"signal_id": None, "ok": False, "error": "missing signal_id"})
                continue
            if not (1.0 <= rating <= 5.0):
                results.append({"signal_id": signal_id, "ok": False, "error": "rating 1-5"})
                continue
            store.update_outcome(row_id=int(signal_id), actual_return_pct=0.0, user_rating=rating)
            results.append({"signal_id": signal_id, "ok": True, "rating": rating})
        return JSONResponse({"ok": True, "count": len(results), "results": results})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/rlhf/summary")
def rlhf_summary():
    """Thống kê tổng quan RLHF: tổng tín hiệu, tỉ lệ thắng, avg reward, weights."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from rlhf_engine import FeedbackStore, WeightAdapter
        import sqlite3
        store = FeedbackStore()
        with sqlite3.connect(store.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM rlhf_feedback").fetchone()[0]
            rated = conn.execute("SELECT COUNT(*) FROM rlhf_feedback WHERE user_rating IS NOT NULL").fetchone()[0]
            resolved = conn.execute("SELECT COUNT(*) FROM rlhf_feedback WHERE reward IS NOT NULL").fetchone()[0]
            avg_reward = conn.execute(
                "SELECT AVG(reward) FROM rlhf_feedback WHERE reward IS NOT NULL"
            ).fetchone()[0]
            win_count = conn.execute(
                "SELECT COUNT(*) FROM rlhf_feedback WHERE reward > 0"
            ).fetchone()[0]
            # Tính win rate
            win_rate = (win_count / resolved * 100) if resolved > 0 else None
            # Đếm tín hiệu theo loại
            by_signal = {}
            rows = conn.execute(
                "SELECT signal, COUNT(*) FROM rlhf_feedback GROUP BY signal"
            ).fetchall()
            for s, c in rows:
                by_signal[s] = c
        # Global weights
        adapter = WeightAdapter.load()
        return JSONResponse({
            "ok": True,
            "stats": {
                "total_signals":    total,
                "rated_by_user":    rated,
                "resolved_signals": resolved,
                "pending_rating":   total - rated,
                "avg_reward":       round(avg_reward, 4) if avg_reward is not None else None,
                "win_rate_pct":     round(win_rate, 1) if win_rate is not None else None,
                "by_signal":        by_signal,
            },
            "weights": adapter.weights,
            "updated_at": pd.Timestamp.utcnow().isoformat(),
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/financials/{ticker}")
def financials(ticker: str, refresh: bool = False):
    """Trả về toàn bộ chỉ tiêu tài chính cơ bản + điểm AI cho một mã."""
    ticker = ticker.upper()
    try:
        summary = get_financial_summary(ticker)
        scoring = score_fundamentals(ticker)
        return JSONResponse({
            "ok": True,
            "ticker": ticker,
            "data": summary,
            "scoring": {
                "fundamental_score": scoring.get("fundamental_score"),
                "rating": scoring.get("rating"),
                "signals": scoring.get("signals", {}),
            },
        })
    except Exception as e:
        return JSONResponse({"ok": False, "ticker": ticker, "error": str(e)}, status_code=500)


@app.get("/api/price/{ticker}")

def price_history(ticker: str, limit: int = 90):
    """Trả về N phiên gần nhất (OHLCV) để vẽ chart."""
    ticker = ticker.upper()
    pq = os.path.join(DATA_DIR, "raw", "parquet", f"{ticker}_history.parquet")
    csv = os.path.join(DATA_DIR, "raw", "csv", f"{ticker}_history.csv")
    try:
        if os.path.exists(pq):
            df = pd.read_parquet(pq, engine="pyarrow")
        elif os.path.exists(csv):
            df = pd.read_csv(csv)
        else:
            return JSONResponse({"ticker": ticker, "data": []}, status_code=404)
        df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
        df = df.sort_values("time").tail(limit)
        cols = [c for c in ["time", "open", "high", "low", "close", "volume"] if c in df.columns]
        return JSONResponse({"ticker": ticker, "data": df[cols].to_dict(orient="records")})
    except Exception as e:
        return JSONResponse({"ticker": ticker, "data": [], "error": str(e)}, status_code=500)


@app.websocket("/ws/status")
async def ws_status(ws: WebSocket):
  await ws.accept()
  try:
    while True:
      payload = _safe_read_json(STATUS_PATH)
      if not payload:
        payload = {"stage": "idle", "progress": 0}
      await ws.send_json(payload)
      await asyncio.sleep(1)
  except WebSocketDisconnect:
    return
  except Exception:
    return


if __name__ == "__main__":
    uvicorn.run("web_dashboard:app", host="0.0.0.0", port=8088, reload=False)
