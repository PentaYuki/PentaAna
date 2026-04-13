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

app = FastAPI(title="Stock-AI Dashboard", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return HTMLResponse("""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Stock-AI Realtime Dashboard</title>
  <style>
    :root { --bg:#f4f6f8; --card:#ffffff; --ink:#1b1f24; --accent:#0a7ea4; --ok:#0f9d58; --warn:#e67e22; }
    body { margin:0; font-family: 'Avenir Next', 'Helvetica Neue', sans-serif; background: linear-gradient(120deg,#f4f6f8,#e8edf2); color:var(--ink); }
    .wrap { max-width:1100px; margin:20px auto; padding:0 12px; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit,minmax(260px,1fr)); gap:12px; }
    .card { background:var(--card); border-radius:14px; padding:14px; box-shadow:0 8px 24px rgba(0,0,0,.08); }
    h1 { margin:0 0 12px; }
    .progress { width:100%; height:14px; background:#dfe6ec; border-radius:999px; overflow:hidden; }
    .bar { height:100%; width:0%; background:linear-gradient(90deg,var(--accent),#26b3d9); transition:width .4s ease; }
    .row { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }
    input { padding:8px; border:1px solid #c8d4df; border-radius:8px; width:120px; }
    button { border:none; background:var(--accent); color:#fff; padding:10px 12px; border-radius:10px; cursor:pointer; }
    .mono { font-family: Menlo, monospace; font-size:12px; white-space:pre-wrap; }
    .ok { color:var(--ok); font-weight:700; }
    .warn { color:var(--warn); font-weight:700; }
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>Stock-AI Fine-tune Realtime</h1>
    <div class=\"grid\">
      <div class=\"card\">
        <h3>Training Status</h3>
        <div id=\"stage\">stage: idle</div>
        <div class=\"progress\"><div id=\"bar\" class=\"bar\"></div></div>
        <div id=\"meta\" class=\"mono\"></div>
      </div>
      <div class=\"card\">
        <h3>Start Fine-tune</h3>
        <div class=\"row\">
          <input id=\"epochs\" value=\"5\" placeholder=\"epochs\" />
          <input id=\"ctx\" value=\"128\" placeholder=\"context\" />
          <input id=\"batch\" value=\"2\" placeholder=\"batch\" />
          <input id=\"lr\" value=\"0.0001\" placeholder=\"lr\" />
        </div>
        <div class=\"row\">
          <input id=\"alpha\" value=\"0.15\" placeholder=\"sent alpha\" />
          <button onclick=\"startTrain()\">Run</button>
        </div>
        <div id=\"runResp\" class=\"mono\"></div>
      </div>
      <div class=\"card\">
        <h3>Data Snapshot</h3>
        <div id=\"snapshot\" class=\"mono\"></div>
      </div>
    </div>

    <div class=\"card\" style=\"margin-top:12px\">
      <h3>Latest Metrics</h3>
      <div id=\"metrics\" class=\"mono\"></div>
    </div>
  </div>

<script>
let ws;
function connectWS(){
  ws = new WebSocket(`ws://${location.host}/ws/status`);
  ws.onmessage = (ev) => {
    const d = JSON.parse(ev.data || '{}');
    const p = d.progress || 0;
    document.getElementById('bar').style.width = `${p}%`;
    document.getElementById('stage').textContent = `stage: ${d.stage || 'idle'} | progress: ${p}%`;
    document.getElementById('meta').textContent = JSON.stringify(d, null, 2);
  };
  ws.onclose = () => setTimeout(connectWS, 1200);
}

async function refreshData(){
  const s = await fetch('/api/snapshot').then(r=>r.json());
  document.getElementById('snapshot').textContent = JSON.stringify(s, null, 2);
  const m = await fetch('/api/metrics').then(r=>r.json());
  document.getElementById('metrics').textContent = JSON.stringify(m, null, 2);
}

async function startTrain(){
  const payload = {
    epochs: Number(document.getElementById('epochs').value),
    context_len: Number(document.getElementById('ctx').value),
    batch_size: Number(document.getElementById('batch').value),
    learning_rate: Number(document.getElementById('lr').value),
    sentiment_alpha: Number(document.getElementById('alpha').value),
    use_sentiment: true
  };
  const resp = await fetch('/api/train/start', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(payload)
  }).then(r=>r.json());
  document.getElementById('runResp').textContent = JSON.stringify(resp, null, 2);
  refreshData();
}

connectWS();
setInterval(refreshData, 3000);
refreshData();
</script>
</body>
</html>
    """)


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
    """
    rating = float(params.get("rating", 3))
    if not (1.0 <= rating <= 5.0):
        return JSONResponse({"ok": False, "error": "rating phải trong khoảng 1-5"}, status_code=400)
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from rlhf_engine import FeedbackStore
        store = FeedbackStore()
        store.update_outcome(row_id=signal_id, actual_return_pct=0.0, user_rating=rating)
        return JSONResponse({"ok": True, "signal_id": signal_id, "rating": rating})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/rlhf/signals")
def rlhf_signals_list():
    """Trả về 20 tín hiệu gần nhất chờ đánh giá từ người dùng."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from rlhf_engine import FeedbackStore
        store = FeedbackStore()
        pending = store.get_recent_rewards(ticker="ALL", lookback_days=60)
        return JSONResponse({"ok": True, "signals": pending[-20:]})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


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
