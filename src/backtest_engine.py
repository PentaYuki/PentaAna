"""
backtest_engine.py — Walk-forward backtest thực chiến cho Multi-Agent Coordinator.

Khác với backtest_comparison.py (dùng Kronos forecast thật):
  - Backtest theo từng phiên lịch sử, giả lập quyết định BUY/HOLD/SELL
  - Hold tối đa 30 phiên rồi cưỡng bán (giống quy trình đầu tư ngắn hạn)
  - Tính Sharpe, MaxDrawdown, WinRate cho từng bộ trọng số agent

Dùng trong:
  - coordinator_tuner.py (Optuna / Grid Search)
  - phase4_orchestrator.py (cross-validation)
"""

import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class BacktestEngineConfig:
    ticker: str          = "VNM"
    start_date: str      = "2022-01-01"   # inclusive
    end_date: str        = "2024-12-31"   # inclusive
    hold_days: int       = 30             # Bán sau tối đa N phiên
    cost_bps: float      = 35.0           # Chi phí giao dịch (mua + bán, bps)
    warmup_bars: int     = 20             # Phiên đầu cần indicators — bỏ qua
    min_confidence: float = 0.55          # Chỉ vào lệnh nếu confidence >= ngưỡng


@dataclass
class BacktestEngineResult:
    ticker: str
    start_date: str
    end_date: str
    n_trades: int        = 0
    win_rate: float      = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio: float  = 0.0
    max_drawdown_pct: float = 0.0
    avg_hold_days: float = 0.0
    results: dict        = field(default_factory=dict)   # alias cho coordinator_tuner

    def __post_init__(self):
        # coordinator_tuner đọc report["results"]["sharpe_ratio"]
        self.results = {
            "sharpe_ratio": self.sharpe_ratio,
            "total_return_pct": self.total_return_pct,
            "win_rate_pct": self.win_rate * 100,
            "max_drawdown_pct": self.max_drawdown_pct,
        }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_price_df(ticker: str) -> Optional[pd.DataFrame]:
    """Load OHLCV. Ưu tiên with_indicators (có RSI/MACD) rồi raw parquet/csv."""
    ind_path = os.path.join(DATA_DIR, "analyzed", "with_indicators", f"{ticker}_with_indicators.parquet")
    pq_path  = os.path.join(DATA_DIR, "raw", "parquet", f"{ticker}_history.parquet")
    csv_path = os.path.join(DATA_DIR, "raw", "csv", f"{ticker}_history.csv")
    for path in (ind_path, pq_path):
        if os.path.exists(path):
            df = pd.read_parquet(path, engine="pyarrow")
            df["time"] = pd.to_datetime(df["time"])
            return df.sort_values("time").reset_index(drop=True)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["time"] = pd.to_datetime(df["time"])
        return df.sort_values("time").reset_index(drop=True)
    return None


def _compute_sharpe(returns: list[float], periods_per_year: float = 252.0) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=float)
    std = float(np.std(arr))
    if std < 1e-9:
        return 0.0
    return float(np.mean(arr) / std * np.sqrt(periods_per_year))


def _compute_max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    eq = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-9)
    return float(np.min(dd) * 100)


# ─── Core engine ──────────────────────────────────────────────────────────────

def run_backtest(
    cfg: BacktestEngineConfig,
    weights: Optional[dict] = None,
) -> BacktestEngineResult:
    """
    Walk-forward backtest: mỗi phiên gọi run_multi_agent_analysis (mock hoặc thật),
    nếu BUY + confidence >= ngưỡng → mở vị thế, hold đến tín hiệu SELL hoặc tối đa hold_days.

    Args:
        cfg:     Cấu hình backtest
        weights: Dict {technical, sentiment, macro, risk} để override env vars.
                 None → dùng defaults / rlhf_weights.json.

    Returns:
        BacktestEngineResult với đầy đủ metrics
    """
    df = _load_price_df(cfg.ticker)
    if df is None:
        raise FileNotFoundError(f"Không có dữ liệu cho {cfg.ticker}")

    # Lọc theo ngày
    mask = (df["time"] >= cfg.start_date) & (df["time"] <= cfg.end_date)
    df = df[mask].reset_index(drop=True)
    if len(df) < cfg.warmup_bars + cfg.hold_days:
        raise ValueError(f"Không đủ dữ liệu ({len(df)} phiên) cho {cfg.ticker} [{cfg.start_date}→{cfg.end_date}]")

    # Set env vars nếu có weights
    _env_backup: dict = {}
    if weights:
        from coordinator_tuner import WeightSet
        ws = WeightSet(**weights) if isinstance(weights, dict) else weights
        _env_backup = {
            "PHASE3_W_TECH":  os.environ.get("PHASE3_W_TECH"),
            "PHASE3_W_SENT":  os.environ.get("PHASE3_W_SENT"),
            "PHASE3_W_MACRO": os.environ.get("PHASE3_W_MACRO"),
            "PHASE3_W_RISK":  os.environ.get("PHASE3_W_RISK"),
        }
        os.environ["PHASE3_W_TECH"]  = str(ws.technical)
        os.environ["PHASE3_W_SENT"]  = str(ws.sentiment)
        os.environ["PHASE3_W_MACRO"] = str(ws.macro)
        os.environ["PHASE3_W_RISK"]  = str(ws.risk)

    try:
        return _run_loop(df, cfg)
    finally:
        # Khôi phục env vars
        for k, v in _env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_loop(df: pd.DataFrame, cfg: BacktestEngineConfig) -> BacktestEngineResult:
    """Vòng lặp chính — không phụ thuộc env vars."""
    from phase3_multi_agent import run_multi_agent_analysis

    cost_one_way = cfg.cost_bps / 10_000 / 2   # mỗi chiều

    trades: list[dict]        = []
    equity_curve: list[float] = [1.0]
    trade_returns: list[float] = []

    in_position = False
    buy_index   = None
    buy_price   = None

    for i in range(cfg.warmup_bars, len(df) - 1):
        row       = df.iloc[i]
        as_of_str = str(row["time"])[:10]   # YYYY-MM-DD

        if in_position:
            held = i - buy_index
            current_price = float(row["close"])
            # Tín hiệu SELL hoặc đủ hold_days → đóng vị thế
            try:
                result = run_multi_agent_analysis(
                    cfg.ticker,
                    as_of_date=as_of_str,
                    use_llm=False,
                )
                should_sell = result["final_signal"] == "SELL" or held >= cfg.hold_days
            except Exception:
                should_sell = (held >= cfg.hold_days)

            if should_sell:
                raw_ret = (current_price - buy_price) / (buy_price + 1e-9)
                net_ret = raw_ret - cost_one_way * 2
                trade_returns.append(net_ret)
                equity_curve.append(equity_curve[-1] * (1 + net_ret))
                trades.append({
                    "buy_date":  str(df.iloc[buy_index]["time"])[:10],
                    "sell_date": as_of_str,
                    "hold_days": held,
                    "return_pct": round(net_ret * 100, 3),
                })
                in_position = False
        else:
            try:
                result = run_multi_agent_analysis(
                    cfg.ticker,
                    as_of_date=as_of_str,
                    use_llm=False,
                )
                confidence = float(result.get("forecast_confidence") or 0.0)
                if result["final_signal"] == "BUY" and confidence >= cfg.min_confidence:
                    in_position = True
                    buy_index   = i
                    buy_price   = float(row["close"]) * (1 + cost_one_way)
            except Exception:
                pass

    # Đóng vị thế cuối nếu còn
    if in_position and buy_index is not None:
        last_price = float(df.iloc[-1]["close"])
        raw_ret    = (last_price - buy_price) / (buy_price + 1e-9)
        net_ret    = raw_ret - cost_one_way
        trade_returns.append(net_ret)
        equity_curve.append(equity_curve[-1] * (1 + net_ret))

    n_trades   = len(trades)
    win_rate   = float(np.mean([1 if t["return_pct"] > 0 else 0 for t in trades])) if trades else 0.0
    total_ret  = (equity_curve[-1] - 1.0) * 100 if equity_curve else 0.0
    sharpe     = _compute_sharpe(trade_returns)
    max_dd     = _compute_max_drawdown(equity_curve)
    avg_hold   = float(np.mean([t["hold_days"] for t in trades])) if trades else 0.0

    return BacktestEngineResult(
        ticker=cfg.ticker,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        n_trades=n_trades,
        win_rate=win_rate,
        total_return_pct=round(total_ret, 3),
        sharpe_ratio=round(sharpe, 4),
        max_drawdown_pct=round(max_dd, 3),
        avg_hold_days=round(avg_hold, 1),
    )


# ─── CLI smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = BacktestEngineConfig(
        ticker="VNM",
        start_date="2023-01-01",
        end_date="2024-06-30",
    )
    print(f"[BacktestEngine] Chạy backtest {cfg.ticker} {cfg.start_date} → {cfg.end_date}...")
    result = run_backtest(cfg)
    print(f"\n  Số giao dịch : {result.n_trades}")
    print(f"  Win rate     : {result.win_rate*100:.1f}%")
    print(f"  Tổng lợi tức : {result.total_return_pct:+.2f}%")
    print(f"  Sharpe       : {result.sharpe_ratio:.3f}")
    print(f"  Max Drawdown : {result.max_drawdown_pct:.2f}%")
    print(f"  Giữ TB       : {result.avg_hold_days:.1f} phiên")
