import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sentiment_features import load_daily_sentiment, build_sentiment_series

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PQ_DIR = os.path.join(DATA_DIR, "raw", "parquet")
DB_PATH = os.path.join(DATA_DIR, "news.db")
REPORT_PATH = os.path.join(DATA_DIR, "reports", "json", "backtest_report.json")
COMPARE_PATH = os.path.join(DATA_DIR, "reports", "json", "backtest_compare_sentiment.json")


@dataclass
class BacktestConfig:
    ticker: str = "VNM"
    context_len: int = 128
    pred_len: int = 20
    start_index: int = 180
    buy_threshold_pct: float = 3.5
    sell_threshold_pct: float = -3.5
    cost_bps: float = 35.0
    use_sentiment_filter: bool = True
    volume_filter_ratio: float = 0.5
    settlement_delay_days: int = 2


def run_backtest(cfg: BacktestConfig) -> dict:
    parquet_path = os.path.join(RAW_PQ_DIR, f"{cfg.ticker}_history.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Khong tim thay du lieu: {parquet_path}")

    df = pd.read_parquet(parquet_path, engine="pyarrow").sort_values("time").reset_index(drop=True)
    closes = df["close"].astype("float32").to_numpy()
    volumes = df["volume"].astype("float64").to_numpy() if "volume" in df.columns else np.ones(len(df), dtype="float64")
    vol_sma20 = pd.Series(volumes).rolling(20, min_periods=1).mean().to_numpy()
    dates = pd.to_datetime(df["time"], errors="coerce")

    sentiment_lookup = load_daily_sentiment(DB_PATH) if cfg.use_sentiment_filter else {}
    sentiment_series = build_sentiment_series(cfg.ticker, dates, sentiment_lookup)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map=device,
        dtype=torch.float32,
    )

    position = 0
    pending_target = 0
    pending_days = 0
    equity = 1.0
    equity_curve = []
    trades = []
    horizon_mae_list = []

    for i in range(cfg.start_index, len(closes) - cfg.pred_len - 1):
        ctx = closes[i - cfg.context_len:i]
        if len(ctx) != cfg.context_len:
            continue

        context = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            forecast = pipeline.predict(context, prediction_length=cfg.pred_len, num_samples=20)

        pred = np.median(forecast[0].numpy(), axis=0)
        actual_horizon = closes[i + 1 : i + 1 + cfg.pred_len]
        if len(actual_horizon) == cfg.pred_len:
            horizon_mae = float(np.mean(np.abs(pred - actual_horizon)))
            horizon_mae_list.append(horizon_mae)
        pred_return_pct = (float(pred[-1]) - float(closes[i])) / float(closes[i]) * 100.0

        signal = "HOLD"
        if pred_return_pct >= cfg.buy_threshold_pct:
            signal = "BUY"
        elif pred_return_pct <= cfg.sell_threshold_pct:
            signal = "SELL"

        # Volume filter for liquidity risk on VN market.
        if signal == "BUY" and volumes[i] < vol_sma20[i] * cfg.volume_filter_ratio:
            signal = "HOLD"

        # Sentiment filter: sentiment am thi han che mua, sentiment duong thi han che ban
        s = float(sentiment_series[i]) if i < len(sentiment_series) else 0.0
        if cfg.use_sentiment_filter:
            if signal == "BUY" and s < -0.15:
                signal = "HOLD"
            if signal == "SELL" and s > 0.15:
                signal = "HOLD"

        next_ret = (float(closes[i + 1]) - float(closes[i])) / float(closes[i])

        target_pos = position
        if signal == "BUY":
            target_pos = 1
        elif signal == "SELL":
            target_pos = -1

        # T+2 settlement simulation: apply position change after delay.
        if pending_days == 0 and target_pos != position:
            pending_target = target_pos
            pending_days = max(0, int(cfg.settlement_delay_days))

        if pending_days > 0:
            pending_days -= 1

        turnover = 0
        if pending_days == 0 and pending_target != position:
            turnover = abs(pending_target - position)
            position = pending_target

        tx_cost = (cfg.cost_bps / 10000.0) * turnover
        pnl = position * next_ret - tx_cost

        equity *= (1.0 + pnl)
        equity_curve.append(equity)
        trades.append(
            {
                "date": str(dates.iloc[i].date()) if pd.notna(dates.iloc[i]) else str(i),
                "price": float(closes[i]),
                "signal": signal,
                "position": int(position),
                "pred_return_pct": round(pred_return_pct, 4),
                "horizon_mae": round(float(horizon_mae_list[-1]), 6) if horizon_mae_list else None,
                "sentiment": round(s, 4),
                "next_ret": round(next_ret, 6),
                "equity": round(equity, 6),
            }
        )

    del pipeline
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    if not equity_curve:
        raise RuntimeError("Backtest khong tao duoc equity curve")

    equity_np = np.array(equity_curve, dtype="float64")
    returns = np.diff(equity_np, prepend=1.0) / np.clip(np.concatenate([[1.0], equity_np[:-1]]), 1e-9, None)
    win_rate = float(np.mean(returns > 0)) * 100.0
    max_dd = float(np.max(1.0 - equity_np / np.maximum.accumulate(equity_np))) * 100.0

    report = {
        "ticker": cfg.ticker,
        "config": cfg.__dict__,
        "n_steps": len(trades),
        "final_equity": round(float(equity_np[-1]), 6),
        "total_return_pct": round((float(equity_np[-1]) - 1.0) * 100.0, 2),
        "win_rate_pct": round(win_rate, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_horizon_mae": round(float(np.mean(horizon_mae_list)), 4) if horizon_mae_list else None,
        "trades": trades[-200:],
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def run_backtest_comparison(base_cfg: BacktestConfig | None = None) -> dict:
    """Run backtest before/after sentiment filter and save comparison report."""
    if base_cfg is None:
        base_cfg = BacktestConfig()

    no_sent_cfg = BacktestConfig(**{**base_cfg.__dict__, "use_sentiment_filter": False})
    with_sent_cfg = BacktestConfig(**{**base_cfg.__dict__, "use_sentiment_filter": True})

    report_no_sent = run_backtest(no_sent_cfg)
    report_with_sent = run_backtest(with_sent_cfg)

    compare = {
        "ticker": base_cfg.ticker,
        "without_sentiment": {
            "total_return_pct": report_no_sent["total_return_pct"],
            "win_rate_pct": report_no_sent["win_rate_pct"],
            "max_drawdown_pct": report_no_sent["max_drawdown_pct"],
            "avg_horizon_mae": report_no_sent["avg_horizon_mae"],
        },
        "with_sentiment": {
            "total_return_pct": report_with_sent["total_return_pct"],
            "win_rate_pct": report_with_sent["win_rate_pct"],
            "max_drawdown_pct": report_with_sent["max_drawdown_pct"],
            "avg_horizon_mae": report_with_sent["avg_horizon_mae"],
        },
    }

    if report_no_sent["avg_horizon_mae"] is not None and report_with_sent["avg_horizon_mae"] is not None:
        mae_before = float(report_no_sent["avg_horizon_mae"])
        mae_after = float(report_with_sent["avg_horizon_mae"])
        compare["delta"] = {
            "mae_change": round(mae_after - mae_before, 4),
            "mae_change_pct": round(((mae_after - mae_before) / mae_before * 100.0), 2) if mae_before else None,
            "return_change_pct": round(float(report_with_sent["total_return_pct"]) - float(report_no_sent["total_return_pct"]), 2),
            "winrate_change_pct": round(float(report_with_sent["win_rate_pct"]) - float(report_no_sent["win_rate_pct"]), 2),
            "maxdd_change_pct": round(float(report_with_sent["max_drawdown_pct"]) - float(report_no_sent["max_drawdown_pct"]), 2),
        }

    os.makedirs(os.path.dirname(COMPARE_PATH), exist_ok=True)
    with open(COMPARE_PATH, "w", encoding="utf-8") as f:
        json.dump(compare, f, indent=2, ensure_ascii=False)

    return compare


if __name__ == "__main__":
    cfg = BacktestConfig()
    r = run_backtest(cfg)
    print("=== BACKTEST REPORT ===")
    print(f"Ticker          : {r['ticker']}")
    print(f"Total return    : {r['total_return_pct']}%")
    print(f"Win rate        : {r['win_rate_pct']}%")
    print(f"Max drawdown    : {r['max_drawdown_pct']}%")
    print(f"Avg Horizon MAE : {r['avg_horizon_mae']}")
    print(f"Saved report    : {REPORT_PATH}")
