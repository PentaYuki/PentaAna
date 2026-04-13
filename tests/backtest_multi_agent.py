"""
backtest_multi_agent.py — Backtest Multi-Agent Coordinator Strategy

Test the final_signal from the coordinator instead of just Kronos forecasts.
Compare results with backtest_basic.py to measure the impact of multi-agent.

This is Priority #2 in Phase 3 improvements.
"""

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phase3_multi_agent import run_multi_agent_analysis
from sentiment_features import load_daily_sentiment, build_sentiment_series
from kronos_test import load_and_prepare_data

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "news.db")
RAW_PQ_DIR = os.path.join(DATA_DIR, "raw", "parquet")
REPORT_PATH = os.path.join(DATA_DIR, "reports", "json", "backtest_multi_agent.json")
COMPARE_PATH = os.path.join(DATA_DIR, "reports", "json", "backtest_compare_agents.json")


@dataclass
class BacktestConfig:
    ticker: str = "VNM"
    context_len: int = 128
    pred_len: int = 20
    start_index: int = 180
    buy_threshold_score: float = 0.12  # coordinator final_score threshold
    sell_threshold_score: float = -0.12
    cost_bps: float = 35.0
    use_sentiment_filter: bool = True
    volume_filter_ratio: float = 0.5
    settlement_delay_days: int = 2


@dataclass
class Trade:
    index: int
    date: str
    signal: str
    price: float
    position_size: float
    equity_before: float
    equity_after: float
    cost: float


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def run_multi_agent_backtest(cfg: BacktestConfig) -> dict:
    """
    Backtest using multi-agent coordinator's final_signal.
    
    Returns: {
        'ticker': str,
        'config': dict,
        'results': {
            'total_trades': int,
            'winning_trades': int,
            'losing_trades': int,
            'win_rate_pct': float,
            'total_return_pct': float,
            'sharpe_ratio': float,
            'max_drawdown_pct': float,
        },
        'equity_curve': list,
        'trades': list,
        'signals_distribution': dict,
        'agent_performance': dict,
    }
    """
    
    # Load data
    parquet_path = os.path.join(RAW_PQ_DIR, f"{cfg.ticker}_history.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Data not found: {parquet_path}")

    df = pd.read_parquet(parquet_path, engine="pyarrow").sort_values("time").reset_index(drop=True)
    closes = df["close"].astype("float64").to_numpy()
    volumes = df["volume"].astype("float64").to_numpy() if "volume" in df.columns else np.ones(len(df), dtype="float64")
    vol_sma20 = pd.Series(volumes).rolling(20, min_periods=1).mean().to_numpy()
    dates = pd.to_datetime(df["time"], errors="coerce")

    # Guard for empty data
    if len(closes) == 0:
        return {
            "error": f"No data for {cfg.ticker}",
            "results": {"total_return_pct": 0, "win_rate_pct": 0, "sharpe_ratio": 0, "max_drawdown_pct": 0}
        }
    
    print(f"\n{'='*60}")
    print(f"  BACKTEST: Multi-Agent Coordinator")
    print(f"  Ticker: {cfg.ticker} | Data points: {len(closes)}")
    if len(dates) > 0:
        print(f"  Date range: {dates.iloc[0]} to {dates.iloc[-1]}")
    print(f"  Config: buy_threshold={cfg.buy_threshold_score}, sell_threshold={cfg.sell_threshold_score}")
    print(f"{'='*60}\n")

    position = 0  # 0: cash, 1: long position
    pending_signal = None
    pending_days = 0
    equity = 1.0
    equity_curve = [1.0]
    trades = []
    signals = []  # Track coordinator signals
    agent_votes_list = []  # Track individual agent votes
    
    agent_vote_counter = {"technical": {}, "sentiment": {}, "macro": {}, "risk": {}}
    
    # ═══════════════════════════════════════════════════════════════
    # Backtest Loop
    # ═══════════════════════════════════════════════════════════════
    
    prog_every = max(1, (len(closes) - cfg.start_index) // 10)
    
    for i in range(cfg.start_index, len(closes) - cfg.pred_len - 1):
        if (i - cfg.start_index) % prog_every == 0:
            progress_pct = 100 * (i - cfg.start_index) / (len(closes) - cfg.start_index - cfg.pred_len - 1)
            print(f"  Progress: {progress_pct:.0f}% ({i}/{len(closes)})")

        current_price = _safe_float(closes[i])
        current_date = str(dates[i].date()) if pd.notna(dates[i]) else f"index_{i}"

        try:
            # Get multi-agent analysis for this date
            analysis = run_multi_agent_analysis(cfg.ticker, as_of_index=i)
            
            final_signal = analysis.get("final_signal", "HOLD")
            final_score = _safe_float(analysis.get("final_score", 0.0))
            agent_votes = analysis.get("agent_votes", {})
            agent_scores = analysis.get("agent_scores", {})
            
            # Track signals
            signals.append({
                "index": i,
                "date": current_date,
                "signal": final_signal,
                "score": final_score,
                "price": current_price,
                "agent_votes": agent_votes,
            })
            
            # Count agent votes
            for agent_name, vote in agent_votes.items():
                if agent_name not in agent_vote_counter:
                    agent_vote_counter[agent_name] = {}
                agent_vote_counter[agent_name][vote] = agent_vote_counter[agent_name].get(vote, 0) + 1
            
        except Exception as e:
            # Fallback: if analysis fails, use HOLD
            print(f"  ⚠️  Analysis failed at index {i}: {str(e)}")
            final_signal = "HOLD"
            final_score = 0.0

        # Volume filter for liquidity-risk sessions.
        if final_signal == "BUY" and volumes[i] < vol_sma20[i] * cfg.volume_filter_ratio:
            final_signal = "HOLD"

        # T+2 delay before signal execution.
        exec_signal = "HOLD"
        if pending_signal is None and final_signal in {"BUY", "SELL"}:
            pending_signal = final_signal
            pending_days = max(0, int(cfg.settlement_delay_days))

        if pending_signal is not None:
            if pending_days > 0:
                pending_days -= 1
            if pending_days == 0:
                exec_signal = pending_signal
                pending_signal = None

        # ───────────────────────────────────────────────────────────
        # Trading Logic
        # ───────────────────────────────────────────────────────────
        
        action = "HOLD"
        
        if position == 0 and exec_signal == "BUY":
            # Open long position
            position = 1
            entry_price = current_price
            action = "OPEN_LONG"
            cost = cfg.cost_bps / 10000.0
            equity *= (1.0 - cost)
            
            trade = Trade(
                index=i,
                date=current_date,
                signal="BUY",
                price=current_price,
                position_size=1.0,
                equity_before=equity / (1.0 - cost),
                equity_after=equity,
                cost=cost,
            )
            trades.append(trade)
            
        elif position == 1 and exec_signal == "SELL":
            # Close long position
            position = 0
            exit_price = current_price
            
            # Calculate P&L
            ret = (exit_price - entry_price) / entry_price
            equity *= (1.0 + ret)
            
            cost = cfg.cost_bps / 10000.0
            equity *= (1.0 - cost)
            
            trade = Trade(
                index=i,
                date=current_date,
                signal="SELL",
                price=current_price,
                position_size=1.0,
                equity_before=1.0,  # Simplified
                equity_after=equity,
                cost=cost,
            )
            trades.append(trade)
            
        elif position == 1:
            # In position, update with daily mark-to-market
            mark_price = current_price
            ret = (mark_price - entry_price) / entry_price
            equity_mark = 1.0 * (1.0 + ret) * np.exp(-i / 252.0 * 0.02)  # Simplified
            
        equity_curve.append(equity)

    # ═══════════════════════════════════════════════════════════════
    # Calculate Metrics
    # ═══════════════════════════════════════════════════════════════
    
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    
    # Sharpe ratio (assuming 0 risk-free rate)
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
    
    # Max drawdown
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (np.array(equity_curve) - cummax) / cummax
    max_drawdown_pct = float(abs(np.min(drawdown) * 100)) if len(drawdown) > 0 else 0.0
    
    # Win rate
    winning_trades = sum(1 for t in trades if "SELL" in str(t.signal) and t.equity_after > t.equity_before)
    losing_trades = sum(1 for t in trades if "SELL" in str(t.signal) and t.equity_after < t.equity_before)
    win_rate = 100 * winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0.0
    
    total_return_pct = 100 * (equity_curve[-1] - 1.0) if len(equity_curve) > 0 else 0.0
    
    # Signal distribution
    signal_dist = {}
    for sig in signals:
        s = sig["signal"]
        signal_dist[s] = signal_dist.get(s, 0) + 1
    
    # Agent performance
    agent_perf = {}
    for agent_name, votes in agent_vote_counter.items():
        agent_perf[agent_name] = dict(votes)
    
    # ═══════════════════════════════════════════════════════════════
    # Report
    # ═══════════════════════════════════════════════════════════════
    
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": cfg.ticker,
        "test_type": "multi_agent_coordinator_backtest",
        "config": asdict(cfg),
        "data": {
            "total_bars": len(closes),
            "test_bars": len(closes) - cfg.start_index - cfg.pred_len - 1,
            "date_range": [str(dates.iloc[cfg.start_index].date()), str(dates.iloc[-cfg.pred_len].date())] if len(dates) > cfg.pred_len else ["N/A", "N/A"],
        },
        "results": {
            "total_trades": len([t for t in trades if t.signal in ["BUY", "SELL"]]),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate_pct": round(win_rate, 2),
            "total_return_pct": round(total_return_pct, 2),
            "annual_return_pct": round(total_return_pct * 252 / (len(closes) - cfg.start_index), 2),
            "sharpe_ratio": round(float(sharpe_ratio), 4),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "final_equity": round(equity_curve[-1], 4),
        },
        "signals": {
            "total_signals": len(signals),
            "distribution": signal_dist,
            "last_5_signals": [
                {
                    "date": s["date"],
                    "signal": s["signal"],
                    "score": round(s["score"], 4),
                    "price": round(s["price"], 1),
                }
                for s in signals[-5:]
            ],
        },
        "agent_votes": agent_perf,
        "equity_curve": [round(e, 6) for e in equity_curve],
        "recommendations": [
            f"Win rate: {win_rate:.1f}% - {'Good' if win_rate > 50 else 'Poor'}",
            f"Sharpe: {sharpe_ratio:.2f} - {'Strong' if sharpe_ratio > 1.0 else 'Weak'}",
            f"Max drawdown: {abs(max_drawdown_pct):.1f}% - {'Acceptable' if abs(max_drawdown_pct) < 15 else 'High'}",
            f"Total return: {total_return_pct:.1f}% - Compare with buy-and-hold baseline",
        ],
    }

    # Save report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Report saved: {REPORT_PATH}")

    # Print summary
    print(f"\n{'='*60}")
    print("  BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Total trades: {report['results']['total_trades']}")
    print(f"Win rate: {report['results']['win_rate_pct']}%")
    print(f"Total return: {report['results']['total_return_pct']}%")
    print(f"Sharpe ratio: {report['results']['sharpe_ratio']}")
    print(f"Max drawdown: {report['results']['max_drawdown_pct']}%")
    print(f"Final equity: {report['results']['final_equity']}")
    print(f"\nSignal distribution: {signal_dist}")
    print(f"Agent votes: {agent_perf}")
    print(f"{'='*60}\n")

    return report


if __name__ == "__main__":
    config = BacktestConfig(ticker="VNM")
    try:
        result = run_multi_agent_backtest(config)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    except Exception as e:
        print(f"Backtest failed: {str(e)}")
        import traceback
        traceback.print_exc()
