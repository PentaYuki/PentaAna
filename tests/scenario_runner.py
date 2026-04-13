"""
scenario_runner.py — Chạy backtest trên 4 kịch bản thị trường cụ thể.

Kịch bản 1: Bull market   (VNM 2021-01 → 2022-01)
Kịch bản 2: Bear market   (VNM 2022-01 → 2023-01)
Kịch bản 3: Sideways      (2023-01 → 2024-01)
Kịch bản 4: Recent data   (2024-01 → 2025-06)

Output:
  data/reports/json/scenario_results.json    — dữ liệu đầy đủ
  (stdout)                                    — bảng tổng hợp dạng text
"""

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PQ_DIR = os.path.join(DATA_DIR, "raw", "parquet")
SCENARIO_REPORT_PATH = os.path.join(DATA_DIR, "reports", "json", "scenario_results.json")

os.makedirs(os.path.dirname(SCENARIO_REPORT_PATH), exist_ok=True)


# ─── Scenario definition ───────────────────────────────────────────────────────

@dataclass
class Scenario:
    name: str
    description: str
    ticker: str
    start_date: str   # YYYY-MM-DD inclusive
    end_date: str     # YYYY-MM-DD inclusive
    expected_regime: str  # "bull" | "bear" | "sideways" | "mixed"


SCENARIOS = [
    Scenario(
        name="S1_Bull_2021",
        description="Bull market — VNM 2021 recovery",
        ticker="VNM",
        start_date="2021-01-01",
        end_date="2022-01-01",
        expected_regime="bull",
    ),
    Scenario(
        name="S2_Bear_2022",
        description="Bear market — VNM 2022 downturn",
        ticker="VNM",
        start_date="2022-01-01",
        end_date="2023-01-01",
        expected_regime="bear",
    ),
    Scenario(
        name="S3_Sideways_2023",
        description="Sideways / volatile — VNM 2023",
        ticker="VNM",
        start_date="2023-01-01",
        end_date="2024-01-01",
        expected_regime="sideways",
    ),
    Scenario(
        name="S4_Recent_2024",
        description="Recent data — VNM 2024-2025",
        ticker="VNM",
        start_date="2024-01-01",
        end_date="2025-07-01",
        expected_regime="mixed",
    ),
]


# ─── Backtest runner ───────────────────────────────────────────────────────────

def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _compute_metrics(equity_curve: list[float], returns: list[float]) -> dict:
    """Tính Sharpe, max drawdown, total return từ equity curve."""
    if not equity_curve or len(equity_curve) < 2:
        return {
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
        }

    eq = np.array(equity_curve, dtype="float64")
    total_return = float((eq[-1] - eq[0]) / eq[0] * 100.0)

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    drawdowns = (eq - peak) / np.maximum(peak, 1e-9) * 100.0
    max_dd = float(np.min(drawdowns))

    # Sharpe (annualized, 252 trading days)
    r = np.array(returns, dtype="float64")
    if len(r) > 1 and np.std(r) > 1e-9:
        sharpe = float(np.mean(r) / np.std(r) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Win rate from trade returns (returns > 0)
    wins = int(np.sum(r > 0))
    total_trades = int(len(r))
    win_rate = float(wins / total_trades * 100) if total_trades > 0 else 0.0

    return {
        "total_return_pct": round(total_return, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "win_rate_pct": round(win_rate, 4),
    }


def run_scenario_backtest(scenario: Scenario, verbose: bool = True) -> dict:
    """
    Simplified backtest cho một kịch bản: dùng Kronos forecast signal.
    Mục đích so sánh hiệu suất qua các regime khác nhau, không cần full multi-agent.
    """
    from phase3_multi_agent import run_multi_agent_analysis
    from kronos_test import load_and_prepare_data

    parquet_path = os.path.join(RAW_PQ_DIR, f"{scenario.ticker}_history.parquet")
    if not os.path.exists(parquet_path):
        return {
            "scenario": scenario.name,
            "status": "error",
            "error": f"Data not found: {parquet_path}",
        }

    df = pd.read_parquet(parquet_path, engine="pyarrow").sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    start_ts = pd.Timestamp(scenario.start_date)
    end_ts = pd.Timestamp(scenario.end_date)

    # Filter to scenario window
    mask = (df["time"] >= start_ts) & (df["time"] <= end_ts)
    df_window = df[mask].reset_index(drop=True)

    if len(df_window) < 60:
        return {
            "scenario": scenario.name,
            "status": "insufficient_data",
            "n_rows": len(df_window),
        }

    closes = df_window["close"].astype("float64").to_numpy()
    dates = df_window["time"].tolist()

    # We need the original indices in the full df to call run_multi_agent_analysis
    df_indices = df[mask].index.tolist()

    # Backtest loop: every 20 trading days, get signal
    step = 20
    min_context = 128
    equity = 1.0
    equity_curve = [equity]
    trade_returns = []
    signals_log = []
    position = 0  # 0=cash, 1=long

    if verbose:
        print(f"\n  Running {scenario.name}: {scenario.ticker} {scenario.start_date}→{scenario.end_date}")
        print(f"  Data points in window: {len(closes)}")

    i = min_context
    while i < len(closes) - step:
        orig_idx = df_indices[i] if i < len(df_indices) else None
        if orig_idx is None:
            break

        try:
            analysis = run_multi_agent_analysis(scenario.ticker, as_of_index=int(orig_idx))
            signal = analysis.get("final_signal", "HOLD")
            score = _safe_float(analysis.get("final_score", 0.0))
        except Exception as e:
            signal = "HOLD"
            score = 0.0

        current_price = _safe_float(closes[i])
        future_price = _safe_float(closes[min(i + step, len(closes) - 1)])
        if current_price <= 0:
            i += step
            continue

        fwd_return = (future_price - current_price) / current_price

        # Simple long-only strategy
        if signal == "BUY" and position == 0:
            position = 1
        elif signal == "SELL" and position == 1:
            position = 0

        # Update equity if in position
        if position == 1:
            equity *= (1.0 + fwd_return)
            trade_returns.append(fwd_return)
        else:
            equity *= 1.0  # Cash: no change
            trade_returns.append(0.0)

        equity_curve.append(equity)
        signals_log.append({
            "date": str(dates[i].date()) if pd.notna(dates[i]) else f"idx_{i}",
            "signal": signal,
            "score": round(score, 4),
            "fwd_return_pct": round(fwd_return * 100, 4),
            "equity": round(equity, 6),
        })

        i += step

    metrics = _compute_metrics(equity_curve, trade_returns)

    # Buy & hold benchmark
    if len(closes) > 0:
        bh_return = float((closes[-1] - closes[0]) / closes[0] * 100)
    else:
        bh_return = 0.0

    result = {
        "scenario": scenario.name,
        "description": scenario.description,
        "ticker": scenario.ticker,
        "period": f"{scenario.start_date} → {scenario.end_date}",
        "expected_regime": scenario.expected_regime,
        "status": "ok",
        "data_points": len(closes),
        "n_steps": len(signals_log),
        "metrics": metrics,
        "buy_and_hold_return_pct": round(bh_return, 4),
        "alpha_pct": round(metrics["total_return_pct"] - bh_return, 4),
        "signals": signals_log,
        "equity_curve": [round(e, 6) for e in equity_curve],
    }

    if verbose:
        print(f"  ✓ Return: {metrics['total_return_pct']:+.2f}% "
              f"| B&H: {bh_return:+.2f}% "
              f"| Alpha: {result['alpha_pct']:+.2f}% "
              f"| Sharpe: {metrics['sharpe_ratio']:.3f} "
              f"| MaxDD: {metrics['max_drawdown_pct']:.2f}%")

    return result


# ─── Main runner ───────────────────────────────────────────────────────────────

def run_all_scenarios(
    scenarios: Optional[list[Scenario]] = None,
    verbose: bool = True,
) -> list[dict]:
    if scenarios is None:
        scenarios = SCENARIOS

    print(f"\n{'='*70}")
    print(f"  SCENARIO RUNNER — {len(scenarios)} scenario(s)")
    print(f"  Started at: {datetime.utcnow().isoformat()}")
    print(f"{'='*70}")

    results = []
    for scenario in scenarios:
        result = run_scenario_backtest(scenario, verbose=verbose)
        results.append(result)

    # Save results
    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "n_scenarios": len(results),
        "scenarios": results,
    }
    with open(SCENARIO_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if verbose:
        _print_summary(results)
        print(f"\n✓ Results saved: {SCENARIO_REPORT_PATH}")

    return results


def _print_summary(results: list[dict]):
    print(f"\n{'─'*70}")
    print(f"  {'Scenario':<22} {'Return':>8} {'B&H':>8} {'Alpha':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print(f"{'─'*70}")
    for r in results:
        if r.get("status") != "ok":
            print(f"  {r.get('scenario', '?'):<22} {'ERROR':>8}")
            continue
        m = r.get("metrics", {})
        print(
            f"  {r.get('description', r.get('scenario', '?')):<22} "
            f"{m.get('total_return_pct', 0):>+7.1f}% "
            f"{r.get('buy_and_hold_return_pct', 0):>+7.1f}% "
            f"{r.get('alpha_pct', 0):>+7.1f}% "
            f"{m.get('sharpe_ratio', 0):>8.3f} "
            f"{m.get('max_drawdown_pct', 0):>7.1f}%"
        )
    print(f"{'─'*70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scenario Runner")
    parser.add_argument(
        "--scenario",
        choices=[s.name for s in SCENARIOS] + ["all"],
        default="all",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.scenario == "all":
        run_all_scenarios(verbose=not args.quiet)
    else:
        scenario = next(s for s in SCENARIOS if s.name == args.scenario)
        result = run_scenario_backtest(scenario, verbose=not args.quiet)
        print(json.dumps(result, indent=2, ensure_ascii=False))
