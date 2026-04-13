"""
backtest_walk_forward.py — Rolling window validation for robustness testing

Tests strategy across multiple time periods to avoid overfitting:
  Strategy: Split 5-year history into rolling windows:
    Train: 2 years (24 months)  
    Test: 6 months
    Roll: 6 months forward, repeat

  Returns: Avg Sharpe, Avg Return, Stability across periods
"""

import sys
import os
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_PQ_DIR = os.path.join(DATA_DIR, "raw", "parquet")
REPORTS_DIR = os.path.join(DATA_DIR, "reports", "json")

os.makedirs(REPORTS_DIR, exist_ok=True)


@dataclass
class WindowResult:
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_data_points: int
    test_data_points: int
    
    # Results
    total_return_pct: float
    win_rate_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    num_trades: int
    
    def to_dict(self):
        return {
            "window_id": self.window_id,
            "train_period": f"{self.train_start} to {self.train_end}",
            "test_period": f"{self.test_start} to {self.test_end}",
            "train_bars": self.train_data_points,
            "test_bars": self.test_data_points,
            "results": {
                "total_return": round(self.total_return_pct, 2),
                "win_rate": round(self.win_rate_pct, 2),
                "sharpe_ratio": round(self.sharpe_ratio, 4),
                "max_drawdown": round(self.max_drawdown_pct, 2),
                "num_trades": int(self.num_trades),
            }
        }


def split_data_into_windows(df: pd.DataFrame, 
                             train_months: int = 24,
                             test_months: int = 6,
                             roll_months: int = 6) -> List[tuple]:
    """
    Split 5-year data into rolling windows.
    
    Returns: List of (train_df, test_df, window_info) tuples
    """
    df = df.copy().sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    
    windows = []
    current_date = df["time"].min()
    end_date = df["time"].max()
    
    window_id = 1
    while (current_date + pd.DateOffset(months=train_months + test_months)) <= end_date:
        # Define train period
        train_start = current_date
        train_end = train_start + pd.DateOffset(months=train_months)
        
        # Define test period
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        
        # Extract data
        train_df = df[(df["time"] >= train_start) & (df["time"] < train_end)]
        test_df = df[(df["time"] >= test_start) & (df["time"] < test_end)]
        
        if len(train_df) > 0 and len(test_df) > 0:
            windows.append((
                train_df,
                test_df,
                {
                    "window_id": window_id,
                    "train_start": str(train_start.date()),
                    "train_end": str(train_end.date()),
                    "test_start": str(test_start.date()),
                    "test_end": str(test_end.date()),
                }
            ))
            window_id += 1
        
        # Roll forward
        current_date += pd.DateOffset(months=roll_months)
    
    return windows


def run_walk_forward_test(ticker: str = "VNM",
                          backtest_fn=None) -> dict:
    """
    Run walk-forward validation.
    
    Args:
        ticker: Stock ticker
        backtest_fn: Function(train_df, test_df) -> report_dict
                    If None, returns mock results
    
    Returns: Walk-forward report with window-by-window stats
    """
    
    print(f"\n{'='*80}")
    print(f"  WALK-FORWARD VALIDATION")
    print(f"  Ticker: {ticker}")
    print(f"{'='*80}\n")
    
    # Load data
    parquet_path = os.path.join(RAW_PQ_DIR, f"{ticker}_history.parquet")
    if not os.path.exists(parquet_path):
        print(f"❌ Data not found: {parquet_path}")
        return None
    
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    
    # Split into windows
    windows = split_data_into_windows(df, train_months=24, test_months=6, roll_months=6)
    print(f"Generated {len(windows)} rolling windows\n")
    
    if len(windows) == 0:
        print("❌ No windows could be created")
        return None
    
    results = []
    for train_df, test_df, info in windows:
        window_id = info["window_id"]
        print(f"[Window {window_id}/{len(windows)}] {info['test_start']} to {info['test_end']}")
        
        try:
            if backtest_fn:
                # Run actual backtest
                report = backtest_fn(train_df, test_df)
                result = report.get("results", {})
            else:
                # Mock results for demo
                result = {
                    "total_return_pct": 12.5 + np.random.normal(0, 3),
                    "win_rate_pct": 52.0 + np.random.normal(0, 2),
                    "sharpe_ratio": 0.8 + np.random.normal(0, 0.2),
                    "max_drawdown_pct": -10.0 - abs(np.random.normal(0, 2)),
                    "num_trades": int(20 + np.random.normal(0, 5)),
                }
            
            wr = WindowResult(
                window_id=window_id,
                train_start=info["train_start"],
                train_end=info["train_end"],
                test_start=info["test_start"],
                test_end=info["test_end"],
                train_data_points=len(train_df),
                test_data_points=len(test_df),
                total_return_pct=result.get("total_return_pct", 0),
                win_rate_pct=result.get("win_rate_pct", 0),
                sharpe_ratio=result.get("sharpe_ratio", 0),
                max_drawdown_pct=result.get("max_drawdown_pct", 0),
                num_trades=result.get("num_trades", 0),
            )
            
            results.append(wr)
            print(f"  ✓ Sharpe: {wr.sharpe_ratio:.4f}, Return: {wr.total_return_pct:.2f}%\n")
            
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
            continue
    
    # Aggregate statistics
    sharpes = [r.sharpe_ratio for r in results]
    returns = [r.total_return_pct for r in results]
    drawdowns = [r.max_drawdown_pct for r in results]
    win_rates = [r.win_rate_pct for r in results]
    
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "total_windows": len(results),
        "statistics": {
            "sharpe_ratio": {
                "mean": round(np.mean(sharpes), 4),
                "std": round(np.std(sharpes), 4),
                "min": round(np.min(sharpes), 4),
                "max": round(np.max(sharpes), 4),
            },
            "total_return": {
                "mean": round(np.mean(returns), 2),
                "std": round(np.std(returns), 2),
                "min": round(np.min(returns), 2),
                "max": round(np.max(returns), 2),
            },
            "max_drawdown": {
                "mean": round(np.mean(drawdowns), 2),
                "std": round(np.std(drawdowns), 2),
                "max_worst": round(np.min(drawdowns), 2),
            },
            "win_rate": {
                "mean": round(np.mean(win_rates), 2),
                "std": round(np.std(win_rates), 2),
                "min": round(np.min(win_rates), 2),
                "max": round(np.max(win_rates), 2),
            }
        },
        "robustness_score": round(
            np.mean(sharpes) - np.std(sharpes),  # Lower std = more robust
            4
        ),
        "windows": [r.to_dict() for r in results],
    }
    
    # Print summary
    print("\n" + "="*80)
    print("  WALK-FORWARD SUMMARY STATISTICS")
    print("="*80)
    print(f"\nTotal Windows: {len(results)}")
    print(f"\nSharpe Ratio:")
    print(f"  Mean: {summary['statistics']['sharpe_ratio']['mean']:.4f}")
    print(f"  Std:  {summary['statistics']['sharpe_ratio']['std']:.4f}")
    print(f"  Min:  {summary['statistics']['sharpe_ratio']['min']:.4f}")
    print(f"  Max:  {summary['statistics']['sharpe_ratio']['max']:.4f}")
    
    print(f"\nTotal Return %:")
    print(f"  Mean: {summary['statistics']['total_return']['mean']:.2f}%")
    print(f"  Std:  {summary['statistics']['total_return']['std']:.2f}%")
    
    print(f"\nRobustness Score: {summary['robustness_score']:.4f}")
    print(f"  (Higher = more stable across periods)")
    
    print("\n" + "="*80)
    
    return summary


if __name__ == "__main__":
    print("\n" + "█"*80)
    print("  WALK-FORWARD VALIDATION TEST")
    print("█"*80)
    
    report = run_walk_forward_test("VNM", backtest_fn=None)
    
    if report:
        # Save report
        output_path = os.path.join(REPORTS_DIR, "walk_forward_results.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Report saved to: {output_path}")
        print("\nInterpretation:")
        print("  • Robustness Score > 0.5: Good (strategy stable across periods)")
        print("  • Sharpe Std < 0.3: Good (consistent performance)")
        print("  • Win Rate Std < 10%: Good (consistent quality)")
