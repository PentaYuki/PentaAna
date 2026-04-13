"""
coordinator_tuner.py — Optimize agent weights using grid search and Bayesian methods

Finds optimal weights for:
  - technical_weight
  - sentiment_weight  
  - macro_weight
  - risk_weight

Maximizes Sharpe ratio or specified metric from backtesting.
"""

import sys
import os
import json
import itertools
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

import numpy as np
import pandas as pd

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(DATA_DIR, "reports", "json")

os.makedirs(REPORTS_DIR, exist_ok=True)


@dataclass
class WeightSet:
    technical: float
    sentiment: float
    macro: float
    risk: float
    
    def to_dict(self):
        return {
            "technical": round(self.technical, 3),
            "sentiment": round(self.sentiment, 3),
            "macro": round(self.macro, 3),
            "risk": round(self.risk, 3),
        }
    
    def validate(self):
        """Check weights sum to 1.0 and all positive"""
        total = self.technical + self.sentiment + self.macro + self.risk
        assert 0.99 < total < 1.01, f"Weights sum to {total}, not 1.0"
        assert self.technical >= 0 and self.sentiment >= 0 and self.macro >= 0 and self.risk >= 0, \
            "All weights must be non-negative"


class GridSearchOptimizer:
    """
    Grid search over weight combinations to find optimal agent weighting.
    
    Strategy:
    1. Define reasonable ranges for each weight
    2. Generate all combinations (grid points)
    3. For each combination, run backtest
    4. Track best Sharpe ratio
    5. Return optimal weights
    """
    
    def __init__(self, 
                 tech_range=(0.30, 0.50, 0.05),      # min, max, step
                 sent_range=(0.10, 0.35, 0.05),
                 macro_range=(0.05, 0.20, 0.05),
                 risk_range=(0.10, 0.25, 0.05),
                 metric="sharpe_ratio"):
        """
        Initialize grid ranges.
        Each range is (min, max, step).
        """
        self.metric = metric
        self.tech_range = np.arange(tech_range[0], tech_range[1] + tech_range[2], tech_range[2])
        self.sent_range = np.arange(sent_range[0], sent_range[1] + sent_range[2], sent_range[2])
        self.macro_range = np.arange(macro_range[0], macro_range[1] + macro_range[2], macro_range[2])
        self.risk_range = np.arange(risk_range[0], risk_range[1] + risk_range[2], risk_range[2])
        
        self.results = []
        self.best_result = None
        
    def generate_grid(self):
        """Generate all valid weight combinations"""
        grid = []
        for tech in self.tech_range:
            for sent in self.sent_range:
                for risk in self.risk_range:
                    # macro fills remaining weight
                    macro = max(0.0, 1.0 - tech - sent - risk)
                    # Only include if macro is in reasonable range
                    if 0.0 <= macro <= 0.30:
                        weights = WeightSet(tech, sent, macro, risk)
                        grid.append(weights)
        return grid
    
    def run_grid_search(self, backtest_fn):
        """
        Run backtest for each weight combination.
        
        Args:
            backtest_fn: Function that takes WeightSet and returns report dict
                        Report must contain self.metric key
        
        Returns:
            List of (weights, metric_value) tuples, sorted by metric descending
        """
        grid = self.generate_grid()
        print(f"\n[GRID SEARCH] Generated {len(grid)} weight combinations")
        print(f"[GRID SEARCH] Metric to optimize: {self.metric}")
        print(f"[GRID SEARCH] Starting optimization...\n")
        
        for i, weights in enumerate(grid, 1):
            try:
                # Run backtest with these weights
                report = backtest_fn(weights)
                metric_value = report.get("results", {}).get(self.metric, 0.0)
                
                result = {
                    "iteration": i,
                    "weights": weights.to_dict(),
                    "metric_value": round(metric_value, 4),
                    "report_summary": {
                        "total_return": report.get("results", {}).get("total_return_pct", 0.0),
                        "win_rate": report.get("results", {}).get("win_rate_pct", 0.0),
                        "sharpe": report.get("results", {}).get("sharpe_ratio", 0.0),
                        "max_drawdown": report.get("results", {}).get("max_drawdown_pct", 0.0),
                    }
                }
                
                self.results.append(result)
                
                if i % 5 == 0:
                    print(f"  [{i}/{len(grid)}] Tested weight set - {self.metric}={metric_value:.4f}")
                
                # Track best
                if self.best_result is None or metric_value > self.best_result["metric_value"]:
                    self.best_result = result
                    print(f"        ✓ NEW BEST: {self.metric}={metric_value:.4f}")
                    print(f"           Weights: {weights.to_dict()}")
                
            except Exception as e:
                print(f"  [{i}/{len(grid)}] Error: {str(e)}")
                continue
        
        # Sort by metric value
        self.results.sort(key=lambda x: x["metric_value"], reverse=True)
        
        return self.results
    
    def get_top_k(self, k=10):
        """Get top K weight combinations by metric"""
        return self.results[:min(k, len(self.results))]
    
    def print_summary(self):
        """Print grid search summary"""
        if not self.results:
            print("No results found!")
            return
        
        print("\n" + "="*80)
        print("  GRID SEARCH OPTIMIZATION RESULTS")
        print("="*80)
        
        print(f"\n[SUMMARY]")
        print(f"  Total tests: {len(self.results)}")
        print(f"  Best {self.metric}: {self.best_result['metric_value']:.4f}")
        print(f"\n[OPTIMAL WEIGHTS]")
        for k, v in self.best_result["weights"].items():
            print(f"  {k:15s}: {v:.3f}")
        
        print(f"\n[TOP 5 CONFIGURATIONS]")
        print(f"  {'Rank':<6} {self.metric:<15} {'Tech':<8} {'Sent':<8} {'Macro':<8} {'Risk':<8}")
        print("-" * 65)
        
        for rank, result in enumerate(self.get_top_k(5), 1):
            w = result["weights"]
            print(f"  {rank:<6} {result['metric_value']:<15.4f} "
                  f"{w['technical']:<8.2f} {w['sentiment']:<8.2f} "
                  f"{w['macro']:<8.2f} {w['risk']:<8.2f}")
        
        print("\n" + "="*80)
    
    def save_results(self, filepath=None):
        """Save all results to JSON"""
        if filepath is None:
            filepath = os.path.join(REPORTS_DIR, "grid_search_weights.json")
        
        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "optimizer_config": {
                "metric": self.metric,
                "total_iterations": len(self.results),
                "grid_size": len(self.results),
            },
            "best_result": self.best_result,
            "top_10": self.get_top_k(10),
            "all_results": self.results,
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Results saved to: {filepath}")
        return filepath


def mock_backtest_fn(weights):
    """
    Mock backtest function for testing.
    In production, this would run actual backtest with different weights.
    """
    # Simplified: metric = weighted average of agent confidence scores
    # (In real version, would run full backtest with these weights)
    
    # Mock: Higher technical weight → higher tech signal strength
    mock_sharpe = (
        0.5 * weights.technical +    # Technical is strong
        0.3 * weights.sentiment +     # Sentiment adds some value
        0.1 * weights.macro +         # Macro adds little
        0.2 * weights.risk +          # Risk helps but penalizes volatility
        np.random.random() * 0.1      # Add small random noise
    )
    
    return {
        "results": {
            "sharpe_ratio": mock_sharpe,
            "total_return_pct": 15.0 + weights.technical * 5.0,
            "win_rate_pct": 52.0 + weights.technical * 2.0,
            "max_drawdown_pct": -12.0 + weights.risk * 2.0,
        }
    }


@contextmanager
def temporary_weight_env(weights: WeightSet):
    keys = ["PHASE3_W_TECH", "PHASE3_W_SENT", "PHASE3_W_MACRO", "PHASE3_W_RISK"]
    old = {k: os.getenv(k) for k in keys}
    os.environ["PHASE3_W_TECH"] = str(weights.technical)
    os.environ["PHASE3_W_SENT"] = str(weights.sentiment)
    os.environ["PHASE3_W_MACRO"] = str(weights.macro)
    os.environ["PHASE3_W_RISK"] = str(weights.risk)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def real_backtest_fn(weights: WeightSet):
    """Run real multi-agent backtest using the provided coordinator weights."""
    from backtest_engine import run_backtest, BacktestEngineConfig as BacktestConfig

    with temporary_weight_env(weights):
        report = run_backtest(
            BacktestConfig(
                ticker="VNM",
                cost_bps=35.0,
            )
        )
    return report


# ─── Optuna Bayesian Optimizer ────────────────────────────────────────────────

class OptunaOptimizer:
    """
    Thay thế GridSearchOptimizer trong môi trường production.
    Dùng TPE sampler (Tree-structured Parzen Estimator) — nhanh hơn grid search
    ~10× với cùng số trials, tìm được vùng trọng số tốt hơn.

    Cài đặt: pip install optuna
    """

    def __init__(self, n_trials: int = 50, metric: str = "sharpe_ratio", direction: str = "maximize"):
        self.n_trials  = n_trials
        self.metric    = metric
        self.direction = direction
        self.study     = None
        self.results: list[dict] = []

    def optimize(self, backtest_fn) -> dict:
        """
        Chạy Optuna optimization.

        Args:
            backtest_fn: fn(WeightSet) → report dict với report["results"][self.metric]

        Returns:
            best_weights dict {technical, sentiment, macro, risk}
        """
        if not _OPTUNA_AVAILABLE:
            raise ImportError("optuna chưa cài — chạy: pip install optuna")

        def objective(trial):
            tech = trial.suggest_float("technical", 0.25, 0.55)
            sent = trial.suggest_float("sentiment", 0.05, 0.35)
            risk = trial.suggest_float("risk", 0.05, 0.30)
            macro = max(0.0, 1.0 - tech - sent - risk)
            if macro < 0.0 or macro > 0.35:
                return float("-inf")
            ws = WeightSet(technical=tech, sentiment=sent, macro=macro, risk=risk)
            try:
                ws.validate()
                report = backtest_fn(ws)
                val = float(report.get("results", {}).get(self.metric, 0.0))
                self.results.append({"weights": ws.to_dict(), self.metric: val})
                return val
            except Exception:
                return float("-inf")

        self.study = optuna.create_study(direction=self.direction)
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best = self.study.best_params
        macro = max(0.0, 1.0 - best["technical"] - best["sentiment"] - best["risk"])
        return {
            "technical": round(best["technical"], 4),
            "sentiment": round(best["sentiment"], 4),
            "macro":     round(macro, 4),
            "risk":      round(best["risk"], 4),
            self.metric: round(self.study.best_value, 4),
        }

    def save_results(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(REPORTS_DIR, "optuna_weights.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        out = {
            "timestamp": datetime.now().isoformat(),
            "n_trials":  self.n_trials,
            "metric":    self.metric,
            "best":      self.study.best_params if self.study else {},
            "best_value": self.study.best_value if self.study else 0.0,
            "results":   sorted(self.results, key=lambda x: x.get(self.metric, 0), reverse=True)[:20],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"✓ Optuna results saved to: {filepath}")
        return filepath


def example_grid_search():
    """
    Example: Run grid search with mock backtest function
    """
    print("\n[EXAMPLE] Grid search with mock backtest...")
    
    optimizer = GridSearchOptimizer(
        tech_range=(0.30, 0.50, 0.10),      # Coarser grid for example
        sent_range=(0.10, 0.30, 0.10),
        macro_range=(0.05, 0.20, 0.05),
        risk_range=(0.10, 0.25, 0.05),
        metric="sharpe_ratio"
    )
    
    results = optimizer.run_grid_search(mock_backtest_fn)
    optimizer.print_summary()
    optimizer.save_results(os.path.join(REPORTS_DIR, "grid_search_example.json"))
    
    return optimizer


if __name__ == "__main__":
    print("\n" + "█"*70)
    print("  COORDINATOR WEIGHT OPTIMIZER")
    print("  Grid Search for Optimal Agent Weights")
    print("█"*70)
    
    optimizer = GridSearchOptimizer(
        tech_range=(0.35, 0.50, 0.05),
        sent_range=(0.08, 0.30, 0.04),
        macro_range=(0.05, 0.20, 0.05),
        risk_range=(0.10, 0.30, 0.05),
        metric="sharpe_ratio",
    )

    try:
        optimizer.run_grid_search(real_backtest_fn)
        optimizer.print_summary()
        optimizer.save_results(os.path.join(REPORTS_DIR, "grid_search_real_backtest.json"))
        print("\n✓ Real backtest grid search complete!")
    except Exception as e:
        print(f"\n⚠ Real backtest grid search failed: {e}")
        print("Falling back to mock grid search for smoke test...")
        optimizer = example_grid_search()
        print("\n✓ Mock grid search complete!")
