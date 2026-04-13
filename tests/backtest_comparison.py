#!/usr/bin/env python3
"""
backtest_comparison.py — Compare Kronos-only vs Multi-Agent strategies
Runs both backtests and generates side-by-side comparison report.
"""

import sys
import os
import json
from datetime import datetime

# Setup paths
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
sys.path.insert(0, os.path.join(os.getcwd(), "tests"))

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(DATA_DIR, "reports", "json")

# Ensure reports directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)


def run_backtest_basic():
    """Run Kronos-only backtest"""
    print("\n" + "="*70)
    print("  BACKTEST 1: Kronos-Only Strategy (Base Model)")
    print("="*70)
    
    try:
        # Try relative import first, then absolute
        try:
            from backtest_basic import run_backtest, BacktestConfig
        except ImportError:
            from tests.backtest_basic import run_backtest, BacktestConfig
        
        cfg = BacktestConfig(ticker="VNM", use_sentiment_filter=False)
        report = run_backtest(cfg)
        return report
    except Exception as e:
        print(f"Error running basic backtest: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_backtest_multi_agent():
    """Run Multi-Agent backtest"""
    print("\n" + "="*70)
    print("  BACKTEST 2: Multi-Agent Strategy (With Coordinator)")
    print("="*70)
    
    try:
        # Try relative import first, then absolute
        try:
            from backtest_multi_agent import run_multi_agent_backtest, BacktestConfig
        except ImportError:
            from tests.backtest_multi_agent import run_multi_agent_backtest, BacktestConfig
        
        cfg = BacktestConfig(ticker="VNM", use_sentiment_filter=True)
        report = run_multi_agent_backtest(cfg)
        return report
    except Exception as e:
        print(f"Error running multi-agent backtest: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_comparison_report(basic_report, multi_report):
    """Create side-by-side comparison"""
    print("\n" + "="*70)
    print("  COMPARISON RESULTS")
    print("="*70)
    
    if basic_report is None or multi_report is None:
        print("❌ Could not generate comparison (missing reports)")
        return None

    # Handle both nested ("results" key) and root-level metric structures
    basic_results = basic_report.get("results", {}) if "results" in basic_report else basic_report
    multi_results = multi_report.get("results", {}) if "results" in multi_report else multi_report

    metrics = [
        ("Total Return", "total_return_pct", "%"),
        ("Win Rate", "win_rate_pct", "%"),
        ("Sharpe Ratio", "sharpe_ratio", ""),
        ("Max Drawdown", "max_drawdown_pct", "%"),
        ("Final Equity", "final_equity", "x"),
    ]

    comparison = {
        "timestamp": datetime.utcnow().isoformat(),
        "backtest_pair": {
            "strategy_1": "Kronos-Only (backtest_basic.py)",
            "strategy_2": "Multi-Agent + Coordinator (backtest_multi_agent.py)",
        },
        "metrics_comparison": [],
        "winner_by_metric": {},
        "recommendation": "",
    }

    print(f"\n{'Metric':<20} {'Kronos-Only':>18} {'Multi-Agent':>18} {'Winner':<12}")
    print("-" * 70)

    multi_agent_wins = 0
    kronos_wins = 0

    for metric_name, key, unit in metrics:
        basic_val = basic_results.get(key, 0)
        multi_val = multi_results.get(key, 0)
        
        # Determine winner (higher is better for most metrics, except drawdown)
        if key == "max_drawdown_pct":
            # For drawdown, lower absolute drawdown is better.
            winner = "Multi-Agent" if abs(multi_val) < abs(basic_val) else "Kronos-Only"
            wins = "multi" if abs(multi_val) < abs(basic_val) else "basic"
        else:
            # For return, win rate, sharpe - higher is better
            winner = "Multi-Agent" if multi_val > basic_val else "Kronos-Only"
            wins = "multi" if multi_val > basic_val else "basic"
        
        if wins == "multi":
            multi_agent_wins += 1
        else:
            kronos_wins += 1

        format_str = f"{basic_val:>15.2f}{unit}" if unit else f"{basic_val:>15.4f}"
        format_str_multi = f"{multi_val:>15.2f}{unit}" if unit else f"{multi_val:>15.4f}"

        print(f"{metric_name:<20} {format_str:>18} {format_str_multi:>18} {winner:<12}")

        comparison["metrics_comparison"].append({
            "metric": metric_name,
            "kronos_only": round(basic_val, 4),
            "multi_agent": round(multi_val, 4),
            "winner": winner,
        })
        comparison["winner_by_metric"][metric_name] = winner

    # Overall recommendation
    print("\n" + "-" * 70)
    print(f"Multi-Agent wins on: {multi_agent_wins} metrics")
    print(f"Kronos-Only wins on: {kronos_wins} metrics")

    if multi_agent_wins >= 3:
        recommendation = "✅ MULTI-AGENT OUTPERFORMS - Use coordinator strategy"
        comparison["recommendation"] = recommendation
        print(f"\n{recommendation}")
    elif kronos_wins >= 3:
        recommendation = "⚠️  KRONOS-ONLY BETTER - Debug multi-agent or optimize weights"
        comparison["recommendation"] = recommendation
        print(f"\n{recommendation}")
    else:
        recommendation = "🔄 INCONCLUSIVE - Both strategies comparable, optimize further"
        comparison["recommendation"] = recommendation
        print(f"\n{recommendation}")

    print("="*70)

    return comparison


def main():
    print("\n" + "█"*70)
    print("  PHASE 3 BACKTEST COMPARISON")
    print("  Kronos-Only vs Multi-Agent Coordinator Strategy")
    print("█"*70)

    # Run both backtests
    print("\n[Step 1/3] Running Kronos-only backtest...")
    basic_report = run_backtest_basic()

    print("\n[Step 2/3] Running multi-agent backtest...")
    multi_report = run_backtest_multi_agent()

    # Create comparison
    print("\n[Step 3/3] Creating comparison report...")
    comparison = create_comparison_report(basic_report, multi_report)

    if comparison:
        # Save comparison
        comparison_path = os.path.join(REPORTS_DIR, "backtest_comparison.json")
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Comparison saved to: {comparison_path}")

        # Also save individual reports if not already saved
        if basic_report:
            basic_path = os.path.join(REPORTS_DIR, "backtest_kronos_only.json")
            with open(basic_path, "w", encoding="utf-8") as f:
                json.dump(basic_report, f, indent=2, ensure_ascii=False)

        if multi_report:
            multi_path = os.path.join(REPORTS_DIR, "backtest_multi_agent_updated.json")
            with open(multi_path, "w", encoding="utf-8") as f:
                json.dump(multi_report, f, indent=2, ensure_ascii=False)

        print("\n" + "="*70)
        print("✓ COMPARISON COMPLETE")
        print("="*70)
        print("\nGenerated files:")
        print(f"  • {comparison_path}")
        print(f"  • {os.path.join(REPORTS_DIR, 'backtest_kronos_only.json')}")
        print(f"  • {os.path.join(REPORTS_DIR, 'backtest_multi_agent_updated.json')}")

        return comparison
    else:
        print("\n❌ Comparison failed - check errors above")
        return None


if __name__ == "__main__":
    main()
