#!/usr/bin/env python3
"""
run_validation_tests.py — Run all Phase 3 validation tests
Executes backtest comparison and shows results
"""

import sys
import os
import json
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "tests"))

def print_header(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def run_backtest_comparison():
    """Run the backtest comparison tool"""
    print_header("RUNNING BACKTEST COMPARISON")
    print("\nStarting: python tests/backtest_comparison.py")
    print("(This may take 5-30 minutes depending on data size...)\n")
    
    try:
        result = subprocess.run(
            [sys.executable, "tests/backtest_comparison.py"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Timeout: Backtest took too long (>1 hour)")
        return False
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        return False

def analyze_results():
    """Read and display backtest results"""
    print_header("ANALYZING RESULTS")
    
    reports_dir = BASE_DIR / "data" / "reports" / "json"
    comparison_file = reports_dir / "backtest_comparison.json"
    
    if not comparison_file.exists():
        print(f"❌ Comparison report not found: {comparison_file}")
        return False
    
    try:
        with open(comparison_file) as f:
            comparison = json.load(f)
        
        print(f"\n📊 BACKTEST COMPARISON RESULTS")
        print(f"Generated: {comparison.get('timestamp', 'N/A')}\n")
        
        # Show metrics comparison
        print("Metrics Comparison:")
        print("-" * 80)
        print(f"{'Metric':<25} {'Kronos-Only':>25} {'Multi-Agent':>25} {'Winner':<15}")
        print("-" * 80)
        
        for mc in comparison.get("metrics_comparison", []):
            metric = mc.get("metric", "")
            kronos = mc.get("kronos_only", 0)
            multi = mc.get("multi_agent", 0)
            winner = mc.get("winner", "")
            
            # Format nicely
            kronos_str = f"{kronos:.4f}" if isinstance(kronos, float) else str(kronos)
            multi_str = f"{multi:.4f}" if isinstance(multi, float) else str(multi)
            
            winner_emoji = "🏆" if winner == "Multi-Agent" else "🥈"
            
            print(f"{metric:<25} {kronos_str:>25} {multi_str:>25} {winner_emoji} {winner:<15}")
        
        # Show recommendation
        print("\n" + "="*80)
        recommendation = comparison.get("recommendation", "UNKNOWN")
        print(f"📋 RECOMMENDATION: {recommendation}")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return False

def main():
    print("\n" + "█"*80)
    print("  PHASE 3 VALIDATION TEST RUNNER")
    print("  Backtest: Kronos-Only vs Multi-Agent with Enhanced Agents")
    print("█"*80)
    
    # Step 1: Run backtest
    success = run_backtest_comparison()
    
    if not success:
        print("\n❌ Backtest comparison failed")
        return 1
    
    # Step 2: Analyze results
    if not analyze_results():
        print("\n❌ Could not analyze results")
        return 1
    
    # Summary
    print_header("✅ VALIDATION COMPLETE")
    print("""
Next Steps:
  1. Review the recommendation above
  2. If Multi-Agent wins: Proceed to weight optimization (Week 2)
  3. If Kronos wins: Debug why coordinator underperforming
  4. Either way: Save the results for reference
    
Detailed reports saved to:
  - data/reports/json/backtest_comparison.json
  - data/reports/json/backtest_kronos_only.json
  - data/reports/json/backtest_multi_agent_updated.json
    """)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
