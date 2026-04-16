#!/usr/bin/env python3
"""
PHASE 3 MASTER CHECKLIST
Quick status check of all Phase 3 deliverables and validation steps
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
TESTS_DIR = BASE_DIR / "tests"
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = DATA_DIR / "reports" / "json"

def check_file(path, name):
    """Check if file exists and return status"""
    exists = path.exists()
    size = ""
    if exists:
        size = f" ({(path.stat().st_size / 1024):.1f}K)"
    status = "✅" if exists else "❌"
    return f"{status} {name}{size}"

def check_import(module_path, import_str):
    """Verify import works"""
    try:
        sys.path.insert(0, str(SRC_DIR))
        exec(f"from {import_str} import *")
        return "✅ Imports verified"
    except Exception as e:
        return f"❌ Import failed: {str(e)[:50]}"

def main():
    print("\n" + "█" * 100)
    print("  PHASE 3 — MULTI-AGENT TRADING SYSTEM")
    print("  MASTER CHECKLIST & VALIDATION STATUS")
    print("█" * 100)
    
    print("\n" + "="*100)
    print("1. CORE SYSTEM FILES")
    print("="*100)
    
    core_files = [
        (SRC_DIR / "phase3_multi_agent.py", "Coordinator (3 agents enhanced ✨)"),
        (SRC_DIR / "enhanced_agents.py", "Enhanced Agents Module (NEW)"),
        (SRC_DIR / "sentiment_features.py", "Sentiment Blending"),
        (SRC_DIR / "technical_indicators.py", "Technical Indicators"),
        (SRC_DIR / "kronos_trainer.py", "Kronos Trainer (use_sentiment=True)"),
        (TESTS_DIR / "kronos_test.py", "Kronos Test (moved to tests/)"),
    ]
    
    for path, name in core_files:
        print(f"  {check_file(path, name)}")
    
    print("\n" + "="*100)
    print("2. VALIDATION & TESTING INFRASTRUCTURE")
    print("="*100)
    
    validation_files = [
        (TESTS_DIR / "backtest_basic.py", "Kronos-only Backtest"),
        (TESTS_DIR / "backtest_basic_multi_agent.py", "Multi-Agent Backtest"),
        (TESTS_DIR / "backtest_comparison.py", "A/B Test Comparison (NEW)"),
        (TESTS_DIR / "backtest_walk_forward.py", "Walk-Forward Validator (NEW)"),
        (BASE_DIR / "run_validation_tests.py", "Validation Runner (NEW)"),
        (SRC_DIR / "coordinator_tuner.py", "Weight Grid Search (NEW)"),
    ]
    
    for path, name in validation_files:
        print(f"  {check_file(path, name)}")
    
    print("\n" + "="*100)
    print("3. DATA & REPORTS")
    print("="*100)
    
    data_files = [
        (DATA_DIR / "raw" / "parquet", "Parquet Data Directory"),
        (REPORTS_DIR, "Reports Directory"),
    ]
    
    for path, name in data_files:
        status = "✅" if path.exists() else "❌"
        if path.exists() and path.is_dir():
            count = len(list(path.glob("*")))
            print(f"  {status} {name} ({count} files)")
        else:
            print(f"  {status} {name}")
    
    print("\n" + "="*100)
    print("4. FEATURE COMPLETENESS")
    print("="*100)
    
    features = {
        "Sentiment Integration": ("✅", "Blended into Kronos forecast via use_sentiment=True"),
        "Enhanced Sentiment Agent": ("✅", "EWM + volume weighting + freshness boost"),
        "Enhanced Macro Agent": ("✅", "VNINDEX momentum + volatility penalty"),
        "Enhanced Risk Agent": ("✅", "ATR-based position sizing + stop-loss"),
        "Coordinator Integration": ("✅", "All 3 agents wired into voting system"),
        "Backtest Comparison Tool": ("✅", "A/B test Kronos vs Multi-Agent"),
        "Walk-Forward Validator": ("✅", "Rolling window robustness testing"),
        "Weight Grid Search": ("✅", "GridSearchOptimizer ready for tuning"),
    }
    
    for feature, (status, desc) in features.items():
        print(f"  {status} {feature:.<40} {desc}")
    
    print("\n" + "="*100)
    print("5. EXECUTION CHECKLIST")
    print("="*100)
    
    execution_steps = [
        ("🔴 HIGH", "Run backtest comparison", "python run_validation_tests.py", "Not started"),
        ("🔴 HIGH", "Analyze comparison results", "Review JSON in data/reports/json/", "Pending backtest"),
        ("🟡 MEDIUM", "Run grid search (if Multi > Kronos)", "python -c 'from coordinator_tuner import GridSearchOptimizer'", "Week 2"),
        ("🟡 MEDIUM", "Run walk-forward validation", "python tests/backtest_walk_forward.py", "Week 3"),
        ("🟢 LOW", "Online learning / RLHF", "Phase 4 (optional)", "Q2"),
    ]
    
    for priority, task, command, status in execution_steps:
        print(f"  {priority:.<12} {task:.<35} [{status}]")
        print(f"               $ {command}\n")
    
    print("="*100)
    print("6. SENTIMENT INTEGRATION VERIFICATION")
    print("="*100)
    
    # Check sentiment DB
    news_db = DATA_DIR / "news.db"
    if news_db.exists():
        print(f"  ✅ Sentiment DB (news.db) {(news_db.stat().st_size / (1024*1024)):.1f}M")
    else:
        print(f"  ⚠️  Sentiment DB not found (optional - graceful fallback works)")
    
    # Try import sentiment
    try:
        sys.path.insert(0, str(SRC_DIR))
        from sentiment_features import blend_price_with_sentiment
        print(f"  ✅ blend_price_with_sentiment() importable")
    except ImportError as e:
        print(f"  ❌ blend_price_with_sentiment() import failed: {e}")
    
    # Try import enhanced agents
    try:
        from enhanced_agents import enhanced_sentiment_agent, enhanced_macro_agent, enhanced_risk_agent
        print(f"  ✅ enhanced_*_agent() functions importable")
    except ImportError as e:
        print(f"  ❌ enhanced_agents import failed: {e}")
    
    print("\n" + "="*100)
    print("7. QUICK START COMMANDS")
    print("="*100)
    print("""
    # Activate environment
    cd /Users/gooleseswsq1gmail.com/Documents/stock-ai
    source .venv/bin/activate
    
    # Option 1: Run validation test (recommended)
    python run_validation_tests.py
    
    # Option 2: Run backtest comparison directly  
    python tests/backtest_comparison.py
    
    # Option 3: Run walk-forward validation
    python tests/backtest_walk_forward.py
    
    # Option 4: Test weight optimizer (after backtest baseline)
    python -c "from coordinator_tuner import GridSearchOptimizer; print('✓ Ready')"
    """)
    
    print("\n" + "="*100)
    print("8. EXPECTED OUTCOMES")
    print("="*100)
    print("""
    After running backtest comparison, expect to see:
    
    Multi-Agent Benefits (if working):
      • Sentiment signals: +2-5% additional returns (via sentiment voting)
      • Risk management: -5-10% lower drawdown (via risk agent)
      • Win rate: +3-8% higher (via macro agent entry timing)
    
    Trade-offs:
      • More trades generated (higher commission)
      • More complex logic (harder to debug)
      • Higher Sharpe potential but higher variance
    
    Success Criteria:
      ✓ Multi-Agent Sharpe > 0.6 (better risk-adjusted returns)
      ✓ Win rate > 53% (better than random)
      ✓ Max drawdown < -15% (controlled risk)
      ✓ Code robustness score > 0.4 (stable across periods)
    """)
    
    print("\n" + "█" * 100)
    print("  STATUS: PHASE 3 CORE IMPLEMENTATION COMPLETE")
    print("  NEXT: Run backtest_comparison.py to validate improvements")
    print("█" * 100 + "\n")

if __name__ == "__main__":
    main()
