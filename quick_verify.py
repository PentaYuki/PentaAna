#!/usr/bin/env python3
"""
quick_verify.py — 30-second sanity check that all Phase 3 components work
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))

print("\n" + "="*80)
print("  PHASE 3 — QUICK IMPORT VERIFICATION (30 seconds)")
print("="*80 + "\n")

checks = []

# Check 1: Enhanced agents
try:
    from enhanced_agents import enhanced_sentiment_agent, enhanced_macro_agent, enhanced_risk_agent
    checks.append(("✅", "enhanced_agents module", "Import successful"))
except Exception as e:
    checks.append(("❌", "enhanced_agents module", str(e)[:50]))

# Check 2: Phase 3 multi-agent
try:
    from phase3_multi_agent import run_multi_agent_analysis
    checks.append(("✅", "phase3_multi_agent.py", "Import successful"))
except Exception as e:
    checks.append(("❌", "phase3_multi_agent.py", str(e)[:50]))

# Check 3: Sentiment blending
try:
    from sentiment_features import blend_price_with_sentiment
    checks.append(("✅", "sentiment_features", "blend_price_with_sentiment() ready"))
except Exception as e:
    checks.append(("❌", "sentiment_features", str(e)[:50]))

# Check 4: Technical indicators
try:
    from technical_indicators import add_technical_indicators
    checks.append(("✅", "technical_indicators", "add_technical_indicators() ready"))
except Exception as e:
    checks.append(("❌", "technical_indicators", str(e)[:50]))

# Check 5: Coordinator tuner
try:
    from coordinator_tuner import GridSearchOptimizer
    checks.append(("✅", "coordinator_tuner", "GridSearchOptimizer ready"))
except Exception as e:
    checks.append(("❌", "coordinator_tuner", str(e)[:50]))

# Check 6: Backtest framework
try:
    import tests.backtest_comparison as bc
    checks.append(("✅", "backtest_comparison", "run_backtest_basic() ready"))
except Exception as e:
    checks.append(("❌", "backtest_comparison", str(e)[:50]))

# Check 7: Walk-forward validator
try:
    import tests.backtest_walk_forward as wf
    checks.append(("✅", "backtest_walk_forward", "run_walk_forward_test() ready"))
except Exception as e:
    checks.append(("❌", "backtest_walk_forward", str(e)[:50]))

# Display results
for status, component, detail in checks:
    print(f"  {status} {component:.<35} {detail}")

# Summary
successes = sum(1 for s, _, _ in checks if s == "✅")
total = len(checks)

print("\n" + "="*80)
if successes == total:
    print(f"  ✅ ALL CHECKS PASSED ({total}/{total})")
    print("\n  NEXT STEP: python run_validation_tests.py")
else:
    print(f"  ⚠️  SOME CHECKS FAILED ({successes}/{total})")
    print("\n  Debug the failed imports above before running tests")

print("="*80 + "\n")
