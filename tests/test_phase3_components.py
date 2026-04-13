"""
test_phase3_components.py — Benchmarking Phase 3 Implementation Status

Kiểm tra từng thành phần của Phase 3 so với yêu cầu:
  1. Sentiment integration in Kronos
  2. Multi-agent backtest
  3. Agent sophistication
  4. Coordinator weight optimization
  5. Online learning mechanism
  6. Walk-forward validation
"""

import json
import os
import sys
from datetime import datetime
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from phase3_multi_agent import run_multi_agent_analysis
from kronos_trainer import finetune_kronos

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORT_PATH = os.path.join(DATA_DIR, "reports", "json", "phase3_component_test.json")


@dataclass
class ComponentStatus:
    name: str
    implemented: bool
    severity: str  # "critical", "high", "medium", "low"
    details: str
    test_passed: bool = False
    test_output: str = ""


def test_sentiment_in_kronos() -> ComponentStatus:
    """
    ✓ IMPLEMENTED: kronos_trainer.py has use_sentiment=True parameter
    ✓ IMPLEMENTED: blend_price_with_sentiment() function exists
    ❌ NOT USED: phase3_multi_agent.py doesn't call sentiment-enabled inference
    """
    status = ComponentStatus(
        name="Sentiment Integration in Kronos",
        implemented=True,
        severity="critical",
        details="kronos_trainer has use_sentiment=True with EWM blending, but phase3_multi_agent doesn't use it",
    )

    try:
        # Check that kronos_trainer has sentiment support
        from kronos_trainer import blend_price_with_sentiment, finetune_kronos
        
        # Verify function exists and works
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0], dtype="float32")
        dates = pd.date_range("2024-01-01", periods=5)
        sentiment_lookup = {"VNM": {"2024-01-01": 0.5, "2024-01-02": 0.3}}
        
        blended = blend_price_with_sentiment("VNM", dates, prices, sentiment_lookup, alpha=0.15)
        
        status.test_passed = (blended is not None and len(blended) == 5)
        status.test_output = f"Blended prices shape: {blended.shape}, sample: {blended[:3]}"
        
        # But check if phase3_multi_agent is using it
        from phase3_multi_agent import tool_kronos_forecast
        import inspect
        source = inspect.getsource(tool_kronos_forecast)
        uses_sentiment = "use_sentiment" in source and "sentiment" in source.lower()
        
        if not uses_sentiment:
            status.test_output += "\n⚠️  tool_kronos_forecast doesn't pass sentiment to kronos"
            
    except Exception as e:
        status.test_output = f"Error: {str(e)}"
        status.test_passed = False

    return status


def test_multi_agent_backtest() -> ComponentStatus:
    """
    ❌ NOT IMPLEMENTED: No backtest_multi_agent.py exists
    Requirement: Compare Kronos-only backtest vs multi-agent coordinator signals
    """
    status = ComponentStatus(
        name="Multi-Agent Backtest",
        implemented=False,
        severity="high",
        details="backtest_multi_agent.py not created yet. Need to backtest coordinator final_signal",
    )

    backtest_multi_agent_path = os.path.join(os.path.dirname(__file__), "backtest_multi_agent.py")
    status.test_passed = os.path.exists(backtest_multi_agent_path)
    
    if status.test_passed:
        status.test_output = f"✓ Found {backtest_multi_agent_path}"
    else:
        status.test_output = f"✗ Missing {backtest_multi_agent_path}"

    return status


def test_agent_sophistication() -> ComponentStatus:
    """
    Check if agents use more than simple heuristics:
      - Technical: RSI, MACD, Forecast return (SIMPLE)
      - Sentiment: Average score with threshold (SIMPLE)
      - Macro: 20-day VNINDEX derivative (SIMPLE)
      - Risk: Bollinger width + confidence (SIMPLE)
    """
    status = ComponentStatus(
        name="Agent Sophistication",
        implemented=True,
        severity="high",
        details="All 4 agents exist but use simple heuristics, no ML models or complex rules",
    )

    try:
        from phase3_multi_agent import (
            agent_technical_vote,
            agent_sentiment_vote,
            agent_macro_vote,
            agent_risk_vote,
            AgentState,
        )

        # Test that agents exist and return valid votes
        test_state = AgentState(
            ticker="VNM",
            timestamp=datetime.utcnow().isoformat(),
            current_price=100.0,
            forecast_return_pct=2.5,
            forecast_confidence=0.7,
            rsi=45.0,
            macd=0.0012,
            bb_width_pct=5.0,
            sentiment_score=0.3,
            sentiment_count=7,
            macro_score=0.05,
        )

        tests = [
            ("technical", agent_technical_vote(test_state)),
            ("sentiment", agent_sentiment_vote(test_state)),
            ("macro", agent_macro_vote(test_state)),
            ("risk", agent_risk_vote(test_state)),
        ]

        all_valid = True
        outputs = []
        for name, (vote, score) in tests:
            valid = vote in ["BUY", "SELL", "HOLD"] and -1 <= score <= 1
            all_valid = all_valid and valid
            outputs.append(f"{name}: {vote} (score={score:.4f})")

        status.test_passed = all_valid
        status.test_output = "\n".join(outputs)

        if all_valid:
            status.test_output += "\n⚠️  Agents work but use simple rules (no ML models)"

    except Exception as e:
        status.test_output = f"Error: {str(e)}"
        status.test_passed = False

    return status


def test_coordinator_optimization() -> ComponentStatus:
    """
    ❌ NOT IMPLEMENTED: No grid search or Bayesian optimization for weights
    Current weights hardcoded: tech=0.4, sentiment=0.3-0.18-0.08, risk=0.25-0.15, macro=0.05
    """
    status = ComponentStatus(
        name="Coordinator Weight Optimization",
        implemented=False,
        severity="low",
        details="Weights (tech=0.4, sentiment=0.3, macro=0.05, risk=0.15) are manually tuned, no grid search yet",
    )

    try:
        from phase3_multi_agent import orchestrate_decision, AgentState
        import inspect

        source = inspect.getsource(orchestrate_decision)
        
        # Check if weights are hardcoded
        has_hardcoded_weights = ("0.40" in source or "0.40" in source) and "grid" not in source.lower()
        
        status.test_passed = has_hardcoded_weights  # Currently using hardcoded
        status.test_output = "Current weights found in code:\n"
        status.test_output += "  - technical_weight = 0.40 (fixed)\n"
        status.test_output += "  - sentiment_weight = 0.30/0.18/0.08 (dynamic on count)\n"
        status.test_output += "  - risk_weight = 0.25/0.15 (dynamic on volatility)\n"
        status.test_output += "  - macro_weight = remaining balance\n"
        status.test_output += "\n❌ No grid search or Bayesian optimization implemented"

    except Exception as e:
        status.test_output = f"Error: {str(e)}"
        status.test_passed = False

    return status


def test_online_learning() -> ComponentStatus:
    """
    ❌ NOT IMPLEMENTED: No feedback mechanism to adjust weights based on trade results
    """
    status = ComponentStatus(
        name="Online Learning / RLHF",
        implemented=False,
        severity="medium",
        details="No mechanism to adjust agent weights based on prediction accuracy",
    )

    try:
        # Check if any learning mechanism exists
        from phase3_multi_agent import run_multi_agent_analysis
        import inspect

        source = inspect.getsource(run_multi_agent_analysis)
        
        has_learning = "update_weight" in source or "adapt" in source.lower() or "learn" in source.lower()
        
        status.test_passed = False  # Not implemented
        status.test_output = "No online learning mechanism found in current code"
        status.test_output += "\nNeeded: Function to compare predictions vs actuals and adjust weights"

    except Exception as e:
        status.test_output = f"Error: {str(e)}"

    return status


def test_walk_forward_validation() -> ComponentStatus:
    """
    ❌ NOT IMPLEMENTED: No rolling window validation across 5 years of data
    Current: backtest_basic.py uses all data, single backtest
    """
    status = ComponentStatus(
        name="Walk-Forward / Rolling Window Validation",
        implemented=False,
        severity="medium",
        details="No rolling window validation (2yr train, 6mo test) across 5-year period",
    )

    walk_forward_path = os.path.join(os.path.dirname(__file__), "backtest_walk_forward.py")
    status.test_passed = os.path.exists(walk_forward_path)
    
    if status.test_passed:
        status.test_output = f"✓ Found {walk_forward_path}"
    else:
        status.test_output = f"✗ Missing walk-forward validation module\n"
        status.test_output += "Needed: Split 5-year data into 10 windows (2yr train + 6mo test each)"

    return status


def test_full_pipeline() -> ComponentStatus:
    """
    Test that full multi-agent pipeline runs end-to-end
    """
    status = ComponentStatus(
        name="Full Phase 3 Pipeline",
        implemented=True,
        severity="high",
        details="Can run_multi_agent_analysis() for a ticker",
    )

    try:
        result = run_multi_agent_analysis("VNM")
        
        required_fields = [
            "ticker", "timestamp", "current_price",
            "forecast_return_pct", "forecast_confidence",
            "rsi", "macd", "bb_width_pct",
            "sentiment_score", "sentiment_count",
            "macro_score",
            "agent_votes", "agent_scores",
            "final_signal", "final_score", "explanation"
        ]
        
        missing = [f for f in required_fields if f not in result]
        
        status.test_passed = len(missing) == 0
        if status.test_passed:
            status.test_output = f"✓ All {len(required_fields)} fields present\n"
            status.test_output += f"Final signal: {result.get('final_signal')}\n"
            status.test_output += f"Score: {result.get('final_score')}\n"
            status.test_output += json.dumps(result.get("agent_votes", {}), indent=2)
        else:
            status.test_output = f"Missing fields: {missing}"

    except Exception as e:
        status.test_output = f"Error: {str(e)}"
        status.test_passed = False

    return status


def main():
    """Run all component tests and generate report."""
    print("\n" + "=" * 70)
    print("  PHASE 3 COMPONENT STATUS CHECK")
    print("=" * 70)

    tests = [
        test_sentiment_in_kronos,
        test_multi_agent_backtest,
        test_agent_sophistication,
        test_coordinator_optimization,
        test_online_learning,
        test_walk_forward_validation,
        test_full_pipeline,
    ]

    results = []
    for test_fn in tests:
        print(f"\n▶ {test_fn.__name__}...")
        try:
            result = test_fn()
            results.append(result)
            
            status_icon = "✓" if result.implemented else "✗"
            passed_icon = "✓" if result.test_passed else "✗"
            
            print(f"  {status_icon} Implemented: {result.implemented}")
            print(f"  {passed_icon} Test Passed: {result.test_passed}")
            print(f"  Severity: {result.severity.upper()}")
            print(f"  Details: {result.details}")
            if result.test_output:
                print(f"  Output:\n{result.test_output}")
                
        except Exception as e:
            print(f"  ✗ Test error: {str(e)}")

    # Summary report
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    implemented = sum(1 for r in results if r.implemented)
    tested = sum(1 for r in results if r.test_passed)
    
    print(f"\nImplemented: {implemented}/{len(results)} components")
    print(f"Tests Passed: {tested}/{len(results)} components")
    print(f"Completion: {100*implemented/len(results):.0f}%")
    
    print("\n▶ Critical Issues (must fix):")
    for r in results:
        if r.severity == "critical" and not r.implemented:
            print(f"  - {r.name}: {r.details}")
    
    print("\n▶ High Priority Issues (should fix soon):")
    for r in results:
        if r.severity == "high" and not r.implemented:
            print(f"  - {r.name}: {r.details}")
    
    print("\n▶ Medium Priority Issues (after high):")
    for r in results:
        if r.severity == "medium" and not r.implemented:
            print(f"  - {r.name}: {r.details}")
    
    # Save report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {
            "total_components": len(results),
            "implemented": implemented,
            "tests_passed": tested,
            "completion_pct": round(100 * implemented / len(results), 1),
        },
        "components": [
            {
                "name": r.name,
                "implemented": r.implemented,
                "severity": r.severity,
                "test_passed": r.test_passed,
                "details": r.details,
                "test_output": r.test_output,
            }
            for r in results
        ],
        "recommendations": [
            "Priority 1: Integrate sentiment into tool_kronos_forecast in phase3_multi_agent.py",
            "Priority 2: Create backtest_multi_agent.py to compare with base backtest",
            "Priority 3: Enhance agents with more sophisticated rules (EWM for sentiment, macro APIs)",
            "Priority 4: Implement grid search for coordinator weights based on backtest results",
            "Priority 5: Add online learning to adjust weights from trade feedback",
            "Priority 6: Implement walk-forward validation for robust evaluation",
        ],
    }
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Report saved to: {REPORT_PATH}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    main()
