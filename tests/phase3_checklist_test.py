import json
import os
import sys
import traceback
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phase3_multi_agent import run_multi_agent_analysis

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_PATH = os.path.join(DATA_DIR, "reports", "json", "phase3_test_report.json")


def _check(condition: bool, name: str, detail: str):
    return {
        "name": name,
        "passed": bool(condition),
        "detail": detail,
    }


def run_phase3_checklist(ticker: str = "VNM") -> dict:
    checks = []
    analysis = {}
    error = None

    try:
        analysis = run_multi_agent_analysis(ticker)

        checks.append(_check("forecast_return_pct" in analysis, "kronos_tool", "Có output forecast_return_pct"))
        checks.append(_check("rsi" in analysis and "macd" in analysis, "technical_tool", "Có RSI/MACD"))
        checks.append(_check("sentiment_score" in analysis, "sentiment_tool", "Có sentiment_score từ DB"))
        checks.append(_check("macro_score" in analysis, "macro_tool", "Có macro proxy score"))
        checks.append(_check(isinstance(analysis.get("agent_votes"), dict), "agent_votes", "Có vote từ các tác tử"))
        checks.append(_check(analysis.get("final_signal") in {"BUY", "SELL", "HOLD"}, "coordinator_signal", "Coordinator trả tín hiệu hợp lệ"))
        checks.append(_check(isinstance(analysis.get("final_score"), (int, float)), "coordinator_score", "Coordinator trả điểm tổng hợp"))
    except Exception as e:
        error = f"{e}\n{traceback.format_exc()}"
        checks.append(_check(False, "pipeline_execution", "Phase 3 pipeline bị lỗi runtime"))

    passed = sum(1 for c in checks if c["passed"])
    total = len(checks)

    report = {
        "phase": "Phase 3 - Multi-Agent",
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "summary": {
            "passed": passed,
            "total": total,
            "pass_rate_pct": round((passed / total * 100.0), 2) if total else 0.0,
            "ready_for_phase3": passed >= 6,
        },
        "checks": checks,
        "analysis_output": analysis,
        "error": error,
        "effectiveness_note": (
            "Đánh giá hiệu quả cần theo dõi thêm qua backtest rolling và PnL thực tế; "
            "report này xác nhận checklist kỹ thuật và khả năng điều phối đa tác tử."
        ),
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    r = run_phase3_checklist("VNM")
    print(json.dumps(r["summary"], ensure_ascii=False, indent=2))
    print(f"Saved: {OUT_PATH}")
