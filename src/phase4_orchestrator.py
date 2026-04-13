"""
phase4_orchestrator.py — Điều phối toàn bộ Phase 4: MLOps + RLHF + LLM.

run_phase4(ticker, mode):
  1. Thu thập macro data thực (macro_data.py)
  2. Chạy multi-agent analysis (phase3_multi_agent.py)
  3. Gọi LLM analyst (llm_analyst.py) — chỉ khi có đủ RAM
  4. Kiểm tra drift (mlops_pipeline.MarketDriftDetector)
  5. Cập nhật RLHF rewards (rlhf_engine.py)
  6. Sinh báo cáo → data/reports/json/phase4_report.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
PHASE4_REPORT_PATH = os.path.join(DATA_DIR, "reports", "json", "phase4_report.json")

os.makedirs(os.path.dirname(PHASE4_REPORT_PATH), exist_ok=True)


# ─── RAM guard ─────────────────────────────────────────────────────────────────

def _check_llm_ram(required_gb: float = 5.5) -> bool:
    """Kiểm tra RAM đủ để chạy LLM không."""
    try:
        from memory_guard import available_ram_gb
        avail = available_ram_gb()
        if avail < required_gb:
            logger.warning(
                f"RAM không đủ cho LLM: {avail:.1f} GB < {required_gb} GB yêu cầu. "
                f"Bỏ qua LLM analysis."
            )
            return False
        return True
    except Exception:
        return False


# ─── Phase 4 steps ─────────────────────────────────────────────────────────────

def step_macro(as_of_date: Optional[str] = None) -> dict:
    """Bước 1: Thu thập macro data."""
    try:
        from macro_data import get_macro_data
        data = get_macro_data(as_of_date=as_of_date)
        logger.info(f"  Macro source: {data.get('source')}, score: {data.get('macro_score')}")
        return {"status": "ok", "data": data}
    except Exception as e:
        logger.error(f"  Macro step failed: {e}")
        return {"status": "error", "error": str(e)}


def step_multi_agent(
    ticker: str,
    as_of_index: Optional[int] = None,
    use_llm: bool = False,
) -> dict:
    """Bước 2+3: Multi-agent analysis + LLM (nếu đủ RAM và use_llm=True)."""
    try:
        from phase3_multi_agent import run_multi_agent_analysis

        # RAM check trước khi kích hoạt LLM
        llm_enabled = use_llm and _check_llm_ram()
        if use_llm and not llm_enabled:
            logger.info("  LLM disabled due to insufficient RAM")

        analysis = run_multi_agent_analysis(
            ticker=ticker,
            as_of_index=as_of_index,
            use_llm=llm_enabled,
        )
        logger.info(
            f"  Signal: {analysis.get('final_signal')} "
            f"(score={analysis.get('final_score')}, "
            f"llm={'yes' if analysis.get('llm_analysis') else 'no'})"
        )
        return {"status": "ok", "analysis": analysis}
    except Exception as e:
        logger.error(f"  Multi-agent step failed: {e}")
        return {"status": "error", "error": str(e)}


def step_drift_check(ticker: str) -> dict:
    """Bước 4: Kiểm tra market drift."""
    try:
        from mlops_pipeline import MarketDriftDetector
        detector = MarketDriftDetector(ticker=ticker)
        result = detector.check_drift()
        if result["drift_detected"]:
            logger.warning(f"  DRIFT detected: {result['drift_reason']}")
        else:
            logger.info(f"  No drift (PSI={result['psi']:.3f})")
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"  Drift check failed: {e}")
        return {"status": "error", "error": str(e)}


def step_rlhf_update(ticker: str) -> dict:
    """Bước 5: Cập nhật RLHF rewards + adapted weights."""
    try:
        from rlhf_engine import run_rlhf_cycle
        result = run_rlhf_cycle(ticker=ticker)
        logger.info(
            f"  RLHF: {result['rewards_processed']} rewards processed, "
            f"{result['outcomes_filled']} outcomes filled"
        )
        return {"status": "ok", "result": result}
    except Exception as e:
        logger.error(f"  RLHF step failed: {e}")
        return {"status": "error", "error": str(e)}


def step_record_signal(
    ticker: str,
    analysis: dict,
    as_of_date: Optional[str] = None,
) -> dict:
    """Record signal vào RLHF store để theo dõi kết quả sau này."""
    try:
        from rlhf_engine import FeedbackStore
        if not analysis.get("final_signal"):
            return {"status": "skipped"}
        store = FeedbackStore()
        signal_date = as_of_date or datetime.utcnow().strftime("%Y-%m-%d")
        row_id = store.record_signal(
            ticker=ticker,
            signal_date=signal_date,
            signal=analysis.get("final_signal", "HOLD"),
            forecast_return_pct=analysis.get("forecast_return_pct") or 0.0,
            confidence=analysis.get("forecast_confidence") or 0.5,
            agent_scores=analysis.get("agent_scores"),
        )
        return {"status": "ok", "row_id": row_id}
    except Exception as e:
        logger.error(f"  Record signal failed: {e}")
        return {"status": "error", "error": str(e)}


# ─── Main orchestrator ─────────────────────────────────────────────────────────

def run_phase4(
    ticker: str = "VNM",
    mode: str = "full",
    use_llm: bool = False,
    as_of_index: Optional[int] = None,
    as_of_date: Optional[str] = None,
) -> dict:
    """
    Điều phối toàn bộ Phase 4.

    Args:
        ticker: Mã cổ phiếu
        mode: "full" | "analysis_only" | "drift_only" | "rlhf_only"
        use_llm: Bật LLM analysis (sẽ kiểm tra RAM trước khi gọi)
        as_of_index: Index điểm dữ liệu cho backtest
        as_of_date: Ngày giới hạn dữ liệu (YYYY-MM-DD) cho macro/sentiment

    Returns:
        Báo cáo tổng hợp Phase 4
    """
    started_at = datetime.utcnow().isoformat()
    logger.info(f"\n{'='*60}")
    logger.info(f"  Phase 4 Orchestrator — Ticker: {ticker}, Mode: {mode}")
    logger.info(f"{'='*60}")

    report: dict = {
        "ticker": ticker,
        "mode": mode,
        "started_at": started_at,
        "steps": {},
    }

    # ── Step 1: Macro data ──────────────────────────────────────────────────
    if mode in ("full", "analysis_only"):
        logger.info("\n[1/5] Macro data...")
        report["steps"]["macro"] = step_macro(as_of_date=as_of_date)

    # ── Step 2+3: Multi-agent + LLM ────────────────────────────────────────
    if mode in ("full", "analysis_only"):
        logger.info("\n[2/5] Multi-agent analysis...")
        ma_result = step_multi_agent(ticker, as_of_index=as_of_index, use_llm=use_llm)
        report["steps"]["multi_agent"] = ma_result
        if ma_result["status"] == "ok":
            report["analysis"] = ma_result["analysis"]

    # ── Step 3: Record signal ───────────────────────────────────────────────
    if mode in ("full", "analysis_only") and report.get("analysis"):
        logger.info("\n[3/5] Recording signal for RLHF tracking...")
        report["steps"]["record_signal"] = step_record_signal(
            ticker=ticker,
            analysis=report["analysis"],
            as_of_date=as_of_date,
        )

    # ── Step 4: Drift check ─────────────────────────────────────────────────
    if mode in ("full", "drift_only"):
        logger.info("\n[4/5] Market drift check...")
        report["steps"]["drift"] = step_drift_check(ticker)

    # ── Step 5: RLHF update ─────────────────────────────────────────────────
    if mode in ("full", "rlhf_only"):
        logger.info("\n[5/5] RLHF update...")
        report["steps"]["rlhf"] = step_rlhf_update(ticker)

    report["completed_at"] = datetime.utcnow().isoformat()
    report["status"] = "ok"

    # Save report
    try:
        with open(PHASE4_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"\n✓ Phase 4 report saved: {PHASE4_REPORT_PATH}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")

    return report


# ─── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Phase 4 Orchestrator")
    parser.add_argument("--ticker", default="VNM", help="Stock ticker (default: VNM)")
    parser.add_argument(
        "--mode",
        default="full",
        choices=["full", "analysis_only", "drift_only", "rlhf_only"],
        help="Execution mode",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM analysis (requires Ollama running + ≥5.5GB RAM)",
    )
    args = parser.parse_args()

    report = run_phase4(ticker=args.ticker, mode=args.mode, use_llm=args.use_llm)

    print("\n=== Phase 4 Summary ===")
    if report.get("analysis"):
        a = report["analysis"]
        print(f"  Ticker      : {a.get('ticker')}")
        print(f"  Signal      : {a.get('final_signal')} (score={a.get('final_score')})")
        print(f"  Forecast    : {a.get('forecast_return_pct', 0):+.2f}% (confidence={a.get('forecast_confidence', 0):.2f})")
        if a.get("llm_analysis"):
            print(f"\n  LLM Analysis:\n  {a['llm_analysis']}")

    if report["steps"].get("drift", {}).get("result", {}).get("drift_detected"):
        dr = report["steps"]["drift"]["result"]
        print(f"\n  ⚠️  DRIFT DETECTED: {dr['drift_reason']}")

    rlhf = report["steps"].get("rlhf", {}).get("result", {})
    if rlhf:
        print(f"\n  RLHF weights: {rlhf.get('adapted_weights')}")
