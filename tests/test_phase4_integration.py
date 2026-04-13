"""
test_phase4_integration.py — Smoke test toàn bộ Phase 4 pipeline.

Tests:
  - run_phase4("VNM", mode="analysis_only") không crash
  - phase4_report.json được tạo với đủ fields bắt buộc
  - drift check trả về dict hợp lệ
  - mode="drift_only" và "rlhf_only" cũng chạy được
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

REQUIRED_ANALYSIS_FIELDS = {
    "ticker",
    "final_signal",
    "final_score",
    "forecast_return_pct",
    "forecast_confidence",
    "agent_votes",
    "agent_scores",
    "explanation",
}

REQUIRED_REPORT_FIELDS = {"ticker", "mode", "started_at", "completed_at", "status", "steps"}


class TestPhase4Pipeline(unittest.TestCase):

    def test_analysis_only_does_not_crash(self):
        """run_phase4 mode='analysis_only' hoàn thành không có exception."""
        from phase4_orchestrator import run_phase4
        report = run_phase4("VNM", mode="analysis_only", use_llm=False)
        self.assertEqual(report.get("status"), "ok")

    def test_report_has_required_fields(self):
        """phase4_report.json có đủ các fields bắt buộc."""
        from phase4_orchestrator import run_phase4, PHASE4_REPORT_PATH
        run_phase4("VNM", mode="analysis_only", use_llm=False)
        self.assertTrue(os.path.exists(PHASE4_REPORT_PATH), "phase4_report.json not created")
        with open(PHASE4_REPORT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        for field in REQUIRED_REPORT_FIELDS:
            self.assertIn(field, data, f"Missing field '{field}' in phase4_report.json")

    def test_analysis_contains_signal(self):
        """Phân tích phải trả về signal BUY/SELL/HOLD hợp lệ."""
        from phase4_orchestrator import run_phase4
        report = run_phase4("VNM", mode="analysis_only", use_llm=False)
        analysis = report.get("analysis", {})
        self.assertIn(
            analysis.get("final_signal"),
            ("BUY", "SELL", "HOLD"),
            f"Invalid signal: {analysis.get('final_signal')}",
        )

    def test_analysis_has_required_fields(self):
        """Analysis result có đủ fields bắt buộc."""
        from phase4_orchestrator import run_phase4
        report = run_phase4("VNM", mode="analysis_only", use_llm=False)
        analysis = report.get("analysis", {})
        for field in REQUIRED_ANALYSIS_FIELDS:
            self.assertIn(field, analysis, f"Missing field '{field}' in analysis")

    def test_drift_check_returns_valid_dict(self):
        """mode='drift_only' trả về kết quả drift hợp lệ."""
        from phase4_orchestrator import run_phase4
        report = run_phase4("VNM", mode="drift_only")
        drift_step = report.get("steps", {}).get("drift", {})
        self.assertEqual(drift_step.get("status"), "ok", f"Drift step failed: {drift_step}")
        result = drift_step.get("result", {})
        self.assertIn("psi", result)
        self.assertIn("drift_detected", result)
        self.assertIsInstance(result["psi"], float)
        self.assertIsInstance(result["drift_detected"], bool)

    def test_rlhf_only_mode(self):
        """mode='rlhf_only' không crash."""
        from phase4_orchestrator import run_phase4
        report = run_phase4("VNM", mode="rlhf_only")
        self.assertEqual(report.get("status"), "ok")
        rlhf_step = report.get("steps", {}).get("rlhf", {})
        self.assertEqual(rlhf_step.get("status"), "ok", f"RLHF step failed: {rlhf_step}")

    def test_llm_disabled_by_default(self):
        """use_llm=False → llm_analysis không được điền (tránh OOM)."""
        from phase4_orchestrator import run_phase4
        report = run_phase4("VNM", mode="analysis_only", use_llm=False)
        llm_val = report.get("analysis", {}).get("llm_analysis")
        self.assertIsNone(llm_val, "llm_analysis should be None when use_llm=False")

    def test_full_mode_completes(self):
        """mode='full' chạy tất cả 5 bước mà không crash."""
        from phase4_orchestrator import run_phase4
        report = run_phase4("VNM", mode="full", use_llm=False)
        self.assertEqual(report.get("status"), "ok")
        steps = report.get("steps", {})
        # Kiểm tra các bước quan trọng đều có kết quả
        for step_name in ("macro", "multi_agent", "drift", "rlhf"):
            self.assertIn(step_name, steps, f"Missing step '{step_name}'")
            self.assertEqual(
                steps[step_name].get("status"),
                "ok",
                f"Step '{step_name}' failed: {steps[step_name]}",
            )


class TestBacktestMultiAgentModule(unittest.TestCase):
    """Kiểm tra tests/backtest_multi_agent.py là module hợp lệ có thể import."""

    def test_module_importable(self):
        from backtest_multi_agent import run_multi_agent_backtest, BacktestConfig
        self.assertTrue(callable(run_multi_agent_backtest))

    def test_backtest_config_defaults(self):
        from backtest_multi_agent import BacktestConfig
        cfg = BacktestConfig()
        self.assertEqual(cfg.ticker, "VNM")
        self.assertEqual(cfg.cost_bps, 35.0)
        self.assertTrue(cfg.use_sentiment_filter)

    def test_run_returns_results_dict(self):
        """Backtest chạy và trả về dict có đủ metrics."""
        from backtest_multi_agent import run_multi_agent_backtest, BacktestConfig
        cfg = BacktestConfig(ticker="VNM", start_index=200)
        result = run_multi_agent_backtest(cfg)
        self.assertIn("results", result)
        metrics = result["results"]
        for key in ("sharpe_ratio", "win_rate_pct", "total_return_pct", "max_drawdown_pct"):
            self.assertIn(key, metrics, f"Missing metric: {key}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
