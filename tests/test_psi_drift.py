"""
test_psi_drift.py — Kiểm tra PSI drift detector.

Tests:
  - inject synthetic return spike → PSI > 0.2 → drift_detected=True
  - stable uniform returns → PSI < 0.1 → drift_detected=False
  - rolling Sharpe thấp → trigger drift mà không cần PSI cao
  - insufficient data → drift_detected=False (graceful)
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlops_pipeline import MarketDriftDetector


def _create_temp_parquet(returns: np.ndarray, ticker: str = "TEST") -> tuple[str, str]:
    """Tạo parquet tạm với chuỗi giá từ daily returns (%). Trả về (tmpdir, path)."""
    tmpdir = tempfile.mkdtemp()
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r / 100.0))
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    df = pd.DataFrame({"time": dates, "close": prices})
    path = os.path.join(tmpdir, f"{ticker}_history.parquet")
    df.to_parquet(path, engine="pyarrow", index=False)
    return tmpdir, path


class TestPSIDriftDetector(unittest.TestCase):

    def _make_detector(self, returns: np.ndarray, ticker: str = "TEST") -> "MarketDriftDetector":
        tmpdir, _ = _create_temp_parquet(returns, ticker)
        det = MarketDriftDetector(ticker=ticker)
        det._load_data(parquet_dir=tmpdir)
        return det

    def test_stable_returns_no_drift(self):
        """
        Returns dao động nhỏ quanh mức tích cực ổn định →
        PSI thấp (cùng phân phối) + Sharpe > 0.3 → drift_detected=False.
        """
        # Pattern lặp lại với mean dương, đủ variance để Sharpe >> 0.3
        # Mỗi chu kỳ 10 giá trị: mean ≈ 0.45%, std ≈ 0.2% → Sharpe ≈ 35 (annualized)
        unit = np.array([0.3, 0.5, 0.4, 0.7, 0.6, 0.3, 0.5, 0.4, 0.6, 0.7])
        returns = np.tile(unit, 40)  # 400 điểm, cùng phân phối ở mọi cửa sổ
        det = self._make_detector(returns)
        result = det.check_drift(recent_days=30, reference_days=90)
        self.assertFalse(
            result["drift_detected"],
            f"Stable repeating returns should not trigger drift. "
            f"PSI={result['psi']:.3f}, Sharpe30d={result['rolling_sharpe_30d']:.3f}, "
            f"reason={result.get('drift_reason')}",
        )

    def test_return_spike_triggers_psi_drift(self):
        """
        Recent window có return spike lớn → PSI > 0.2 → drift_detected=True.
        """
        rng = np.random.default_rng(0)
        # 120 ngày bình thường (reference)
        reference = rng.normal(0.05, 0.5, 120)
        # 30 ngày gần đây: biến động gấp 10 lần
        recent_spike = rng.normal(0.0, 5.0, 30)
        returns = np.concatenate([reference, recent_spike])
        det = self._make_detector(returns)
        result = det.check_drift(recent_days=30, reference_days=90)
        self.assertTrue(
            result["drift_detected"],
            f"Return spike should trigger drift. PSI={result['psi']:.3f}",
        )
        self.assertGreater(result["psi"], 0.2, f"PSI={result['psi']:.3f} should be > 0.2 for spike data")

    def test_low_sharpe_triggers_drift(self):
        """
        Returns liên tục âm (trending down) → Sharpe < 0.3 → drift_detected=True.
        Even if PSI is low.
        """
        # 150 ngày returns âm ổn định (consistent bear)
        returns = np.full(150, -0.3)  # -0.3% mỗi ngày, ổn định nhưng sharpe rất thấp
        det = self._make_detector(returns)
        result = det.check_drift(recent_days=30, reference_days=90)
        self.assertTrue(
            result["drift_detected"],
            f"Consistently negative returns should trigger low-Sharpe drift. "
            f"Sharpe30d={result['rolling_sharpe_30d']:.3f}",
        )
        self.assertLess(result["rolling_sharpe_30d"], MarketDriftDetector.SHARPE_THRESHOLD)

    def test_insufficient_data_graceful(self):
        """
        Ít hơn recent+reference ngày → drift_detected=False (không crash).
        """
        returns = np.array([0.1] * 50)  # Chỉ 50 ngày, cần 30+90=120
        det = self._make_detector(returns)
        result = det.check_drift(recent_days=30, reference_days=90)
        self.assertFalse(result["drift_detected"])
        self.assertIn(result.get("drift_reason"), ("insufficient_data", None))

    def test_missing_data_graceful(self):
        """Ticker không có data → không crash, drift_detected=False."""
        det = MarketDriftDetector(ticker="NONEXISTENT_TICKER_XYZ")
        det._load_data(parquet_dir="/tmp")  # No file there
        result = det.check_drift()
        self.assertFalse(result["drift_detected"])
        self.assertIn(result.get("drift_reason"), ("data_unavailable", "insufficient_data", None))

    def test_result_has_required_fields(self):
        """Kết quả check_drift có đủ các fields bắt buộc."""
        returns = np.random.default_rng(1).normal(0.05, 0.5, 200)
        det = self._make_detector(returns)
        result = det.check_drift()
        for field in ("ticker", "psi", "rolling_sharpe_30d", "drift_detected", "checked_at"):
            self.assertIn(field, result, f"Missing field '{field}' in drift result")

    def test_compute_psi_known_values(self):
        """Test PSI static method với giá trị đã biết."""
        # Hai phân phối giống nhau → PSI gần 0
        rng = np.random.default_rng(7)
        same1 = rng.normal(0, 1, 500)
        same2 = rng.normal(0, 1, 500)
        psi_same = MarketDriftDetector._compute_psi(same1, same2)
        self.assertLess(psi_same, 0.1, f"Same distributions should have low PSI, got {psi_same:.4f}")

        # Hai phân phối khác nhau nhiều → PSI cao
        different = rng.normal(5, 1, 200)  # shifted
        psi_diff = MarketDriftDetector._compute_psi(different, same1[:200])
        self.assertGreater(psi_diff, 0.2, f"Different distributions should have high PSI, got {psi_diff:.4f}")


class TestAutoRetrainScheduler(unittest.TestCase):
    """Smoke tests cho AutoRetrainScheduler."""

    def test_run_now_dry_run(self):
        """run_now() với dry_run=True không gọi retrain thực."""
        from mlops_pipeline import AutoRetrainScheduler
        import shutil, tempfile
        tmpdir = tempfile.mkdtemp()
        # Tạo dữ liệu giả cho VNM
        rng = np.random.default_rng(42)
        returns = rng.normal(0.05, 0.5, 200)
        prices = [100.0]
        for r in returns:
            prices.append(prices[-1] * (1 + r / 100.0))
        dates = pd.date_range("2020-01-01", periods=len(prices), freq="B")
        df = pd.DataFrame({"time": dates, "close": prices})
        df.to_parquet(os.path.join(tmpdir, "VNM_history.parquet"), engine="pyarrow", index=False)

        # Monkeypatch RAW_PQ_DIR
        import mlops_pipeline as mlops
        orig = mlops.RAW_PQ_DIR
        mlops.RAW_PQ_DIR = tmpdir
        try:
            scheduler = AutoRetrainScheduler(tickers=["VNM"], dry_run=True)
            results = scheduler.run_now()
            self.assertIsInstance(results, list)
            self.assertEqual(len(results), 1)
        finally:
            mlops.RAW_PQ_DIR = orig
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
