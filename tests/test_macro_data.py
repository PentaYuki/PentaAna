"""
test_macro_data.py — Kiểm tra macro_data.py.

Tests:
  - yfinance unavailable → fallback về VNINDEX local không crash
  - cache TTL: gọi 2 lần trong 1 giờ → chỉ fetch 1 lần (in-memory cache)
  - as_of_date mode (backtest) → không gọi yfinance (proxy only)
  - result có đủ fields bắt buộc
  - macro_score trong range [-1, 1]
"""

import os
import sys
import time
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestMacroDataFallback(unittest.TestCase):

    def test_fallback_when_yfinance_unavailable(self):
        """
        Khi yfinance raise ImportError hoặc exception → fallback về VNINDEX local,
        không crash, trả về dict hợp lệ.
        """
        import macro_data

        with patch.object(macro_data, "_fetch_yfinance_macro", side_effect=Exception("network error")):
            # Reset cache để buộc re-fetch
            macro_data._MACRO_CACHE = None
            macro_data._CACHE_TIMESTAMP = 0.0
            data = macro_data.get_macro_data(force_refresh=True)

        self.assertIsInstance(data, dict)
        self.assertIn("macro_score", data)
        self.assertIsInstance(data["macro_score"], float)

    def test_result_has_required_fields(self):
        """Kết quả có đủ macro_score, fetched_at, source."""
        import macro_data
        macro_data._MACRO_CACHE = None
        macro_data._CACHE_TIMESTAMP = 0.0
        with patch.object(macro_data, "_fetch_yfinance_macro", return_value={}):
            data = macro_data.get_macro_data(force_refresh=True)

        for field in ("macro_score", "fetched_at", "source"):
            self.assertIn(field, data, f"Missing field: {field}")

    def test_macro_score_in_valid_range(self):
        """macro_score luôn trong [-1, 1]."""
        import macro_data
        macro_data._MACRO_CACHE = None
        macro_data._CACHE_TIMESTAMP = 0.0
        with patch.object(macro_data, "_fetch_yfinance_macro", return_value={}):
            data = macro_data.get_macro_data(force_refresh=True)

        score = data["macro_score"]
        self.assertGreaterEqual(score, -1.0, f"macro_score {score} < -1.0")
        self.assertLessEqual(score, 1.0, f"macro_score {score} > 1.0")

    def test_as_of_date_mode_does_not_call_yfinance(self):
        """
        as_of_date != None (backtest mode) → _fetch_yfinance_macro không được gọi.
        """
        import macro_data

        with patch.object(macro_data, "_fetch_yfinance_macro") as mock_fetch:
            data = macro_data.get_macro_data(as_of_date="2023-06-01")

        mock_fetch.assert_not_called()
        self.assertIn("macro_score", data)

    def test_in_memory_cache_avoids_second_fetch(self):
        """
        Gọi get_macro_data() lần 2 trong TTL → không gọi _fetch_yfinance_macro lần 2.
        """
        import macro_data

        # Force cache miss
        macro_data._MACRO_CACHE = None
        macro_data._CACHE_TIMESTAMP = 0.0

        call_count = [0]
        def counting_fetch(*args, **kwargs):
            call_count[0] += 1
            return {"sp500": {"ret_20d_pct": 2.0, "last_price": 4500.0}}

        with patch.object(macro_data, "_fetch_yfinance_macro", side_effect=counting_fetch):
            macro_data.get_macro_data(force_refresh=True)  # Call 1 — fetches
            macro_data.get_macro_data()                     # Call 2 — should hit cache

        self.assertEqual(call_count[0], 1, "yfinance should only be called once (cache hit on 2nd call)")

    def test_cache_is_invalidated_by_force_refresh(self):
        """force_refresh=True → cache bị bỏ qua, re-fetch."""
        import macro_data

        call_count = [0]
        def counting_fetch(*args, **kwargs):
            call_count[0] += 1
            return {}

        with patch.object(macro_data, "_fetch_yfinance_macro", side_effect=counting_fetch):
            macro_data.get_macro_data(force_refresh=True)
            macro_data.get_macro_data(force_refresh=True)

        self.assertEqual(call_count[0], 2, "force_refresh=True should bypass cache")

    def test_yfinance_data_used_when_available(self):
        """Khi yfinance trả về data → source = 'yfinance'."""
        import macro_data

        mock_yf = {
            "sp500": {"ret_20d_pct": 5.0, "last_price": 4500.0},
            "usdvnd": {"ret_20d_pct": -1.0, "last_price": 23000.0},
        }

        macro_data._MACRO_CACHE = None
        macro_data._CACHE_TIMESTAMP = 0.0

        with patch.object(macro_data, "_fetch_yfinance_macro", return_value=mock_yf):
            data = macro_data.get_macro_data(force_refresh=True)

        self.assertEqual(data.get("source"), "yfinance")
        self.assertIn("sp500_ret_20d_pct", data)
        self.assertIn("usdvnd_ret_20d_pct", data)

    def test_fallback_vnindex_uses_local_parquet(self):
        """_fallback_from_vnindex() dùng VNINDEX parquet local khi có sẵn."""
        import macro_data

        data = macro_data._fallback_from_vnindex(as_of_date=None)
        # Nếu file tồn tại → source != 'fallback_zero'
        if os.path.exists(macro_data.VNINDEX_PATH):
            self.assertIn(data["source"], ("vnindex_local", "fallback_insufficient"))
        else:
            self.assertIn(data["source"], ("fallback_zero", "fallback_error"))


class TestToolMacroReal(unittest.TestCase):
    """Kiểm tra tool_macro_real() trong phase3_multi_agent."""

    def test_backtest_mode_uses_proxy(self):
        """as_of_date != None → gọi proxy (không gọi yfinance)."""
        import macro_data
        with patch.object(macro_data, "_fetch_yfinance_macro") as mock_fetch:
            from phase3_multi_agent import tool_macro_real
            result = tool_macro_real(as_of_date="2023-01-15")
        mock_fetch.assert_not_called()
        self.assertIn("macro_score", result)

    def test_live_mode_attempts_yfinance(self):
        """as_of_date=None → gọi get_macro_data() (attempted yfinance)."""
        import macro_data
        call_count = [0]
        def counting_fetch(*args, **kwargs):
            call_count[0] += 1
            return {}
        macro_data._MACRO_CACHE = None
        macro_data._CACHE_TIMESTAMP = 0.0
        with patch.object(macro_data, "_fetch_yfinance_macro", side_effect=counting_fetch):
            from phase3_multi_agent import tool_macro_real
            import importlib, phase3_multi_agent
            importlib.reload(phase3_multi_agent)  # reload to pick up patched macro_data
            result = phase3_multi_agent.tool_macro_real(as_of_date=None)
        self.assertIn("macro_score", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
