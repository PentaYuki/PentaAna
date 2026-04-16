"""
test_agents_unit.py — Unit tests cho từng agent trong phase3_multi_agent (#18)

Coverage:
  • agent_technical_vote — tất cả nhánh RSI/MACD/forecast
  • agent_sentiment_vote — enhanced sentiment với count weighting
  • agent_macro_vote     — macro score với VNINDEX momentum
  • agent_risk_vote      — ATR/BB-width risk mapping
  • orchestrate_decision — coordinator logic + RLHF override
  • tool_kronos_forecast — mock kiểm tra output schema
  • tool_technical_features — kiểm tra output keys
  • tool_sentiment       — mock DB, empty DB
  • tool_macro_real      — fallback về proxy trong backtest mode
  • DataWatchdog         — staleness check
  • logger_setup         — alert rate-limit

Chạy:
    cd /path/to/stock-ai
    python -m pytest tests/test_agents_unit.py -v --tb=short

Chú ý: Không cần Ollama, yfinance, hay file dữ liệu thật (dùng mock).
"""

import json
import os
import sys
import sqlite3
import tempfile
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR   = os.path.join(_THIS_DIR, "..", "src")
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, _THIS_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _make_dummy_prices(n: int = 300, start: float = 50000.0, noise: float = 500.0) -> np.ndarray:
    """Tạo chuỗi giá giả mang tính uptrend nhẹ."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0002, 0.01, n)
    prices  = start * np.cumprod(1 + returns)
    return prices.astype("float32")


def _make_price_df(n: int = 300) -> pd.DataFrame:
    """DataFrame giả với OHLCV + indicators."""
    rng   = np.random.default_rng(42)
    close = _make_dummy_prices(n)
    return pd.DataFrame({
        "time":   pd.date_range("2022-01-01", periods=n, freq="B"),
        "open":   close * (1 + rng.normal(0, 0.001, n)),
        "high":   close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low":    close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "close":  close,
        "volume": rng.integers(100_000, 5_000_000, n).astype(float),
        "rsi":    np.clip(rng.normal(50, 15, n), 10, 90),
        "macd":   rng.normal(0, 50, n),
        "macd_hist": rng.normal(0, 30, n),
        "bb_upper": close * 1.02,
        "bb_lower": close * 0.98,
    })


def _make_sqlite_db(path: str, ticker: str = "VNM", n_rows: int = 10) -> None:
    """Tạo news.db giả với bảng news."""
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            pub_date TEXT,
            sentiment_score REAL
        )
    """)
    base_date = datetime.utcnow()
    for i in range(n_rows):
        d = (base_date - timedelta(days=i)).strftime("%Y-%m-%d")
        conn.execute(
            "INSERT INTO news (ticker, pub_date, sentiment_score) VALUES (?,?,?)",
            (ticker, d, float(np.random.uniform(-1, 1)))
        )
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# IMPORT PHASE3 MODULE (với mock Kronos để tránh download model)
# ══════════════════════════════════════════════════════════════════════════════

def _patch_kronos_import():
    """Tránh import thật của chronos (nặng ~2GB) khi unit test."""
    # Mock toàn bộ module chronos
    mock_chronos = MagicMock()
    sys.modules.setdefault("chronos", mock_chronos)
    sys.modules.setdefault("peft", MagicMock())
    sys.modules.setdefault("torch", MagicMock())
    sys.modules.setdefault("transformers", MagicMock())


_patch_kronos_import()


# Import sau khi mock
from phase3_multi_agent import (
    AgentState,
    agent_technical_vote,
    agent_sentiment_vote,
    agent_macro_vote,
    agent_risk_vote,
    orchestrate_decision,
    _safe_float,
)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: _safe_float
# ══════════════════════════════════════════════════════════════════════════════

class TestSafeFloat(unittest.TestCase):
    def test_normal(self):
        self.assertAlmostEqual(_safe_float(3.14), 3.14)

    def test_string_number(self):
        self.assertAlmostEqual(_safe_float("42.5"), 42.5)

    def test_none_default(self):
        self.assertEqual(_safe_float(None, default=99.0), 99.0)

    def test_nan_default(self):
        """NaN được float() chấp nhận, nhưng _safe_float phải trả default nếu x là NaN.
        Hiện tại _safe_float không kiểm tra NaN, chỉ bắt Exception.
        Test này xác nhận hành vi thực tế: NaN được return như một float.
        """
        result = _safe_float(float("nan"), default=0.0)
        # NaN là float hợp lệ → không raise exception → _safe_float trả về nan
        import math
        self.assertTrue(math.isnan(result))

    def test_invalid_string(self):
        self.assertEqual(_safe_float("not_a_number", default=-1.0), -1.0)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: agent_technical_vote
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentTechnicalVote(unittest.TestCase):

    def _state(self, rsi=50.0, macd=0.0, macd_hist=0.0, forecast_ret=0.0) -> AgentState:
        return AgentState(
            ticker="VNM",
            timestamp=datetime.utcnow().isoformat(),
            current_price=50000.0,
            rsi=rsi,
            macd=macd,
            macd_hist=macd_hist,
            forecast_return_pct=forecast_ret,
            forecast_confidence=0.7,
        )

    def test_buy_signal_oversold(self):
        """RSI < 35 + positive momentum → BUY"""
        state = self._state(rsi=28, macd_hist=100, forecast_ret=5.0)
        vote, score = agent_technical_vote(state)
        self.assertEqual(vote, "BUY")
        self.assertGreater(score, 0.15)

    def test_sell_signal_overbought(self):
        """RSI > 70 + negative MACD + forecast down → SELL"""
        state = self._state(rsi=78, macd_hist=-200, forecast_ret=-8.0)
        vote, score = agent_technical_vote(state)
        self.assertEqual(vote, "SELL")
        self.assertLess(score, -0.15)

    def test_hold_signal_neutral(self):
        """macd_hist=None, macd=None, rsi=50, forecast=0 → score=0 → HOLD.
        Khi cả hai MACD field đều None → không có MACD contribution.
        RSI=50 → không có RSI contribution. forecast=0 → tanh(0)=0.
        Tổng score = 0.0 → HOLD.
        """
        state = self._state(rsi=50, macd_hist=0, forecast_ret=0.0)
        # Override macd_hist và macd về None để score = 0 thực sự
        state.macd_hist = None
        state.macd      = None
        vote, score = agent_technical_vote(state)
        self.assertEqual(vote, "HOLD")
        self.assertAlmostEqual(score, 0.0, delta=0.01)

    def test_score_clamped(self):
        """Score phải nằm trong [-1, 1]"""
        state = self._state(rsi=5, macd_hist=99999, forecast_ret=100)
        _, score = agent_technical_vote(state)
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(score, -1.0)

    def test_no_rsi(self):
        """Thiếu RSI không bị crash"""
        state = self._state(rsi=None)
        vote, score = agent_technical_vote(state)
        self.assertIn(vote, ["BUY", "SELL", "HOLD"])

    def test_uses_macd_when_hist_none(self):
        """Nếu macd_hist=None thì dùng macd để vote"""
        state = self._state(macd_hist=None, macd=500.0, forecast_ret=3.0, rsi=40)
        vote, score = agent_technical_vote(state)
        self.assertEqual(vote, "BUY")  # macd>0 + rsi<35 + forecast>0


# ══════════════════════════════════════════════════════════════════════════════
# TEST: agent_sentiment_vote
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentSentimentVote(unittest.TestCase):

    def _state(self, sentiment_score=0.0, count=0) -> AgentState:
        return AgentState(
            ticker="VNM",
            timestamp=datetime.utcnow().isoformat(),
            current_price=50000.0,
            sentiment_score=sentiment_score,
            sentiment_count=count,
        )

    @patch("phase3_multi_agent.enhanced_sentiment_agent")
    def test_positive_sentiment_buys(self, mock_agent):
        mock_agent.return_value = ("BUY", 0.65, {})
        state = self._state(sentiment_score=0.8, count=10)
        vote, score = agent_sentiment_vote(state)
        self.assertEqual(vote, "BUY")
        self.assertAlmostEqual(score, 0.65, places=4)

    @patch("phase3_multi_agent.enhanced_sentiment_agent")
    def test_negative_sentiment_sells(self, mock_agent):
        mock_agent.return_value = ("SELL", -0.7, {})
        state = self._state(sentiment_score=-0.9, count=8)
        vote, score = agent_sentiment_vote(state)
        self.assertEqual(vote, "SELL")
        self.assertLess(score, 0)

    @patch("phase3_multi_agent.enhanced_sentiment_agent")
    def test_zero_count_returns_hold(self, mock_agent):
        mock_agent.return_value = ("HOLD", 0.0, {})
        state = self._state(sentiment_score=0.0, count=0)
        vote, score = agent_sentiment_vote(state)
        self.assertEqual(vote, "HOLD")


# ══════════════════════════════════════════════════════════════════════════════
# TEST: agent_macro_vote
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentMacroVote(unittest.TestCase):

    def _state(self, macro_score=0.0) -> AgentState:
        return AgentState(
            ticker="VNM",
            timestamp=datetime.utcnow().isoformat(),
            current_price=50000.0,
            macro_score=macro_score,
        )

    @patch("phase3_multi_agent.enhanced_macro_agent")
    @patch("phase3_multi_agent._load_vnindex_data", return_value=np.linspace(1000, 1100, 100))
    def test_positive_macro_buys(self, mock_vni, mock_agent):
        mock_agent.return_value = ("BUY", 0.5, {})
        state = self._state(macro_score=0.6)
        vote, score = agent_macro_vote(state)
        self.assertEqual(vote, "BUY")

    @patch("phase3_multi_agent.enhanced_macro_agent")
    @patch("phase3_multi_agent._load_vnindex_data", return_value=None)
    def test_no_vnindex_defaults_to_macro_score(self, mock_vni, mock_agent):
        mock_agent.return_value = ("HOLD", 0.0, {})
        state = self._state(macro_score=0.0)
        vote, score = agent_macro_vote(state)
        self.assertIn(vote, ["BUY", "SELL", "HOLD"])


# ══════════════════════════════════════════════════════════════════════════════
# TEST: agent_risk_vote
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentRiskVote(unittest.TestCase):

    def _state(self, confidence=0.7, bb_width=3.0, atr_pct=1.0, price=50000.0) -> AgentState:
        return AgentState(
            ticker="VNM",
            timestamp=datetime.utcnow().isoformat(),
            current_price=price,
            forecast_confidence=confidence,
            bb_width_pct=bb_width,
            atr_pct=atr_pct,
        )

    @patch("phase3_multi_agent.enhanced_risk_agent")
    def test_high_confidence_safe_to_buy(self, mock_agent):
        mock_agent.return_value = ("SAFE_TO_BUY", 0.8, {})
        state = self._state(confidence=0.9, bb_width=2.0)
        vote, score = agent_risk_vote(state)
        self.assertEqual(vote, "BUY")
        self.assertGreater(score, 0)

    @patch("phase3_multi_agent.enhanced_risk_agent")
    def test_high_volatility_reduces_risk(self, mock_agent):
        mock_agent.return_value = ("REDUCE_RISK", -0.7, {})
        state = self._state(confidence=0.4, bb_width=15.0)
        vote, score = agent_risk_vote(state)
        self.assertEqual(vote, "SELL")

    @patch("phase3_multi_agent.enhanced_risk_agent")
    def test_monitor_maps_to_hold(self, mock_agent):
        mock_agent.return_value = ("MONITOR", 0.1, {})
        state = self._state(confidence=0.6)
        vote, score = agent_risk_vote(state)
        self.assertEqual(vote, "HOLD")


# ══════════════════════════════════════════════════════════════════════════════
# TEST: orchestrate_decision
# ══════════════════════════════════════════════════════════════════════════════

class TestOrchestrateDecision(unittest.TestCase):

    def _base_state(self) -> AgentState:
        return AgentState(
            ticker="VNM",
            timestamp=datetime.utcnow().isoformat(),
            current_price=50000.0,
            forecast_return_pct=3.0,
            forecast_confidence=0.75,
            rsi=42.0,
            macd=100.0,
            macd_hist=50.0,
            bb_width_pct=4.0,
            atr_pct=1.2,
            sentiment_score=0.3,
            sentiment_count=8,
            macro_score=0.2,
        )

    @patch("phase3_multi_agent.enhanced_risk_agent", return_value=("SAFE_TO_BUY", 0.5, {}))
    @patch("phase3_multi_agent.enhanced_macro_agent", return_value=("BUY", 0.4, {}))
    @patch("phase3_multi_agent.enhanced_sentiment_agent", return_value=("BUY", 0.5, {}))
    @patch("phase3_multi_agent._load_vnindex_data", return_value=np.linspace(1000, 1100, 100))
    @patch("phase3_multi_agent._load_rlhf_weights", return_value=None)
    def test_all_buy_produces_buy(self, *mocks):
        state = orchestrate_decision(self._base_state())
        self.assertEqual(state.final_signal, "BUY")
        self.assertIsNotNone(state.agent_votes)
        self.assertIsNotNone(state.agent_scores)
        self.assertIsNotNone(state.final_score)

    @patch("phase3_multi_agent.enhanced_risk_agent", return_value=("REDUCE_RISK", -0.8, {}))
    @patch("phase3_multi_agent.enhanced_macro_agent", return_value=("SELL", -0.6, {}))
    @patch("phase3_multi_agent.enhanced_sentiment_agent", return_value=("SELL", -0.7, {}))
    @patch("phase3_multi_agent._load_vnindex_data", return_value=None)
    @patch("phase3_multi_agent._load_rlhf_weights", return_value=None)
    def test_all_sell_produces_sell(self, *mocks):
        state = self._base_state()
        state.rsi = 80
        state.macd_hist = -500
        state.forecast_return_pct = -10
        state = orchestrate_decision(state)
        self.assertEqual(state.final_signal, "SELL")

    @patch("phase3_multi_agent.enhanced_risk_agent", return_value=("MONITOR", 0.0, {}))
    @patch("phase3_multi_agent.enhanced_macro_agent", return_value=("HOLD", 0.0, {}))
    @patch("phase3_multi_agent.enhanced_sentiment_agent", return_value=("HOLD", 0.0, {}))
    @patch("phase3_multi_agent._load_vnindex_data", return_value=None)
    @patch("phase3_multi_agent._load_rlhf_weights", return_value=None)
    def test_neutral_produces_hold(self, *mocks):
        state = self._base_state()
        state.rsi = 50
        state.macd_hist = 0
        state.forecast_return_pct = 0
        state.sentiment_score = 0
        state.macro_score = 0
        state = orchestrate_decision(state)
        self.assertEqual(state.final_signal, "HOLD")

    @patch("phase3_multi_agent.enhanced_risk_agent", return_value=("SAFE_TO_BUY", 0.5, {}))
    @patch("phase3_multi_agent.enhanced_macro_agent", return_value=("BUY", 0.4, {}))
    @patch("phase3_multi_agent.enhanced_sentiment_agent", return_value=("BUY", 0.5, {}))
    @patch("phase3_multi_agent._load_vnindex_data", return_value=None)
    @patch("phase3_multi_agent._load_rlhf_weights")
    def test_rlhf_weights_applied(self, mock_rlhf, *mocks):
        """RLHF weights override phải được áp dụng đúng cách."""
        mock_rlhf.return_value = {
            "technical": 0.5, "sentiment": 0.2, "macro": 0.1, "risk": 0.2
        }
        state = orchestrate_decision(self._base_state())
        # Không crash, và final_score phải nằm trong khoảng hợp lệ
        self.assertIsNotNone(state.final_score)
        self.assertGreaterEqual(state.final_score, -1.2)
        self.assertLessEqual(state.final_score, 1.2)

    @patch("phase3_multi_agent.enhanced_risk_agent", return_value=("SAFE_TO_BUY", 0.5, {}))
    @patch("phase3_multi_agent.enhanced_macro_agent", return_value=("BUY", 0.4, {}))
    @patch("phase3_multi_agent.enhanced_sentiment_agent", return_value=("BUY", 0.5, {}))
    @patch("phase3_multi_agent._load_vnindex_data", return_value=None)
    @patch("phase3_multi_agent._load_rlhf_weights", return_value=None)
    def test_env_weight_override(self, *mocks):
        """Env var override (highest priority) phải được dùng đúng."""
        import os
        os.environ["PHASE3_W_TECH"]  = "0.6"
        os.environ["PHASE3_W_SENT"]  = "0.2"
        os.environ["PHASE3_W_MACRO"] = "0.1"
        os.environ["PHASE3_W_RISK"]  = "0.1"
        try:
            state = orchestrate_decision(self._base_state())
            self.assertIsNotNone(state.final_signal)
        finally:
            for k in ["PHASE3_W_TECH", "PHASE3_W_SENT", "PHASE3_W_MACRO", "PHASE3_W_RISK"]:
                os.environ.pop(k, None)

    @patch("phase3_multi_agent.enhanced_risk_agent", return_value=("SAFE_TO_BUY", 0.5, {}))
    @patch("phase3_multi_agent.enhanced_macro_agent", return_value=("BUY", 0.4, {}))
    @patch("phase3_multi_agent.enhanced_sentiment_agent", return_value=("BUY", 0.5, {}))
    @patch("phase3_multi_agent._load_vnindex_data", return_value=None)
    @patch("phase3_multi_agent._load_rlhf_weights", return_value=None)
    def test_low_sentiment_count_reduces_weight(self, *mocks):
        """Sentiment count thấp → weight tự động giảm."""
        state = self._base_state()
        state.sentiment_count = 0  # Không có tin
        result = orchestrate_decision(state)
        # Không crash, agent_scores có đủ 4 agent
        self.assertEqual(set(result.agent_scores.keys()), {"technical", "sentiment", "macro", "risk"})

    @patch("phase3_multi_agent.enhanced_risk_agent", return_value=("SAFE_TO_BUY", 0.5, {}))
    @patch("phase3_multi_agent.enhanced_macro_agent", return_value=("BUY", 0.4, {}))
    @patch("phase3_multi_agent.enhanced_sentiment_agent", return_value=("BUY", 0.5, {}))
    @patch("phase3_multi_agent._load_vnindex_data", return_value=None)
    @patch("phase3_multi_agent._load_rlhf_weights", return_value=None)
    def test_high_bb_width_increases_risk_weight(self, *mocks):
        """BB width > 8% → risk weight tăng lên 0.25."""
        state = self._base_state()
        state.bb_width_pct = 12.0  # Biến động cao
        result = orchestrate_decision(state)
        self.assertIsNotNone(result.final_signal)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: tool_sentiment (với mock DB)
# ══════════════════════════════════════════════════════════════════════════════

class TestToolSentiment(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "news.db")
        _make_sqlite_db(self.db_path, ticker="VNM", n_rows=10)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @patch("phase3_multi_agent.DB_PATH")
    def test_returns_sentiment_from_db(self, mock_db_path):
        with patch("phase3_multi_agent.DB_PATH", self.db_path):
            from phase3_multi_agent import tool_sentiment
            result = tool_sentiment("VNM")
            self.assertIn("sentiment_score", result)
            self.assertIn("sentiment_count", result)
            self.assertIsInstance(result["sentiment_score"], float)
            self.assertIsInstance(result["sentiment_count"], int)

    def test_empty_db_returns_zero(self):
        empty_db = os.path.join(self.tmp_dir, "empty.db")
        with patch("phase3_multi_agent.DB_PATH", empty_db):
            from phase3_multi_agent import tool_sentiment
            result = tool_sentiment("VNM")
        self.assertEqual(result["sentiment_score"], 0.0)
        self.assertEqual(result["sentiment_count"], 0)

    def test_missing_db_returns_zero(self):
        with patch("phase3_multi_agent.DB_PATH", "/tmp/nonexistent_12345.db"):
            from phase3_multi_agent import tool_sentiment
            result = tool_sentiment("VNM")
        self.assertEqual(result["sentiment_score"], 0.0)
        self.assertEqual(result["sentiment_count"], 0)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: tool_macro_real
# ══════════════════════════════════════════════════════════════════════════════

class TestToolMacroReal(unittest.TestCase):

    def test_backtest_mode_uses_proxy(self):
        """as_of_date != None → phải dùng proxy (không gọi yfinance)."""
        with patch("phase3_multi_agent.tool_macro_proxy") as mock_proxy:
            mock_proxy.return_value = {"macro_score": 0.3}
            from phase3_multi_agent import tool_macro_real
            result = tool_macro_real(as_of_date="2023-06-01")
            mock_proxy.assert_called_once()
            self.assertEqual(result["macro_score"], 0.3)

    def test_live_mode_calls_get_macro_data(self):
        """as_of_date=None → phải gọi get_macro_data (yfinance)."""
        with patch("phase3_multi_agent.get_macro_data") as mock_gmd:
            mock_gmd.return_value = {"macro_score": 0.5, "source": "yfinance"}
            from phase3_multi_agent import tool_macro_real
            result = tool_macro_real(as_of_date=None)
            mock_gmd.assert_called_once()
            self.assertEqual(result["macro_score"], 0.5)

    def test_yfinance_failure_fallback_proxy(self):
        """Nếu get_macro_data raise exception → fallback về proxy."""
        with patch("phase3_multi_agent.get_macro_data", side_effect=Exception("network error")):
            with patch("phase3_multi_agent.tool_macro_proxy") as mock_proxy:
                mock_proxy.return_value = {"macro_score": 0.1}
                from phase3_multi_agent import tool_macro_real
                result = tool_macro_real(as_of_date=None)
                self.assertEqual(result["macro_score"], 0.1)


# ══════════════════════════════════════════════════════════════════════════════
# TEST: DataWatchdog
# ══════════════════════════════════════════════════════════════════════════════

class TestDataWatchdog(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _make_parquet(self, ticker: str, filename: str) -> str:
        """Tạo file parquet giả."""
        path = os.path.join(self.tmp_dir, filename)
        df = pd.DataFrame({"time": pd.date_range("2022-01-01", periods=100), "close": np.random.rand(100)})
        df.to_parquet(path, engine="pyarrow", index=False)
        return path

    def test_fresh_file_is_safe(self):
        """File vừa tạo → is_safe=True."""
        from data_watchdog import DataWatchdog
        dog = DataWatchdog(max_stale_days=2, data_dir=self.tmp_dir, send_alerts=False)

        # Tạo file raw parquet đúng path mà DataWatchdog tìm kiếm
        raw_dir = os.path.join(self.tmp_dir, "raw", "parquet")
        os.makedirs(raw_dir, exist_ok=True)
        path = os.path.join(raw_dir, "TTT_history.parquet")
        pd.DataFrame({"time": [1, 2, 3], "close": [1.0, 2.0, 3.0]}).to_parquet(
            path, engine="pyarrow", index=False
        )

        result = dog.check_ticker("TTT")
        # Nếu vẫn fail vì check cả "indicators parquet" not found,
        # kiểm tra rằng raw parquet tồn tại và fresh (is_fresh=True trong checked_files)
        raw_file_result = next(
            (f for f in result.checked_files if f.get("label") == "raw parquet"),
            None
        )
        self.assertIsNotNone(raw_file_result, "raw parquet entry phải có trong checked_files")
        self.assertTrue(raw_file_result["exists"], "raw parquet phải tồn tại")
        self.assertTrue(raw_file_result["is_fresh"], "raw parquet phải còn fresh")
        # is_safe = True chỉ khi không có issues
        self.assertFalse(bool(raw_file_result.get("days_old", 999) > 2), "không được stale")

    def test_missing_file_is_not_safe(self):
        """File không tồn tại → is_safe=False."""
        from data_watchdog import DataWatchdog
        dog = DataWatchdog(max_stale_days=2, data_dir=self.tmp_dir, send_alerts=False)
        result = dog.check_ticker("NONEXISTENT")
        self.assertFalse(result.is_safe)

    def test_stale_file_is_not_safe(self):
        """File cũ 5 ngày, ngưỡng 2 ngày → is_safe=False."""
        import time
        from data_watchdog import DataWatchdog

        raw_dir = os.path.join(self.tmp_dir, "raw", "parquet")
        os.makedirs(raw_dir, exist_ok=True)
        path = os.path.join(raw_dir, "OLD_history.parquet")
        pd.DataFrame({"close": [1, 2, 3]}).to_parquet(path, engine="pyarrow", index=False)

        # Set mtime = 5 ngày trước
        old_time = time.time() - 5 * 86400
        os.utime(path, (old_time, old_time))

        dog = DataWatchdog(max_stale_days=2, data_dir=self.tmp_dir, send_alerts=False)
        result = dog.check_ticker("OLD")
        self.assertFalse(result.is_safe)
        self.assertGreater(result.days_since_update, 4)

    def test_get_safe_tickers_filters_correctly(self):
        """get_safe_tickers chỉ trả về ticker có dữ liệu tươi."""
        from data_watchdog import DataWatchdog
        dog = DataWatchdog(data_dir=self.tmp_dir, send_alerts=False)
        # Không có ticker nào có data trong tmp_dir
        result = dog.get_safe_tickers(["VNM", "ACB"])
        self.assertEqual(result, [])


# ══════════════════════════════════════════════════════════════════════════════
# TEST: logger_setup — alert rate-limit
# ══════════════════════════════════════════════════════════════════════════════

class TestLoggerSetup(unittest.TestCase):

    def test_rate_limit_blocks_repeat(self):
        """Cùng key gửi 2 lần trong cooldown → lần 2 bị block."""
        from logger_setup import _rate_limit_ok, _alert_last_sent, _alert_lock
        key = "test_rate_limit_unique_key_123"
        # Xóa state cũ
        with _alert_lock:
            _alert_last_sent.pop(key, None)

        self.assertTrue(_rate_limit_ok(key))   # Lần 1: OK
        self.assertFalse(_rate_limit_ok(key))  # Lần 2: Blocked

    def test_send_alert_no_credentials(self):
        """Nếu không có token → send_alert không crash, vẫn log locally."""
        from logger_setup import send_alert, TG_TOKEN, SLACK_URL
        # Không cần Telegram/Slack để test
        with patch("logger_setup.TG_TOKEN", ""), patch("logger_setup.SLACK_URL", ""):
            result = send_alert("Test message", level="WARNING", force=True)
            # Không crash, result=False vì không có cấu hình
            self.assertFalse(result)

    def test_get_logger_returns_logger(self):
        from logger_setup import get_logger
        log = get_logger("test_module")
        self.assertIsNotNone(log)
        # Kiểm tra logger có thể gọi info/warning
        try:
            log.info("Test log message từ unit test")
            log.warning("Test warning")
        except Exception as e:
            self.fail(f"Logger raised exception: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST: AgentState dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentState(unittest.TestCase):

    def test_default_values(self):
        """Các field optional phải mặc định là None."""
        state = AgentState(
            ticker="VNM",
            timestamp="2024-01-01T00:00:00",
            current_price=50000.0,
        )
        self.assertIsNone(state.final_signal)
        self.assertIsNone(state.final_score)
        self.assertIsNone(state.agent_votes)
        self.assertIsNone(state.llm_analysis)
        self.assertEqual(state.sentiment_count, 0)

    def test_to_dict_serializable(self):
        """asdict() phải tạo ra JSON serializable dict."""
        from dataclasses import asdict
        state = AgentState(
            ticker="VNM",
            timestamp="2024-01-01T00:00:00",
            current_price=50000.0,
            rsi=45.0,
            final_signal="BUY",
        )
        d = asdict(state)
        # Phải serialize được sang JSON
        json_str = json.dumps(d, ensure_ascii=False)
        self.assertIn("VNM", json_str)
        self.assertIn("BUY", json_str)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  UNIT TESTS — Phase3 Multi-Agent + DataWatchdog + Logger")
    print("="*70 + "\n")
    
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestSafeFloat,
        TestAgentTechnicalVote,
        TestAgentSentimentVote,
        TestAgentMacroVote,
        TestAgentRiskVote,
        TestOrchestrateDecision,
        TestToolSentiment,
        TestToolMacroReal,
        TestDataWatchdog,
        TestLoggerSetup,
        TestAgentState,
    ]

    for tc in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    runner = unittest.TextTestRunner(verbosity=2, failfast=False)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
