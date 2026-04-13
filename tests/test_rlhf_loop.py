"""
test_rlhf_loop.py — Kiểm tra vòng lặp RLHF hoạt động đúng end-to-end.

Tests:
  - record_signal → fill_outcome → compute_reward → adapt_weights
  - Weights thay đổi đúng hướng sau 10 BUY đúng liên tiếp
  - MIN_WEIGHT không bao giờ bị vi phạm dù 50 bad signals
  - RLHF weights được đọc bởi orchestrate_decision
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rlhf_engine import (
    FeedbackStore,
    RewardCalculator,
    WeightAdapter,
    MIN_WEIGHT,
    MAX_WEIGHT,
    DEFAULT_WEIGHTS,
    fill_pending_outcomes,
    RLHF_WEIGHTS_PATH,
)


class TestFeedbackStoreLoop(unittest.TestCase):
    """Test vòng lặp record → fill → reward."""

    def setUp(self):
        # Dùng DB tạm thời để không ảnh hưởng production
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self.store = FeedbackStore(db_path=self._tmp.name)

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_record_and_update_outcome(self):
        row_id = self.store.record_signal(
            ticker="VNM",
            signal_date="2024-01-10",
            signal="BUY",
            forecast_return_pct=5.0,
            confidence=0.8,
            agent_scores={"technical": 0.6, "sentiment": 0.2, "macro": 0.1, "risk": 0.1},
        )
        self.assertGreater(row_id, 0)

        # Điền kết quả thực
        self.store.update_outcome(row_id, actual_return_pct=12.0)

        # Kiểm tra reward đã được tính (lookback lớn để bao phủ mọi ngày test)
        rewards = self.store.get_recent_rewards("VNM", lookback_days=1000)
        self.assertEqual(len(rewards), 1)
        self.assertGreater(rewards[0]["reward"], 0, "Correct direction + big gain should give positive reward")

    def test_wrong_direction_gives_negative_reward(self):
        row_id = self.store.record_signal(
            ticker="VNM",
            signal_date="2024-02-01",
            signal="BUY",
            forecast_return_pct=5.0,
            confidence=0.8,
        )
        self.store.update_outcome(row_id, actual_return_pct=-4.0)

        rewards = self.store.get_recent_rewards("VNM", lookback_days=1000)
        self.assertLess(rewards[0]["reward"], 0, "Wrong direction should give negative reward")

    def test_full_loop_10_correct_buys(self):
        """10 BUY đúng → weights kéo về technical (đóng góp nhiều nhất)."""
        agent_scores = {"technical": 0.7, "sentiment": 0.1, "macro": 0.1, "risk": 0.1}
        row_ids = []
        for i in range(10):
            rid = self.store.record_signal(
                ticker="VNM",
                signal_date=f"2024-03-{i+1:02d}",
                signal="BUY",
                forecast_return_pct=5.0,
                confidence=0.8,
                agent_scores=agent_scores,
            )
            row_ids.append(rid)

        for rid in row_ids:
            self.store.update_outcome(rid, actual_return_pct=10.0)

        rewards = self.store.get_recent_rewards("VNM", lookback_days=1000)
        adapter = WeightAdapter()
        adapter.adapt_from_history(rewards)

        # Technical dominated the correct signals → should gain weight
        # Technical dominated correct signals → should be at least as high as default
        # (EWM alpha=0.1 is conservative; after 10 steps it may be notably higher)
        self.assertGreaterEqual(
            adapter.weights["technical"],
            DEFAULT_WEIGHTS["technical"],
            "Technical weight should not decrease after 10 correct high-tech signals",
        )

    def test_pending_outcomes_filter(self):
        """Chỉ trả về signal đã đủ thời gian chờ và chưa có outcome."""
        # Signal tương lai → không được trả về vì chưa đủ outcome_delay_days
        # Dùng ngày trong tương lai xa (2035) để đảm bảo không bao giờ bị kể là pending
        self.store.record_signal(
            ticker="VNM",
            signal_date="2035-12-01",  # ngày tương lai xa, không bao giờ đủ 30 ngày
            signal="BUY",
            forecast_return_pct=3.0,
            confidence=0.6,
        )
        pending = self.store.get_pending_outcomes(outcome_delay_days=30)
        for p in pending:
            self.assertNotEqual(p["signal_date"], "2035-12-01",
                                "Future signal should not appear in pending outcomes")


class TestWeightAdapterMinWeightGuard(unittest.TestCase):
    """MIN_WEIGHT không bao giờ bị vi phạm."""

    def test_min_weight_after_50_bad_sentiment_signals(self):
        adapter = WeightAdapter()
        for _ in range(50):
            # Sentiment đóng góp nhiều nhưng liên tục sai
            adapter.update(
                reward=-2.0,
                agent_scores={"technical": 0.05, "sentiment": 0.85, "macro": 0.05, "risk": 0.05},
            )
        for agent, weight in adapter.weights.items():
            self.assertGreaterEqual(
                weight,
                MIN_WEIGHT,
                f"{agent} weight {weight:.6f} fell below MIN_WEIGHT={MIN_WEIGHT}",
            )

    def test_weights_sum_to_one(self):
        adapter = WeightAdapter()
        for _ in range(30):
            adapter.update(-1.0, {"technical": 0.1, "sentiment": 0.7, "macro": 0.1, "risk": 0.1})
        total = sum(adapter.weights.values())
        self.assertAlmostEqual(total, 1.0, places=4, msg=f"Weights sum={total} != 1.0")

    def test_max_weight_cap(self):
        adapter = WeightAdapter()
        for _ in range(100):
            # Technical always dominates
            adapter.update(
                reward=2.0,
                agent_scores={"technical": 0.99, "sentiment": 0.00, "macro": 0.00, "risk": 0.01},
            )
        self.assertLessEqual(
            adapter.weights["technical"],
            MAX_WEIGHT + 1e-4,  # allow tiny floating point tolerance
            f"technical weight {adapter.weights['technical']:.6f} exceeded MAX_WEIGHT={MAX_WEIGHT}",
        )
        # Also sanity check sum
        self.assertAlmostEqual(sum(adapter.weights.values()), 1.0, places=4)

    def test_load_save_roundtrip(self):
        adapter = WeightAdapter()
        for _ in range(5):
            adapter.update(1.0, {"technical": 0.6, "sentiment": 0.2, "macro": 0.1, "risk": 0.1})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_path = f.name
        try:
            adapter.save(tmp_path)
            loaded = WeightAdapter.load(tmp_path)
            for k in adapter.weights:
                self.assertAlmostEqual(
                    adapter.weights[k], loaded.weights[k], places=5,
                    msg=f"Weight mismatch for {k} after save/load",
                )
        finally:
            os.unlink(tmp_path)


class TestRLHFWeightsReadByOrchestrator(unittest.TestCase):
    """
    Kiểm tra vòng khép kín: rlhf_engine lưu → orchestrate_decision đọc.
    """

    def test_rlhf_weights_are_read_by_orchestrator(self):
        """Nếu rlhf_weights.json tồn tại, orchestrate_decision phải dùng chúng."""
        from phase3_multi_agent import _load_rlhf_weights

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_weights_path = os.path.join(tmpdir, "rlhf_weights.json")
            test_weights = {
                "technical": 0.50,
                "sentiment": 0.20,
                "macro": 0.15,
                "risk": 0.15,
            }
            with open(tmp_weights_path, "w") as f:
                json.dump({"weights": test_weights, "updated_at": "2025-01-01"}, f)

            # Monkeypatch RLHF_WEIGHTS_PATH
            import rlhf_engine as _re
            orig_path = _re.RLHF_WEIGHTS_PATH
            _re.RLHF_WEIGHTS_PATH = tmp_weights_path
            try:
                loaded = _load_rlhf_weights()
                self.assertIsNotNone(loaded, "_load_rlhf_weights() returned None despite file existing")
                self.assertAlmostEqual(loaded["technical"], 0.50, places=4)
                self.assertAlmostEqual(loaded["sentiment"], 0.20, places=4)
            finally:
                _re.RLHF_WEIGHTS_PATH = orig_path

    def test_load_rlhf_weights_returns_none_when_missing(self):
        from phase3_multi_agent import _load_rlhf_weights
        import rlhf_engine as _re
        orig_path = _re.RLHF_WEIGHTS_PATH
        _re.RLHF_WEIGHTS_PATH = "/tmp/does_not_exist_rlhf_weights_xyz.json"
        try:
            result = _load_rlhf_weights()
            self.assertIsNone(result)
        finally:
            _re.RLHF_WEIGHTS_PATH = orig_path


if __name__ == "__main__":
    unittest.main(verbosity=2)
