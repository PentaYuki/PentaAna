"""
test_reward_calculator.py — Kiểm tra chi tiết công thức reward RLHF.

Tests:
  - forecast +5%, actual +15%, conf 0.8 → reward > 0
  - forecast +5%, actual -3%, conf 0.8 → reward < 0 và penalty lớn
  - confidence thấp → penalty nhỏ hơn khi sai (proportional)
  - reward bị clip trong [-2.0, +2.0]
  - user_rating ảnh hưởng đến reward
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rlhf_engine import RewardCalculator, REWARD_CLIP


class TestRewardCalculatorFormula(unittest.TestCase):

    def test_correct_direction_large_return_positive(self):
        """forecast +5%, actual +15%, conf 0.8 → reward > 0."""
        r = RewardCalculator.compute_reward(5.0, 15.0, 0.8)
        self.assertGreater(r, 0, f"Expected positive reward, got {r}")

    def test_wrong_direction_negative(self):
        """forecast +5%, actual -3%, conf 0.8 → reward < 0."""
        r = RewardCalculator.compute_reward(5.0, -3.0, 0.8)
        self.assertLess(r, 0, f"Expected negative reward (penalty), got {r}")

    def test_penalty_proportional_to_confidence(self):
        """Sai chiều: confidence cao → penalty lớn; confidence thấp → penalty nhỏ."""
        penalty_high_conf = RewardCalculator.compute_reward(5.0, -3.0, 0.9)
        penalty_low_conf = RewardCalculator.compute_reward(5.0, -3.0, 0.1)
        self.assertLess(
            penalty_high_conf,
            penalty_low_conf,
            f"High-conf penalty ({penalty_high_conf}) should be more negative than low-conf ({penalty_low_conf})",
        )

    def test_larger_actual_return_bigger_reward(self):
        """Đúng chiều: actual +15% nên thưởng nhiều hơn +0.1%."""
        r_big = RewardCalculator.compute_reward(5.0, 15.0, 0.8)
        r_small = RewardCalculator.compute_reward(5.0, 0.1, 0.8)
        self.assertGreater(r_big, r_small, f"Big return ({r_big}) should have larger reward than small ({r_small})")

    def test_reward_clipped_at_plus_two(self):
        """reward không vượt quá +REWARD_CLIP."""
        r = RewardCalculator.compute_reward(5.0, 100.0, 1.0)
        self.assertLessEqual(r, REWARD_CLIP, f"Reward {r} exceeds clip {REWARD_CLIP}")

    def test_reward_clipped_at_minus_two(self):
        """reward không thấp hơn -REWARD_CLIP."""
        r = RewardCalculator.compute_reward(5.0, -100.0, 1.0)
        self.assertGreaterEqual(r, -REWARD_CLIP, f"Reward {r} below clip {-REWARD_CLIP}")

    def test_zero_forecast_gives_zero_reward(self):
        """forecast = 0 → sign = 0 → reward = 0."""
        r = RewardCalculator.compute_reward(0.0, 10.0, 0.8)
        self.assertEqual(r, 0.0, f"Zero forecast should give zero reward, got {r}")

    def test_user_rating_5_increases_positive_reward(self):
        """
        Đúng chiều + user_rating=5 → reward cao hơn user_rating=1.
        Note: khi base reward đã ở clip ceiling (2.0), formula là base*0.7 + user_component*0.3.
        Dùng actual_return nhỏ để base không bị clip.
        """
        # base = 1.0 * 1.0 * 0.5 = 0.5 (không bị clip)
        r_no_rating = RewardCalculator.compute_reward(5.0, 1.0, 0.5)
        r_rated_5 = RewardCalculator.compute_reward(5.0, 1.0, 0.5, user_rating=5.0)
        r_rated_1 = RewardCalculator.compute_reward(5.0, 1.0, 0.5, user_rating=1.0)
        self.assertGreater(r_rated_5, r_rated_1, "user_rating=5 should give higher reward than user_rating=1")

    def test_user_rating_1_decreases_positive_reward(self):
        """Đúng chiều outcome nhưng user_rating=1 → reward thấp hơn không có rating."""
        r_no_rating = RewardCalculator.compute_reward(5.0, 5.0, 0.7)
        r_rated_1 = RewardCalculator.compute_reward(5.0, 5.0, 0.7, user_rating=1.0)
        self.assertLessEqual(r_rated_1, r_no_rating, "user_rating=1 should reduce reward")

    def test_user_rating_3_is_neutral(self):
        """
        user_rating=3 → user_component=0 → reward = clip(base*0.7, ...) → không đổi chiều.
        Dùng actual nhỏ để base không bị clip và có thể so sánh rõ ràng.
        """
        # base = 1.0 * 1.0 * 0.5 = 0.5; with rating=3: clip(0.5*0.7+0.0, -2, 2) = 0.35
        r_no_rating = RewardCalculator.compute_reward(5.0, 1.0, 0.5)  # 0.5
        r_neutral = RewardCalculator.compute_reward(5.0, 1.0, 0.5, user_rating=3.0)  # 0.35
        # user_rating=3 scales reward to 70% of base (doesn’t flip direction)
        self.assertGreater(r_no_rating, 0, "Base reward should be positive")
        self.assertGreater(r_neutral, 0, "Neutral user_rating should preserve sign")

    def test_sell_signal_correct_direction(self):
        """forecast -5%, actual -8% → reward > 0 (đúng chiều sell)."""
        r = RewardCalculator.compute_reward(-5.0, -8.0, 0.7)
        self.assertGreater(r, 0, f"Correct SELL direction should be rewarded, got {r}")

    def test_sell_signal_wrong_direction(self):
        """forecast -5%, actual +3% → reward < 0."""
        r = RewardCalculator.compute_reward(-5.0, 3.0, 0.7)
        self.assertLess(r, 0, f"Wrong SELL direction should be penalized, got {r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
