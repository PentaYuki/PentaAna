"""
rlhf_engine.py — Reinforcement Learning from Human Feedback

FeedbackStore: SQLite lưu signal, actual outcome, user rating.
RewardCalculator: reward dựa trên magnitude + confidence (không binary).
WeightAdapter: cập nhật agent weights qua EWM với MIN_WEIGHT guard.
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "news.db")
RLHF_WEIGHTS_PATH = os.path.join(DATA_DIR, "reports", "json", "rlhf_weights.json")

os.makedirs(os.path.dirname(RLHF_WEIGHTS_PATH), exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
MIN_WEIGHT = 0.05   # Mỗi agent giữ tối thiểu 5%
MAX_WEIGHT = 0.60   # Không agent nào chiếm quá 60%
REWARD_CLIP = 2.0   # Giới hạn reward để tránh update quá lớn
EWM_ALPHA = 0.1     # Smoothing factor mặc định cho EWM weight update

# Per-agent EWM_ALPHA: sentiment phản ứng nhanh hơn technical/macro
AGENT_EWM_ALPHA: dict[str, float] = {
    "technical": 0.08,   # Chỉ số kỹ thuật — lagging, thích nghi chậm
    "sentiment": 0.15,   # Tin tức thay đổi nhanh — thích nghi nhanh hơn
    "macro": 0.05,       # Xu hướng vĩ mô bền vững — thích nghi chậm nhất
    "risk": 0.10,        # Mức mặc định
}

VNINDEX_PARQUET_PATH = os.path.join(DATA_DIR, "raw", "parquet", "VNINDEX_history.parquet")


def _load_vnindex_return_pct(signal_date: str, outcome_date: str) -> float:
    """VNINDEX return (%) giữa signal_date và outcome_date. Trả 0.0 nếu thiếu dữ liệu."""
    try:
        df = pd.read_parquet(VNINDEX_PARQUET_PATH, engine="pyarrow").sort_values("time")
        df["time"] = pd.to_datetime(df["time"])
        p0_rows = df[df["time"] <= pd.to_datetime(signal_date)]
        p1_rows = df[df["time"] <= pd.to_datetime(outcome_date)]
        if p0_rows.empty or p1_rows.empty:
            return 0.0
        p0 = float(p0_rows.iloc[-1]["close"])
        p1 = float(p1_rows.iloc[-1]["close"])
        return float((p1 - p0) / p0 * 100.0) if p0 > 0 else 0.0
    except Exception:
        return 0.0


def _ticker_weights_path(ticker: str) -> str:
    """Path cho file RLHF weights riêng theo từng ticker."""
    return os.path.join(DATA_DIR, "reports", "json", f"rlhf_weights_{ticker}.json")


DEFAULT_WEIGHTS = {
    "technical": 0.40,
    "sentiment": 0.25,
    "macro": 0.20,
    "risk": 0.15,
}

# ─── FeedbackStore ─────────────────────────────────────────────────────────────

class FeedbackStore:
    """
    Lưu feedback RLHF vào SQLite (cùng file news.db).
    Schema: rlhf_feedback(id, ticker, signal_date, ticker, signal,
                           forecast_return_pct, confidence, actual_return_pct,
                           user_rating, reward, recorded_at)
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_table()

    def _init_table(self):
        _dir = os.path.dirname(self.db_path)
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rlhf_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    signal_date TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    forecast_return_pct REAL,
                    confidence REAL,
                    agent_scores TEXT,
                    actual_return_pct REAL,
                    user_rating REAL,
                    reward REAL,
                    recorded_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.commit()

    def record_signal(
        self,
        ticker: str,
        signal_date: str,
        signal: str,
        forecast_return_pct: float,
        confidence: float,
        agent_scores: Optional[dict] = None,
    ) -> int:
        """Ghi lại signal tại thời điểm phát sinh. Trả về row id."""
        agent_scores_json = json.dumps(agent_scores or {})
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO rlhf_feedback
                   (ticker, signal_date, signal, forecast_return_pct, confidence, agent_scores)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ticker, signal_date, signal, forecast_return_pct, confidence, agent_scores_json),
            )
            conn.commit()
            return cur.lastrowid

    def update_outcome(
        self,
        row_id: int,
        actual_return_pct: float,
        user_rating: Optional[float] = None,
        vnindex_return_pct: float = 0.0,
    ):
        """Cập nhật kết quả thực tế và tính reward (alpha-based nếu có VNINDEX)."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT forecast_return_pct, confidence FROM rlhf_feedback WHERE id=?",
                (row_id,),
            ).fetchone()
            if row is None:
                return
            forecast_return_pct, confidence = row
            reward = RewardCalculator.compute_reward(
                forecast_return_pct=forecast_return_pct or 0.0,
                actual_return_pct=actual_return_pct,
                confidence=confidence or 0.5,
                user_rating=user_rating,
                vnindex_return_pct=vnindex_return_pct,
            )
            conn.execute(
                """UPDATE rlhf_feedback
                   SET actual_return_pct=?, user_rating=?, reward=?
                   WHERE id=?""",
                (actual_return_pct, user_rating, reward, row_id),
            )
            conn.commit()

    def get_pending_outcomes(self, outcome_delay_days: int = 30) -> list[dict]:
        """Trả về các signal chưa có actual_return_pct và đã đủ thời gian chờ."""
        cutoff_date = (datetime.utcnow() - timedelta(days=outcome_delay_days)).strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT id, ticker, signal_date, signal, forecast_return_pct, confidence
                   FROM rlhf_feedback
                   WHERE actual_return_pct IS NULL
                     AND signal_date <= ?
                   ORDER BY signal_date""",
                (cutoff_date,),
            ).fetchall()
        return [
            {
                "id": r[0],
                "ticker": r[1],
                "signal_date": r[2],
                "signal": r[3],
                "forecast_return_pct": r[4],
                "confidence": r[5],
            }
            for r in rows
        ]

    def get_recent_rewards(self, ticker: str, lookback_days: int = 90) -> list[dict]:
        """Lấy rewards gần đây để tính adapted weights.
        ticker="ALL" → trả về toàn bộ tickers.
        """
        since = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            if ticker == "ALL":
                rows = conn.execute(
                    """SELECT signal_date, signal, forecast_return_pct, confidence,
                              agent_scores, actual_return_pct, reward
                       FROM rlhf_feedback
                       WHERE signal_date >= ? AND reward IS NOT NULL
                       ORDER BY signal_date""",
                    (since,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT signal_date, signal, forecast_return_pct, confidence,
                              agent_scores, actual_return_pct, reward
                       FROM rlhf_feedback
                       WHERE ticker=? AND signal_date >= ? AND reward IS NOT NULL
                       ORDER BY signal_date""",
                    (ticker, since),
                ).fetchall()
        return [
            {
                "signal_date": r[0],
                "signal": r[1],
                "forecast_return_pct": r[2],
                "confidence": r[3],
                "agent_scores": json.loads(r[4] or "{}"),
                "actual_return_pct": r[5],
                "reward": r[6],
            }
            for r in rows
        ]


# ─── RewardCalculator ──────────────────────────────────────────────────────────

class RewardCalculator:
    """
    Reward dựa trên magnitude — phân biệt đúng 0.1% với đúng 15%.

    Formula:
        base_reward = actual_return_pct * sign(forecast_return) * confidence
        reward = clip(base_reward, -REWARD_CLIP, +REWARD_CLIP)

    Nếu có user_rating (1-5):
        reward = reward * 0.7 + (user_rating - 3) * 0.5  (normalized -1..+1 range)
        reward = clip(reward, -REWARD_CLIP, +REWARD_CLIP)
    """

    @staticmethod
    def compute_reward(
        forecast_return_pct: float,
        actual_return_pct: float,
        confidence: float = 0.5,
        user_rating: Optional[float] = None,
        vnindex_return_pct: float = 0.0,
    ) -> float:
        confidence = float(np.clip(confidence, 0.01, 1.0))
        # Alpha-based reward: trừ VNINDEX return để đo excess return (alpha).        # Nếu không có VNINDEX data (vnindex_return_pct=0.0) → hành vi giống cũ.
        alpha_return = actual_return_pct - vnindex_return_pct
        sign_forecast = float(np.sign(forecast_return_pct)) if forecast_return_pct != 0 else 0.0
        base_reward = float(alpha_return * sign_forecast * confidence)
        reward = float(np.clip(base_reward, -REWARD_CLIP, REWARD_CLIP))

        if user_rating is not None:
            # user_rating 1-5 → normalized -1..+1 (3 = neutral)
            user_component = float(np.clip((user_rating - 3.0) / 2.0, -1.0, 1.0))
            reward = float(np.clip(reward * 0.7 + user_component * 0.3, -REWARD_CLIP, REWARD_CLIP))

        return round(reward, 6)


# ─── WeightAdapter ─────────────────────────────────────────────────────────────

class WeightAdapter:
    """
    Cập nhật agent weights dựa trên reward signal (gradient-free, EWM).

    Bảo vệ khỏi weight collapse:
      - MIN_WEIGHT = 0.05 — mỗi agent giữ ít nhất 5%
      - MAX_WEIGHT = 0.60 — không agent nào chiếm quá 60%
      - Renormalize sau mỗi update để tổng = 1.0
    """

    def __init__(self, initial_weights: Optional[dict] = None):
        self.weights = dict(DEFAULT_WEIGHTS)
        if initial_weights:
            self.weights.update(initial_weights)
        self._normalize()

    def _normalize(self):
        """
        Clip to non-negative, apply [MIN_WEIGHT, MAX_WEIGHT] bounds, renormalize.
        Uses iterative lifting to guarantee every agent has ≥ MIN_WEIGHT.
        """
        keys = list(self.weights.keys())

        # Step 1: Clip to non-negative (EWM update can push weights below zero)
        for k in keys:
            self.weights[k] = max(0.0, float(self.weights[k]))

        total = sum(self.weights.values())
        if total <= 0:
            self.weights = dict(DEFAULT_WEIGHTS)
            return

        # Step 2: Normalize to sum=1
        for k in keys:
            self.weights[k] /= total

        # Step 3: Iteratively lift weights below MIN_WEIGHT
        for _ in range(20):  # max iterations
            below = [k for k in keys if self.weights[k] < MIN_WEIGHT]
            if not below:
                break
            # Fix below-min weights
            for k in below:
                self.weights[k] = MIN_WEIGHT
            # How much we "spent" on the floor
            floor_total = len(below) * MIN_WEIGHT
            remaining = 1.0 - floor_total
            free_keys = [k for k in keys if k not in below]
            if not free_keys or remaining <= 0:
                # Redistribute evenly
                eq = 1.0 / len(keys)
                for k in keys:
                    self.weights[k] = eq
                break
            free_sum = sum(self.weights[k] for k in free_keys)
            if free_sum <= 0:
                eq = remaining / len(free_keys)
                for k in free_keys:
                    self.weights[k] = eq
            else:
                scale = remaining / free_sum
                for k in free_keys:
                    self.weights[k] = min(self.weights[k] * scale, MAX_WEIGHT)

        # Step 4: Iteratively cap weights above MAX_WEIGHT and renormalize
        for _ in range(20):
            above = [k for k in keys if self.weights[k] > MAX_WEIGHT]
            if not above:
                break
            for k in above:
                self.weights[k] = MAX_WEIGHT
            used_by_capped = len(above) * MAX_WEIGHT
            remaining = 1.0 - used_by_capped
            free_keys_upper = [k for k in keys if k not in above]
            if not free_keys_upper or remaining <= 0:
                eq = 1.0 / len(keys)
                for k in keys:
                    self.weights[k] = eq
                break
            free_sum = sum(self.weights[k] for k in free_keys_upper)
            if free_sum <= 0:
                eq = remaining / len(free_keys_upper)
                for k in free_keys_upper:
                    self.weights[k] = max(eq, MIN_WEIGHT)
            else:
                scale = remaining / free_sum
                for k in free_keys_upper:
                    self.weights[k] = self.weights[k] * scale

        # Final renormalize
        total2 = sum(self.weights.values())
        if total2 > 0:
            for k in keys:
                self.weights[k] = round(self.weights[k] / total2, 6)

    def update(self, reward: float, agent_scores: dict, signal: str = "HOLD") -> dict:
        """
        Cập nhật weights theo hướng agent nào đóng góp vào kết quả.

        Nếu reward > 0: tăng weight các agent đồng ý với kết quả đúng.
        Nếu reward < 0: giảm weight các agent đồng ý với quyết định sai.

        agent_scores: {"technical": float, "sentiment": float, ...}
        signal: Tín hiệu cuối cùng đã phát ("BUY"/"SELL"/"HOLD") để tính
                directional attribution — agent vote đúng chiều được ghi nhận.
        """
        if not agent_scores or reward == 0:
            return dict(self.weights)

        scores_arr = np.array([agent_scores.get(k, 0.0) for k in self.weights], dtype="float64")
        abs_sum = np.sum(np.abs(scores_arr))
        if abs_sum < 1e-9:
            return dict(self.weights)

        # Directional attribution: agent có score cùng chiều signal được ghi nhận chính xác hơn.
        # BUY→+1, SELL→-1, HOLD→0 (fallback to magnitude-only contributions)
        signal_direction = 1.0 if signal == "BUY" else -1.0 if signal == "SELL" else 0.0
        if signal_direction != 0.0:
            alignment = scores_arr * signal_direction  # >0 nếu agent đồng ý với signal
            aligned_sum = np.sum(np.abs(alignment))
            contributions = alignment / aligned_sum if aligned_sum > 1e-9 else scores_arr / abs_sum
        else:
            contributions = scores_arr / abs_sum

        # EWM-style update với per-agent alpha (sentiment nhanh hơn macro)
        for i, k in enumerate(self.weights):
            alpha = AGENT_EWM_ALPHA.get(k, EWM_ALPHA)
            delta = alpha * reward * float(contributions[i])
            self.weights[k] = self.weights[k] + delta

        self._normalize()
        return dict(self.weights)

    def adapt_from_history(self, reward_history: list[dict]) -> dict:
        """
        Cập nhật weights từ lịch sử reward (batch update).
        reward_history: list of {"reward": float, "agent_scores": dict, "signal": str}
        """
        for record in reward_history:
            self.update(
                reward=record.get("reward", 0.0),
                agent_scores=record.get("agent_scores", {}),
                signal=record.get("signal", "HOLD"),
            )
        return dict(self.weights)

    def save(self, path: str = RLHF_WEIGHTS_PATH, ticker: Optional[str] = None):
        """Lưu weights. Nếu ticker được cung cấp, lưu vào file riêng theo ticker."""
        if ticker:
            path = _ticker_weights_path(ticker)
        data = {
            "weights": self.weights,
            "updated_at": datetime.utcnow().isoformat(),
            "min_weight": MIN_WEIGHT,
            "max_weight": MAX_WEIGHT,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str = RLHF_WEIGHTS_PATH, ticker: Optional[str] = None) -> "WeightAdapter":
        """Load weights. Ưu tiên file per-ticker nếu tồn tại, fallback về global."""
        if ticker:
            ticker_path = _ticker_weights_path(ticker)
            if os.path.exists(ticker_path):
                path = ticker_path
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                return cls(initial_weights=data.get("weights", {}))
            except Exception:
                pass
        return cls()


# ─── Auto-fill outcomes ────────────────────────────────────────────────────────

def fill_pending_outcomes(
    store: FeedbackStore,
    outcome_delay_days: int = 5,
    raw_parquet_dir: Optional[str] = None,
):
    """
    Tự động điền actual_return_pct cho các signal chưa có kết quả.
    Dùng giá đóng cửa sau outcome_delay_days từ parquet files.
    """
    if raw_parquet_dir is None:
        raw_parquet_dir = os.path.join(DATA_DIR, "raw", "parquet")

    pending = store.get_pending_outcomes(outcome_delay_days=outcome_delay_days)
    if not pending:
        return 0

    filled = 0
    for record in pending:
        ticker = record["ticker"]
        signal_date = record["signal_date"]
        parquet_path = os.path.join(raw_parquet_dir, f"{ticker}_history.parquet")
        if not os.path.exists(parquet_path):
            continue
        try:
            df = pd.read_parquet(parquet_path, engine="pyarrow").sort_values("time")
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.dropna(subset=["time"])

            signal_ts = pd.to_datetime(signal_date, errors="coerce")
            # Fix: BDay(N) thay vì timedelta(days=N) để tránh nhảy qua kỳ nghỉ dài
            outcome_ts = signal_ts + BDay(outcome_delay_days)
            outcome_str = str(outcome_ts.normalize().date())

            # Giá tại ngày signal
            at_signal = df[df["time"] <= signal_ts]
            # Giá tại khoảng outcome
            at_outcome = df[df["time"] <= outcome_ts]

            if at_signal.empty or at_outcome.empty:
                continue

            p0 = float(at_signal.iloc[-1]["close"])
            p1 = float(at_outcome.iloc[-1]["close"])
            if p0 <= 0:
                continue

            actual_return_pct = (p1 - p0) / p0 * 100.0
            # Alpha-based reward: lấy VNINDEX return cùng kỳ để normalize
            vnindex_ret = _load_vnindex_return_pct(signal_date, outcome_str)
            store.update_outcome(
                record["id"],
                actual_return_pct=actual_return_pct,
                vnindex_return_pct=vnindex_ret,
            )
            filled += 1
        except Exception:
            continue

    return filled


# ─── Public entrypoint ─────────────────────────────────────────────────────────

def run_rlhf_cycle(
    ticker: str,
    outcome_delay_days: int = 5,
    min_samples: int = 5,
) -> dict:
    """
    Chạy một chu kỳ RLHF:
    1. Fill pending outcomes từ price data
    2. Lấy reward history
    3. Cập nhật và lưu adapted weights per-ticker

    Args:
        ticker: Mã cổ phiếu
        outcome_delay_days: Số ngày giao dịch chờ kết quả
        min_samples: Tối thiểu số lượng reward cần có trước khi adapt weights

    Returns: {"adapted_weights": dict, "rewards_processed": int, "skipped": bool}
    """
    store = FeedbackStore()
    filled = fill_pending_outcomes(store, outcome_delay_days=outcome_delay_days)

    reward_history = store.get_recent_rewards(ticker, lookback_days=180)
    adapter = WeightAdapter.load(ticker=ticker)

    if len(reward_history) < min_samples:
        return {
            "adapted_weights": adapter.weights,
            "rewards_processed": len(reward_history),
            "outcomes_filled": filled,
            "skipped": True,
            "reason": f"Insufficient samples ({len(reward_history)} < {min_samples})",
            "updated_at": datetime.utcnow().isoformat(),
        }

    adapter.adapt_from_history(reward_history)
    adapter.save(ticker=ticker)

    return {
        "adapted_weights": adapter.weights,
        "rewards_processed": len(reward_history),
        "outcomes_filled": filled,
        "skipped": False,
        "updated_at": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    # Smoke test
    store = FeedbackStore()
    calculator = RewardCalculator()

    # Test reward formula
    print("=== Reward Calculator Test ===")
    cases = [
        (5.0, 15.0, 0.8, None),    # Đúng chiều, return lớn → reward cao
        (5.0, 0.1, 0.8, None),     # Đúng chiều, return nhỏ → reward nhỏ
        (5.0, -3.0, 0.8, None),    # Sai chiều với confidence cao → penalty nhiều
        (5.0, -3.0, 0.2, None),    # Sai chiều với confidence thấp → penalty ít
        (5.0, 15.0, 0.8, 5.0),     # Đúng + user rating 5 → reward cao nhất
        (5.0, 15.0, 0.8, 1.0),     # Đúng outcome nhưng user rating 1 → giảm
    ]
    for fc, ac, conf, ur in cases:
        r = calculator.compute_reward(fc, ac, conf, ur)
        print(f"  forecast={fc:+.1f}%, actual={ac:+.1f}%, conf={conf:.1f}, user_rating={ur} → reward={r:.4f}")

    # Test WeightAdapter
    print("\n=== WeightAdapter Test ===")
    adapter = WeightAdapter()
    print(f"  Initial: {adapter.weights}")

    # Simulate sideways market (many small signals)
    for _ in range(50):
        adapter.update(reward=0.0, agent_scores={"technical": 0.5, "sentiment": 0.1, "macro": 0.2, "risk": 0.2})
    print(f"  After 50 neutral updates: {adapter.weights}")
    assert all(v >= MIN_WEIGHT for v in adapter.weights.values()), "MIN_WEIGHT violated!"

    # Force sentiment to collapse then recover
    for _ in range(30):
        adapter.update(reward=-1.0, agent_scores={"technical": 0.1, "sentiment": 0.8, "macro": 0.05, "risk": 0.05})
    print(f"  After 30 bad sentiment signals: {adapter.weights}")
    assert adapter.weights["sentiment"] >= MIN_WEIGHT, "Sentiment weight collapsed below MIN_WEIGHT!"
    print(f"  ✓ MIN_WEIGHT guard working correctly")
