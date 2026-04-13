"""
mlops_pipeline.py — MLOps tự động: drift detection + auto-retrain scheduling.

MarketDriftDetector:
  - PSI (Population Stability Index) giữa 30 ngày gần nhất vs 90 ngày trước
  - Ngưỡng PSI > 0.2 → trigger retrain
  - Cũng trigger nếu rolling Sharpe 30d < 0.3

AutoRetrainScheduler:
  - APScheduler CronTrigger: chạy mỗi Chủ nhật 02:00
  - Nếu drift detected → gọi kronos_trainer
  - Ghi log vào data/reports/json/mlops_log.json
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MLOPS_LOG_PATH = os.path.join(DATA_DIR, "reports", "json", "mlops_log.json")
DRIFT_FLAG_PATH = os.path.join(DATA_DIR, "reports", "json", "drift_retrain_queue.json")
RAW_PQ_DIR = os.path.join(DATA_DIR, "raw", "parquet")

os.makedirs(os.path.dirname(MLOPS_LOG_PATH), exist_ok=True)


# ─── MarketDriftDetector ───────────────────────────────────────────────────────

class MarketDriftDetector:
    """
    Phát hiện market regime drift bằng PSI (Population Stability Index).

    PSI so sánh phân phối returns giữa hai cửa sổ thời gian:
      - Recent window: 30 ngày gần nhất
      - Reference window: 90 ngày trước đó

    Diễn giải PSI:
      PSI < 0.1  → Phân phối ổn định
      0.1-0.2   → Thay đổi nhỏ, theo dõi
      > 0.2     → Drift đáng kể → trigger retrain
    """

    PSI_THRESHOLD = 0.20
    SHARPE_THRESHOLD = 0.30
    N_BINS = 10

    def __init__(self, ticker: str = "VNM"):
        self.ticker = ticker
        self._df: Optional[pd.DataFrame] = None

    def _load_data(self, parquet_dir: str = RAW_PQ_DIR) -> bool:
        path = os.path.join(parquet_dir, f"{self.ticker}_history.parquet")
        if not os.path.exists(path):
            logger.warning(f"Data not found: {path}")
            return False
        try:
            df = pd.read_parquet(path, engine="pyarrow").sort_values("time")
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.dropna(subset=["time", "close"])
            df["return_pct"] = df["close"].pct_change() * 100.0
            self._df = df.dropna(subset=["return_pct"])
            return len(self._df) > 0
        except Exception as e:
            logger.error(f"Failed to load data for {self.ticker}: {e}")
            return False

    @staticmethod
    def _compute_psi(actual: np.ndarray, expected: np.ndarray, n_bins: int = 10) -> float:
        """
        PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
        Bins từ quantile của expected window.
        """
        if len(actual) < 2 or len(expected) < 2:
            return 0.0

        # Build bins from expected (reference) distribution
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(expected, percentiles)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0

        eps = 1e-6
        actual_counts, _ = np.histogram(actual, bins=bin_edges)
        expected_counts, _ = np.histogram(expected, bins=bin_edges)

        actual_pct = actual_counts / (len(actual) + eps)
        expected_pct = expected_counts / (len(expected) + eps)

        # Avoid division by zero / log(0)
        actual_pct = np.maximum(actual_pct, eps)
        expected_pct = np.maximum(expected_pct, eps)

        psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
        return round(psi, 6)

    @staticmethod
    def _compute_rolling_sharpe(returns: np.ndarray, window: int = 30) -> float:
        """Annualized Sharpe ratio cho rolling window."""
        if len(returns) < window:
            return 0.0
        r = returns[-window:]
        mean_r = np.mean(r)
        std_r = np.std(r)
        if std_r < 1e-9:
            return 0.0
        # Annualize: 252 trading days
        return float(mean_r / std_r * np.sqrt(252))

    def check_drift(self, recent_days: int = 30, reference_days: int = 90) -> dict:
        """
        Kiểm tra drift.

        Returns:
            {
                "ticker": str,
                "psi": float,
                "rolling_sharpe_30d": float,
                "drift_detected": bool,
                "drift_reason": str | None,
                "checked_at": str,
            }
        """
        if self._df is None and not self._load_data():
            return {
                "ticker": self.ticker,
                "psi": 0.0,
                "rolling_sharpe_30d": 0.0,
                "drift_detected": False,
                "drift_reason": "data_unavailable",
                "checked_at": datetime.utcnow().isoformat(),
            }

        returns = self._df["return_pct"].to_numpy()
        total = len(returns)

        if total < recent_days + reference_days:
            return {
                "ticker": self.ticker,
                "psi": 0.0,
                "rolling_sharpe_30d": 0.0,
                "drift_detected": False,
                "drift_reason": "insufficient_data",
                "checked_at": datetime.utcnow().isoformat(),
            }

        recent_returns = returns[-recent_days:]
        reference_returns = returns[-(recent_days + reference_days):-recent_days]

        psi = self._compute_psi(recent_returns, reference_returns, n_bins=self.N_BINS)
        sharpe = self._compute_rolling_sharpe(returns, window=recent_days)

        drift_detected = False
        drift_reason = None

        if psi > self.PSI_THRESHOLD:
            drift_detected = True
            drift_reason = f"PSI={psi:.3f} > threshold={self.PSI_THRESHOLD}"
        elif sharpe < self.SHARPE_THRESHOLD:
            drift_detected = True
            drift_reason = f"Sharpe30d={sharpe:.3f} < threshold={self.SHARPE_THRESHOLD}"

        return {
            "ticker": self.ticker,
            "psi": psi,
            "rolling_sharpe_30d": round(sharpe, 4),
            "drift_detected": drift_detected,
            "drift_reason": drift_reason,
            "checked_at": datetime.utcnow().isoformat(),
        }


# ─── Log helpers ───────────────────────────────────────────────────────────────

def _load_log() -> list:
    if os.path.exists(MLOPS_LOG_PATH):
        try:
            with open(MLOPS_LOG_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _append_log(entry: dict):
    log = _load_log()
    log.append(entry)
    # Keep last 500 entries
    if len(log) > 500:
        log = log[-500:]
    with open(MLOPS_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


# ─── AutoRetrainScheduler ──────────────────────────────────────────────────────

class AutoRetrainScheduler:
    """
    Lên lịch kiểm tra drift và retrain tự động dùng APScheduler.

    Usage:
        scheduler = AutoRetrainScheduler(tickers=["VNM", "VCB", "FPT"])
        scheduler.start()   # Non-blocking, chạy nền
        # ... app code ...
        scheduler.stop()
    """

    def __init__(
        self,
        tickers: list[str] = None,
        cron: dict = None,
        dry_run: bool = False,
    ):
        """
        Args:
            tickers: Danh sách ticker để kiểm tra drift.
            cron: APScheduler cron kwargs (mặc định: Chủ nhật 02:00).
            dry_run: Nếu True, log nhưng không chạy retrain thực.
        """
        self.tickers = tickers or ["VNM", "VCB", "FPT", "HPG", "TCB"]
        self.cron = cron or {"day_of_week": "sun", "hour": 2, "minute": 0}
        self.dry_run = dry_run
        self._scheduler = None

    def _run_check(self):
        """Được gọi bởi APScheduler theo lịch."""
        logger.info(f"[MLOps] Drift check started for {self.tickers}")
        results = []
        for ticker in self.tickers:
            detector = MarketDriftDetector(ticker=ticker)
            result = detector.check_drift()
            results.append(result)

            if result["drift_detected"]:
                logger.warning(f"[MLOps] Drift detected for {ticker}: {result['drift_reason']}")
                if not self.dry_run:
                    self._trigger_retrain(ticker, result)
                else:
                    logger.info(f"[MLOps] dry_run=True — skipping retrain for {ticker}")

        _append_log({
            "event": "scheduled_check",
            "checked_at": datetime.utcnow().isoformat(),
            "results": results,
        })
        return results

    def _trigger_retrain(self, ticker: str, drift_info: dict):
        """Đặt flag vào drift_retrain_queue.json để pipeline_manager đọc và trigger retrain.
        Không gọi finetune_kronos() trực tiếp — tránh race condition với pipeline_manager."""
        logger.info(f"[MLOps] Queuing drift-retrain flag for {ticker}...")
        try:
            queue: list = []
            if os.path.exists(DRIFT_FLAG_PATH):
                try:
                    with open(DRIFT_FLAG_PATH, encoding="utf-8") as f:
                        queue = json.load(f)
                except Exception:
                    queue = []
            if ticker not in [e.get("ticker") for e in queue]:
                queue.append({
                    "ticker": ticker,
                    "drift_info": drift_info,
                    "queued_at": datetime.now(timezone.utc).isoformat(),
                })
            with open(DRIFT_FLAG_PATH, "w", encoding="utf-8") as f:
                json.dump(queue, f, ensure_ascii=False, indent=2)
            log_entry = {
                "event": "retrain_queued",
                "ticker": ticker,
                "drift_info": drift_info,
                "triggered_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"[MLOps] Failed to queue retrain for {ticker}: {e}")
            log_entry = {
                "event": "retrain_queue_failed",
                "ticker": ticker,
                "drift_info": drift_info,
                "error": str(e),
                "triggered_at": datetime.now(timezone.utc).isoformat(),
            }

        _append_log(log_entry)

    def start(self):
        """Khởi động APScheduler background."""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
        except ImportError:
            logger.error("APScheduler chưa cài — chạy: pip install apscheduler")
            return

        self._scheduler = BackgroundScheduler()
        self._scheduler.add_job(
            self._run_check,
            CronTrigger(**self.cron),
            id="drift_check",
            max_instances=1,
            coalesce=True,
        )
        self._scheduler.start()
        logger.info(
            f"[MLOps] Scheduler started — cron={self.cron}, "
            f"tickers={self.tickers}, dry_run={self.dry_run}"
        )

    def stop(self):
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("[MLOps] Scheduler stopped")

    def run_now(self) -> list:
        """Chạy ngay lập tức (không cần lịch) — dùng cho testing."""
        return self._run_check()


# ─── Standalone check ──────────────────────────────────────────────────────────

def check_all_tickers(tickers: Optional[list[str]] = None) -> list[dict]:
    """
    Kiểm tra drift cho tất cả (hoặc danh sách) tickers.
    Hàm tiện ích dùng từ phase4_orchestrator.
    """
    if tickers is None:
        tickers = ["VNM", "VCB", "FPT", "HPG", "TCB", "MWG", "ACB"]
    results = []
    for ticker in tickers:
        detector = MarketDriftDetector(ticker=ticker)
        result = detector.check_drift()
        results.append(result)
        status = "⚠️  DRIFT" if result["drift_detected"] else "✓ Stable"
        print(f"  [{ticker}] PSI={result['psi']:.3f} Sharpe30d={result['rolling_sharpe_30d']:.3f} → {status}")
        if result["drift_detected"]:
            print(f"         Reason: {result['drift_reason']}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== Market Drift Detection ===")
    results = check_all_tickers()
    print(f"\nDrift detected in {sum(r['drift_detected'] for r in results)}/{len(results)} tickers")
