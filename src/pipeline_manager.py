import logging
import os
import json
import time
from pathlib import Path
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SENTINEL_FILE = Path(os.path.join(BASE_DIR, "data", ".sentiment_done"))
DRIFT_FLAG_PATH = Path(os.path.join(BASE_DIR, "data", "reports", "json", "drift_retrain_queue.json"))
LOG_DIR = Path(os.path.join(BASE_DIR, "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "pipeline_manager.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("pipeline_manager")


def job_build_analyzed_data():
    """Tạo data/analyzed/{indicators,with_indicators} từ toàn bộ raw parquet."""
    raw_dir = Path(os.path.join(BASE_DIR, "data", "raw", "parquet"))
    out_ind = Path(os.path.join(BASE_DIR, "data", "analyzed", "indicators"))
    out_full = Path(os.path.join(BASE_DIR, "data", "analyzed", "with_indicators"))
    out_ind.mkdir(parents=True, exist_ok=True)
    out_full.mkdir(parents=True, exist_ok=True)

    try:
        from technical_indicators import add_technical_indicators
    except Exception as e:
        logger.error("[JOB ANALYZED] Không import được technical_indicators: %s", e, exc_info=True)
        return

    files = sorted(raw_dir.glob("*_history.parquet"))
    if not files:
        logger.warning("[JOB ANALYZED] Không có file parquet trong %s", raw_dir)
        return

    built = 0
    for fp in files:
        ticker = fp.name.replace("_history.parquet", "")
        try:
            df = pd.read_parquet(fp, engine="pyarrow")
            df_ta = add_technical_indicators(df)
            if df_ta.empty:
                continue

            full_path = out_full / f"{ticker}_with_indicators.csv"
            ind_path = out_ind / f"{ticker}_indicators.csv"

            indicator_cols = [
                c for c in [
                    "time", "ticker", "rsi", "macd", "macd_sig", "macd_hist",
                    "sma_20", "sma_50", "ema_12", "ema_26",
                    "bb_upper", "bb_lower", "bb_mid", "atr", "obv", "vol_sma",
                    "golden_cross", "rsi_oversold", "rsi_overbought",
                ] if c in df_ta.columns
            ]

            df_ta.to_csv(full_path, index=False)
            df_ta[indicator_cols].to_csv(ind_path, index=False)
            built += 1
        except Exception as e:
            logger.error("[JOB ANALYZED] Lỗi khi build %s: %s", ticker, e, exc_info=True)

    logger.info("[JOB ANALYZED] Hoàn tất: %d/%d ticker", built, len(files))


def job_run_sentiment():
    logger.info("[JOB 00:00] Bắt đầu chấm điểm Sentiment hàng loạt (Batch Scoring)...")
    SENTINEL_FILE.unlink(missing_ok=True)  # Xóa cờ hiệu cũ

    try:
        from sentiment_analyzer import batch_score_news
        batch_score_news()
        SENTINEL_FILE.touch()
        logger.info("[JOB 00:00] Hoàn thành! Đã cắm cờ Sentinel.")
    except Exception as e:
        logger.error("[JOB 00:00] Thất bại: %s", e, exc_info=True)


def job_crawl_news():
    """Mỗi 2 giờ: crawl tin tiếng Việt (CafeF, VNExpress) + tiếng Anh (Reuters, Yahoo)."""
    logger.info("[JOB CRAWL] Bắt đầu thu thập tin tức (mỗi 2 giờ)...")
    try:
        from pipeline import WATCHLIST
    except ImportError:
        WATCHLIST = ["VNM", "HPG", "VCB", "FPT", "TCB", "MWG", "ACB"]
    try:
        from news_crawler import crawl_all, crawl_english_rss
        vi_results = crawl_all(WATCHLIST)
        en_count = crawl_english_rss()
        total = sum(vi_results.values()) + en_count
        logger.info("[JOB CRAWL] %d bài mới (VI: %d, EN: %d)", total, sum(vi_results.values()), en_count)
    except Exception as e:
        logger.error("[JOB CRAWL] Lỗi crawl: %s", e, exc_info=True)


def job_train_kronos():
    logger.info("[JOB 02:00] Bắt đầu Train mô hình Kronos...")

    # 1. Chống Race Condition bằng Sentinel File
    if not SENTINEL_FILE.exists():
        logger.warning("[JOB 02:00] Chưa thấy cờ `.sentiment_done`. Bỏ qua huấn luyện để tránh OOM.")
        return

    logger.info("[JOB 02:00] Sentinel File mở khóa. Xóa cờ...")
    SENTINEL_FILE.unlink()

    # 2. Xoá RAM an toàn bằng Memory Guard
    from memory_guard import prepare_for_training
    if not prepare_for_training():
        return

    # 3. Tiến hành Training
    try:
        from kronos_trainer import finetune_kronos

        # Kiểm tra drift queue từ mlops_pipeline (tránh race condition)
        drift_tickers: list[str] = []
        if DRIFT_FLAG_PATH.exists():
            try:
                with open(DRIFT_FLAG_PATH, encoding="utf-8") as _f:
                    _queue = json.load(_f)
                drift_tickers = [e.get("ticker") for e in _queue if e.get("ticker")]
                DRIFT_FLAG_PATH.unlink()
                if drift_tickers:
                    logger.info("[JOB 02:00] Drift queue: %s — retrain priority tickers first.", drift_tickers)
                    finetune_kronos(tickers=drift_tickers, epochs=2, max_samples_ticker=50)
            except Exception as _de:
                logger.error("[JOB 02:00] Drift queue processing failed: %s", _de)

        finetune_kronos()
        logger.info("[JOB 02:00] Training Kronos hoàn tất.")
    except Exception as e:
        logger.error("[JOB 02:00] Lỗi Training: %s", e, exc_info=True)


if __name__ == "__main__":
    logger.info("Khởi động nền tảng Điều phối (Jobs Manager)...")
    logger.info("- Crawl tin: mỗi 2 giờ | Sentiment: 00:00 | Training: 02:00")

    scheduler = BackgroundScheduler()
    scheduler.add_job(job_crawl_news,    'interval', hours=2, id="crawl_news")
    scheduler.add_job(job_run_sentiment, 'cron', hour=0, minute=0)
    scheduler.add_job(job_train_kronos,  'cron', hour=2, minute=0)
    scheduler.add_job(job_build_analyzed_data, 'cron', hour=3, minute=0)
    scheduler.start()

    # Chạy crawl ngay lần đầu khi khởi động
    job_crawl_news()
    # Build analyzed data ngay khi khởi động để dashboard/report có dữ liệu
    job_build_analyzed_data()

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Tắt Scheduler.")
