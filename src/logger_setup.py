"""
logger_setup.py — Centralized logging & realtime alerting (#16)

Tính năng:
  • loguru thay thế logging.basicConfig rải rác — format thống nhất, màu sắc
  • Telegram Bot Alert: gửi tin nhắn khi crawl lỗi, drift, hoặc data stale
  • Slack Webhook Alert: tùy chọn thay thế / song song Telegram
  • Rate-limit alert: tối đa 1 alert/chủ đề/10 phút để tránh spam
  • File rotation: logs/{date}.log, giữ 14 ngày

Cấu hình qua .env:
  TELEGRAM_BOT_TOKEN  = "123456:ABCdef..."
  TELEGRAM_CHAT_ID    = "-100123456789"
  SLACK_WEBHOOK_URL   = "https://hooks.slack.com/services/..."
  LOG_LEVEL           = "INFO"            # DEBUG | INFO | WARNING | ERROR
"""

import os
import sys
import time
import threading
from datetime import datetime
from typing import Optional

try:
    from loguru import logger as _loguru_logger
    _HAS_LOGURU = True
except ImportError:
    import logging
    _HAS_LOGURU = False

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# ── Alert rate-limit state ─────────────────────────────────────────────────
_alert_last_sent: dict[str, float] = {}
_alert_lock = threading.Lock()
ALERT_COOLDOWN_SEC = 600   # 10 phút / chủ đề

# ── Cấu hình từ .env ───────────────────────────────────────────────────────
TG_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
SLACK_URL  = os.getenv("SLACK_WEBHOOK_URL", "")
LOG_LEVEL  = os.getenv("LOG_LEVEL", "INFO").upper()


# ══════════════════════════════════════════════════════════════════════════════
# LOGGER SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_logger(
    name: str = "stock-ai",
    log_level: str = LOG_LEVEL,
    log_to_file: bool = True,
) -> "logger":
    """
    Khởi tạo logger tập trung.

    Dùng loguru nếu đã cài, fallback về stdlib logging.
    Gọi 1 lần ở đầu mỗi module:
        from logger_setup import get_logger
        log = get_logger(__name__)

    Returns: logger object (loguru hoặc std-logging)
    """
    if _HAS_LOGURU:
        # Xóa handler mặc định, cấu hình lại cho sạch
        _loguru_logger.remove()

        # Console handler — màu + format ngắn gọn
        _loguru_logger.add(
            sys.stderr,
            level=log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
            ),
            colorize=True,
            enqueue=True,           # thread-safe
        )

        if log_to_file:
            log_path = os.path.join(LOGS_DIR, "{time:YYYY-MM-DD}.log")
            _loguru_logger.add(
                log_path,
                level=log_level,
                rotation="00:00",   # Rotate mỗi ngày
                retention="14 days",
                compression="zip",
                enqueue=True,
                encoding="utf-8",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} — {message}",
            )

        return _loguru_logger.bind(app=name)
    else:
        # Fallback: stdlib logging
        import logging
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(sys.stderr),
                *(
                    [logging.FileHandler(
                        os.path.join(LOGS_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log"),
                        encoding="utf-8"
                    )]
                    if log_to_file else []
                ),
            ],
        )
        return logging.getLogger(name)


def get_logger(module_name: str = __name__):
    """Lấy logger cho module cụ thể — gọi ở đầu mỗi file."""
    if _HAS_LOGURU:
        return _loguru_logger.bind(name=module_name)
    else:
        import logging
        return logging.getLogger(module_name)


# ── Singleton init ────────────────────────────────────────────────────────
logger = setup_logger()


# ══════════════════════════════════════════════════════════════════════════════
# ALERT SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def _rate_limit_ok(key: str) -> bool:
    """Kiểm tra xem có được gửi alert không (theo rate-limit)."""
    now = time.time()
    with _alert_lock:
        last = _alert_last_sent.get(key, 0)
        if now - last >= ALERT_COOLDOWN_SEC:
            _alert_last_sent[key] = now
            return True
        return False


def _send_telegram(message: str, parse_mode: str = "HTML") -> bool:
    """Gửi tin nhắn qua Telegram Bot API."""
    if not _HAS_REQUESTS or not TG_TOKEN or not TG_CHAT_ID:
        return False
    try:
        resp = _requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={
                "chat_id": TG_CHAT_ID,
                "text": message,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        logger.warning(f"Telegram alert failed: {e}")
        return False


def _send_slack(message: str) -> bool:
    """Gửi tin nhắn qua Slack Incoming Webhook."""
    if not _HAS_REQUESTS or not SLACK_URL:
        return False
    try:
        resp = _requests.post(
            SLACK_URL,
            json={"text": message},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        logger.warning(f"Slack alert failed: {e}")
        return False


def send_alert(
    message: str,
    level: str = "WARNING",
    key: Optional[str] = None,
    force: bool = False,
) -> bool:
    """
    Gửi cảnh báo realtime qua Telegram và/hoặc Slack.

    Args:
        message: Nội dung cảnh báo (hỗ trợ HTML cho Telegram)
        level:   "INFO" | "WARNING" | "ERROR" | "CRITICAL"
        key:     ID duy nhất cho rate-limit (dùng nếu không muốn spam)
                 None → không rate-limit (luôn gửi)
        force:   Bỏ qua rate-limit nếu True

    Returns:
        True nếu gửi thành công ít nhất 1 kênh
    """
    # Rate limit check
    if key and not force:
        if not _rate_limit_ok(key):
            logger.debug(f"Alert '{key}' bị rate-limit, bỏ qua.")
            return False

    # Log locally
    icon = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌", "CRITICAL": "🚨"}.get(level, "🔔")
    log_fn = {
        "INFO": logger.info,
        "WARNING": logger.warning,
        "ERROR": logger.error,
        "CRITICAL": logger.critical,
    }.get(level, logger.warning)
    log_fn(f"[ALERT] {message}")

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = (
        f"{icon} <b>[stock-ai/{level}]</b>\n"
        f"🕐 {now_str}\n\n"
        f"{message}"
    )

    sent = False
    # Gửi Telegram trong background thread để không block main thread
    if TG_TOKEN and TG_CHAT_ID:
        t = threading.Thread(target=_send_telegram, args=(full_msg,), daemon=True)
        t.start()
        sent = True

    if SLACK_URL:
        slack_msg = f"{icon} [stock-ai/{level}] {now_str}\n{message}"
        t2 = threading.Thread(target=_send_slack, args=(slack_msg,), daemon=True)
        t2.start()
        sent = True

    return sent


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE SHORTCUTS
# ══════════════════════════════════════════════════════════════════════════════

def alert_crawl_error(ticker: str, error: str):
    """Cảnh báo khi crawl dữ liệu thất bại."""
    send_alert(
        f"🕷️ Crawl lỗi cho <b>{ticker}</b>:\n<code>{error[:500]}</code>",
        level="ERROR",
        key=f"crawl_error_{ticker}",
    )


def alert_data_stale(ticker: str, days_since_update: float):
    """Cảnh báo khi dữ liệu không được cập nhật đủ lâu."""
    send_alert(
        f"📉 Dữ liệu <b>{ticker}</b> chưa cập nhật {days_since_update:.1f} ngày!\n"
        f"Hệ thống <b>TẠM DỪNG GIAO DỊCH</b> cho mã này.",
        level="CRITICAL",
        key=f"data_stale_{ticker}",
    )


def alert_model_drift(ticker: str, drift_score: float, threshold: float):
    """Cảnh báo khi phát hiện model drift."""
    send_alert(
        f"📊 Model drift phát hiện cho <b>{ticker}</b>\n"
        f"PSI score: <code>{drift_score:.4f}</code> (ngưỡng: {threshold:.4f})\n"
        f"Khuyến nghị: retrain model.",
        level="WARNING",
        key=f"drift_{ticker}",
    )


def alert_low_ram(available_gb: float, required_gb: float):
    """Cảnh báo khi RAM không đủ cho training."""
    send_alert(
        f"🧠 RAM không đủ cho training!\n"
        f"Có sẵn: <code>{available_gb:.1f} GB</code> / Yêu cầu: {required_gb:.1f} GB\n"
        f"Training bị hủy bỏ tự động.",
        level="ERROR",
        key="ram_oom",
    )


def alert_backtest_complete(ticker: str, sharpe: float, total_return: float, windows: int):
    """Thông báo khi walk-forward backtest hoàn tất."""
    send_alert(
        f"✅ Walk-forward backtest hoàn tất: <b>{ticker}</b>\n"
        f"📈 Avg Sharpe: <code>{sharpe:.4f}</code>\n"
        f"💰 Avg Return: <code>{total_return:+.2f}%</code>\n"
        f"🔲 Số windows: {windows}",
        level="INFO",
        key=f"backtest_done_{ticker}",
        force=True,   # Luôn gửi kết quả backtest
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log = get_logger(__name__)
    log.info("Logger setup test bắt đầu")
    log.warning("Đây là warning test")
    log.error("Đây là error test")

    print("\n--- Test alert system ---")
    print(f"  Telegram configured: {bool(TG_TOKEN and TG_CHAT_ID)}")
    print(f"  Slack configured:    {bool(SLACK_URL)}")

    # Test alert (sẽ gửi nếu đã cấu hình token)
    result = send_alert(
        "🧪 Test alert từ stock-ai logger_setup.py\nNếu bạn thấy tin này, alerting đã hoạt động!",
        level="INFO",
        force=True,
    )
    print(f"  Alert sent: {result}")
    log.info("Logger setup test hoàn thành ✓")
