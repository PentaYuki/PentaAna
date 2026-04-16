"""
macro_data.py — Thu thập macro data thực cho thị trường Việt Nam.

Ưu tiên nguồn theo thứ tự độ ổn định:
  1. yfinance: USDVND=X (tỷ giá), ^VNINDEX (đã có sẵn qua parquet)
  2. VN30 Index qua yfinance làm proxy PMI/macro trend
  3. Fallback về VNINDEX parquet local nếu tất cả online request fail

Cache vào data/raw/parquet/macro_vn.parquet — tái sử dụng trong ngày.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MACRO_CACHE_PATH = os.path.join(DATA_DIR, "raw", "parquet", "macro_vn.parquet")
VNINDEX_PATH = os.path.join(DATA_DIR, "raw", "parquet", "VNINDEX_history.parquet")

os.makedirs(os.path.dirname(MACRO_CACHE_PATH), exist_ok=True)

# Cache in-memory
_MACRO_CACHE: Optional[dict] = None
_CACHE_TIMESTAMP: float = 0.0

# (#21) 1-day disk cache: TTL = 24 giờ
CACHE_TTL_HOURS   = float(os.getenv("MACRO_CACHE_TTL_HOURS", "24"))
_CACHE_TTL_SECONDS = CACHE_TTL_HOURS * 3600

# (#21) Retry/alert config cho yfinance rate-limit
_YF_RETRY_DELAYS  = [2, 5, 15]   # giây chờ giữa các retry (exponential-ish)
_YF_FAIL_COUNT    = 0            # số lần yfinance liên tiếp thất bại
YF_FAIL_THRESHOLD = 3            # gửi alert nếu fail ≥ N lần liên tiếp


def _fetch_yfinance_macro(lookback_days: int = 90) -> dict:
    """
    Lấy dữ liệu macro từ yfinance với per-symbol retry + exponential backoff. (#21)

    Nếu một symbol bị rate-limit, chờ rồi thử lại (không bỏ qua ngay).
    Tickers:
      - USDVND=X: tỷ giá USD/VND
      - ^GSPC: S&P 500
      - GC=F: Gold futures
      - CL=F: Crude oil
    """
    global _YF_FAIL_COUNT

    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance không có — chạy: pip install yfinance")
        return {}

    end_date   = datetime.today()
    start_date = end_date - timedelta(days=lookback_days)

    tickers = {
        "usdvnd": "USDVND=X",
        "sp500":  "^GSPC",
        "gold":   "GC=F",
        "oil":    "CL=F",
    }

    results = {}
    any_success = False

    for name, symbol in tickers.items():
        fetched = False
        for attempt, delay in enumerate([0] + _YF_RETRY_DELAYS, start=0):
            if delay > 0:
                logger.debug(f"yfinance {symbol}: retry #{attempt}, chờ {delay}s...")
                time.sleep(delay)
            try:
                df = yf.download(
                    symbol,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=True,
                )
                if df is None or df.empty:
                    continue
                close_col = "Close" if "Close" in df.columns else df.columns[0]
                series = df[close_col].dropna()
                if len(series) < 2:
                    continue
                prices   = series.values.astype("float64")
                last_p   = float(prices[-1])
                ret_5d   = float((prices[-1] - prices[min(5,  len(prices)-1)]) / prices[min(5,  len(prices)-1)] * 100)
                ret_20d  = float((prices[-1] - prices[min(20, len(prices)-1)]) / prices[min(20, len(prices)-1)] * 100)
                results[name] = {
                    "last_price":  round(last_p,  4),
                    "ret_5d_pct":  round(ret_5d,  4),
                    "ret_20d_pct": round(ret_20d, 4),
                }
                fetched     = True
                any_success = True
                break   # thành công, không retry nữa
            except Exception as e:
                logger.debug(f"yfinance {symbol} attempt {attempt+1} failed: {e}")

        if not fetched:
            logger.warning(f"yfinance {symbol}: tất cả retry thất bại")

    # Cập nhật fail counter và cảnh báo nếu cần
    if any_success:
        _YF_FAIL_COUNT = 0
    else:
        _YF_FAIL_COUNT += 1
        logger.warning(f"yfinance macro: thất bại liên tiếp lần {_YF_FAIL_COUNT}")
        if _YF_FAIL_COUNT >= YF_FAIL_THRESHOLD:
            try:
                from logger_setup import send_alert
                send_alert(
                    f"📶 yfinance rate-limit liên tục {_YF_FAIL_COUNT} lần!\n"
                    f"Hệ thống đang dùng VNINDEX local thay thế.\n"
                    f"Kiểm tra kết nối internet hoặc thử lại sau.",
                    level="WARNING",
                    key="yfinance_ratelimit",
                )
            except Exception:
                pass

    return results


def _compute_macro_score_from_yfinance(yf_data: dict) -> float:
    """
    Tổng hợp các chỉ số yfinance → macro_score trong [-1, 1].

    Logic:
    - sp500 tăng → risk-on → tích cực cho VN (hệ số +0.4)
    - gold tăng mạnh → risk-off → tiêu cực (-0.2)
    - oil tăng mạnh → tốn chi phí → tiêu cực (-0.1)
    - usdvnd tăng → VND mất giá → tiêu cực (-0.3)
    """
    score = 0.0

    if "sp500" in yf_data:
        sp_ret = yf_data["sp500"].get("ret_20d_pct", 0.0)
        score += np.tanh(sp_ret / 8.0) * 0.4

    if "usdvnd" in yf_data:
        usd_ret = yf_data["usdvnd"].get("ret_20d_pct", 0.0)
        # VND mất giá (USDVND tăng) → tiêu cực
        score -= np.tanh(usd_ret / 3.0) * 0.3

    if "gold" in yf_data:
        gold_ret = yf_data["gold"].get("ret_20d_pct", 0.0)
        # Gold tăng mạnh > 5% → risk-off rõ ràng
        if gold_ret > 5.0:
            score -= 0.2
        elif gold_ret < -3.0:
            score += 0.1

    if "oil" in yf_data:
        oil_ret = yf_data["oil"].get("ret_20d_pct", 0.0)
        score -= np.tanh(oil_ret / 10.0) * 0.1

    return float(np.clip(score, -1.0, 1.0))


def _fallback_from_vnindex(as_of_date: Optional[str] = None) -> dict:
    """Fallback: tính macro_score từ VNINDEX local."""
    if not os.path.exists(VNINDEX_PATH):
        return {"macro_score": 0.0, "source": "fallback_zero"}
    try:
        df = pd.read_parquet(VNINDEX_PATH, engine="pyarrow").sort_values("time")
        if as_of_date is not None:
            as_of_ts = pd.to_datetime(as_of_date, errors="coerce")
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df[df["time"] <= as_of_ts]
        if len(df) < 22:
            return {"macro_score": 0.0, "source": "fallback_insufficient"}
        c = df["close"].astype("float64").to_numpy()
        r20 = (c[-1] - c[-21]) / c[-21] * 100.0 if c[-21] else 0.0
        macro_score = float(np.tanh(r20 / 5.0))
        return {
            "macro_score": round(macro_score, 4),
            "vnindex_ret_20d_pct": round(r20, 4),
            "source": "vnindex_local",
        }
    except Exception as e:
        logger.debug(f"VNINDEX fallback failed: {e}")
        return {"macro_score": 0.0, "source": "fallback_error"}


def get_macro_data(as_of_date: Optional[str] = None, force_refresh: bool = False) -> dict:
    """
    Lấy macro data tổng hợp, ưu tiên theo thứ tự:
      1. Disk cache (macro_vn.parquet) nếu TTL < {CACHE_TTL_HOURS}h và không force  (#21)
      2. yfinance với retry + backoff                                               (#21)
      3. Fallback VNINDEX local

    Returns:
        {{
            "macro_score": float,
            "usdvnd_ret_20d_pct": float,
            "sp500_ret_20d_pct": float,
            "gold_ret_20d_pct": float,
            "oil_ret_20d_pct": float,
            "vnindex_ret_20d_pct": float,
            "source": str,   # "yfinance" | "disk_cache" | "vnindex_local" | ...
            "fetched_at": str,
            "cache_age_hours": float,
        }}
    """
    global _MACRO_CACHE, _CACHE_TIMESTAMP

    # (A) In-memory cache check (live-mode only)
    if (
        not force_refresh
        and as_of_date is None
        and _MACRO_CACHE is not None
        and (time.time() - _CACHE_TIMESTAMP) < _CACHE_TTL_SECONDS
    ):
        return _MACRO_CACHE

    # (B) (#21) Disk cache check — tái sử dụng nếu file đủ mới
    if not force_refresh and as_of_date is None and os.path.exists(MACRO_CACHE_PATH):
        try:
            file_age_sec = time.time() - os.path.getmtime(MACRO_CACHE_PATH)
            cache_age_h  = file_age_sec / 3600
            if file_age_sec < _CACHE_TTL_SECONDS:
                cached_df = pd.read_parquet(MACRO_CACHE_PATH, engine="pyarrow")
                if not cached_df.empty:
                    cached = cached_df.iloc[-1].to_dict()
                    cached["source"]         = "disk_cache"
                    cached["cache_age_hours"] = round(cache_age_h, 2)
                    logger.debug(f"[macro] Dùng disk cache ({cache_age_h:.1f}h tuổi)")
                    _MACRO_CACHE     = cached
                    _CACHE_TIMESTAMP = time.time()
                    return cached
        except Exception as e:
            logger.debug(f"[macro] Disk cache read failed: {e}")

    result: dict = {}
    yf_data = {}

    # (C) Chỉ gọi yfinance nếu không có as_of_date (backtest dùng local data)
    if as_of_date is None:
        try:
            yf_data = _fetch_yfinance_macro(lookback_days=90)
        except Exception as e:
            logger.warning(f"yfinance macro fetch failed: {e}")

    if yf_data:
        macro_score = _compute_macro_score_from_yfinance(yf_data)
        result["macro_score"] = round(macro_score, 4)
        result["source"]      = "yfinance"
        for key in ("usdvnd", "sp500", "gold", "oil"):
            if key in yf_data:
                result[f"{key}_ret_20d_pct"]  = yf_data[key].get("ret_20d_pct", 0.0)
                result[f"{key}_last_price"]   = yf_data[key].get("last_price",  0.0)
    else:
        # (D) (#21) Fallback về VNINDEX local khi yfinance không khả dụng
        fallback = _fallback_from_vnindex(as_of_date)
        result.update(fallback)

    # Luôn thêm VNINDEX local để bổ sung
    vnindex_info = _fallback_from_vnindex(as_of_date)
    result["vnindex_ret_20d_pct"] = vnindex_info.get("vnindex_ret_20d_pct", 0.0)
    result["fetched_at"]          = datetime.utcnow().isoformat()
    result["cache_age_hours"]     = 0.0

    # (E) Cập nhật in-memory cache (chỉ cho live data)
    if as_of_date is None:
        _MACRO_CACHE     = result
        _CACHE_TIMESTAMP = time.time()

        # (F) (#21) Persist cache to parquet (1-day disk cache)
        try:
            cache_df = pd.DataFrame([result])
            cache_df.to_parquet(MACRO_CACHE_PATH, engine="pyarrow", index=False)
            logger.debug(f"[macro] Disk cache cập nhật: {MACRO_CACHE_PATH}")
        except Exception as e:
            logger.debug(f"Failed to persist macro cache: {e}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = get_macro_data(force_refresh=True)
    print("\n=== Macro Data ===")
    for k, v in data.items():
        print(f"  {k}: {v}")
    print(f"\n  macro_score: {data.get('macro_score', 'N/A')}")
    print(f"  source: {data.get('source', 'N/A')}")
