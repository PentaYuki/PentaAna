"""
data_watchdog.py — Tự động kiểm tra dữ liệu missing/stale (#17)

Logic:
  • Quét tất cả parquet trong data/raw/parquet/ và data/analyzed/
  • Nếu file không cập nhật trong MAX_STALE_DAYS → báo động, block giao dịch
  • Tích hợp alert_data_stale() từ logger_setup.py → gửi Telegram/Slack
  • Dùng như guard trong pipeline.py hoặc phase4_orchestrator.py:

      from data_watchdog import DataWatchdog
      dog = DataWatchdog()
      if not dog.is_safe_to_trade("VNM"):
          raise RuntimeError("Dữ liệu VNM stale — dừng giao dịch")

  • CLI: python data_watchdog.py --ticker VNM
"""

import os
import json
import argparse
from datetime import datetime, timedelta
from typing import Optional

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR  = os.path.join(BASE_DIR, "data")
REPORT_DIR = os.path.join(DATA_DIR, "reports", "json")

# Ngưỡng: nếu file không cập nhật quá N ngày → stale
MAX_STALE_DAYS = float(os.getenv("DATA_MAX_STALE_DAYS", "2"))

# Mapping: ticker → các file cần kiểm tra theo thứ tự ưu tiên
_TICKER_PATHS = [
    # Format: (relative_path_template, human_label)
    ("raw/parquet/{ticker}_history.parquet", "raw parquet"),
    ("analyzed/with_indicators/{ticker}_with_indicators.parquet", "indicators parquet"),
]

_VNINDEX_PATHS = [
    ("raw/parquet/VNINDEX_history.parquet", "VNINDEX parquet"),
]


class DataWatchdog:
    """
    Kiểm tra độ tươi mới của dữ liệu và block giao dịch nếu stale.

    Ví dụ:
        dog = DataWatchdog()
        status = dog.check_ticker("VNM")
        if not status.is_safe:
            logger.error(status.message)
    """

    def __init__(
        self,
        max_stale_days: float = MAX_STALE_DAYS,
        data_dir: str = DATA_DIR,
        send_alerts: bool = True,
    ):
        self.max_stale_days = max_stale_days
        self.data_dir       = data_dir
        self.send_alerts    = send_alerts
        self._results_cache: dict = {}

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    def check_ticker(self, ticker: str) -> "WatchdogResult":
        """
        Kiểm tra độ tươi mới của dữ liệu cho một mã.

        Returns:
            WatchdogResult với is_safe=True nếu an toàn giao dịch
        """
        ticker = ticker.upper()
        issues = []
        checked_files = []
        most_recent_mtime: Optional[float] = None

        paths_to_check = _TICKER_PATHS[:]
        if ticker == "VNINDEX":
            paths_to_check = _VNINDEX_PATHS[:]

        for path_template, label in paths_to_check:
            rel_path = path_template.format(ticker=ticker)
            abs_path = os.path.join(self.data_dir, rel_path)

            file_result = self._check_file(abs_path, label)
            checked_files.append(file_result)

            if file_result["exists"]:
                mtime = file_result["mtime"]
                if most_recent_mtime is None or mtime > most_recent_mtime:
                    most_recent_mtime = mtime
                if not file_result["is_fresh"]:
                    issues.append(
                        f"{label}: {file_result['days_old']:.1f} ngày chưa cập nhật"
                    )
            else:
                issues.append(f"{label}: FILE KHÔNG TỒN TẠI ({abs_path})")

        # Kiểm tra data hiện tại đủ dùng không (phải có ít nhất raw parquet)
        primary_exists = any(
            r["exists"] for r in checked_files
            if r.get("label") == "raw parquet"
        )
        if not primary_exists:
            # Fallback: nếu ít nhất 1 file tồn tại
            primary_exists = any(r["exists"] for r in checked_files)

        is_safe = primary_exists and len(issues) == 0
        days_since = (
            (datetime.now().timestamp() - most_recent_mtime) / 86400
            if most_recent_mtime else float("inf")
        )

        result = WatchdogResult(
            ticker=ticker,
            is_safe=is_safe,
            days_since_update=round(days_since, 2),
            max_stale_days=self.max_stale_days,
            issues=issues,
            checked_files=checked_files,
            checked_at=datetime.utcnow().isoformat(),
        )

        # Gửi alert nếu stale
        if not is_safe and self.send_alerts and issues:
            try:
                from logger_setup import alert_data_stale
                alert_data_stale(ticker, days_since)
            except Exception:
                pass  # Không để alert failure block watchdog

        return result

    def is_safe_to_trade(self, ticker: str) -> bool:
        """Shortcut: True nếu dữ liệu đủ tươi để giao dịch."""
        return self.check_ticker(ticker).is_safe

    def check_all_tickers(self, tickers: list) -> dict:
        """
        Kiểm tra nhiều mã cùng lúc.

        Returns: {ticker: WatchdogResult}
        """
        results = {}
        for ticker in tickers:
            results[ticker] = self.check_ticker(ticker)
        return results

    def get_safe_tickers(self, tickers: list) -> list:
        """Trả về danh sách mã có dữ liệu tươi (an toàn giao dịch)."""
        return [t for t in tickers if self.is_safe_to_trade(t)]

    def check_vnindex(self) -> "WatchdogResult":
        """Kiểm tra VNINDEX — dùng cho macro data."""
        return self.check_ticker("VNINDEX")

    # ──────────────────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────────────────

    def _check_file(self, abs_path: str, label: str) -> dict:
        if not os.path.exists(abs_path):
            return {
                "path":     abs_path,
                "label":    label,
                "exists":   False,
                "is_fresh": False,
                "days_old": float("inf"),
                "mtime":    None,
                "size_kb":  0,
            }

        try:
            stat = os.stat(abs_path)
        except OSError:
            return {
                "path":     abs_path,
                "label":    label,
                "exists":   False,
                "is_fresh": False,
                "days_old": float("inf"),
                "mtime":    None,
                "size_kb":  0,
            }

        mtime    = stat.st_mtime
        days_old = (datetime.now().timestamp() - mtime) / 86400.0
        is_fresh = days_old <= self.max_stale_days
        size_kb  = round(stat.st_size / 1024, 1)

        return {
            "path":      abs_path,
            "label":     label,
            "exists":    True,
            "is_fresh":  is_fresh,
            "days_old":  round(days_old, 2),
            "mtime":     mtime,
            "mtime_str": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "size_kb":   size_kb,
        }


class WatchdogResult:
    """Kết quả kiểm tra staleness của một ticker."""

    def __init__(
        self,
        ticker: str,
        is_safe: bool,
        days_since_update: float,
        max_stale_days: float,
        issues: list,
        checked_files: list,
        checked_at: str,
    ):
        self.ticker            = ticker
        self.is_safe           = is_safe
        self.days_since_update = days_since_update
        self.max_stale_days    = max_stale_days
        self.issues            = issues
        self.checked_files     = checked_files
        self.checked_at        = checked_at

    @property
    def message(self) -> str:
        if self.is_safe:
            return f"✓ {self.ticker}: dữ liệu tươi ({self.days_since_update:.1f} ngày)"
        if self.days_since_update == float("inf"):
            return f"✗ {self.ticker}: KHÔNG tìm thấy file dữ liệu"
        return (
            f"✗ {self.ticker}: dữ liệu STALE ({self.days_since_update:.1f} ngày >"
            f" {self.max_stale_days} ngày cho phép)"
        )

    def to_dict(self) -> dict:
        return {
            "ticker":            self.ticker,
            "is_safe":           self.is_safe,
            "days_since_update": self.days_since_update,
            "max_stale_days":    self.max_stale_days,
            "message":           self.message,
            "issues":            self.issues,
            "checked_files":     self.checked_files,
            "checked_at":        self.checked_at,
        }

    def __repr__(self):
        return f"WatchdogResult({self.message})"


# ══════════════════════════════════════════════════════════════════════════════
# GUARD DECORATOR
# ══════════════════════════════════════════════════════════════════════════════

def require_fresh_data(ticker_arg: str = "ticker", max_stale_days: float = MAX_STALE_DAYS):
    """
    Decorator tự động block hàm nếu dữ liệu của ticker stale.

    Dùng:
        @require_fresh_data("ticker", max_stale_days=2)
        def run_multi_agent_analysis(ticker: str, ...):
            ...
    """
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Lấy ticker từ args hoặc kwargs
            ticker = kwargs.get(ticker_arg)
            if ticker is None and args:
                ticker = args[0]
            if ticker:
                dog = DataWatchdog(max_stale_days=max_stale_days)
                result = dog.check_ticker(str(ticker))
                if not result.is_safe:
                    raise RuntimeError(
                        f"[DataWatchdog] Dừng giao dịch — {result.message}\n"
                        f"Issues: {'; '.join(result.issues)}"
                    )
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# ══════════════════════════════════════════════════════════════════════════════
# FULL SYSTEM SCAN
# ══════════════════════════════════════════════════════════════════════════════

def scan_all_data(
    tickers: Optional[list] = None,
    save_report: bool = True,
) -> dict:
    """
    Quét toàn bộ dữ liệu trong hệ thống.

    Args:
        tickers:      Danh sách mã cần quét; None → tự detect
        save_report:  Lưu kết quả vào reports/json/watchdog_report.json

    Returns:
        Báo cáo tổng hợp {safe: [], stale: [], missing: []}
    """
    if tickers is None:
        # Tự detect từ file trong data/raw/parquet/
        raw_pq_dir = os.path.join(DATA_DIR, "raw", "parquet")
        tickers = []
        if os.path.isdir(raw_pq_dir):
            for fname in os.listdir(raw_pq_dir):
                if fname.endswith("_history.parquet") and fname != "VNINDEX_history.parquet":
                    t = fname.replace("_history.parquet", "")
                    tickers.append(t)

    dog = DataWatchdog()
    safe_list    = []
    stale_list   = []
    missing_list = []

    for ticker in sorted(tickers):
        r = dog.check_ticker(ticker)
        if r.is_safe:
            safe_list.append(ticker)
        elif r.days_since_update == float("inf"):
            missing_list.append(ticker)
        else:
            stale_list.append({"ticker": ticker, "days": r.days_since_update})

    # Kiểm tra VNINDEX riêng
    vnindex_ok = dog.check_vnindex().is_safe

    report = {
        "scanned_at":    datetime.utcnow().isoformat(),
        "max_stale_days": MAX_STALE_DAYS,
        "total_tickers": len(tickers),
        "safe":          safe_list,
        "stale":         stale_list,
        "missing":       missing_list,
        "vnindex_ok":    vnindex_ok,
        "summary": {
            "n_safe":    len(safe_list),
            "n_stale":   len(stale_list),
            "n_missing": len(missing_list),
            "all_clear": len(stale_list) == 0 and len(missing_list) == 0 and vnindex_ok,
        },
    }

    if save_report:
        os.makedirs(REPORT_DIR, exist_ok=True)
        out_path = os.path.join(REPORT_DIR, "watchdog_report.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[DataWatchdog] Báo cáo lưu tại: {out_path}")

    return report


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Watchdog — kiểm tra độ tươi mới của dữ liệu")
    parser.add_argument("--ticker", "-t", default=None, help="Mã cổ phiếu cụ thể (để trống → quét tất cả)")
    parser.add_argument("--max-stale-days", type=float, default=MAX_STALE_DAYS)
    parser.add_argument("--no-save", action="store_true", help="Không lưu báo cáo JSON")
    args = parser.parse_args()

    dog = DataWatchdog(max_stale_days=args.max_stale_days)

    print(f"\n{'='*60}")
    print(f"  DATA WATCHDOG — Stock-AI")
    print(f"  Ngưỡng stale: {args.max_stale_days} ngày")
    print(f"{'='*60}\n")

    if args.ticker:
        result = dog.check_ticker(args.ticker)
        print(result.message)
        for issue in result.issues:
            print(f"    • {issue}")
        for f in result.checked_files:
            status = "✓" if f.get("is_fresh") else ("✗" if f.get("exists") else "?")
            age    = f"{f.get('days_old', '?'):.1f}d" if f.get("exists") else "MISSING"
            print(f"    [{status}] {f.get('label', '')}: {age}")
        print(f"\n  is_safe_to_trade: {result.is_safe}")
    else:
        report = scan_all_data(save_report=not args.no_save)
        print(f"  ✓ An toàn  ({len(report['safe'])}): {', '.join(report['safe'])}")
        if report["stale"]:
            for s in report["stale"]:
                print(f"  ⚠ Stale: {s['ticker']} ({s['days']:.1f}d)")
        if report["missing"]:
            print(f"  ✗ Missing ({len(report['missing'])}): {', '.join(report['missing'])}")
        print(f"  VNINDEX: {'✓ OK' if report['vnindex_ok'] else '✗ STALE/MISSING'}")
        print(f"\n  ALL CLEAR: {report['summary']['all_clear']}")
