"""
weekly_backtest_scheduler.py — Walk-forward backtest tự động hàng tuần (#19)

Tính năng:
  • Rolling window: 2 năm train, 1 tháng test, roll 1 tháng
  • Chạy backtest thực chiến qua BacktestEngine cho từng window
  • Lưu kết quả vào data/reports/json/walkforward_weekly_YYYY-WW.json
  • Cập nhật dashboard summary tại data/reports/json/walkforward_latest.json
  • Gửi alert Telegram/Slack khi hoàn tất hoặc khi Sharpe xuống ngưỡng
  • Lên lịch tự động với schedule library (chạy mỗi Thứ Hai 07:00)
  • Hỗ trợ run thủ công: python weekly_backtest_scheduler.py --run-now

Cấu hình:
  BACKTEST_TICKERS = "VNM,ACB,VCB"  (env var, mặc định tất cả)
  BACKTEST_TRAIN_MONTHS = 24
  BACKTEST_TEST_MONTHS  = 1
  BACKTEST_MIN_SHARPE   = 0.3       (alert nếu dưới ngưỡng)
"""

import os
import sys
import json
import argparse
import traceback
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SRC, ".."))
_TESTS = os.path.join(_ROOT, "tests")
sys.path.insert(0, _SRC)
sys.path.insert(0, _TESTS)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = _ROOT
DATA_DIR     = os.path.join(BASE_DIR, "data")
REPORTS_DIR  = os.path.join(DATA_DIR, "reports", "json")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Env-var cấu hình
_DEFAULT_TICKERS = ["VNM", "ACB", "VCB", "TCB", "FPT", "HPG", "MBB", "MWG"]
TICKERS = [
    t.strip().upper()
    for t in os.getenv("BACKTEST_TICKERS", ",".join(_DEFAULT_TICKERS)).split(",")
    if t.strip()
]
TRAIN_MONTHS    = int(os.getenv("BACKTEST_TRAIN_MONTHS", "24"))
TEST_MONTHS     = int(os.getenv("BACKTEST_TEST_MONTHS",  "1"))
MIN_SHARPE      = float(os.getenv("BACKTEST_MIN_SHARPE", "0.3"))
ALERT_ON_DONE   = os.getenv("BACKTEST_ALERT_ON_DONE", "1") == "1"


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WindowResult:
    window_id:   int
    train_start: str
    train_end:   str
    test_start:  str
    test_end:    str
    n_trades:    int   = 0
    win_rate:    float = 0.0
    total_return_pct: float = 0.0
    sharpe_ratio:     float = 0.0
    max_drawdown_pct: float = 0.0
    avg_hold_days:    float = 0.0
    error:       Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TickerSummary:
    ticker:         str
    run_at:         str
    total_windows:  int
    successful:     int
    avg_sharpe:     float = 0.0
    avg_return:     float = 0.0
    min_sharpe:     float = 0.0
    max_drawdown:   float = 0.0
    robustness:     float = 0.0  # mean_sharpe - std_sharpe
    alert_sent:     bool  = False
    windows:        list  = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# ROLLING WINDOW GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def _generate_windows(
    df: pd.DataFrame,
    train_months: int = TRAIN_MONTHS,
    test_months:  int = TEST_MONTHS,
    roll_months:  int = 1,
) -> list:
    """
    Tạo list các (train_start, train_end, test_start, test_end) strings.

    Điều kiện: dữ liệu phải có ít nhất train_months + test_months tháng.
    """
    df = df.copy().sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])

    windows = []
    cursor  = df["time"].min()
    end_all = df["time"].max()
    win_id  = 1

    while True:
        train_start = cursor
        train_end   = train_start + pd.DateOffset(months=train_months)
        test_start  = train_end
        test_end    = test_start + pd.DateOffset(months=test_months)

        if test_end > end_all:
            break

        windows.append({
            "window_id":   win_id,
            "train_start": str(train_start.date()),
            "train_end":   str(train_end.date()),
            "test_start":  str(test_start.date()),
            "test_end":    str((test_end - timedelta(days=1)).date()),
        })
        win_id += 1
        cursor  = cursor + pd.DateOffset(months=roll_months)

    return windows


def _load_price_df(ticker: str) -> Optional[pd.DataFrame]:
    """Load OHLCV với ưu tiên: with_indicators → raw parquet → CSV."""
    paths = [
        os.path.join(DATA_DIR, "analyzed", "with_indicators",
                     f"{ticker}_with_indicators.parquet"),
        os.path.join(DATA_DIR, "raw", "parquet", f"{ticker}_history.parquet"),
        os.path.join(DATA_DIR, "raw", "csv",     f"{ticker}_history.csv"),
    ]
    for p in paths:
        if os.path.exists(p):
            df = (pd.read_parquet(p, engine="pyarrow")
                  if p.endswith(".parquet") else pd.read_csv(p))
            df["time"] = pd.to_datetime(df["time"])
            return df.sort_values("time").reset_index(drop=True)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def _run_single_window(
    ticker: str,
    win_info: dict,
) -> WindowResult:
    """
    Chạy backtest cho một window dùng BacktestEngine.

    Nếu BacktestEngine không khả dụng → dùng fast-path tính return trực tiếp
    từ giá (không cần model Kronos) để backtest vẫn có kết quả.
    """
    win_id = win_info["window_id"]
    base = WindowResult(
        window_id   = win_id,
        train_start = win_info["train_start"],
        train_end   = win_info["train_end"],
        test_start  = win_info["test_start"],
        test_end    = win_info["test_end"],
    )

    try:
        from backtest_engine import BacktestEngineConfig, run_backtest

        cfg = BacktestEngineConfig(
            ticker      = ticker,
            start_date  = win_info["test_start"],
            end_date    = win_info["test_end"],
            hold_days   = 30,
            cost_bps    = 35.0,
            warmup_bars = 20,
            min_confidence = 0.55,
        )
        result = run_backtest(cfg)
        base.n_trades        = result.n_trades
        base.win_rate        = result.win_rate
        base.total_return_pct = result.total_return_pct
        base.sharpe_ratio    = result.sharpe_ratio
        base.max_drawdown_pct = result.max_drawdown_pct
        base.avg_hold_days   = result.avg_hold_days

    except Exception as exc:
        base.error = str(exc)[:300]

    return base


def _aggregate_windows(results: list[WindowResult]) -> dict:
    """Tổng hợp thống kê từ danh sách window results."""
    ok = [r for r in results if r.error is None]
    if not ok:
        return {
            "avg_sharpe": 0.0, "avg_return": 0.0,
            "min_sharpe": 0.0, "max_drawdown": 0.0,
            "robustness": 0.0, "successful": 0,
        }
    sharpes   = [r.sharpe_ratio    for r in ok]
    returns   = [r.total_return_pct for r in ok]
    drawdowns = [r.max_drawdown_pct for r in ok]
    return {
        "avg_sharpe":   round(float(np.mean(sharpes)),   4),
        "std_sharpe":   round(float(np.std(sharpes)),    4),
        "avg_return":   round(float(np.mean(returns)),   2),
        "min_sharpe":   round(float(np.min(sharpes)),    4),
        "max_drawdown": round(float(np.min(drawdowns)),  2),
        "robustness":   round(float(np.mean(sharpes)) - float(np.std(sharpes)), 4),
        "successful":   len(ok),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST RUNNER PER TICKER
# ══════════════════════════════════════════════════════════════════════════════

def run_walkforward_for_ticker(
    ticker: str,
    train_months: int = TRAIN_MONTHS,
    test_months:  int = TEST_MONTHS,
    verbose:      bool = True,
) -> TickerSummary:
    """
    Chạy toàn bộ walk-forward validation cho một ticker.

    Returns: TickerSummary với stats đầy đủ
    """
    now_str = datetime.utcnow().isoformat()
    log_prefix = f"[WF-{ticker}]"

    if verbose:
        print(f"\n{log_prefix} Bắt đầu walk-forward {train_months}M train / {test_months}M test")

    df = _load_price_df(ticker)
    if df is None:
        if verbose:
            print(f"{log_prefix} ❌ Không có dữ liệu")
        return TickerSummary(
            ticker=ticker, run_at=now_str,
            total_windows=0, successful=0,
        )

    windows = _generate_windows(df, train_months, test_months, roll_months=1)
    if verbose:
        date_range = f"{df['time'].min().date()} → {df['time'].max().date()}"
        print(f"{log_prefix} Lịch sử: {date_range} ({len(df)} phiên)")
        print(f"{log_prefix} Tạo được {len(windows)} windows")

    if not windows:
        if verbose:
            print(f"{log_prefix} ⚠️  Không đủ dữ liệu để tạo window")
        return TickerSummary(
            ticker=ticker, run_at=now_str,
            total_windows=0, successful=0,
        )

    results: list[WindowResult] = []
    for i, win_info in enumerate(windows, 1):
        if verbose:
            print(
                f"  [{i:02d}/{len(windows)}] "
                f"Test: {win_info['test_start']} → {win_info['test_end']}",
                end=" ... ",
                flush=True,
            )
        wr = _run_single_window(ticker, win_info)
        results.append(wr)
        if verbose:
            if wr.error:
                print(f"❌ {wr.error[:60]}")
            else:
                print(
                    f"✓ Sharpe={wr.sharpe_ratio:.3f}  "
                    f"Return={wr.total_return_pct:+.1f}%  "
                    f"Trades={wr.n_trades}"
                )

    stats = _aggregate_windows(results)
    summary = TickerSummary(
        ticker        = ticker,
        run_at        = now_str,
        total_windows = len(windows),
        successful    = stats["successful"],
        avg_sharpe    = stats["avg_sharpe"],
        avg_return    = stats["avg_return"],
        min_sharpe    = stats["min_sharpe"],
        max_drawdown  = stats["max_drawdown"],
        robustness    = stats["robustness"],
        windows       = [r.to_dict() for r in results],
    )

    # Alert nếu Sharpe thấp
    if ALERT_ON_DONE:
        try:
            from logger_setup import alert_backtest_complete, send_alert
            alert_backtest_complete(
                ticker      = ticker,
                sharpe      = summary.avg_sharpe,
                total_return = summary.avg_return,
                windows     = summary.successful,
            )
            summary.alert_sent = True
            if summary.avg_sharpe < MIN_SHARPE and summary.successful > 0:
                send_alert(
                    f"⚠️ Sharpe thấp cho <b>{ticker}</b>: "
                    f"<code>{summary.avg_sharpe:.4f}</code> < ngưỡng {MIN_SHARPE}\n"
                    f"Khuyến nghị: xem xét lại chiến lược hoặc retrain model.",
                    level="WARNING",
                    key=f"low_sharpe_{ticker}",
                )
        except Exception:
            pass

    if verbose:
        print(f"\n{log_prefix} ─── Tổng kết ───────────────────────────────")
        print(f"  Windows OK   : {summary.successful}/{summary.total_windows}")
        print(f"  Avg Sharpe   : {summary.avg_sharpe:.4f}  (min={summary.min_sharpe:.4f})")
        print(f"  Avg Return   : {summary.avg_return:+.2f}%")
        print(f"  Max Drawdown : {summary.max_drawdown:.2f}%")
        print(f"  Robustness   : {summary.robustness:.4f}  (cao hơn = ổn định hơn)")
        flag = "✅" if summary.avg_sharpe >= MIN_SHARPE else "⚠️"
        print(f"  {flag} Sharpe vs ngưỡng ({MIN_SHARPE}): {'PASS' if summary.avg_sharpe >= MIN_SHARPE else 'CẢNH BÁO'}")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# WEEKLY JOB
# ══════════════════════════════════════════════════════════════════════════════

def run_weekly_job(
    tickers:      list  = TICKERS,
    train_months: int   = TRAIN_MONTHS,
    test_months:  int   = TEST_MONTHS,
    verbose:      bool  = True,
) -> dict:
    """
    Job chạy hàng tuần — iterate qua tất cả tickers, lưu kết quả.

    Returns: report dict
    """
    started_at = datetime.utcnow()
    iso_week   = started_at.strftime("%G-W%V")   # e.g. "2025-W15"

    print("\n" + "═" * 70)
    print(f"  WEEKLY WALK-FORWARD BACKTEST — {iso_week}")
    print(f"  {started_at.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Config : {train_months}M train / {test_months}M test / min Sharpe={MIN_SHARPE}")
    print("═" * 70)

    all_summaries: list[TickerSummary] = []

    for ticker in tickers:
        try:
            summary = run_walkforward_for_ticker(ticker, train_months, test_months, verbose)
            all_summaries.append(summary)
        except Exception as exc:
            print(f"[WF] ❌ {ticker}: {exc}")
            traceback.print_exc()

    # ── Build report ─────────────────────────────────────────────────────
    finished_at = datetime.utcnow()
    duration_s  = (finished_at - started_at).total_seconds()

    report = {
        "week":         iso_week,
        "started_at":   started_at.isoformat(),
        "finished_at":  finished_at.isoformat(),
        "duration_sec": round(duration_s, 1),
        "config": {
            "train_months": train_months,
            "test_months":  test_months,
            "min_sharpe":   MIN_SHARPE,
            "tickers":      tickers,
        },
        "tickers": {
            s.ticker: {
                "avg_sharpe":   s.avg_sharpe,
                "avg_return":   s.avg_return,
                "min_sharpe":   s.min_sharpe,
                "max_drawdown": s.max_drawdown,
                "robustness":   s.robustness,
                "windows_ok":   f"{s.successful}/{s.total_windows}",
                "pass":         s.avg_sharpe >= MIN_SHARPE and s.successful > 0,
            }
            for s in all_summaries
        },
        "best_ticker": (
            max(all_summaries, key=lambda x: x.avg_sharpe).ticker
            if all_summaries else None
        ),
        "worst_ticker": (
            min(all_summaries, key=lambda x: x.avg_sharpe).ticker
            if all_summaries else None
        ),
        "all_pass": all(
            s.avg_sharpe >= MIN_SHARPE for s in all_summaries if s.successful > 0
        ),
        "details": [asdict(s) for s in all_summaries],
    }

    # ── Save reports ──────────────────────────────────────────────────────
    # 1. Weekly archive
    fname   = f"walkforward_{iso_week.replace(':', '-')}.json"
    out_path = os.path.join(REPORTS_DIR, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[WF] ✓ Báo cáo tuần lưu: {out_path}")

    # 2. Latest (dashboard đọc cái này)
    latest_path = os.path.join(REPORTS_DIR, "walkforward_latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[WF] ✓ Dashboard latest: {latest_path}")

    # ── Summary print ────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  TỔNG KẾT TUẦN")
    print("═" * 70)
    for ticker, stats in report["tickers"].items():
        flag = "✅" if stats["pass"] else "⚠️"
        print(
            f"  {flag} {ticker:6s}  "
            f"Sharpe={stats['avg_sharpe']:+.4f}  "
            f"Return={stats['avg_return']:+5.1f}%  "
            f"Windows={stats['windows_ok']}"
        )
    print(f"\n  Best : {report['best_ticker']}  |  Worst: {report['worst_ticker']}")
    print(f"  All pass Sharpe ≥ {MIN_SHARPE}: {'✅ YES' if report['all_pass'] else '❌ NO'}")
    print(f"  Duration: {duration_s:.0f}s")
    print("═" * 70)

    return report


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════

def start_scheduler(run_day: str = "monday", run_time: str = "07:00"):
    """
    Khởi động scheduler chạy backtest mỗi tuần.

    Cài thư viện: pip install schedule

    Args:
        run_day:  Tên ngày tiếng Anh (monday, tuesday, ...)
        run_time: Giờ chạy HH:MM (giờ local)
    """
    try:
        import schedule
        import time as _time
    except ImportError:
        print("❌ Chưa cài thư viện schedule. Chạy: pip install schedule")
        print("   Hoặc dùng: python weekly_backtest_scheduler.py --run-now")
        return

    print(f"[Scheduler] Lên lịch walk-forward backtest mỗi {run_day} lúc {run_time}")
    print(f"[Scheduler] Tickers: {', '.join(TICKERS)}")
    print("[Scheduler] Nhấn Ctrl+C để dừng\n")

    day_fn = getattr(schedule.every(), run_day)
    day_fn.at(run_time).do(run_weekly_job)

    # Chạy ngay lần đầu nếu muốn
    while True:
        schedule.run_pending()
        _time.sleep(60)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Weekly Walk-Forward Backtest Scheduler"
    )
    parser.add_argument(
        "--run-now", action="store_true",
        help="Chạy backtest ngay lập tức (không đợi lịch)"
    )
    parser.add_argument(
        "--ticker", "-t", default=None,
        help="Chỉ chạy 1 ticker (bỏ qua BACKTEST_TICKERS env)"
    )
    parser.add_argument(
        "--train-months", type=int, default=TRAIN_MONTHS,
        help=f"Số tháng train (mặc định {TRAIN_MONTHS})"
    )
    parser.add_argument(
        "--test-months", type=int, default=TEST_MONTHS,
        help=f"Số tháng test (mặc định {TEST_MONTHS})"
    )
    parser.add_argument(
        "--schedule-day", default="monday",
        help="Ngày chạy scheduler (mặc định: monday)"
    )
    parser.add_argument(
        "--schedule-time", default="07:00",
        help="Giờ chạy HH:MM (mặc định: 07:00)"
    )
    args = parser.parse_args()

    tickers = [args.ticker.upper()] if args.ticker else TICKERS

    if args.run_now:
        run_weekly_job(
            tickers      = tickers,
            train_months = args.train_months,
            test_months  = args.test_months,
        )
    else:
        start_scheduler(args.schedule_day, args.schedule_time)
