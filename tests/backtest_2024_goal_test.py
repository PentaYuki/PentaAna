"""
backtest_2024_goal_test.py
==========================
Kịch bản kiểm thử: Năm 2026, giả lập hệ thống hoạt động ở năm 2024
- Dữ liệu thật: 2024-01-01 → 2025-12-31
- Vốn ban đầu : 5,000,000 VND
- Mục tiêu    : 9,000,000 VND lợi nhuận (tổng NAV = 14,000,000 VND)
- Output      : reports/backtest_2024_goal_report.json + .txt

Chạy: python tests/backtest_2024_goal_test.py
"""

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR    = os.path.join(BASE_DIR, "src")
DATA_DIR   = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
JSON_DIR   = os.path.join(REPORTS_DIR, "json")
TXT_REPORT = os.path.join(REPORTS_DIR, "backtest_2024_goal_report.txt")
JSON_REPORT = os.path.join(JSON_DIR, "backtest_2024_goal_report.json")

sys.path.insert(0, SRC_DIR)

# ── Config ────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 5_000_000   # 5 triệu VND
TARGET_PROFIT   = 9_000_000   # mục tiêu lời 9 triệu
TARGET_NAV      = INITIAL_CAPITAL + TARGET_PROFIT  # 14 triệu

START_DATE = "2024-01-01"
END_DATE   = "2025-12-31"

# Danh sách mã kiểm thử
TICKERS = ["VNM", "VCB", "FPT", "HPG", "MBB", "TCB", "ACB", "MWG", "SSI", "VHM"]

DASHBOARD_URL = "http://localhost:8088"
FEE_PCT       = 0.2   # phí 0.2% mỗi chiều
LOT_SIZE      = 100   # lô tối thiểu HOSE

# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str, file=None):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if file:
        file.write(line + "\n")
        file.flush()


def sep(title: str = "", file=None):
    line = "─" * 60
    if title:
        line = f"── {title} " + "─" * max(0, 56 - len(title))
    log(line, file)


def load_parquet(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, "raw", "parquet", f"{ticker}_history.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path, engine="pyarrow")
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    return df


def signal(row) -> str:
    try:
        if any(pd.isna([row["macd"], row["macd_sig"], row["ema20"], row["rsi"]])):
            return "HOLD"
        if row["macd"] > row["macd_sig"] and row["close"] > row["ema20"] and row["rsi"] < 70:
            return "BUY"
        if row["macd"] < row["macd_sig"] or row["close"] < row["ema50"]:
            return "SELL"
    except Exception:
        pass
    return "HOLD"


# ── Core simulation ───────────────────────────────────────────────────────────

def run_simulation(ticker: str) -> dict:
    df_full = load_parquet(ticker)
    if df_full is None:
        return {"ticker": ticker, "status": "NO_DATA", "passed": False}

    df_full = add_indicators(df_full)
    mask = (df_full["time"] >= START_DATE) & (df_full["time"] <= END_DATE)
    df = df_full[mask].reset_index(drop=True)

    if len(df) < 30:
        return {"ticker": ticker, "status": "INSUFFICIENT_DATA", "rows": len(df), "passed": False}

    cash   = float(INITIAL_CAPITAL)
    shares = 0
    ep     = 0.0   # entry price
    fee_r  = FEE_PCT / 100.0

    equity_curve = []
    trades       = []
    target_hit   = None
    peak_nav     = cash
    max_dd       = 0.0

    for i, row in df.iterrows():
        price    = float(row["close"])
        date_str = str(row["time"])[:10]
        sig      = signal(row)

        if sig == "BUY" and cash > 0 and shares == 0:
            max_sh = int(cash / (price * (1 + fee_r)))
            actual = (max_sh // LOT_SIZE) * LOT_SIZE
            if actual > 0:
                cost   = actual * price
                fee    = cost * fee_r
                cash  -= (cost + fee)
                shares = actual
                ep     = price
                trades.append({"action": "BUY", "date": date_str, "price": price,
                               "shares": actual, "fee": fee})

        elif sig == "SELL" and shares > 0:
            proceeds = shares * price
            fee      = proceeds * fee_r
            profit   = proceeds - fee - shares * ep * (1 + fee_r)
            cash    += (proceeds - fee)
            trades.append({"action": "SELL", "date": date_str, "price": price,
                           "shares": shares, "fee": fee,
                           "profit": round(profit, 0),
                           "profit_pct": round((price - ep) / ep * 100, 2)})
            shares = 0
            ep     = 0.0

        nav = cash + shares * price
        if nav > peak_nav:
            peak_nav = nav
        dd = (peak_nav - nav) / peak_nav * 100
        if dd > max_dd:
            max_dd = dd

        equity_curve.append({"date": date_str, "nav": round(nav, 0), "drawdown": round(dd, 2)})

        if target_hit is None and nav >= TARGET_NAV:
            target_hit = date_str

    # Liquidate cuối kỳ
    if shares > 0:
        last_price = float(df.iloc[-1]["close"])
        proceeds   = shares * last_price
        fee        = proceeds * fee_r
        profit     = proceeds - fee - shares * ep * (1 + fee_r)
        cash      += (proceeds - fee)
        trades.append({"action": "SELL(LIQ)", "date": str(df.iloc[-1]["time"])[:10],
                       "price": last_price, "shares": shares, "fee": fee,
                       "profit": round(profit, 0),
                       "profit_pct": round((last_price - ep) / ep * 100, 2)})

    final_nav    = cash
    total_ret    = (final_nav - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    sell_trades  = [t for t in trades if "SELL" in t["action"]]
    win_trades   = [t for t in sell_trades if t.get("profit", 0) > 0]
    win_rate     = len(win_trades) / len(sell_trades) * 100 if sell_trades else 0.0
    goal_reached = final_nav >= TARGET_NAV

    returns = []
    prev    = INITIAL_CAPITAL
    for pt in equity_curve:
        r = (pt["nav"] - prev) / prev if prev else 0
        returns.append(r)
        prev = pt["nav"]
    arr = np.array(returns)
    std = arr.std()
    sharpe = float(arr.mean() / std * np.sqrt(252)) if std > 1e-9 else 0.0

    gap_pct = (TARGET_NAV - final_nav) / TARGET_NAV * 100 if not goal_reached else 0.0

    return {
        "ticker":          ticker,
        "start_date":      START_DATE,
        "end_date":        END_DATE,
        "initial_capital": INITIAL_CAPITAL,
        "target_nav":      TARGET_NAV,
        "final_nav":       round(final_nav, 0),
        "total_return_pct": round(total_ret, 2),
        "profit_vnd":      round(final_nav - INITIAL_CAPITAL, 0),
        "goal_reached":    goal_reached,
        "target_hit_date": target_hit,
        "gap_to_goal_pct": round(gap_pct, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio":    round(sharpe, 4),
        "total_trades":    len(sell_trades),
        "win_rate_pct":    round(win_rate, 2),
        "trade_log":       trades,
        "equity_curve":    equity_curve,
        "data_rows":       len(df),
        "status":          "OK",
        "passed":          True,
    }


# ── Dashboard API check ───────────────────────────────────────────────────────

def check_dashboard(logf):
    sep("KIỂM TRA DASHBOARD API", logf)
    try:
        import urllib.request
        import urllib.error

        endpoints = [
            ("GET", f"{DASHBOARD_URL}/api/snapshot",        None),
            ("GET", f"{DASHBOARD_URL}/api/metrics",         None),
            ("GET", f"{DASHBOARD_URL}/api/mlops/status",    None),
            ("GET", f"{DASHBOARD_URL}/api/rlhf/summary",    None),
            ("GET", f"{DASHBOARD_URL}/api/price/VNM",       None),
            ("GET", f"{DASHBOARD_URL}/api/financials/VNM",  None),
            ("POST", f"{DASHBOARD_URL}/api/phase3/analyze", b'{"ticker":"VNM"}'),
        ]

        results = []
        for method, url, data in endpoints:
            try:
                req = urllib.request.Request(url, data=data,
                    headers={"Content-Type": "application/json"}, method=method)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    body = json.loads(resp.read())
                    ok   = resp.status == 200
                    log(f"  {'✓' if ok else '✗'} {method} {url.replace(DASHBOARD_URL,'')} → {resp.status}", logf)
                    results.append({"endpoint": url, "status": resp.status, "ok": ok, "sample": str(body)[:120]})
            except urllib.error.URLError as e:
                log(f"  ✗ {method} {url.replace(DASHBOARD_URL,'')} → {e}", logf)
                results.append({"endpoint": url, "ok": False, "error": str(e)})

        passed = sum(1 for r in results if r.get("ok"))
        log(f"  Dashboard: {passed}/{len(results)} endpoints OK", logf)
        return {"dashboard_url": DASHBOARD_URL, "results": results,
                "passed": passed, "total": len(results)}

    except Exception as e:
        log(f"  ✗ Dashboard check lỗi: {e}", logf)
        return {"dashboard_url": DASHBOARD_URL, "error": str(e), "passed": 0}


# ── Simulate strategy via API ─────────────────────────────────────────────────

def check_simulate_api(ticker: str, logf) -> dict:
    try:
        import urllib.request
        payload = json.dumps({
            "ticker":          ticker,
            "initial_capital": INITIAL_CAPITAL,
            "target_profit":   TARGET_PROFIT,
            "start_date":      START_DATE,
            "end_date":        END_DATE,
        }).encode()
        req = urllib.request.Request(
            f"{DASHBOARD_URL}/api/simulate_strategy",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
            if body.get("ok"):
                r = body["result"]
                log(f"  API simulate {ticker}: final_nav={r.get('final_nav'):,} | "
                    f"goal={'✓' if r.get('target_hit_date') else '✗'}", logf)
                return {"ok": True, "result": r}
            return {"ok": False, "error": body.get("error")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Analysis: Why goal not reached ───────────────────────────────────────────

def explain_result(res: dict) -> list[str]:
    reasons = []
    if res.get("goal_reached"):
        reasons.append(f"✅ Đạt mục tiêu vào ngày {res['target_hit_date']}")
        return reasons

    ret = res.get("total_return_pct", 0)
    dd  = res.get("max_drawdown_pct", 0)
    wr  = res.get("win_rate_pct", 0)
    tr  = res.get("total_trades", 0)
    gap = res.get("gap_to_goal_pct", 0)

    reasons.append(f"❌ Không đạt mục tiêu 14M VND (còn thiếu {gap:.1f}% so với mục tiêu)")

    if ret < 0:
        reasons.append(f"  → Tổng lợi nhuận âm ({ret:+.1f}%): chiến lược thua lỗ trong giai đoạn này")
    elif ret < 80:
        reasons.append(f"  → Lợi nhuận {ret:.1f}% chưa đủ (cần +180% để đạt 9M từ 5M)")

    if dd > 30:
        reasons.append(f"  → MaxDrawdown quá lớn ({dd:.1f}%): rủi ro cao, nên giảm position size")
    elif dd > 15:
        reasons.append(f"  → MaxDrawdown {dd:.1f}%: cần cải thiện stop-loss")

    if wr < 50:
        reasons.append(f"  → Win rate thấp ({wr:.1f}%): tín hiệu mua/bán chưa chính xác")

    if tr < 5:
        reasons.append(f"  → Chỉ có {tr} lệnh: không đủ giao dịch để tích lũy lợi nhuận")
    elif tr > 60:
        reasons.append(f"  → {tr} lệnh quá nhiều: phí giao dịch bào mòn lợi nhuận")

    reasons.append("  → Gợi ý: Dùng Multi-Agent AI (phase3) thay Fast-Mode để tăng chính xác")
    return reasons


# ── Print report ──────────────────────────────────────────────────────────────

def print_sim_result(res: dict, logf):
    ticker = res["ticker"]
    if not res.get("passed"):
        log(f"  [{ticker}] SKIP — {res.get('status','ERR')}", logf)
        return

    goal_mark = "✅ ĐẠT" if res["goal_reached"] else "❌ KHÔNG ĐẠT"
    log(f"", logf)
    log(f"  [{ticker}] {goal_mark}", logf)
    log(f"    Vốn ban đầu : {res['initial_capital']:>12,.0f} VND", logf)
    log(f"    NAV cuối kỳ : {res['final_nav']:>12,.0f} VND", logf)
    log(f"    Lợi nhuận   : {res['profit_vnd']:>+12,.0f} VND  ({res['total_return_pct']:+.2f}%)", logf)
    log(f"    Mục tiêu    : {res['target_nav']:>12,.0f} VND", logf)
    log(f"    Sharpe      : {res['sharpe_ratio']:.3f}", logf)
    log(f"    MaxDrawdown : {res['max_drawdown_pct']:.2f}%", logf)
    log(f"    Win rate    : {res['win_rate_pct']:.1f}%  ({res['total_trades']} lệnh)", logf)
    if res.get("target_hit_date"):
        log(f"    Đạt mục tiêu: {res['target_hit_date']}", logf)
    for reason in explain_result(res):
        log(f"    {reason}", logf)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    t0 = time.time()
    with open(TXT_REPORT, "w", encoding="utf-8") as logf:

        log("═" * 60, logf)
        log("  STOCK-AI — BACKTEST KỊCH BẢN 2024-2025", logf)
        log(f"  Thời điểm chạy : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", logf)
        log(f"  Giả lập hệ thống hoạt động: 2024-01-01 → 2025-12-31", logf)
        log(f"  Vốn ban đầu    : {INITIAL_CAPITAL:>12,.0f} VND", logf)
        log(f"  Mục tiêu lời   : {TARGET_PROFIT:>12,.0f} VND", logf)
        log(f"  Mục tiêu NAV   : {TARGET_NAV:>12,.0f} VND (+180%)", logf)
        log("═" * 60, logf)

        # 1. Dashboard check
        dashboard_result = check_dashboard(logf)

        # 2. Per-ticker simulation
        sep("MÔ PHỎNG TỪNG MÃ (Fast-Mode)", logf)
        sim_results = []
        for ticker in TICKERS:
            log(f"  Đang chạy {ticker}...", logf)
            try:
                res = run_simulation(ticker)
            except Exception as e:
                res = {"ticker": ticker, "status": f"ERROR: {e}", "passed": False}
            sim_results.append(res)
            print_sim_result(res, logf)

        # 3. API-based simulation (nếu dashboard up)
        sep("KỂT QUẢ QUA API /simulate_strategy", logf)
        api_results = []
        if dashboard_result.get("passed", 0) > 0:
            for ticker in TICKERS[:3]:  # test 3 mã qua API
                api_r = check_simulate_api(ticker, logf)
                api_results.append({"ticker": ticker, **api_r})
        else:
            log("  Dashboard không hoạt động — bỏ qua API test", logf)

        # 4. Summary
        sep("TỔNG KẾT", logf)
        ok_sims   = [r for r in sim_results if r.get("passed")]
        goal_hits = [r for r in ok_sims if r.get("goal_reached")]
        best      = max(ok_sims, key=lambda r: r.get("total_return_pct", -999)) if ok_sims else None
        worst     = min(ok_sims, key=lambda r: r.get("total_return_pct", 999)) if ok_sims else None

        log(f"", logf)
        log(f"  Tổng mã kiểm thử  : {len(TICKERS)}", logf)
        log(f"  Có dữ liệu        : {len(ok_sims)}", logf)
        log(f"  Đạt mục tiêu 14M  : {len(goal_hits)} / {len(ok_sims)}", logf)

        if best:
            log(f"  Mã tốt nhất       : {best['ticker']}  ({best['total_return_pct']:+.2f}%)", logf)
        if worst:
            log(f"  Mã kém nhất       : {worst['ticker']} ({worst['total_return_pct']:+.2f}%)", logf)

        log(f"", logf)
        log(f"  Nhận xét mô hình Fast-Mode:", logf)
        log(f"  • Chiến lược EMA+MACD+RSI hiệu quả với trend rõ ràng", logf)
        log(f"  • Mục tiêu +180% trong 2 năm là rất tham vọng cho 1 mã duy nhất", logf)
        log(f"  • Nên phân bổ vốn đa mã + dùng Multi-Agent để tăng chính xác", logf)
        log(f"  • Dashboard API: {dashboard_result.get('passed',0)}/{dashboard_result.get('total',7)} endpoints hoạt động", logf)

        elapsed = round(time.time() - t0, 1)
        log(f"", logf)
        log(f"  Thời gian chạy    : {elapsed}s", logf)
        log(f"  Báo cáo TXT       : {TXT_REPORT}", logf)
        log(f"  Báo cáo JSON      : {JSON_REPORT}", logf)
        log("═" * 60, logf)

    # Save JSON
    report = {
        "generated_at":    datetime.now().isoformat(),
        "scenario": {
            "run_year":        2026,
            "simulate_period": f"{START_DATE} → {END_DATE}",
            "initial_capital": INITIAL_CAPITAL,
            "target_profit":   TARGET_PROFIT,
            "target_nav":      TARGET_NAV,
            "tickers":         TICKERS,
        },
        "dashboard_check": dashboard_result,
        "simulation_results": sim_results,
        "api_results":     api_results,
        "summary": {
            "total_tickers":  len(TICKERS),
            "ok_tickers":     len(ok_sims),
            "goal_reached":   len(goal_hits),
            "goal_tickers":   [r["ticker"] for r in goal_hits],
            "best_ticker":    best["ticker"] if best else None,
            "best_return_pct": best["total_return_pct"] if best else None,
            "elapsed_sec":    elapsed,
        },
    }
    with open(JSON_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✓ Hoàn thành. Report: {TXT_REPORT}")


if __name__ == "__main__":
    main()
