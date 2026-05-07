"""
backtest_portfolio_2024.py — Portfolio đa mã v3 với Market Regime Detection
"""
import json, os, sys, time
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR     = os.path.join(BASE_DIR, "src")
DATA_DIR    = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
JSON_DIR    = os.path.join(REPORTS_DIR, "json")
TXT_OUT     = os.path.join(REPORTS_DIR, "backtest_portfolio_2024_report.txt")
JSON_OUT    = os.path.join(JSON_DIR,    "backtest_portfolio_2024_report.json")
sys.path.insert(0, SRC_DIR)

# ── Config ────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 5_000_000
TARGET_PROFIT   = 9_000_000
TARGET_NAV      = INITIAL_CAPITAL + TARGET_PROFIT

START_DATE = "2024-01-01"
END_DATE   = "2025-12-31"

MAX_POSITIONS = 3
FEE_PCT       = 0.2
LOT_SIZE      = 100
STOP_LOSS_PCT = 7.0
TRAILING_PCT  = 4.0
COOLDOWN_DAYS = 5

# Ngưỡng động theo Market Regime — FIX CHÍNH: tránh dùng 1 ngưỡng cho mọi pha
REGIME_THRESHOLDS = {
    "BULL":     {"buy": 0.35, "sell": -0.25},
    "SIDEWAYS": {"buy": 0.55, "sell": -0.30},
    "BEAR":     {"buy": 0.70, "sell": -0.20},
}

UNIVERSE = [
    "VNM", "VCB", "FPT", "HPG", "MBB",
    "TCB", "ACB", "MWG", "SSI", "VHM",
    "BID", "CTG", "GAS", "MSN", "PNJ",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def log(msg, f=None):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if f:
        f.write(line + "\n"); f.flush()

def sep(title="", f=None):
    n = max(0, 56 - len(title))
    log((f"── {title} " + "─" * n) if title else "─" * 60, f)

def load_df(ticker):
    for ext, reader in [(".parquet", pd.read_parquet), (".csv", pd.read_csv)]:
        sub = "parquet" if ext == ".parquet" else "csv"
        p   = os.path.join(DATA_DIR, "raw", sub, f"{ticker}_history{ext}")
        if os.path.exists(p):
            kw = {"engine": "pyarrow"} if ext == ".parquet" else {}
            df = reader(p, **kw)
            df["time"] = pd.to_datetime(df["time"])
            return df.sort_values("time").reset_index(drop=True)
    return None

def add_indicators(df):
    c = df["close"].copy()
    df["ema20"]   = c.ewm(span=20, adjust=False).mean()
    df["ema50"]   = c.ewm(span=50, adjust=False).mean()
    df["ema200"]  = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    g = delta.where(delta > 0, 0).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"]      = 100 - 100 / (1 + g / (l + 1e-9))
    df["macd"]     = c.ewm(span=12,adjust=False).mean() - c.ewm(span=26,adjust=False).mean()
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]= df["macd"] - df["macd_sig"]
    df["hl"]       = df["high"] - df["low"] if "high" in df.columns else c * 0.02
    df["atr14"]    = df["hl"].rolling(14).mean()
    if "volume" in df.columns:
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"]= df["volume"] / (df["vol_ma20"] + 1)
    else:
        df["vol_ratio"] = 1.0
    df["ema20_slope"] = df["ema20"].diff(5) / df["ema20"].shift(5) * 100
    return df

def detect_regime(vnindex_df, as_of_date: str) -> str:
    """Xác định Market Regime: BULL / SIDEWAYS / BEAR"""
    try:
        if vnindex_df is None or len(vnindex_df) < 210:
            return "SIDEWAYS"
        sub = vnindex_df[vnindex_df["time"].dt.strftime("%Y-%m-%d") <= as_of_date]
        if len(sub) < 210:
            return "SIDEWAYS"
        c   = sub["close"].values.astype(float)
        e20 = pd.Series(c).ewm(span=20, adjust=False).mean().values
        e50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
        e200= pd.Series(c).ewm(span=200,adjust=False).mean().values
        slope = (e20[-1] - e20[-6]) / (e20[-6] + 1e-9) * 100
        if e20[-1] > e50[-1] > e200[-1] and slope > 0.3:
            return "BULL"
        if e20[-1] < e50[-1] < e200[-1] and slope < -0.3:
            return "BEAR"
    except Exception:
        pass
    return "SIDEWAYS"

def score_signal(row, rlhf_w: dict) -> float:
    """
    Chấm điểm -1.0 → +1.0. Tích hợp RLHF weights nếu có.
    FIX: Yêu cầu nhiều điều kiện đồng thuận, EMA200 uptrend là bắt buộc.
    """
    score = 0.0
    try:
        if pd.isna(row["macd"]) or pd.isna(row["ema200"]):
            return 0.0
        c, e20, e50, e200 = row["close"], row["ema20"], row["ema50"], row["ema200"]
        rsi  = row["rsi"]
        macd, msig, mhst = row["macd"], row["macd_sig"], row["macd_hist"]
        vr   = row["vol_ratio"] if not pd.isna(row["vol_ratio"]) else 1.0
        slope= row["ema20_slope"] if not pd.isna(row["ema20_slope"]) else 0.0

        # FIX 1: Bắt buộc ở trên EMA200 mới BUY
        if c < e200 * 0.99:
            return max(-0.5, -0.5)

        # Trend (RLHF: tăng/giảm weight technical)
        tech_w = rlhf_w.get("technical", 1.0)
        if c > e20 > e50 > e200:  score += 0.30 * tech_w
        elif c > e20 > e50:       score += 0.20 * tech_w
        elif c > e20:             score += 0.10 * tech_w
        if c < e20 < e50:         score -= 0.25 * tech_w
        if c < e50 < e200:        score -= 0.45 * tech_w

        # EMA slope xác nhận (mới)
        if slope > 0.5:           score += 0.10
        elif slope < -0.5:        score -= 0.10

        # MACD momentum — cần cả 2 điều kiện
        if macd > msig and mhst > 0:  score += 0.25
        elif macd > msig:             score += 0.10
        if macd < msig and mhst < 0:  score -= 0.25
        elif macd < msig:             score -= 0.10

        # RSI quality
        if 45 < rsi < 65:   score += 0.15
        if 35 < rsi <= 45:  score += 0.10
        if rsi > 70:        score -= 0.20
        if rsi > 78:        score -= 0.35
        if rsi < 30:        score -= 0.20

        # Volume confirm
        if vr > 1.5 and score > 0:  score += 0.10
        if vr < 0.5:                score -= 0.05
    except Exception:
        pass
    return max(-1.0, min(1.0, score))

def _load_rlhf_weights() -> dict:
    path = os.path.join(JSON_DIR, "rlhf_weights.json")
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f).get("weights", {})
    except Exception:
        pass
    return {}

# ── Slot ─────────────────────────────────────────────────────────────────────
class Slot:
    def __init__(self, sid, alloc):
        self.id = sid; self.cash = alloc
        self.shares = 0; self.ticker = None
        self.ep = 0.0; self.ep_date = None
        self.peak_price = 0.0; self.cur_price = 0.0
        self.trades = []

    @property
    def nav(self):
        if self.shares > 0 and self.cur_price > 0:
            return self.cash + self.shares * self.cur_price
        return self.cash

    def open(self, ticker, price, date, fee_r):
        max_sh = int(self.cash / (price * (1 + fee_r)))
        actual = (max_sh // LOT_SIZE) * LOT_SIZE
        if actual <= 0: return False
        fee = actual * price * fee_r
        self.cash -= (actual * price + fee)
        self.shares = actual; self.ticker = ticker
        self.ep = price; self.ep_date = date
        self.peak_price = price; self.cur_price = price
        self.trades.append({"action": "BUY", "date": date, "ticker": ticker,
                            "price": price, "shares": actual, "fee": round(fee)})
        return True

    def close(self, price, date, fee_r, reason="SELL"):
        if self.shares == 0: return 0.0
        proceeds = self.shares * price
        fee = proceeds * fee_r
        profit = proceeds - fee - self.shares * self.ep * (1 + fee_r)
        self.cash += (proceeds - fee)
        self.trades.append({"action": reason, "date": date, "ticker": self.ticker,
                            "price": price, "shares": self.shares, "fee": round(fee),
                            "profit": round(profit), "profit_pct": round((price-self.ep)/self.ep*100, 2),
                            "entry_price": self.ep, "entry_date": self.ep_date})
        self.shares = 0; self.ticker = None; self.ep = 0.0
        self.ep_date = None; self.peak_price = 0.0
        return profit

    def update(self, price):
        self.cur_price = price
        if price > self.peak_price: self.peak_price = price

    def check_stop(self, price) -> str:
        if not self.shares or not self.ep: return ""
        if (price - self.ep) / self.ep * 100 <= -STOP_LOSS_PCT: return "STOP_LOSS"
        if self.peak_price > 0 and (price-self.peak_price)/self.peak_price*100 <= -TRAILING_PCT and price < self.ep*0.98:
            return "TRAILING_STOP"
        return ""

# ── Portfolio Engine ──────────────────────────────────────────────────────────
def run_portfolio():
    all_dfs = {}
    for t in UNIVERSE:
        df = load_df(t)
        if df is None: continue
        df = add_indicators(df)
        mask = (df["time"] >= START_DATE) & (df["time"] <= END_DATE)
        df = df[mask].reset_index(drop=True)
        if len(df) >= 50: all_dfs[t] = df

    # Load VNINDEX cho regime detection
    vnindex_df = load_df("VNINDEX")
    if vnindex_df is not None:
        vnindex_df = add_indicators(vnindex_df)

    # Load RLHF weights (kết nối vòng học)
    rlhf_w = _load_rlhf_weights()

    dates = sorted(set(
        d for df in all_dfs.values()
        for d in df["time"].dt.strftime("%Y-%m-%d").tolist()
    ))

    alloc = INITIAL_CAPITAL / MAX_POSITIONS
    slots = [Slot(i, alloc) for i in range(MAX_POSITIONS)]

    fee_r = FEE_PCT / 100.0
    portfolio_nav = []
    target_hit = None
    peak_pnav = INITIAL_CAPITAL
    max_dd = 0.0
    daily_returns = []
    cooldowns: dict = {}
    regime_log: dict = {}

    def get_row(ticker, date_str):
        df = all_dfs.get(ticker)
        if df is None: return None
        sub = df[df["time"].dt.strftime("%Y-%m-%d") == date_str]
        return sub.iloc[0] if len(sub) > 0 else None

    prev_nav = INITIAL_CAPITAL
    prev_regime = "SIDEWAYS"

    for date_str in dates:
        # Detect regime ngày hôm nay
        regime = detect_regime(vnindex_df, date_str)
        if regime != prev_regime:
            regime_log[date_str] = regime
            prev_regime = regime

        thresh = REGIME_THRESHOLDS[regime]
        buy_th  = thresh["buy"]
        sell_th = thresh["sell"]

        # Stop-loss check
        for s in slots:
            if s.shares and s.ticker:
                row = get_row(s.ticker, date_str)
                if row is not None:
                    price = float(row["close"])
                    s.update(price)
                    reason = s.check_stop(price)
                    if reason:
                        tk = s.ticker
                        s.close(price, date_str, fee_r, reason)
                        cooldowns[tk] = date_str

        # Score tất cả mã
        scores = {}
        for t in all_dfs:
            row = get_row(t, date_str)
            if row is not None:
                scores[t] = score_signal(row, rlhf_w)

        # Đóng vị thế nếu score < ngưỡng SELL của regime
        for s in slots:
            if s.shares and s.ticker:
                row = get_row(s.ticker, date_str)
                if row is not None and scores.get(s.ticker, 0) < sell_th:
                    tk = s.ticker
                    s.close(float(row["close"]), date_str, fee_r, "SELL")
                    cooldowns[tk] = date_str

        # Mở vị thế mới vào slot trống
        occupied = {s.ticker for s in slots if s.ticker}
        on_cd = {t for t, d in cooldowns.items()
                 if (pd.Timestamp(date_str) - pd.Timestamp(d)).days < COOLDOWN_DAYS}
        candidates = sorted(
            [(t, sc) for t, sc in scores.items()
             if t not in occupied and t not in on_cd and sc >= buy_th],
            key=lambda x: -x[1]
        )
        for s in slots:
            if not s.shares and candidates:
                ticker, sc = candidates.pop(0)
                row = get_row(ticker, date_str)
                if row is not None:
                    s.open(ticker, float(row["close"]), date_str, fee_r)

        # Cập nhật giá & tính NAV
        pnav = 0.0
        for s in slots:
            if s.shares and s.ticker:
                row = get_row(s.ticker, date_str)
                if row is not None:
                    s.cur_price = float(row["close"])
            pnav += s.nav

        if pnav > peak_pnav: peak_pnav = pnav
        dd = (peak_pnav - pnav) / peak_pnav * 100
        if dd > max_dd: max_dd = dd
        daily_returns.append((pnav - prev_nav) / prev_nav if prev_nav > 0 else 0)
        prev_nav = pnav
        portfolio_nav.append({"date": date_str, "nav": round(pnav), "drawdown": round(dd, 2), "regime": regime})
        if target_hit is None and pnav >= TARGET_NAV:
            target_hit = date_str

    # Liquidate cuối kỳ
    last = dates[-1] if dates else END_DATE
    for s in slots:
        if s.shares and s.ticker:
            row = get_row(s.ticker, last)
            if row is not None:
                s.close(float(row["close"]), last, fee_r, "LIQUIDATE")

    final_nav = sum(s.cash for s in slots)
    arr = np.array(daily_returns)
    sharpe = float(arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 1e-9 else 0.0

    all_trades = []
    for s in slots: all_trades.extend(s.trades)
    all_trades.sort(key=lambda x: x["date"])

    sell_t = [t for t in all_trades if any(k in t["action"] for k in ("SELL","STOP","LIQ"))]
    win_t  = [t for t in sell_t if t.get("profit", 0) > 0]
    total_ret = (final_nav - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    ticker_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "profit": 0.0})
    for t in sell_t:
        tk = t.get("ticker", "?")
        ticker_stats[tk]["trades"] += 1
        ticker_stats[tk]["profit"] += t.get("profit", 0)
        if t.get("profit", 0) > 0: ticker_stats[tk]["wins"] += 1

    return {
        "final_nav": round(final_nav),
        "total_return_pct": round(total_ret, 2),
        "profit_vnd": round(final_nav - INITIAL_CAPITAL),
        "goal_reached": final_nav >= TARGET_NAV,
        "target_hit_date": target_hit,
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 4),
        "total_trades": len(sell_t),
        "win_rate_pct": round(len(win_t)/len(sell_t)*100 if sell_t else 0, 2),
        "win_trades": len(win_t),
        "rlhf_weights_used": rlhf_w,
        "regime_changes": regime_log,
        "tickers_used": sorted(all_dfs.keys()),
        "portfolio_nav": portfolio_nav,
        "all_trades": all_trades,
        "ticker_stats": {k: dict(v) for k, v in ticker_stats.items()},
    }

# ── Explain ───────────────────────────────────────────────────────────────────
def explain(res, f):
    if res["goal_reached"]:
        log(f"  ✅ ĐẠT MỤC TIÊU! Ngày: {res['target_hit_date']}", f)
    else:
        gap = TARGET_NAV - res["final_nav"]
        log(f"  ❌ Chưa đạt — còn thiếu {gap:,.0f} VND ({gap/TARGET_NAV*100:.1f}%)", f)
    log("", f)
    ret = res["total_return_pct"]
    dd  = res["max_drawdown_pct"]
    wr  = res["win_rate_pct"]
    if ret >= 180: log("  ✅ Lợi nhuận đạt/vượt +180%!", f)
    elif ret > 0:  log(f"  ⚠  Lợi nhuận {ret:+.1f}% — cần thêm {180-ret:.0f}% nữa", f)
    else:          log(f"  ❌ Lợi nhuận âm ({ret:+.1f}%)", f)
    if dd > 25:    log(f"  ❌ MaxDrawdown {dd:.1f}% quá cao", f)
    elif dd > 15:  log(f"  ⚠  MaxDrawdown {dd:.1f}% — cần cải thiện stop-loss", f)
    else:          log(f"  ✅ MaxDrawdown {dd:.1f}% — kiểm soát tốt", f)
    if wr >= 55:   log(f"  ✅ Win rate {wr:.1f}% — tốt", f)
    elif wr >= 40: log(f"  ⚠  Win rate {wr:.1f}% — cần tăng ngưỡng BUY", f)
    else:          log(f"  ❌ Win rate {wr:.1f}% — nhiễu nhiều", f)
    log("", f)
    rc = res.get("regime_changes", {})
    log(f"  📊 Market Regime thay đổi {len(rc)} lần:", f)
    for d, r in list(rc.items())[:8]:
        log(f"     {d}: → {r}", f)
    log("", f)
    rw = res.get("rlhf_weights_used", {})
    if rw:
        log(f"  🧠 RLHF weights đã áp dụng: {rw}", f)
    else:
        log("  ⚠  RLHF weights chưa có — chạy phase3 để tích lũy dữ liệu học", f)
    log("", f)
    log("  💡 Nguyên nhân RL không tiến bộ (theo phân tích):", f)
    log("  1. EWM không phải RL thực — không có State/Policy/Memory", f)
    log("  2. Reward signal quá thô (chỉ +/-1), không phạt sai số lớn", f)
    log("  3. Backtest v1/v2 không kết nối RLHF weights — v3 này đã sửa", f)
    log("  4. Thị trường 2024-2025 sideway 60% thời gian — EMA/MACD báo nhiễu", f)
    log("  5. Regime thay đổi không được detect — v3 này đã thêm", f)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(JSON_DIR, exist_ok=True)
    t0 = time.time()
    with open(TXT_OUT, "w", encoding="utf-8") as f:
        log("═" * 60, f)
        log("  STOCK-AI — PORTFOLIO v3 (Regime + RLHF) 2024-2025", f)
        log(f"  Chạy lúc      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f)
        log(f"  Vốn           : {INITIAL_CAPITAL:>12,.0f} VND", f)
        log(f"  Mục tiêu NAV  : {TARGET_NAV:>12,.0f} VND (+180%)", f)
        log(f"  Slots         : {MAX_POSITIONS} | SL: -{STOP_LOSS_PCT}% | Trail: -{TRAILING_PCT}%", f)
        log(f"  Universe      : {len(UNIVERSE)} mã", f)
        log("═" * 60, f)
        sep("CHẠY MÔ PHỎNG", f)
        log("  Đang chạy...", f)
        try:
            res = run_portfolio()
        except Exception as e:
            import traceback
            log(f"  ❌ LỖI: {e}", f)
            log(traceback.format_exc(), f)
            return
        elapsed = round(time.time() - t0, 2)
        sep("KẾT QUẢ PORTFOLIO", f)
        log(f"  NAV cuối kỳ   : {res['final_nav']:>12,.0f} VND", f)
        log(f"  Lợi nhuận     : {res['profit_vnd']:>+12,.0f} VND  ({res['total_return_pct']:+.2f}%)", f)
        log(f"  Mục tiêu      : {TARGET_NAV:>12,.0f} VND", f)
        log(f"  Sharpe        : {res['sharpe_ratio']:.4f}", f)
        log(f"  MaxDrawdown   : {res['max_drawdown_pct']:.2f}%", f)
        log(f"  Win Rate      : {res['win_rate_pct']:.1f}%  ({res['win_trades']}/{res['total_trades']})", f)
        log("", f)
        sep("HIỆU SUẤT TỪNG MÃ", f)
        ts = res["ticker_stats"]
        log(f"  {'Mã':<6} {'Lệnh':>5} {'Win':>5} {'Win%':>7} {'Lợi nhuận':>14}", f)
        log(f"  {'─'*6} {'─'*5} {'─'*5} {'─'*7} {'─'*14}", f)
        for tk in sorted(ts, key=lambda x: -ts[x]["profit"]):
            d  = ts[tk]
            wr = d["wins"]/d["trades"]*100 if d["trades"] > 0 else 0
            log(f"  {tk:<6} {d['trades']:>5} {d['wins']:>5} {wr:>6.1f}% {d['profit']:>+14,.0f}", f)
        log("", f)
        sep("PHÂN TÍCH", f)
        explain(res, f)
        log(f"  ⏱ {elapsed}s | TXT: {TXT_OUT}", f)
        log("═" * 60, f)

    report = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "initial_capital": INITIAL_CAPITAL, "target_nav": TARGET_NAV,
            "start_date": START_DATE, "end_date": END_DATE,
            "max_positions": MAX_POSITIONS, "stop_loss_pct": STOP_LOSS_PCT,
            "trailing_pct": TRAILING_PCT, "fee_pct": FEE_PCT,
            "regime_thresholds": REGIME_THRESHOLDS, "universe": UNIVERSE,
        },
        "result":        {k: v for k, v in res.items() if k not in ("portfolio_nav","all_trades")},
        "portfolio_nav": res["portfolio_nav"],
        "all_trades":    res["all_trades"],
        "ticker_stats":  res["ticker_stats"],
        "elapsed_sec":   elapsed,
    }
    with open(JSON_OUT, "w", encoding="utf-8") as jf:
        json.dump(report, jf, indent=2, ensure_ascii=False, default=str)
    print(f"\n✓ Hoàn thành! Report: {TXT_OUT}")


if __name__ == "__main__":
    main()
