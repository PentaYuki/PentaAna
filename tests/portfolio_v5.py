"""
portfolio_v5.py — Portfolio với Blacklist + Sentiment + 5yr data
Chạy: python tests/portfolio_v5.py
"""
import os, sys, json, sqlite3, time
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
JSON_DIR = os.path.join(DATA_DIR, "reports", "json")
DB_PATH  = os.path.join(DATA_DIR, "news.db")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# ── Config ────────────────────────────────────────────────────────────────────
START, END      = "2024-01-01", "2025-12-31"
INITIAL, TARGET = 5_000_000, 14_000_000
FEE_R, LOT      = 0.002, 100
SL, TRAIL, CD   = 7.0, 4.0, 5
MAX_POS         = 3

REGIME_BUY = {"BULL": 0.28, "SIDEWAYS": 0.45, "BEAR": 0.60}
REGIME_SEL = {"BULL": -0.22, "SIDEWAYS": -0.28, "BEAR": -0.18}

FULL_UNIVERSE = ["VNM","VCB","FPT","HPG","MBB","TCB","ACB","MWG","SSI","VHM","BID","CTG","GAS","MSN","PNJ"]

# ── Load Blacklist ─────────────────────────────────────────────────────────────
def load_blacklist():
    p = os.path.join(JSON_DIR, "dynamic_blacklist.json")
    if os.path.exists(p):
        return list(json.load(open(p)).keys())
    return []

# ── Load RLHF weights ─────────────────────────────────────────────────────────
def load_rlhf():
    p = os.path.join(JSON_DIR, "rlhf_weights.json")
    if os.path.exists(p):
        return json.load(open(p)).get("weights", {})
    return {}

# ── Sentiment từ DB ───────────────────────────────────────────────────────────
_SENT_CACHE: dict = {}

def get_sentiment(ticker: str, as_of_date: str, lookback: int = 90) -> float:
    key = f"{ticker}_{as_of_date}"
    if key in _SENT_CACHE:
        return _SENT_CACHE[key]
    try:
        d1 = as_of_date
        d0 = (datetime.strptime(d1, "%Y-%m-%d") - timedelta(days=lookback)).strftime("%Y-%m-%d")
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT AVG(sentiment_score) FROM news WHERE ticker=? AND pub_date>=? AND pub_date<=? AND sentiment_score IS NOT NULL",
                (ticker, d0, d1)
            ).fetchone()
        val = float(row[0]) if row and row[0] is not None else 0.0
    except Exception:
        val = 0.0
    _SENT_CACHE[key] = val
    return val

# ── Synthetic Index ───────────────────────────────────────────────────────────
def build_synthetic_index():
    dfs = []
    for t in ["VNM", "MBB"]:
        p = os.path.join(DATA_DIR, "raw", "parquet", f"{t}_history.parquet")
        if not os.path.exists(p): continue
        df = pd.read_parquet(p, engine="pyarrow")
        df["time"] = pd.to_datetime(df["time"])
        dfs.append(df[["time","close"]].rename(columns={"close": t}).set_index("time"))
    if not dfs: return None
    merged = dfs[0]
    for d in dfs[1:]: merged = merged.join(d, how="outer")
    merged = merged.ffill().dropna()
    normed = merged / merged.iloc[0] * 1000
    merged["syn"] = normed.mean(axis=1)
    return merged.reset_index()

def detect_regime(idx_df, date_str: str) -> str:
    try:
        sub = idx_df[idx_df["time"].dt.strftime("%Y-%m-%d") <= date_str]
        if len(sub) < 60: return "SIDEWAYS"
        c   = sub["syn"].values.astype(float)
        e20 = pd.Series(c).ewm(span=20, adjust=False).mean().values
        e50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
        sl5 = (e20[-1] - e20[-min(6,len(e20))]) / (e20[-min(6,len(e20))] + 1e-9) * 100
        if e20[-1] > e50[-1] and sl5 > 0.2: return "BULL"
        if e20[-1] < e50[-1] and sl5 < -0.2: return "BEAR"
    except Exception:
        pass
    return "SIDEWAYS"

# ── Indicators ────────────────────────────────────────────────────────────────
def add_indicators(df):
    c = df["close"].copy()
    df["ema20"]  = c.ewm(span=20, adjust=False).mean()
    df["ema50"]  = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    d = c.diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    df["rsi"]      = 100 - 100 / (1 + g / (l + 1e-9))
    df["macd"]     = c.ewm(span=12,adjust=False).mean() - c.ewm(span=26,adjust=False).mean()
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]= df["macd"] - df["macd_sig"]
    if "volume" in df.columns:
        df["vol_ma20"]  = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / (df["vol_ma20"] + 1)
    else:
        df["vol_ratio"] = 1.0
    return df

# ── Signal v5 ─────────────────────────────────────────────────────────────────
def score_v5(row, ticker: str, rlhf_w: dict, date_str: str) -> float:
    score = 0.0
    try:
        if pd.isna(row.get("macd", float("nan"))) or pd.isna(row.get("ema200", float("nan"))):
            return 0.0
        c, e20, e50, e200 = row["close"], row["ema20"], row["ema50"], row["ema200"]
        rsi   = row["rsi"]
        macd, msig, mhst = row["macd"], row["macd_sig"], row["macd_hist"]
        vr    = row["vol_ratio"] if not pd.isna(row.get("vol_ratio", float("nan"))) else 1.0
        tw    = rlhf_w.get("technical", 0.40)
        sw    = rlhf_w.get("sentiment", 0.25)

        # Bắt buộc: trên EMA200
        if c < e200 * 0.99: return -0.4

        # Trend
        if c > e20 > e50 > e200:   score += 0.30 * tw * 2.5
        elif c > e20 > e50:        score += 0.20 * tw * 2.5
        elif c > e20:              score += 0.10 * tw * 2.5
        if c < e20 < e50:          score -= 0.30
        if c < e50 < e200:         score -= 0.50

        # MACD
        if macd > msig and mhst > 0:   score += 0.25
        elif macd > msig:              score += 0.10
        if macd < msig and mhst < 0:   score -= 0.25
        elif macd < msig:              score -= 0.10

        # RSI
        if 45 < rsi < 65:  score += 0.15
        if 35 < rsi <= 45: score += 0.10
        if rsi > 72:       score -= 0.25
        if rsi < 28:       score -= 0.25

        # Volume
        if vr > 1.5 and score > 0: score += 0.10
        if vr < 0.5:               score -= 0.05

        # Sentiment từ DB thật (price-proxy)
        sent = get_sentiment(ticker, date_str, lookback=90)
        score += sent * sw * 0.5

    except Exception:
        pass
    return float(np.clip(score, -1.0, 1.0))

# ── Slot ──────────────────────────────────────────────────────────────────────
class Slot:
    def __init__(self, alloc):
        self.cash = alloc; self.shares = 0; self.ticker = None
        self.ep = 0.0; self.peak = 0.0; self.cur = 0.0
        self.ep_date = None; self.trades = []

    @property
    def nav(self):
        return self.cash + self.shares * self.cur if self.shares > 0 else self.cash

    def open(self, ticker, price, date):
        sh = (int(self.cash / (price * (1 + FEE_R))) // LOT) * LOT
        if sh <= 0: return False
        fee = sh * price * FEE_R
        self.cash -= sh * price + fee
        self.shares = sh; self.ticker = ticker; self.ep = price
        self.ep_date = date; self.peak = price; self.cur = price
        self.trades.append({"action":"BUY","date":date,"ticker":ticker,"price":price,"shares":sh})
        return True

    def close(self, price, date, reason):
        if not self.shares: return 0.0
        pr = self.shares * price; fee = pr * FEE_R
        pnl = pr - fee - self.shares * self.ep * (1 + FEE_R)
        self.cash += pr - fee
        self.trades.append({"action":reason,"date":date,"ticker":self.ticker,
            "price":price,"profit":round(pnl),"profit_pct":round((price-self.ep)/self.ep*100,2)})
        self.shares = 0; self.ticker = None; self.ep = 0; self.peak = 0
        return pnl

    def upd(self, p):
        self.cur = p
        if p > self.peak: self.peak = p

    def stop_reason(self, p):
        if not self.shares or not self.ep: return ""
        if (p - self.ep) / self.ep * 100 <= -SL: return "STOP_LOSS"
        if self.peak > 0 and (p - self.peak) / self.peak * 100 <= -TRAIL and p < self.ep * 0.98:
            return "TRAILING_STOP"
        return ""

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 60)
    print("  PORTFOLIO v5 — Blacklist + Sentiment + 5yr data")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    blacklist = load_blacklist()
    rlhf_w    = load_rlhf()
    universe  = [t for t in FULL_UNIVERSE if t not in blacklist]
    print(f"\n  Blacklist ({len(blacklist)}): {blacklist}")
    print(f"  Universe  ({len(universe)}): {universe}")
    print(f"  RLHF: tech={rlhf_w.get('technical',0):.3f} sent={rlhf_w.get('sentiment',0):.3f}")

    # Load data
    all_dfs = {}
    for t in universe:
        p = os.path.join(DATA_DIR, "raw", "parquet", f"{t}_history.parquet")
        if not os.path.exists(p): continue
        df = pd.read_parquet(p, engine="pyarrow")
        df["time"] = pd.to_datetime(df["time"])
        all_dfs[t] = add_indicators(df).sort_values("time").reset_index(drop=True)
    print(f"  Data loaded: {len(all_dfs)} mã")

    idx_df = build_synthetic_index()
    print(f"  Synthetic Index: {len(idx_df)} rows ({idx_df['time'].min().date()} -> {idx_df['time'].max().date()})")

    # Test regime
    test_r = {d: detect_regime(idx_df, d) for d in ["2024-03-01","2024-07-01","2024-12-01","2025-06-01"]}
    print(f"  Regime samples: {test_r}")

    # Get trading dates
    sk    = "MBB" if "MBB" in all_dfs else next(iter(all_dfs))
    dates = all_dfs[sk][(all_dfs[sk]["time"] >= START) & (all_dfs[sk]["time"] <= END)]["time"].dt.strftime("%Y-%m-%d").tolist()
    print(f"  Trading dates : {len(dates)} phiên")

    def gr(ticker, date_str):
        df  = all_dfs.get(ticker)
        if df is None: return None
        sub = df[df["time"].dt.strftime("%Y-%m-%d") == date_str]
        return sub.iloc[0] if len(sub) > 0 else None

    # Simulation
    slots    = [Slot(INITIAL / MAX_POS) for _ in range(MAX_POS)]
    cooldowns= {}; peak_nav = INITIAL; max_dd = 0.0
    prev_nav = INITIAL; daily_r = []; target_hit = None
    regime_log = {}; prev_regime = "SIDEWAYS"
    ts = defaultdict(lambda: {"t": 0, "w": 0, "p": 0.0})

    print("\n  Đang chạy simulation...")

    for date_str in dates:
        regime = detect_regime(idx_df, date_str)
        if regime != prev_regime:
            regime_log[date_str] = regime; prev_regime = regime
        bt = REGIME_BUY[regime]; st = REGIME_SEL[regime]

        # Stop-loss check
        for s in slots:
            if s.shares and s.ticker:
                row = gr(s.ticker, date_str)
                if row is not None:
                    p = float(row["close"]); s.upd(p)
                    reason = s.stop_reason(p)
                    if reason:
                        tk = s.ticker; pnl = s.close(p, date_str, reason); cooldowns[tk] = date_str
                        ts[tk]["t"] += 1; ts[tk]["p"] += pnl; ts[tk]["w"] += (1 if pnl > 0 else 0)

        # Score all
        scores = {}
        for t in all_dfs:
            row = gr(t, date_str)
            if row is not None:
                scores[t] = score_v5(row, t, rlhf_w, date_str)

        # Sell check
        for s in slots:
            if s.shares and s.ticker:
                row = gr(s.ticker, date_str)
                if row is not None and scores.get(s.ticker, 0) < st:
                    tk = s.ticker; p = float(row["close"])
                    pnl = s.close(p, date_str, "SELL"); cooldowns[tk] = date_str
                    ts[tk]["t"] += 1; ts[tk]["p"] += pnl; ts[tk]["w"] += (1 if pnl > 0 else 0)

        # Buy
        occ   = {s.ticker for s in slots if s.ticker}
        on_cd = {t for t, d in cooldowns.items()
                 if (pd.Timestamp(date_str) - pd.Timestamp(d)).days < CD}
        cands = sorted(
            [(t, sc) for t, sc in scores.items() if t not in occ and t not in on_cd and sc >= bt],
            key=lambda x: -x[1]
        )
        for s in slots:
            if not s.shares and cands:
                tk, sc = cands.pop(0)
                row = gr(tk, date_str)
                if row is not None: s.open(tk, float(row["close"]), date_str)

        # NAV
        pnav = 0.0
        for s in slots:
            if s.shares and s.ticker:
                row = gr(s.ticker, date_str)
                if row is not None: s.cur = float(row["close"])
            pnav += s.nav

        if pnav > peak_nav: peak_nav = pnav
        dd = (peak_nav - pnav) / peak_nav * 100
        if dd > max_dd: max_dd = dd
        daily_r.append((pnav - prev_nav) / prev_nav if prev_nav > 0 else 0)
        prev_nav = pnav
        if target_hit is None and pnav >= TARGET: target_hit = date_str

    # Liquidate
    last = dates[-1]
    for s in slots:
        if s.shares and s.ticker:
            row = gr(s.ticker, last)
            if row is not None:
                tk = s.ticker; pnl = s.close(float(row["close"]), last, "LIQUIDATE")
                ts[tk]["t"] += 1; ts[tk]["p"] += pnl; ts[tk]["w"] += (1 if pnl > 0 else 0)

    final = sum(s.cash for s in slots)
    ret   = (final - INITIAL) / INITIAL * 100
    arr   = np.array(daily_r)
    sharpe= float(arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 1e-9 else 0.0
    all_t = []
    for s in slots: all_t.extend(s.trades)
    sell_t = [t for t in all_t if any(k in t["action"] for k in ("SELL","STOP","LIQ"))]
    win_t  = [t for t in sell_t if t.get("profit", 0) > 0]
    win_rate = len(win_t) / len(sell_t) * 100 if sell_t else 0

    elapsed = round(time.time() - t0, 2)

    print()
    print("=" * 60)
    print("  KET QUA PORTFOLIO v5")
    print("=" * 60)
    goal_str = "DAT" if final >= TARGET else "Chua dat"
    print(f"  NAV cuoi ky  : {final:>12,.0f} VND")
    print(f"  Loi nhuan    : {final-INITIAL:>+12,.0f} VND  ({ret:+.2f}%)")
    print(f"  Muc tieu     : [{goal_str}] 14,000,000 VND")
    if target_hit: print(f"  Dat muc tieu : {target_hit}")
    print(f"  Sharpe       : {sharpe:.4f}")
    print(f"  MaxDrawdown  : {max_dd:.2f}%")
    print(f"  Win Rate     : {win_rate:.1f}%  ({len(win_t)}/{len(sell_t)} lenh)")
    print(f"  Regime change: {len(regime_log)} lan")
    print(f"  Universe     : {len(universe)} ma | Blacklist: {len(blacklist)} ma")
    print()
    print(f"  {'Ma':<6} {'Lenh':>5} {'Win':>5} {'Win%':>7} {'Loi nhuan':>14}")
    print(f"  {'-'*6} {'-'*5} {'-'*5} {'-'*7} {'-'*14}")
    for tk in sorted(ts, key=lambda x: -ts[x]["p"]):
        d  = ts[tk]; wr = d["w"]/d["t"]*100 if d["t"] > 0 else 0
        print(f"  {tk:<6} {d['t']:>5} {d['w']:>5} {wr:>6.1f}% {d['p']:>+14,.0f}")
    print()
    print("  So sanh cac phien ban:")
    print("  v1: -11% | v2: -32% | v3: -22% | v4: +5.6% | v5: ???")
    print()
    print(f"  Regime thay doi ({len(regime_log)} lan):")
    for k, v in list(regime_log.items())[:8]:
        print(f"    {k}: -> {v}")
    print()
    print(f"  Blacklist ap dung: {blacklist}")
    print(f"  Sentiment records: 330 records (price-proxy)")
    print(f"  Time: {elapsed}s")
    print("=" * 60)

    # Save
    report = {
        "generated_at": datetime.now().isoformat(),
        "version": "v5",
        "config": {"universe": universe, "blacklist": blacklist, "rlhf_weights": rlhf_w},
        "result": {
            "final_nav": round(final), "profit": round(final-INITIAL),
            "return_pct": round(ret, 2), "goal": final >= TARGET,
            "target_hit": target_hit, "sharpe": round(sharpe, 4),
            "max_dd": round(max_dd, 2), "win_rate": round(win_rate, 2),
            "trades": len(sell_t), "wins": len(win_t),
            "regime_changes": len(regime_log),
        },
        "ticker_stats": {k: dict(v) for k, v in ts.items()},
        "elapsed_sec": elapsed,
    }
    out = os.path.join(JSON_DIR, "portfolio_v5_report.json")
    with open(out, "w", encoding="utf-8") as jf:
        json.dump(report, jf, indent=2, ensure_ascii=False, default=str)
    print(f"  Report: {out}")

if __name__ == "__main__":
    main()
