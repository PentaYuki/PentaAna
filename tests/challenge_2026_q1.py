"""
challenge_2026_q1.py — Thử thách 3 tháng đầu năm 2026
===================================================
"""
import os, sys, json, pandas as pd, numpy as np
from datetime import datetime
from collections import defaultdict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# --- CONFIG Q1 2026 ---
START, END = "2026-01-01", "2026-03-31"
INITIAL    = 5_000_000
UNIVERSE   = ["FPT", "MBB", "TCB", "ACB", "SSI", "CTG", "PNJ"]

def get_max_positions(regime):
    if regime == "BULL": return 6
    if regime == "SIDEWAYS": return 3
    if regime == "BEAR": return 1
    return 3

class Slot:
    def __init__(self, id):
        self.id = id; self.cash = 0; self.shares = 0; self.ticker = None
        self.ep = 0.0; self.cur = 0.0; self.peak = 0.0; self.trades = []
    @property
    def nav(self): return self.cash + (self.shares * self.cur if self.shares > 0 else 0)
    def open(self, t, p, d):
        sh = (int(self.cash / (p * 1.002)) // 100) * 100
        if sh <= 0: return False
        self.cash -= sh*p*1.002; self.shares = sh; self.ticker = t; self.ep = p; self.cur = p; self.peak = p
        self.trades.append(f"BUY {t} @ {p:,.0f}")
        return True
    def close(self, p, d, r):
        if not self.shares: return 0
        v = self.shares * p * 0.998; pnl = v - self.shares*self.ep*1.002
        self.cash += v; self.shares = 0; self.ticker = None
        self.trades.append(f"SELL {r} @ {p:,.0f} PNL: {pnl:,.0f}")
        return pnl

def run_challenge():
    print("="*60); print(f"  BACKTEST THỰC TẾ Q1/2026 (01/01 - 31/03)"); print("="*60)
    
    all_dfs = {}
    for t in UNIVERSE:
        p = os.path.join(DATA_DIR, "raw", "parquet", f"{t}_history.parquet")
        df = pd.read_parquet(p); df["time"] = pd.to_datetime(df["time"])
        c = df["close"]
        df["ema20"] = c.ewm(span=20).mean(); df["ema200"] = c.ewm(span=200).mean()
        all_dfs[t] = df.sort_values("time")

    idx_df = all_dfs["MBB"].copy() # Proxy
    dates = idx_df[(idx_df["time"] >= START) & (idx_df["time"] <= END)]["time"].dt.strftime("%Y-%m-%d").tolist()
    
    slots = [Slot(i) for i in range(6)]
    for s in slots: s.cash = INITIAL / 3
    
    peak_nav = INITIAL; max_dd = 0; ts_pnl = defaultdict(float)

    for d_str in dates:
        # Detect Regime đơn giản cho Q1
        sub = idx_df[idx_df["time"].dt.strftime("%Y-%m-%d") <= d_str]
        regime = "SIDEWAYS"
        if len(sub) > 2:
            c = sub.iloc[-1]["close"]; e20 = sub.iloc[-1]["ema20"]
            regime = "BULL" if c > e20 else "BEAR"
        
        max_p = get_max_positions(regime)
        
        # Update
        for s in slots:
            if s.shares:
                r = all_dfs[s.ticker][all_dfs[s.ticker]["time"].dt.strftime("%Y-%m-%d")==d_str]
                if not r.empty:
                    p = float(r.iloc[0]["close"]); s.cur = p
                    if p > s.peak: s.peak = p
                    if (p-s.ep)/s.ep*100 <= -7: s.close(p, d_str, "SL")

        # Logic mua bán
        for t in UNIVERSE:
            r = all_dfs[t][all_dfs[t]["time"].dt.strftime("%Y-%m-%d")==d_str]
            if not r.empty:
                row = r.iloc[0]; p = float(row["close"])
                # Mua nếu giá > EMA20 và còn slot
                occ = {s.ticker for s in slots if s.ticker}
                if t not in occ and p > row["ema20"] and p > row["ema200"]:
                    for i in range(max_p):
                        if not slots[i].shares:
                            slots[i].cash = sum(sl.nav for sl in slots) / max_p
                            slots[i].open(t, p, d_str); break
                # Bán nếu giá < EMA20
                elif t in occ and p < row["ema20"]*0.99:
                    for s in slots:
                        if s.ticker == t:
                            pnl = s.close(p, d_str, "SELL"); ts_pnl[t] += pnl

        nav = sum(s.nav for s in slots)
        if nav > peak_nav: peak_nav = nav
        dd = (peak_nav - nav)/peak_nav * 100
        if dd > max_dd: max_dd = dd

    print(f"KẾT QUẢ SAU 3 THÁNG (Q1/2026):")
    print(f"Vốn ban đầu  : {INITIAL:,.0f} VND")
    print(f"NAV hiện tại : {nav:,.0f} VND")
    print(f"Lợi nhuận    : {nav-INITIAL:+,,.0f} VND ({(nav-INITIAL)/INITIAL*100:+.2f}%)")
    print(f"Max Drawdown : {max_dd:.2f}%")
    print("-" * 30)
    for t, p in ts_pnl.items():
        print(f"  {t}: {p:+,..0f} VND")
    print("="*60)

if __name__ == "__main__":
    run_challenge()
