"""
challenge_2026_q1_v3.py — Backtest Q1/2026 với logic tiền mặt chuẩn
==================================================================
"""
import os, sys, pandas as pd, numpy as np
from datetime import datetime
from collections import defaultdict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

START, END = "2026-01-01", "2026-03-31"
INITIAL    = 5_000_000
UNIVERSE   = ["FPT", "MBB", "TCB", "ACB", "SSI", "CTG", "PNJ"]

class PortfolioManager:
    def __init__(self, initial_cash):
        self.cash = float(initial_cash)
        self.positions = {} # ticker -> {shares, entry_price, peak_price}
        self.history = []
        self.total_pnl = 0.0

    def get_nav(self, current_prices):
        val = self.cash
        for tk, pos in self.positions.items():
            if tk in current_prices:
                val += pos["shares"] * current_prices[tk]
        return val

    def buy(self, ticker, price, date):
        if len(self.positions) >= 3: return # Giới hạn 3 mã cho an toàn
        # Phân bổ 1/3 vốn khởi tạo cho mỗi mã
        alloc = INITIAL / 3
        if self.cash < alloc: alloc = self.cash
        
        shares = (int(alloc / (price * 1.002)) // 100) * 100
        if shares <= 0: return
        
        cost = shares * price * 1.002
        self.cash -= cost
        self.positions[ticker] = {"shares": shares, "entry_price": price, "peak_price": price, "date": date}
        # print(f"[{date}] BUY {ticker}: {shares} cổ @ {price:,.0f}")

    def sell(self, ticker, price, date, reason):
        if ticker not in self.positions: return
        pos = self.positions.pop(ticker)
        val = pos["shares"] * price * 0.998
        pnl = val - pos["shares"] * pos["entry_price"] * 1.002
        self.cash += val
        self.total_pnl += pnl
        # print(f"[{date}] SELL {ticker} ({reason}): {pos['shares']} cổ @ {price:,.0f} | PNL: {pnl:+,.0f}")

def run_challenge():
    print("="*60); print(f"  BACKTEST Q1/2026 - CHIẾN THUẬT V6 (CHUẨN VỐN)"); print("="*60)
    
    all_dfs = {}
    for t in UNIVERSE:
        p = os.path.join(DATA_DIR, "raw", "parquet", f"{t}_history.parquet")
        df = pd.read_parquet(p); df["time"] = pd.to_datetime(df["time"])
        c = df["close"]
        df["ema20"] = c.ewm(span=20, adjust=False).mean()
        df["ema200"] = c.ewm(span=200, adjust=False).mean()
        all_dfs[t] = df.sort_values("time").reset_index(drop=True)

    pm = PortfolioManager(INITIAL)
    idx_df = all_dfs["MBB"]
    dates = idx_df[(idx_df["time"] >= START) & (idx_df["time"] <= END)]["time"].dt.strftime("%Y-%m-%d").tolist()
    
    peak_nav = INITIAL; max_dd = 0

    for d_str in dates:
        prices = {}
        for t in UNIVERSE:
            r = all_dfs[t][all_dfs[t]["time"].dt.strftime("%Y-%m-%d")==d_str]
            if not r.empty: prices[t] = float(r.iloc[0]["close"])

        # Update peak & check SL
        for tk in list(pm.positions.keys()):
            if tk in prices:
                p = prices[tk]
                pos = pm.positions[tk]
                if p > pos["peak_price"]: pos["peak_price"] = p
                # Stop loss 7%
                if (p - pos["entry_price"])/pos["entry_price"]*100 <= -7:
                    pm.sell(tk, p, d_str, "STOP_LOSS")

        # Trading signals
        for t in UNIVERSE:
            r = all_dfs[t][all_dfs[t]["time"].dt.strftime("%Y-%m-%d")==d_str]
            if r.empty: continue
            row = r.iloc[0]; p = prices.get(t)
            if p is None: continue

            # Sell signal (Price < EMA20)
            if t in pm.positions and p < row["ema20"] * 0.99:
                pm.sell(t, p, d_str, "EMA_CROSS")
            
            # Buy signal (Price > EMA20 and Price > EMA200)
            if t not in pm.positions and p > row["ema20"] and p > row["ema200"]:
                pm.buy(t, p, d_str)

        nav = pm.get_nav(prices)
        if nav > peak_nav: peak_nav = nav
        dd = (peak_nav - nav)/peak_nav * 100 if peak_nav > 0 else 0
        if dd > max_dd: max_dd = dd

    print(f"KẾT QUẢ Q1/2026 (01/01 - 31/03):")
    print(f"Vốn ban đầu  : {INITIAL:,.0f} VND")
    print(f"NAV cuối kỳ  : {nav:,.0f} VND")
    print(f"Lợi nhuận    : {nav-INITIAL:+,.0f} VND ({((nav-INITIAL)/INITIAL*100):+.2f}%)")
    print(f"Max Drawdown : {max_dd:.2f}%")
    print(f"Số mã đang giữ: {len(pm.positions)}")
    for t in pm.positions:
        print(f"  - {t}")
    print("="*60)

if __name__ == "__main__":
    run_challenge()
