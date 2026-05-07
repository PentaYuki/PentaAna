"""
challenge_2026_q1_fixed.py — Thử thách 3 tháng đầu năm 2026 (Sửa lỗi logic vốn)
============================================================================
"""
import os, sys, json, pandas as pd, numpy as np
from datetime import datetime
from collections import defaultdict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

START, END = "2026-01-01", "2026-03-31"
INITIAL    = 5_000_000
UNIVERSE   = ["FPT", "MBB", "TCB", "ACB", "SSI", "CTG", "PNJ"]

class Slot:
    def __init__(self, id):
        self.id = id; self.cash = 0.0; self.shares = 0; self.ticker = None
        self.ep = 0.0; self.cur = 0.0; self.peak = 0.0
    @property
    def nav(self): return self.cash + (self.shares * self.cur if self.shares > 0 else 0)
    def open(self, t, p, d):
        sh = (int(self.cash / (p * 1.002)) // 100) * 100
        if sh <= 0: return False
        self.cash -= sh*p*1.002; self.shares = sh; self.ticker = t; self.ep = p; self.cur = p; self.peak = p
        return True
    def close(self, p, d):
        if not self.shares: return 0
        v = self.shares * p * 0.998; pnl = v - self.shares*self.ep*1.002
        self.cash += v; self.shares = 0; self.ticker = None
        return pnl

def run_challenge():
    print("="*60); print(f"  BACKTEST THỰC TẾ Q1/2026 (VỐN THẬT 5M)"); print("="*60)
    
    all_dfs = {}
    for t in UNIVERSE:
        p = os.path.join(DATA_DIR, "raw", "parquet", f"{t}_history.parquet")
        df = pd.read_parquet(p); df["time"] = pd.to_datetime(df["time"])
        c = df["close"]
        df["ema20"] = c.ewm(span=20).mean(); df["ema200"] = c.ewm(span=200).mean()
        all_dfs[t] = df.sort_values("time")

    idx_df = all_dfs["MBB"].copy()
    dates = idx_df[(idx_df["time"] >= START) & (idx_df["time"] <= END)]["time"].dt.strftime("%Y-%m-%d").tolist()
    
    # Khởi tạo vốn thật: Chia đều INITIAL vào các slot trống
    slots = [Slot(i) for i in range(6)]
    slots[0].cash = INITIAL # Dùng 1 quỹ chung cho đơn giản
    
    peak_nav = INITIAL; max_dd = 0; ts_pnl = defaultdict(float)

    for d_str in dates:
        # Cập nhật giá hiện tại cho các slot đang giữ hàng
        for s in slots:
            if s.shares:
                r = all_dfs[s.ticker][all_dfs[s.ticker]["time"].dt.strftime("%Y-%m-%d")==d_str]
                if not r.empty:
                    p = float(r.iloc[0]["close"]); s.cur = p
                    if p > s.peak: s.peak = p
                    if (p-s.ep)/s.ep*100 <= -7:
                        pnl = s.close(p, d_str); ts_pnl[s.ticker] += pnl

        # Tính tổng tiền mặt khả dụng (dồn về slot 0 để tái phân bổ)
        total_cash = sum(s.cash for s in slots)
        for s in slots: s.cash = 0
        slots[0].cash = total_cash

        # Quyết định mua/bán
        for t in UNIVERSE:
            r = all_dfs[t][all_dfs[t]["time"].dt.strftime("%Y-%m-%d")==d_str]
            if not r.empty:
                row = r.iloc[0]; p = float(row["close"])
                occ = {s.ticker for s in slots if s.ticker}
                
                # Bán
                if t in occ and p < row["ema20"]*0.99:
                    for s in slots:
                        if s.ticker == t:
                            pnl = s.close(p, d_str); ts_pnl[t] += pnl
                
                # Mua
                if t not in occ and p > row["ema20"] and p > row["ema200"]:
                    # Tìm slot trống
                    for s in slots:
                        if not s.shares:
                            # Phân bổ tối đa 1/3 vốn hiện có cho mỗi mã (để an toàn)
                            available = slots[0].cash
                            s.cash = min(available, INITIAL / 3) 
                            if s.open(t, p, d_str):
                                slots[0].cash -= s.cash
                                break

        nav = sum(s.nav for s in slots)
        if nav > peak_nav: peak_nav = nav
        dd = (peak_nav - nav)/peak_nav * 100 if peak_nav > 0 else 0
        if dd > max_dd: max_dd = dd

    print(f"KẾT QUẢ SAU 3 THÁNG (Q1/2026):")
    print(f"Vốn ban đầu  : {INITIAL:,.0f} VND")
    print(f"NAV cuối kỳ  : {nav:,.0f} VND")
    print(f"Lợi nhuận    : {nav-INITIAL:,.0f} VND ({((nav-INITIAL)/INITIAL*100):+.2f}%)")
    print(f"Max Drawdown : {max_dd:.2f}%")
    print("-" * 30)
    for t, pnl in ts_pnl.items():
        print(f"  {t}: {pnl:+,.0f} VND")
    print("="*60)

if __name__ == "__main__":
    run_challenge()
