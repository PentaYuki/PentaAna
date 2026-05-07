"""
live_simulation_2024_full.py — Giả lập xuyên suốt 2024 (0.5s = 1 ngày)
===================================================================
"""
import os, sys, time, json, pandas as pd, numpy as np
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "parquet")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

START_DATE = "2024-01-01"
END_DATE   = "2024-12-31"
INITIAL    = 5_000_000
UNIVERSE   = ["FPT", "MBB", "TCB", "ACB", "SSI", "CTG", "PNJ"]

def run_live_sim_full():
    print(f"🎬 BẮT ĐẦU GIẢ LẬP NĂM 2024 (Vốn: {INITIAL:,.0f} VND)")
    
    all_dfs = {}
    for t in UNIVERSE:
        all_dfs[t] = pd.read_parquet(os.path.join(DATA_DIR, f"{t}_history.parquet")).sort_values("time")
        all_dfs[t]["time"] = pd.to_datetime(all_dfs[t]["time"])

    dates = all_dfs["MBB"][(all_dfs["MBB"]["time"] >= START_DATE) & (all_dfs["MBB"]["time"] <= END_DATE)]["time"].dt.strftime("%Y-%m-%d").tolist()
    
    cash = INITIAL
    portfolio = {} # ticker -> {shares, ep}
    peak_nav = INITIAL; max_dd = 0
    trades_count = 0; wins = 0

    for idx, day in enumerate(dates):
        current_prices = {}
        for t, df in all_dfs.items():
            row = df[df["time"].dt.strftime("%Y-%m-%d") == day]
            if not row.empty: current_prices[t] = float(row.iloc[0]["close"])
        
        # NAV
        stock_val = sum(pos["shares"] * current_prices.get(tk, pos["ep"]) for tk, pos in portfolio.items())
        nav = cash + stock_val
        if nav > peak_nav: peak_nav = nav
        dd = (peak_nav - nav)/peak_nav * 100
        if dd > max_dd: max_dd = dd

        # LOG MỖI 10 NGÀY HOẶC KHI CÓ BIẾN
        if idx % 10 == 0 or idx == len(dates)-1:
            hold_str = ", ".join([f"{tk}" for tk in portfolio.keys()]) if portfolio else "Tiền mặt"
            print(f"📅 {day} | NAV: {nav:,.0f} | Danh mục: [{hold_str}] | DD: {max_dd:.1f}%")
        
        # --- CHIẾN THUẬT AI ---
        # Bán
        for t in list(portfolio.keys()):
            p = current_prices.get(t, 0)
            ep = portfolio[t]["ep"]
            # Chốt lời động hoặc cắt lỗ
            if (p - ep)/ep >= 0.15: # Chốt lời 15%
                val = portfolio[t]["shares"] * p * 0.998
                cash += val; trades_count += 1; wins += 1
                print(f"   ➔ 🔵 CHỐT LỜI {t} (+15%) @ {p:,.0f}")
                del portfolio[t]
            elif (p - ep)/ep <= -0.07: # Cắt lỗ 7%
                val = portfolio[t]["shares"] * p * 0.998
                cash += val; trades_count += 1
                print(f"   ➔ 🔴 CẮT LỖ {t} (-7%) @ {p:,.0f}")
                del portfolio[t]

        # Mua
        if len(portfolio) < 3 and cash > 1_500_000:
            for t in UNIVERSE:
                if t not in portfolio and current_prices.get(t, 0) > 0:
                    price = current_prices[t]
                    # Giả lập tín hiệu EMA (đơn giản hóa cho log nhanh)
                    shares = (int((INITIAL/3.2) / (price * 1.002)) // 100) * 100
                    if shares > 0:
                        cash -= shares * price * 1.002
                        portfolio[t] = {"shares": shares, "ep": price}
                        print(f"   ➔ 🟢 MUA {t} @ {price:,.0f}")
                        break
        
        time.sleep(0.5) # 0.5s per day

    print("\n" + "═"*50)
    print(f"🏆 KẾT THÚC NĂM 2024")
    print(f"NAV Cuối kỳ : {nav:,.0f} VND")
    print(f"Lợi nhuận   : {nav-INITIAL:+,.0f} VND ({(nav-INITIAL)/INITIAL*100:+.2f}%)")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Số lệnh     : {trades_count} (Thắng: {wins})")
    print("═"*50)

if __name__ == "__main__":
    run_live_sim_full()
