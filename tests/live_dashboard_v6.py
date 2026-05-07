"""
live_dashboard_v6.py — Bảng điều khiển giả lập thời gian thực
============================================================
Cách dùng: python tests/live_dashboard_v6.py
"""
import os, sys, time, pandas as pd, numpy as np
from datetime import datetime

# Giả lập 1 ngày giao dịch = 0.8 giây để anh xem cho sướng mắt
SPEED = 0.8 

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "parquet")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

INITIAL = 5_000_000
UNIVERSE = ["FPT", "MBB", "TCB", "ACB", "SSI", "CTG", "PNJ"]

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_color(val):
    if val > 0: return "\033[92m" # Xanh
    if val < 0: return "\033[91m" # Đỏ
    return "\033[0m"

def run_dashboard():
    # Load Data
    dfs = {}
    for t in UNIVERSE:
        df = pd.read_parquet(os.path.join(DATA_DIR, f"{t}_history.parquet"))
        df["time"] = pd.to_datetime(df["time"])
        df["ema20"] = df["close"].ewm(span=20).mean()
        dfs[t] = df.sort_values("time")

    dates = dfs["MBB"][dfs["MBB"]["time"] >= "2024-01-01"]["time"].dt.strftime("%Y-%m-%d").tolist()
    
    cash = INITIAL
    portfolio = {} # ticker -> {shares, ep, peak}
    history = []

    for idx, day in enumerate(dates):
        current_prices = {t: float(df[df["time"].dt.strftime("%Y-%m-%d")==day].iloc[0]["close"]) for t, df in dfs.items() if not df[df["time"].dt.strftime("%Y-%m-%d")==day].empty}
        
        # Calculate NAV
        stock_val = sum(pos["shares"] * current_prices.get(tk, pos["ep"]) for tk, pos in portfolio.items())
        nav = cash + stock_val
        profit_total = nav - INITIAL
        pct_total = (profit_total/INITIAL)*100

        # AI THOUGHTS (Logic v6)
        thought = "Đang quét tín hiệu thị trường..."
        
        # Sell Logic
        for t in list(portfolio.keys()):
            p = current_prices.get(t, 0)
            ep = portfolio[t]["ep"]
            row = dfs[t][dfs[t]["time"].dt.strftime("%Y-%m-%d")==day].iloc[0]
            if p < row["ema20"] * 0.99:
                val = portfolio[t]["shares"] * p * 0.998
                cash += val
                del portfolio[t]
                thought = f"🛡 BÁN {t} để bảo vệ vốn (Gãy EMA20)"
            elif (p-ep)/ep >= 0.20:
                val = portfolio[t]["shares"] * p * 0.998
                cash += val
                del portfolio[t]
                thought = f"💰 CHỐT LỜI {t} (+20%) - Đã đạt mục tiêu!"

        # Buy Logic
        if len(portfolio) < 3 and cash > 1_500_000:
            for t in UNIVERSE:
                if t not in portfolio and t in current_prices:
                    row = dfs[t][dfs[t]["time"].dt.strftime("%Y-%m-%d")==day].iloc[0]
                    p = current_prices[t]
                    if p > row["ema20"]:
                        sh = (int((INITIAL/3.2) / (p*1.002)) // 100) * 100
                        if sh > 0:
                            cash -= sh*p*1.002
                            portfolio[t] = {"shares": sh, "ep": p}
                            thought = f"🚀 MUA {t} - Tín hiệu tăng trưởng mạnh!"
                            break

        # UI RENDER
        clear_screen()
        print("="*60)
        print(f" 🤖 PENTA-AI TRADING BOT - LIVE SIMULATION v6")
        print(f" 📅 NGÀY: {day} | 🕒 Trạng thái: ĐANG GIAO DỊCH")
        print("="*60)
        
        print(f" 💰 TÀI SẢN (NAV): {nav:,.0f} VND")
        color = get_color(profit_total)
        print(f" 📈 LỢI NHUẬN: {color}{profit_total:+,.0f} VND ({pct_total:+.2f}%)\033[0m")
        print(f" 💵 TIỀN MẶT: {cash:,.0f} VND")
        print("-" * 60)
        
        print(f" 💼 DANH MỤC ĐẦU TƯ:")
        if not portfolio:
            print("   (Trống - Đang chờ tín hiệu tốt)")
        else:
            print(f"   {'Mã':<8} {'Giá mua':<12} {'Giá hiện tại':<15} {'Lãi/Lỗ'}")
            for tk, pos in portfolio.items():
                p = current_prices.get(tk, 0)
                p_pct = (p-pos["ep"])/pos["ep"]*100
                p_color = get_color(p_pct)
                print(f"   {tk:<8} {pos['ep']:<12,.0f} {p:<15,.0f} {p_color}{p_pct:+.2f}%\033[0m")
        
        print("-" * 60)
        print(f" 💡 AI THOUGHTS: {thought}")
        print("="*60)
        print(" (Nhấn Ctrl+C để dừng cỗ máy thời gian)")
        
        time.sleep(SPEED)

if __name__ == "__main__":
    try:
        run_dashboard()
    except KeyboardInterrupt:
        print("\n⏹ Đã dừng Dashboard.")
