"""
live_dashboard_v6_2.py — Bản Treo Máy Vĩnh Cửu (Infinite Live Mode)
==================================================================
"""
import os, sys, time, pandas as pd, numpy as np
from datetime import datetime
from vnstock.api.quote import Quote

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "parquet")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

INITIAL = 5_000_000
UNIVERSE = ["FPT", "MBB", "TCB", "ACB", "SSI", "CTG", "PNJ"]

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_color(val):
    if val > 0: return "\033[92m"
    if val < 0: return "\033[91m"
    return "\033[0m"

def get_live_price(ticker):
    try:
        q = Quote(symbol=ticker, source='VCI')
        # Lấy giá khớp lệnh gần nhất
        df = q.history(start=datetime.now().strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
        if not df.empty: return float(df.iloc[-1]['close'])
    except: pass
    return None

def run_infinite_dashboard():
    # Load History
    dfs = {}
    for t in UNIVERSE:
        dfs[t] = pd.read_parquet(os.path.join(DATA_DIR, f"{t}_history.parquet")).sort_values("time")
        dfs[t]["ema20"] = dfs[t]["close"].ewm(span=20).mean()

    dates = dfs["MBB"][dfs["MBB"]["time"] >= "2024-01-01"]["time"].dt.strftime("%Y-%m-%d").tolist()
    
    cash = INITIAL
    portfolio = {}
    
    # 1. Chạy nhanh qua lịch sử
    print("🚀 Đang tái hiện lịch sử...")
    for day in dates[:-1]: # Chạy đến sát ngày hôm nay
        current_prices = {t: float(df[df["time"].dt.strftime("%Y-%m-%d")==day].iloc[0]["close"]) for t, df in dfs.items() if not df[df["time"].dt.strftime("%Y-%m-%d")==day].empty}
        # Logic mua bán tối giản
        for t in list(portfolio.keys()):
            p = current_prices.get(t, 0)
            row = dfs[t][dfs[t]["time"].dt.strftime("%Y-%m-%d")==day].iloc[0]
            if p < row["ema20"] * 0.99:
                cash += portfolio[t]["shares"] * p * 0.998
                del portfolio[t]
        if len(portfolio) < 3 and cash > 1_500_000:
            for t in UNIVERSE:
                if t not in portfolio and t in current_prices:
                    row = dfs[t][dfs[t]["time"].dt.strftime("%Y-%m-%d")==day].iloc[0]
                    if current_prices[t] > row["ema20"]:
                        sh = (int((INITIAL/3.2) / (current_prices[t]*1.002)) // 100) * 100
                        if sh > 0:
                            cash -= sh*current_prices[t]*1.002
                            portfolio[t] = {"shares": sh, "ep": current_prices[t]}
                            break

    # 2. Vòng lặp VĨNH CỬU cho Live Mode
    while True:
        today_str = datetime.now().strftime("%Y-%m-%d")
        # Cố gắng lấy giá Live, nếu không lấy được thì lấy giá đóng cửa gần nhất
        current_prices = {}
        for t in UNIVERSE:
            lp = get_live_price(t)
            if lp: current_prices[t] = lp
            else: current_prices[t] = float(dfs[t].iloc[-1]['close'])

        nav = cash + sum(pos["shares"] * current_prices.get(tk, pos["ep"]) for tk, pos in portfolio.items())
        
        clear_screen()
        print("="*65)
        print(f" 🤖 PENTA-AI v6.2 | 🔴 CHẾ ĐỘ LIVE ĐANG CHẠY (TREO MÁY)")
        print(f" 📅 NGÀY: {today_str} | NAV: {nav:,.0f} VND")
        print("="*65)
        profit = nav - INITIAL
        print(f" 📈 LỢI NHUẬN: {get_color(profit)}{profit:+,.0f} VND ({ (profit/INITIAL)*100:+.2f}%)\033[0m")
        print(f" 💵 TIỀN MẶT: {cash:,.0f} VND")
        print("-" * 65)
        print(f" 💼 DANH MỤC TRỰC TUYẾN:")
        for tk, pos in portfolio.items():
            p = current_prices.get(tk, 0)
            p_pct = (p-pos["ep"])/pos["ep"]*100
            print(f"   {tk:<8} {pos['ep']:<12,.0f} {p:<15,.0f} {get_color(p_pct)}{p_pct:+.2f}%\033[0m")
        print("-" * 65)
        print(f" 🕒 Tự động cập nhật sau 30 giây... (Ctrl+C để dừng)")
        time.sleep(30)

if __name__ == "__main__":
    try:
        run_infinite_dashboard()
    except KeyboardInterrupt:
        print("\n⏹ Đã dừng hệ thống.")
