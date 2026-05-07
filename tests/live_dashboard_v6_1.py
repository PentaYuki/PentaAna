"""
live_dashboard_v6_1.py — Phiên bản Real-time (Đồng bộ đến 07/05/2026)
===================================================================
"""
import os, sys, time, pandas as pd, numpy as np
from datetime import datetime
from vnstock.api.quote import Quote # Sử dụng API mới của vnstock 4.0

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "parquet")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

INITIAL = 5_000_000
UNIVERSE = ["FPT", "MBB", "TCB", "ACB", "SSI", "CTG", "PNJ"]
SPEED = 0.5 # Tốc độ chạy lịch sử

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_color(val):
    if val > 0: return "\033[92m"
    if val < 0: return "\033[91m"
    return "\033[0m"

def run_realtime_dashboard():
    # 1. Load Data
    dfs = {}
    for t in UNIVERSE:
        p = os.path.join(DATA_DIR, f"{t}_history.parquet")
        df = pd.read_parquet(p)
        df["time"] = pd.to_datetime(df["time"])
        df["ema20"] = df["close"].ewm(span=20).mean()
        dfs[t] = df.sort_values("time")

    # Danh sách ngày lịch sử (từ 2024 đến nay)
    dates = dfs["MBB"][dfs["MBB"]["time"] >= "2024-01-01"]["time"].dt.strftime("%Y-%m-%d").tolist()
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    cash = INITIAL
    portfolio = {}
    
    # 2. Vòng lặp chính
    for idx, day in enumerate(dates):
        is_today = (day == today_str)
        
        current_prices = {}
        for t, df in dfs.items():
            row = df[df["time"].dt.strftime("%Y-%m-%d") == day]
            if not row.empty:
                current_prices[t] = float(row.iloc[0]["close"])
        
        # --- LOGIC AI (Giữ nguyên v6) ---
        stock_val = sum(pos["shares"] * current_prices.get(tk, pos["ep"]) for tk, pos in portfolio.items())
        nav = cash + stock_val
        thought = "Đang phân tích thị trường..."

        # Sell
        for t in list(portfolio.keys()):
            p = current_prices.get(t, 0)
            row = dfs[t][dfs[t]["time"].dt.strftime("%Y-%m-%d") == day].iloc[0]
            if p < row["ema20"] * 0.99:
                cash += portfolio[t]["shares"] * p * 0.998
                del portfolio[t]
                thought = f"🛡 BÁN {t} (Bảo vệ tài sản)"
        
        # Buy
        if len(portfolio) < 3 and cash > 1_500_000:
            for t in UNIVERSE:
                if t not in portfolio:
                    row = dfs[t][dfs[t]["time"].dt.strftime("%Y-%m-%d") == day].iloc[0]
                    p = current_prices.get(t, 0)
                    if p > row["ema20"]:
                        sh = (int((INITIAL/3.2) / (p*1.002)) // 100) * 100
                        if sh > 0:
                            cash -= sh*p*1.002
                            portfolio[t] = {"shares": sh, "ep": p}
                            thought = f"🚀 MUA {t} (Tín hiệu dẫn sóng)"
                            break

        # UI RENDER
        clear_screen()
        print("="*65)
        print(f" 🤖 PENTA-AI v6.1 | MODE: {'LIVE 🔴' if is_today else 'FAST-SIM ⏩'}")
        print(f" 📅 NGÀY: {day} | NAV: {nav:,.0f} VND")
        print("="*65)
        
        profit = nav - INITIAL
        color = get_color(profit)
        print(f" 📈 LỢI NHUẬN: {color}{profit:+,.0f} VND ({ (profit/INITIAL)*100:+.2f}%)\033[0m")
        print(f" 💵 TIỀN MẶT: {cash:,.0f} VND")
        print("-" * 65)
        
        if portfolio:
            print(f"   {'Mã':<8} {'Giá mua':<12} {'Giá hiện tại':<15} {'Lãi/Lỗ'}")
            for tk, pos in portfolio.items():
                p = current_prices.get(tk, 0)
                p_pct = (p-pos["ep"])/pos["ep"]*100
                print(f"   {tk:<8} {pos['ep']:<12,.0f} {p:<15,.0f} {get_color(p_pct)}{p_pct:+.2f}%\033[0m")
        else:
            print("   (Danh mục trống - Đang chờ sóng mới)")
        
        print("-" * 65)
        print(f" 💡 AI THOUGHT: {thought}")
        print("="*65)

        if is_today:
            print(f" 🕒 Chế độ LIVE đang chạy... (Cập nhật sau 60s)")
            time.sleep(60) # Live mode update every minute
            # Tải giá mới nhất nếu cần...
        else:
            time.sleep(SPEED)

if __name__ == "__main__":
    try:
        run_realtime_dashboard()
    except KeyboardInterrupt:
        print("\n⏹ Đã dừng hệ thống.")
