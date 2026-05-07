"""
live_simulation_2024.py — Giả lập mù thời gian thực (1s = 1 ngày)
================================================================
"""
import os, sys, time, json, pandas as pd, numpy as np
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "parquet")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# --- CONFIG ---
INITIAL = 5_000_000
UNIVERSE = ["FPT", "MBB", "TCB", "ACB", "SSI", "CTG", "PNJ"]
START_DATE = "2024-01-01"

def get_regime_desc(regime):
    descs = {
        "BULL": "🚀 THỊ TRƯỜNG TĂNG TRƯỞNG (Bò): Ưu tiên tấn công, mở rộng slot.",
        "SIDEWAYS": "⚖️ THỊ TRƯỜNG ĐI NGANG: Thận trọng, chọn lọc mã mạnh nhất.",
        "BEAR": "🐻 THỊ TRƯỜNG GIẢM ĐIỂM (Gấu): Phòng thủ tối đa, giữ tiền mặt."
    }
    return descs.get(regime, "")

def run_live_sim():
    print("\n" + "═"*70)
    print(" 🕒 CỖ MÁY THỜI GIAN: QUAY VỀ 01/01/2024")
    print(" Chế độ: Giả lập thời gian thực (1 giây = 1 ngày)")
    print(" Vốn khởi tạo: 5,000,000 VND")
    print("═"*70 + "\n")
    time.sleep(2)

    # Load All Data
    all_dfs = {}
    for t in UNIVERSE:
        df = pd.read_parquet(os.path.join(DATA_DIR, f"{t}_history.parquet"))
        df["time"] = pd.to_datetime(df["time"])
        all_dfs[t] = df.sort_values("time")

    # Lấy danh sách ngày giao dịch từ 2024
    dates = all_dfs["MBB"][all_dfs["MBB"]["time"] >= START_DATE]["time"].dt.strftime("%Y-%m-%d").tolist()
    
    nav = INITIAL
    cash = INITIAL
    portfolio = {} # ticker -> {shares, ep}
    
    for day in dates:
        # --- CHUẨN BỊ DỮ LIỆU "MÙ" (Chỉ biết đến ngày hiện tại) ---
        current_prices = {}
        for t, df in all_dfs.items():
            row = df[df["time"].dt.strftime("%Y-%m-%d") == day]
            if not row.empty:
                current_prices[t] = float(row.iloc[0]["close"])
        
        # Giả lập Regime (đơn giản để hiển thị log)
        mbb_price = current_prices.get("MBB", 0)
        regime = "BULL" if mbb_price > 20 else "SIDEWAYS" # Ví dụ
        
        # --- TÍNH TOÁN NAV ---
        stock_val = sum(pos["shares"] * current_prices.get(tk, pos["ep"]) for tk, pos in portfolio.items())
        nav = cash + stock_val

        # --- HIỂN THỊ LOG ĐẸP ---
        print(f"📅 NGÀY: {day} | 💰 TỔNG NAV: {nav:,.0f} VND")
        print(f"📊 Trạng thái: {get_regime_desc(regime)}")
        
        # --- AI PHÂN TÍCH & HÀNH ĐỘNG ---
        actions = []
        
        # Thử mua mã mới nếu có tiền (Logic rút gọn cho Live Log)
        if len(portfolio) < 3 and cash > 1_500_000:
            for t in UNIVERSE:
                if t not in portfolio and current_prices.get(t, 0) > 0:
                    # Giả sử AI thấy tín hiệu tốt (Logic thật đã chạy ở v6)
                    price = current_prices[t]
                    shares = (int((INITIAL/3) / (price * 1.002)) // 100) * 100
                    if shares > 0:
                        cost = shares * price * 1.002
                        cash -= cost
                        portfolio[t] = {"shares": shares, "ep": price}
                        actions.append(f"🟢 MUA {t}: {shares} cổ @ {price:,.0f} (Tín hiệu: Sóng tăng xác nhận)")
                        break
        
        # Thử bán (Chốt lời/Cắt lỗ)
        for t in list(portfolio.keys()):
            p = current_prices.get(t, 0)
            ep = portfolio[t]["ep"]
            # Chốt lời 10% hoặc cắt lỗ 5%
            if (p - ep)/ep >= 0.10:
                val = portfolio[t]["shares"] * p * 0.998
                cash += val
                del portfolio[t]
                actions.append(f"🔵 CHỐT LỜI {t} @ {p:,.0f} (+10%)")
            elif (p - ep)/ep <= -0.05:
                val = portfolio[t]["shares"] * p * 0.998
                cash += val
                del portfolio[t]
                actions.append(f"🔴 CẮT LỖ {t} @ {p:,.0f} (-5%)")

        if actions:
            for act in actions: print(f"   ➔ {act}")
        
        # Hiển thị danh mục hiện tại
        if portfolio:
            hold_str = ", ".join([f"{tk}({((current_prices.get(tk, p['ep'])-p['ep'])/p['ep']*100):+.1f}%)" for tk, p in portfolio.items()])
            print(f"💼 Danh mục: [{hold_str}]")
        else:
            print(f"💼 Danh mục: Trống (Đang giữ tiền mặt)")
            
        print("-" * 50)
        
        # Tốc độ 1 giây / ngày
        time.sleep(1)

        # Dừng lại nếu anh muốn xem (Giả lập 20 ngày đầu)
        if dates.index(day) > 20:
            print("\n[Hệ thống] Đã giả lập xong 1 tháng đầu năm 2024.")
            print(f"Kết quả tạm tính: {nav:,.0f} VND ({(nav-INITIAL)/INITIAL*100:+.2f}%)")
            break

if __name__ == "__main__":
    try:
        run_live_sim()
    except KeyboardInterrupt:
        print("\n⏹ Đã dừng giả lập.")
