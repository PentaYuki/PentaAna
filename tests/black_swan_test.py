"""
black_swan_test.py — Thử thách sinh tồn tháng 04/2024
=====================================================
"""
import os, sys, pandas as pd, numpy as np
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "parquet")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# Giai đoạn "Thiên nga đen"
START, END = "2024-04-01", "2024-05-15"
INITIAL    = 5_000_000
UNIVERSE   = ["FPT", "MBB", "TCB", "ACB", "SSI", "CTG", "PNJ"]

class SurvivalAI:
    def __init__(self):
        self.cash = float(INITIAL)
        self.pos = {} # ticker -> {shares, ep, peak}
        self.nav_history = []

    def run(self, all_dfs, dates):
        print(f"{'Ngày':<12} {'NAV':<12} {'Trạng thái':<20} {'Hành động'}")
        print("-" * 65)
        
        peak_nav = INITIAL
        max_dd = 0

        for d_str in dates:
            prices = {t: float(df[df["time"].dt.strftime("%Y-%m-%d")==d_str].iloc[0]["close"]) 
                      for t, df in all_dfs.items() if not df[df["time"].dt.strftime("%Y-%m-%d")==d_str].empty}
            
            # 1. Kiểm tra Stop Loss khẩn cấp
            action = ""
            for t in list(self.pos.keys()):
                p = prices.get(t, 0)
                if p == 0: continue
                ep = self.pos[t]["ep"]
                if (p-ep)/ep*100 <= -7.0: # Kỷ luật sắt 7%
                    val = self.pos[t]["shares"] * p * 0.998
                    self.cash += val
                    del self.pos[t]
                    action = f"🔥 THÁO CHẠY: Bán {t} (Cắt lỗ 7%)"

            # 2. Tín hiệu kỹ thuật (EMA20 gãy là bán hết)
            for t in list(self.pos.keys()):
                df = all_dfs[t]; r = df[df["time"].dt.strftime("%Y-%m-%d")==d_str].iloc[0]
                p = prices[t]
                if p < r["ema20"] * 0.98: # Gãy xu hướng
                    val = self.pos[t]["shares"] * p * 0.998
                    self.cash += val
                    del self.pos[t]
                    action = f"🛡 PHÒNG THỦ: Thoát {t} (Gãy xu hướng)"

            # 3. Mua (Chỉ mua nếu thị trường hồi phục rõ rệt)
            if len(self.pos) < 3 and action == "":
                for t in UNIVERSE:
                    if t not in self.pos and t in prices:
                        df = all_dfs[t]; r = df[df["time"].dt.strftime("%Y-%m-%d")==d_str].iloc[0]
                        p = prices[t]
                        if p > r["ema20"] and p > r["ema200"]: # Chỉ mua khi cực kỳ an toàn
                            sh = (int((INITIAL/3.5) / (p*1.002)) // 100) * 100
                            if sh > 0:
                                self.cash -= sh*p*1.002
                                self.pos[t] = {"shares": sh, "ep": p}
                                action = f"🟢 MUA {t} (Dò đường)"
                                break

            nav = self.cash + sum(p["shares"] * prices.get(tk, p["ep"]) for tk, p in self.pos.items())
            if nav > peak_nav: peak_nav = nav
            dd = (peak_nav - nav)/peak_nav * 100
            if dd > max_dd: max_dd = dd
            
            status = f"{len(self.pos)} mã" if self.pos else "TIỀN MẶT"
            print(f"{d_str:<12} {nav:<12,.0f} {status:<20} {action}")
            self.nav_history.append(nav)

        return nav, max_dd

def main():
    all_dfs = {}
    for t in UNIVERSE:
        p = os.path.join(DATA_DIR, f"{t}_history.parquet")
        df = pd.read_parquet(p); df["time"] = pd.to_datetime(df["time"])
        df["ema20"] = df["close"].ewm(span=20).mean()
        df["ema200"] = df["close"].ewm(span=200).mean()
        all_dfs[t] = df

    dates = all_dfs["MBB"][(all_dfs["MBB"]["time"] >= START) & (all_dfs["MBB"]["time"] <= END)]["time"].dt.strftime("%Y-%m-%d").tolist()
    
    ai = SurvivalAI()
    final_nav, max_dd = ai.run(all_dfs, dates)

    print("-" * 65)
    print(f"🚩 KẾT QUẢ SINH TỒN:")
    print(f"   Lợi nhuận: {(final_nav-INITIAL)/INITIAL*100:+.2f}%")
    print(f"   Sụt giảm lớn nhất (Max DD): {max_dd:.2f}%")
    market_drop = -15.0 # Ước tính thị trường chung rơi 15%
    print(f"   So với Thị trường rơi: {market_drop}%")
    print(f"   Kết luận: {'THẮNG' if max_dd < abs(market_drop) else 'THUA'} THỊ TRƯỜNG")
    print("-" * 65)

if __name__ == "__main__":
    main()
