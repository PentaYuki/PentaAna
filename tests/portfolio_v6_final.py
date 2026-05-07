"""
portfolio_v6_final.py — Kết hợp v5 (High Performance) + v6 (Dynamic Slots)
=======================================================================
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

# --- CONFIG ---
START, END      = "2024-01-01", "2025-12-31"
INITIAL, TARGET = 5_000_000, 14_000_000
FEE_R, LOT      = 0.002, 100
SL, TRAIL, CD   = 7.0, 4.0, 5

def get_max_positions(regime):
    if regime == "BULL": return 6      # Tăng lên 6 mã để tối đa lợi nhuận
    if regime == "SIDEWAYS": return 3
    if regime == "BEAR": return 1
    return 3

REGIME_BUY = {"BULL": 0.22, "SIDEWAYS": 0.42, "BEAR": 0.60}
REGIME_SEL = {"BULL": -0.22, "SIDEWAYS": -0.28, "BEAR": -0.18}

# Universe sạch (sau blacklist v5)
UNIVERSE = ["FPT", "MBB", "TCB", "ACB", "SSI", "CTG", "PNJ"]

def load_rlhf():
    p = os.path.join(JSON_DIR, "rlhf_weights.json")
    if os.path.exists(p): return json.load(open(p)).get("weights", {})
    return {"technical": 0.164, "sentiment": 0.284, "macro": 0.311, "risk": 0.241}

def get_sentiment_v6(ticker, date_str, row):
    db_score = 0.0
    try:
        with sqlite3.connect(DB_PATH) as conn:
            res = conn.execute("SELECT AVG(sentiment_score) FROM news WHERE ticker=? AND pub_date=?", (ticker, date_str)).fetchone()
            if res and res[0]: db_score = float(res[0])
    except: pass
    
    # Kết hợp động lượng nếu không có tin
    if db_score == 0:
        db_score = 0.1 if row["macd_hist"] > 0 else -0.1
    return db_score

def score_v6(row, ticker, rlhf_w, date_str, regime):
    score = 0.0
    try:
        c, e20, e50, e200 = row["close"], row["ema20"], row["ema50"], row["ema200"]
        rsi, macd, msig, mhst = row["rsi"], row["macd"], row["macd_sig"], row["macd_hist"]
        tw, sw = rlhf_w.get("technical", 0.16), rlhf_w.get("sentiment", 0.28)
        
        if c < e200 * 0.99: return -0.4
        
        # Tech Score (v5 logic)
        if c > e20 > e50 > e200: score += 0.30 * tw * 2.5
        elif c > e20 > e50: score += 0.20 * tw * 2.5
        if macd > msig and mhst > 0: score += 0.25
        if 45 < rsi < 65: score += 0.15
        
        # Sentiment v6
        sent = get_sentiment_v6(ticker, date_str, row)
        score += sent * sw * 0.8
    except: pass
    return float(np.clip(score, -1.0, 1.0))

class SlotV6:
    def __init__(self, id):
        self.id = id; self.cash = 0; self.shares = 0; self.ticker = None
        self.ep = 0.0; self.peak = 0.0; self.cur = 0.0; self.trades = []
    @property
    def nav(self): return self.cash + (self.shares * self.cur if self.shares > 0 else 0)
    def open(self, ticker, price, date):
        sh = (int(self.cash / (price * (1+FEE_R))) // LOT) * LOT
        if sh <= 0: return False
        self.cash -= (sh*price*(1+FEE_R)); self.shares = sh; self.ticker = ticker
        self.ep = price; self.peak = price; self.cur = price
        self.trades.append({"a":"BUY","d":date,"t":ticker,"p":price})
        return True
    def close(self, price, date, reason):
        if not self.shares: return 0.0
        val = self.shares * price * (1-FEE_R)
        pnl = val - self.shares * self.ep * (1+FEE_R)
        self.cash += val; self.shares = 0; self.ticker = None
        self.trades.append({"a":reason,"d":date,"p":price,"pnl":round(pnl)})
        return pnl

def run_v6_final():
    rlhf_w = load_rlhf()
    all_dfs = {}
    for t in UNIVERSE:
        p = os.path.join(DATA_DIR, "raw", "parquet", f"{t}_history.parquet")
        df = pd.read_parquet(p); df["time"] = pd.to_datetime(df["time"])
        # Add Indicators (v5 style)
        c = df["close"]
        df["ema20"]=c.ewm(span=20).mean(); df["ema50"]=c.ewm(span=50).mean(); df["ema200"]=c.ewm(span=200).mean()
        df["macd"]=c.ewm(span=12).mean()-c.ewm(span=26).mean(); df["macd_sig"]=df["macd"].ewm(span=9).mean(); df["macd_hist"]=df["macd"]-df["macd_sig"]
        d = c.diff(); g = d.where(d>0,0).rolling(14).mean(); l = (-d.where(d<0,0)).rolling(14).mean()
        df["rsi"]=100-100/(1+g/(l+1e-9))
        all_dfs[t] = df.sort_values("time").reset_index(drop=True)

    # Synthetic Index
    idx_df = pd.read_parquet(os.path.join(DATA_DIR, "raw", "parquet", "VNM_history.parquet"))
    idx_df["time"] = pd.to_datetime(idx_df["time"])
    idx_df["syn"] = idx_df["close"].rolling(50).mean() # Dùng SMA50 làm trend proxy

    dates = all_dfs["MBB"][(all_dfs["MBB"]["time"] >= START) & (all_dfs["MBB"]["time"] <= END)]["time"].dt.strftime("%Y-%m-%d").tolist()
    slots = [SlotV6(i) for i in range(6)]
    for s in slots: s.cash = INITIAL / 3 # Khởi đầu chia 3
    
    cooldowns = {}; peak_nav = INITIAL; max_dd = 0.0
    
    for date_str in dates:
        # 1. Regime & Max Slots
        sub = idx_df[idx_df["time"].dt.strftime("%Y-%m-%d") <= date_str]
        regime = "SIDEWAYS"
        if len(sub) > 2:
            last = sub.iloc[-1]["close"]; prev = sub.iloc[-2]["close"]
            regime = "BULL" if last > prev * 1.002 else ("BEAR" if last < prev * 0.998 else "SIDEWAYS")
        
        max_p = get_max_positions(regime)
        bt, st = REGIME_BUY[regime], REGIME_SEL[regime]
        
        # 2. Update & Stop-loss (duyệt tất cả slot có hàng)
        for s in slots:
            if s.shares and s.ticker:
                df = all_dfs[s.ticker]; r = df[df["time"].dt.strftime("%Y-%m-%d")==date_str]
                if not r.empty:
                    p = float(r.iloc[0]["close"]); s.cur = p
                    if p > s.peak: s.peak = p
                    if (p-s.ep)/s.ep*100 <= -SL: s.close(p, date_str, "SL")

        # 3. Score & Sell
        scs = {}
        for t in UNIVERSE:
            df = all_dfs[t]; r = df[df["time"].dt.strftime("%Y-%m-%d")==date_str]
            if not r.empty: scs[t] = score_v6(r.iloc[0], t, rlhf_w, date_str, regime)
        
        for s in slots:
            if s.shares and s.ticker and scs.get(s.ticker, 0) < st:
                df = all_dfs[s.ticker]; r = df[df["time"].dt.strftime("%Y-%m-%d")==date_str]
                if not r.empty: s.close(float(r.iloc[0]["close"]), date_str, "SELL")

        # 4. Buy (chỉ buy vào slot trống TRONG PHẠM VI max_p)
        occ = {s.ticker for s in slots if s.ticker}
        cands = sorted([(t, sc) for t, sc in scs.items() if t not in occ and sc >= bt], key=lambda x:-x[1])
        
        for i in range(max_p):
            s = slots[i]
            if not s.shares and cands:
                tk, sc = cands.pop(0)
                df = all_dfs[tk]; r = df[df["time"].dt.strftime("%Y-%m-%d")==date_str]
                if not r.empty:
                    # Tái cân bằng vốn: lấy NAV chia cho số slot max hiện tại
                    cur_nav = sum(sl.nav for sl in slots)
                    s.cash = cur_nav / max_p
                    s.open(tk, float(r.iloc[0]["close"]), date_str)

        # 5. Peak/DD
        nav = sum(s.nav for s in slots)
        if nav > peak_nav: peak_nav = nav
        dd = (peak_nav - nav)/peak_nav * 100
        if dd > max_dd: max_dd = dd

    print("="*60); print(f"KẾT QUẢ v6 FINAL (Dynamic {max_p} Slots)"); print("="*60)
    print(f"NAV Cuối kỳ: {nav:,.0f} VND | Lợi nhuận: {(nav-INITIAL)/INITIAL*100:+.2f}%")
    print(f"Max Drawdown: {max_dd:.2f}% | Mục tiêu 14M: {'ĐẠT' if nav >= TARGET else 'Chưa đạt'}")

if __name__ == "__main__":
    run_v6_final()
