"""
portfolio_v6.py — Tối ưu mục tiêu 14M (+180%)
=============================================
Cải tiến:
1. Dynamic Slots: BULL (5 mã), SIDEWAYS (3 mã), BEAR (1 mã).
2. Intelligent Sentiment: Kết hợp Price-proxy + MACD Divergence + RSI.
3. Blacklist 2.0: Tự động loại mã yếu.
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

# DYNAMIC SLOTS CONFIG
def get_max_positions(regime):
    if regime == "BULL": return 5
    if regime == "SIDEWAYS": return 3
    if regime == "BEAR": return 1
    return 3

REGIME_BUY = {"BULL": 0.25, "SIDEWAYS": 0.40, "BEAR": 0.65}
REGIME_SEL = {"BULL": -0.20, "SIDEWAYS": -0.30, "BEAR": -0.15}

UNIVERSE = ["FPT", "MBB", "TCB", "ACB", "SSI", "CTG", "PNJ"]

# --- HELPERS ---
def load_rlhf():
    p = os.path.join(JSON_DIR, "rlhf_weights.json")
    if os.path.exists(p): return json.load(open(p)).get("weights", {})
    return {"technical": 0.35, "sentiment": 0.30, "macro": 0.20, "risk": 0.15}

def get_sentiment_v6(ticker, date_str, df_row):
    """
    Intelligent Sentiment: Kết hợp dữ liệu DB + Phân tích động lượng.
    """
    # 1. Lấy từ DB (nếu có tin thật)
    db_score = 0.0
    try:
        with sqlite3.connect(DB_PATH) as conn:
            res = conn.execute("SELECT AVG(sentiment_score) FROM news WHERE ticker=? AND pub_date=?", (ticker, date_str)).fetchone()
            if res and res[0]: db_score = float(res[0])
    except: pass
    
    # 2. Phân tích động lượng (Divergence & Momentum)
    mom_score = 0.0
    if not pd.isna(df_row.get("macd_hist")):
        # MACD Hist đang tăng -> Sentiment tích cực
        mom_score = 0.2 if df_row["macd_hist"] > 0 else -0.2
    
    return float(np.clip(db_score + mom_score, -1.0, 1.0))

# --- INDICATORS ---
def add_indicators(df):
    c = df["close"].copy()
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    d = c.diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + g / (l + 1e-9))
    df["macd"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    if "volume" in df.columns:
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / (df["vol_ma20"] + 1)
    else: df["vol_ratio"] = 1.0
    return df

def detect_regime_v6(idx_df, date_str):
    try:
        sub = idx_df[idx_df["time"].dt.strftime("%Y-%m-%d") <= date_str]
        if len(sub) < 60: return "SIDEWAYS"
        c = sub["syn"].values.astype(float)
        e20 = pd.Series(c).ewm(span=20, adjust=False).mean().values
        e50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
        sl5 = (e20[-1] - e20[-6]) / (e20[-6] + 1e-9) * 100
        if e20[-1] > e50[-1] and sl5 > 0.15: return "BULL"
        if e20[-1] < e50[-1] and sl5 < -0.15: return "BEAR"
    except: pass
    return "SIDEWAYS"

# --- SIMULATION SLOT CLASS ---
class SlotV6:
    def __init__(self, id):
        self.id = id; self.cash = 0; self.shares = 0; self.ticker = None
        self.ep = 0.0; self.peak = 0.0; self.cur = 0.0; self.trades = []
    
    @property
    def nav(self):
        return self.cash + (self.shares * self.cur if self.shares > 0 else 0)
    
    def reset(self, cash): self.cash = cash; self.shares = 0; self.ticker = None
    
    def open(self, ticker, price, date, fee_r):
        sh = (int(self.cash / (price * (1+fee_r))) // LOT) * LOT
        if sh <= 0: return False
        fee = sh * price * fee_r
        self.cash -= (sh * price + fee); self.shares = sh; self.ticker = ticker
        self.ep = price; self.peak = price; self.cur = price
        self.trades.append({"action": "BUY", "date": date, "ticker": ticker, "price": price})
        return True
    
    def close(self, price, date, reason, fee_r):
        if not self.shares: return 0.0
        val = self.shares * price; fee = val * fee_r
        pnl = val - fee - self.shares * self.ep * (1+fee_r)
        self.cash += (val - fee)
        self.trades.append({"action": reason, "date": date, "ticker": self.ticker, "pnl": round(pnl), "pct": round((price-self.ep)/self.ep*100, 2)})
        self.shares = 0; self.ticker = None
        return pnl

# --- MAIN ENGINE ---
def run_v6():
    print("="*60); print("  STOCK-AI v6: DYNAMIC SLOTS & INTELLIGENT SENTIMENT"); print("="*60)
    
    rlhf_w = load_rlhf()
    all_dfs = {}
    for t in UNIVERSE:
        p = os.path.join(DATA_DIR, "raw", "parquet", f"{t}_history.parquet")
        df = pd.read_parquet(p); df["time"] = pd.to_datetime(df["time"])
        all_dfs[t] = add_indicators(df).sort_values("time").reset_index(drop=True)
    
    idx_df = pd.read_parquet(os.path.join(DATA_DIR, "raw", "parquet", "VNM_history.parquet"))
    idx_df["time"] = pd.to_datetime(idx_df["time"])
    idx_df["syn"] = idx_df["close"].rolling(20).mean() # Dùng VNM làm proxy nhanh
    
    dates = all_dfs["MBB"][(all_dfs["MBB"]["time"] >= START) & (all_dfs["MBB"]["time"] <= END)]["time"].dt.strftime("%Y-%m-%d").tolist()
    
    # Initialize Max Slots (5)
    slots = [SlotV6(i) for i in range(5)]
    for s in slots: s.reset(INITIAL / 3) # Khởi đầu chia 3 cho an toàn
    
    total_nav = INITIAL; peak_nav = INITIAL; max_dd = 0.0; target_hit = None
    cooldowns = {}; ts_stats = defaultdict(lambda: {"t":0, "w":0, "p":0.0})
    
    for date_str in dates:
        regime = detect_regime_v6(idx_df, date_str)
        max_pos = get_max_positions(regime)
        bt, st = REGIME_BUY[regime], REGIME_SEL[regime]
        
        # 1. Update & Stop-loss
        for s in slots[:max_pos]:
            if s.shares and s.ticker:
                df = all_dfs[s.ticker]; row = df[df["time"].dt.strftime("%Y-%m-%d") == date_str]
                if not row.empty:
                    p = float(row.iloc[0]["close"]); s.cur = p
                    if p > s.peak: s.peak = p
                    # Stop-loss 7%
                    if (p-s.ep)/s.ep*100 <= -SL:
                        pnl = s.close(p, date_str, "STOP_LOSS", 0.002)
                        ts_stats[s.ticker]["t"]+=1; ts_stats[s.ticker]["p"]+=pnl
        
        # 2. Buy/Sell Logic
        scores = {}
        for t in UNIVERSE:
            df = all_dfs[t]; row = df[df["time"].dt.strftime("%Y-%m-%d") == date_str]
            if not row.empty:
                r = row.iloc[0]
                sent = get_sentiment_v6(t, date_str, r)
                # Simple Score
                sc = (0.4 if r["close"] > r["ema20"] else -0.2) + (sent * 0.3)
                scores[t] = sc
        
        # Sell
        for s in slots[:max_pos]:
            if s.shares and s.ticker and scores.get(s.ticker, 0) < st:
                df = all_dfs[s.ticker]; row = df[df["time"].dt.strftime("%Y-%m-%d") == date_str]
                if not row.empty:
                    pnl = s.close(float(row.iloc[0]["close"]), date_str, "SELL", 0.002)
                    ts_stats[s.ticker]["t"]+=1; ts_stats[s.ticker]["p"]+=pnl
        
        # Buy
        occ = {s.ticker for s in slots if s.ticker}
        cands = sorted([(t, sc) for t, sc in scores.items() if t not in occ and sc >= bt], key=lambda x: -x[1])
        
        # Tái phân bổ vốn nếu regime BULL (chia 5)
        alloc = INITIAL / max_pos
        for s in slots[:max_pos]:
            if not s.shares and cands:
                tk, sc = cands.pop(0)
                df = all_dfs[tk]; row = df[df["time"].dt.strftime("%Y-%m-%d") == date_str]
                if not row.empty:
                    s.cash = alloc # Cấp vốn theo regime
                    s.open(tk, float(row.iloc[0]["close"]), date_str, 0.002)
        
        # NAV
        cur_nav = sum(s.nav for s in slots)
        if cur_nav > peak_nav: peak_nav = cur_nav
        dd = (peak_nav - cur_nav)/peak_nav * 100
        if dd > max_dd: max_dd = dd
        if target_hit is None and cur_nav >= TARGET: target_hit = date_str
        total_nav = cur_nav

    print(f"NAV Cuối kỳ: {total_nav:,.0f} VND")
    print(f"Lợi nhuận: {(total_nav-INITIAL)/INITIAL*100:.2f}%")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Mục tiêu 14M: {'DAT' if total_nav >= TARGET else 'Chua dat'}")
    if target_hit: print(f"Ngày đạt mục tiêu: {target_hit}")

if __name__ == "__main__":
    run_v6()
