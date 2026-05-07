"""
improve_pipeline.py — Nâng cấp toàn bộ pipeline 4 bước
Chạy: python tests/improve_pipeline.py
"""
import os, sys, time, json
from datetime import datetime
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
JSON_DIR = os.path.join(DATA_DIR, "reports", "json")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# ── Fix 1: Synthetic Market Index từ VNM + MBB (đủ data 2021-2026) ──────────
PROXY_TICKERS = ["VNM", "MBB"]  # Hai mã có đủ data dài hạn

def build_synthetic_index(data_dir=DATA_DIR):
    dfs = []
    for t in PROXY_TICKERS:
        p = os.path.join(data_dir, "raw", "parquet", f"{t}_history.parquet")
        if not os.path.exists(p): continue
        df = pd.read_parquet(p, engine="pyarrow")
        df["time"] = pd.to_datetime(df["time"])
        df = df[["time","close"]].rename(columns={"close": t})
        dfs.append(df.set_index("time"))
    if not dfs: return None
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.join(d, how="outer")
    merged = merged.ffill().dropna()
    # Normalize và average
    normed = merged / merged.iloc[0] * 1000
    merged["synthetic"] = normed.mean(axis=1)
    merged = merged.reset_index()
    print(f"  [Synthetic Index] {len(merged)} rows | {merged['time'].min().date()} -> {merged['time'].max().date()}")
    return merged

def detect_regime_synthetic(idx_df, as_of_date: str) -> str:
    try:
        if idx_df is None or len(idx_df) < 60: return "SIDEWAYS"
        sub = idx_df[idx_df["time"].dt.strftime("%Y-%m-%d") <= as_of_date]
        if len(sub) < 60: return "SIDEWAYS"
        c    = sub["synthetic"].values.astype(float)
        e20  = pd.Series(c).ewm(span=20, adjust=False).mean().values
        e50  = pd.Series(c).ewm(span=50, adjust=False).mean().values
        e200 = pd.Series(c).ewm(span=min(200,len(c)-1), adjust=False).mean().values
        sl5  = (e20[-1] - e20[-min(6,len(e20))]) / (e20[-min(6,len(e20))] + 1e-9) * 100
        if e20[-1] > e50[-1] and sl5 > 0.2:  return "BULL"
        if e20[-1] < e50[-1] and sl5 < -0.2: return "BEAR"
    except Exception:
        pass
    return "SIDEWAYS"

# ── Fix 2: Signal Filter nâng cao ─────────────────────────────────────────────
LONG_DATA = {"VNM", "MBB"}   # Mã có đủ data dài → ưu tiên

def score_v4(row, regime: str, is_long_data: bool, rlhf_w: dict) -> float:
    """Signal v4: thêm regime-aware + ưu tiên mã có data dài."""
    score = 0.0
    try:
        if pd.isna(row.get("macd", float("nan"))) or pd.isna(row.get("ema200", float("nan"))):
            return 0.0
        c   = row["close"]
        e20 = row["ema20"]
        e50 = row["ema50"]
        e200= row["ema200"]
        rsi = row["rsi"]
        mhst= row["macd_hist"]
        macd= row["macd"]
        msig= row["macd_sig"]
        vr  = row["vol_ratio"] if not pd.isna(row.get("vol_ratio", float("nan"))) else 1.0

        tech_w = rlhf_w.get("technical", 0.40)

        # Bắt buộc: trên EMA200
        if c < e200 * 0.99:
            return max(-0.5, -0.4)

        # Trend
        if c > e20 > e50 > e200:  score += 0.30 * tech_w * 2.5
        elif c > e20 > e50:       score += 0.20 * tech_w * 2.5
        elif c > e20:             score += 0.10 * tech_w * 2.5
        if c < e20 < e50:         score -= 0.30
        if c < e50 < e200:        score -= 0.50

        # MACD — cả 2 điều kiện
        if macd > msig and mhst > 0:  score += 0.25
        elif macd > msig:             score += 0.10
        if macd < msig and mhst < 0:  score -= 0.25
        elif macd < msig:             score -= 0.10

        # RSI
        if 45 < rsi < 65: score += 0.15
        if 35 < rsi <= 45:score += 0.10
        if rsi > 72:      score -= 0.25
        if rsi < 28:      score -= 0.25

        # Volume
        if vr > 1.5 and score > 0: score += 0.10
        if vr < 0.5:               score -= 0.05

        # Bonus nếu là mã có data dài (ổn định hơn)
        if is_long_data and score > 0: score += 0.05

    except Exception:
        pass
    return float(np.clip(score, -1.0, 1.0))

# ── Fix 3: Prioritized Experience Replay ─────────────────────────────────────
def run_prioritized_replay(all_dfs, idx_df, f=None):
    try:
        from rlhf_engine import FeedbackStore, WeightAdapter
    except ImportError as e:
        print(f"  ❌ RLHF import fail: {e}")
        return {}

    START = "2024-01-01"
    END   = "2025-12-31"
    OUTCOME_DAYS = 15   # tăng từ 10 → 15 ngày
    UPDATE_EVERY = 15

    # Lấy ngày giao dịch
    sample_key = "VNM" if "VNM" in all_dfs else next(iter(all_dfs))
    sample = all_dfs[sample_key]
    dates = sample[(sample["time"] >= START) & (sample["time"] <= END)]["time"].dt.strftime("%Y-%m-%d").tolist()

    store   = FeedbackStore()
    adapter = WeightAdapter.load()

    def get_row(ticker, date_str):
        df  = all_dfs.get(ticker)
        if df is None: return None
        sub = df[df["time"].dt.strftime("%Y-%m-%d") == date_str]
        return sub.iloc[0] if len(sub) > 0 else None

    def get_actual(ticker, date_str, n_days):
        df  = all_dfs.get(ticker)
        if df is None: return None
        mask = df["time"].dt.strftime("%Y-%m-%d") == date_str
        if not mask.any(): return None
        idx0 = df[mask].index[0]
        idx1 = idx0 + n_days
        if idx1 >= len(df): return None
        p0, p1 = float(df.loc[idx0,"close"]), float(df.loc[idx1,"close"])
        return (p1-p0)/p0*100 if p0 > 0 else None

    rlhf_w  = adapter.weights
    pending = []
    sigs, outs, upds = 0, 0, 0
    weight_hist = []
    prev_w = dict(adapter.weights)

    for day_idx, date_str in enumerate(dates):
        regime = detect_regime_synthetic(idx_df, date_str)
        thresh_buy  = {"BULL": 0.30, "SIDEWAYS": 0.45, "BEAR": 0.65}.get(regime, 0.45)

        for ticker, df in all_dfs.items():
            row = get_row(ticker, date_str)
            if row is None: continue
            is_long = ticker in LONG_DATA
            sc = score_v4(row, regime, is_long, rlhf_w)
            if abs(sc) < thresh_buy: continue

            signal = "BUY" if sc > 0 else "SELL"
            fc_ret  = float(sc * 4.0)
            conf    = float(min(abs(sc), 1.0))
            agent_s = {
                "technical": float(np.clip(sc * 1.2, -1, 1)),
                "sentiment": 0.0,
                "macro":     0.05 if regime == "BULL" else (-0.05 if regime == "BEAR" else 0.0),
                "risk":      float(np.clip(-abs(sc) * 0.3 + 0.1, -1, 1)),
            }
            rid = store.record_signal(ticker, date_str, signal, fc_ret, conf, agent_s)
            pending.append({"id": rid, "ticker": ticker, "date": date_str, "signal": signal, "agent_scores": agent_s})
            sigs += 1

        # Fill outcomes
        still = []
        for p in pending:
            td = (pd.Timestamp(date_str) - pd.Timestamp(p["date"])).days
            if td < OUTCOME_DAYS: still.append(p); continue
            actual = get_actual(p["ticker"], p["date"], OUTCOME_DAYS)
            if actual is None: still.append(p); continue
            store.update_outcome(p["id"], actual_return_pct=actual, vnindex_return_pct=0.0)
            outs += 1
        pending = still

        # RLHF update
        if (day_idx + 1) % UPDATE_EVERY == 0:
            rh = store.get_recent_rewards("ALL", lookback_days=999)
            rewarded = [r for r in rh if r.get("reward") is not None]
            if len(rewarded) >= 10:
                # Prioritized: weight các mẫu có |reward| lớn hơn
                sorted_r = sorted(rewarded, key=lambda x: -abs(x.get("reward", 0)))
                top_r    = sorted_r[:min(len(sorted_r), 200)]  # Top 200 mẫu quan trọng nhất
                adapter.adapt_from_history(top_r)
                adapter.save()
                rlhf_w = adapter.weights
                upds += 1
                diff = {k: round(adapter.weights[k] - prev_w.get(k,0), 4) for k in adapter.weights}
                weight_hist.append({"day": date_str, "weights": dict(adapter.weights), "delta": diff})
                prev_w = dict(adapter.weights)

    adapter.save()
    rh  = store.get_recent_rewards("ALL", lookback_days=999)
    rwd = [r for r in rh if r.get("reward") is not None]
    pos = [r["reward"] for r in rwd if r["reward"] > 0]
    neg = [r["reward"] for r in rwd if r["reward"] < 0]
    avg = float(np.mean([r["reward"] for r in rwd])) if rwd else 0.0

    return {
        "signals": sigs, "outcomes": outs, "updates": upds,
        "total_rewarded": len(rwd),
        "positive": len(pos), "negative": len(neg), "avg_reward": round(avg, 6),
        "final_weights": adapter.weights,
        "weight_history": weight_hist,
    }

# ── Fix 4: Portfolio với regime + v4 signal ───────────────────────────────────
def run_portfolio_v4(all_dfs, idx_df, rlhf_w, f=None):
    START, END = "2024-01-01", "2025-12-31"
    FEE_R = 0.002; LOT = 100; SL = 7.0; TRAIL = 4.0; CD = 5
    MAX_POS = 3
    INITIAL = 5_000_000; TARGET = 14_000_000

    REGIME_BUY = {"BULL": 0.30, "SIDEWAYS": 0.50, "BEAR": 0.65}
    REGIME_SEL = {"BULL": -0.25, "SIDEWAYS": -0.30, "BEAR": -0.20}

    sample_key = "VNM" if "VNM" in all_dfs else next(iter(all_dfs))
    sample = all_dfs[sample_key]
    dates = sample[(sample["time"] >= START) & (sample["time"] <= END)]["time"].dt.strftime("%Y-%m-%d").tolist()

    class Slot:
        def __init__(self, alloc):
            self.cash = alloc; self.shares = 0; self.ticker = None
            self.ep = 0.0; self.ep_date = None; self.peak = 0.0; self.cur = 0.0
            self.trades = []
        @property
        def nav(self):
            return self.cash + self.shares * self.cur if self.shares > 0 and self.cur > 0 else self.cash
        def open(self, ticker, price, date):
            sh = (int(self.cash / (price * (1+FEE_R))) // LOT) * LOT
            if sh <= 0: return False
            fee = sh * price * FEE_R
            self.cash -= sh*price+fee; self.shares=sh; self.ticker=ticker
            self.ep=price; self.ep_date=date; self.peak=price; self.cur=price
            self.trades.append({"action":"BUY","date":date,"ticker":ticker,"price":price,"shares":sh,"fee":round(fee)})
            return True
        def close(self, price, date, reason="SELL"):
            if not self.shares: return 0.0
            pr = self.shares*price; fee=pr*FEE_R; pnl=pr-fee-self.shares*self.ep*(1+FEE_R)
            self.cash += pr-fee
            self.trades.append({"action":reason,"date":date,"ticker":self.ticker,
                "price":price,"shares":self.shares,"fee":round(fee),"profit":round(pnl),
                "profit_pct":round((price-self.ep)/self.ep*100,2),"entry_date":self.ep_date})
            self.shares=0; self.ticker=None; self.ep=0; self.peak=0
            return pnl
        def upd(self, price):
            self.cur=price
            if price > self.peak: self.peak=price
        def stop(self, price):
            if not self.shares or not self.ep: return ""
            if (price-self.ep)/self.ep*100 <= -SL: return "STOP_LOSS"
            if self.peak>0 and (price-self.peak)/self.peak*100<=-TRAIL and price<self.ep*0.98: return "TRAILING_STOP"
            return ""

    def get_row(t, d):
        df = all_dfs.get(t)
        if df is None: return None
        sub = df[df["time"].dt.strftime("%Y-%m-%d") == d]
        return sub.iloc[0] if len(sub) > 0 else None

    slots = [Slot(INITIAL/MAX_POS) for _ in range(MAX_POS)]
    cooldowns = {}; pnav_hist = []; peak_nav = INITIAL; max_dd = 0.0
    daily_r = []; prev_nav = INITIAL; target_hit = None
    regime_log = {}; prev_regime = "SIDEWAYS"
    from collections import defaultdict
    ticker_stats = defaultdict(lambda: {"trades":0,"wins":0,"profit":0.0})

    for date_str in dates:
        regime = detect_regime_synthetic(idx_df, date_str)
        if regime != prev_regime:
            regime_log[date_str] = regime; prev_regime = regime

        buy_th = REGIME_BUY[regime]; sell_th = REGIME_SEL[regime]

        for s in slots:
            if s.shares and s.ticker:
                row = get_row(s.ticker, date_str)
                if row is not None:
                    p = float(row["close"]); s.upd(p)
                    reason = s.stop(p)
                    if reason:
                        tk = s.ticker; pnl = s.close(p, date_str, reason); cooldowns[tk] = date_str
                        ticker_stats[tk]["trades"] += 1; ticker_stats[tk]["profit"] += pnl
                        if pnl > 0: ticker_stats[tk]["wins"] += 1

        scores = {}
        for t, df in all_dfs.items():
            row = get_row(t, date_str)
            if row is not None:
                scores[t] = score_v4(row, regime, t in LONG_DATA, rlhf_w)

        for s in slots:
            if s.shares and s.ticker:
                row = get_row(s.ticker, date_str)
                if row is not None and scores.get(s.ticker, 0) < sell_th:
                    tk = s.ticker; p = float(row["close"])
                    pnl = s.close(p, date_str, "SELL"); cooldowns[tk] = date_str
                    ticker_stats[tk]["trades"] += 1; ticker_stats[tk]["profit"] += pnl
                    if pnl > 0: ticker_stats[tk]["wins"] += 1

        occupied = {s.ticker for s in slots if s.ticker}
        on_cd = {t for t,d in cooldowns.items()
                 if (pd.Timestamp(date_str)-pd.Timestamp(d)).days < CD}
        cands = sorted(
            [(t,sc) for t,sc in scores.items() if t not in occupied and t not in on_cd and sc >= buy_th],
            key=lambda x: -x[1]
        )
        for s in slots:
            if not s.shares and cands:
                tk, sc = cands.pop(0)
                row = get_row(tk, date_str)
                if row is not None: s.open(tk, float(row["close"]), date_str)

        pnav = 0.0
        for s in slots:
            if s.shares and s.ticker:
                row = get_row(s.ticker, date_str)
                if row is not None: s.cur = float(row["close"])
            pnav += s.nav

        if pnav > peak_nav: peak_nav = pnav
        dd = (peak_nav - pnav) / peak_nav * 100
        if dd > max_dd: max_dd = dd
        daily_r.append((pnav - prev_nav)/prev_nav if prev_nav > 0 else 0)
        prev_nav = pnav
        pnav_hist.append({"date": date_str, "nav": round(pnav), "regime": regime})
        if target_hit is None and pnav >= TARGET: target_hit = date_str

    last = dates[-1]
    for s in slots:
        if s.shares and s.ticker:
            row = get_row(s.ticker, last)
            if row is not None:
                tk = s.ticker; pnl = s.close(float(row["close"]), last, "LIQUIDATE")
                ticker_stats[tk]["trades"] += 1; ticker_stats[tk]["profit"] += pnl
                if pnl > 0: ticker_stats[tk]["wins"] += 1

    final = sum(s.cash for s in slots)
    arr = np.array(daily_r)
    sharpe = float(arr.mean()/arr.std()*np.sqrt(252)) if arr.std() > 1e-9 else 0.0

    all_t = []
    for s in slots: all_t.extend(s.trades)
    all_t.sort(key=lambda x: x["date"])
    sell_t = [t for t in all_t if any(k in t["action"] for k in ("SELL","STOP","LIQ"))]
    win_t  = [t for t in sell_t if t.get("profit",0) > 0]

    return {
        "final_nav": round(final), "profit": round(final - INITIAL),
        "return_pct": round((final-INITIAL)/INITIAL*100, 2),
        "goal": final >= TARGET, "target_hit": target_hit,
        "max_dd": round(max_dd, 2), "sharpe": round(sharpe, 4),
        "trades": len(sell_t), "wins": len(win_t),
        "win_rate": round(len(win_t)/len(sell_t)*100 if sell_t else 0, 2),
        "regime_changes": len(regime_log), "regime_log": regime_log,
        "ticker_stats": {k: dict(v) for k, v in ticker_stats.items()},
    }

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(JSON_DIR, exist_ok=True)
    t0 = time.time()
    print("═" * 60)
    print("  STOCK-AI — IMPROVE PIPELINE v4")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60)

    # Load data
    print("\n[1/4] Load dữ liệu...")
    all_dfs = {}
    for t in ["VNM","VCB","FPT","HPG","MBB","TCB","ACB","MWG","SSI","VHM","BID","CTG","GAS","MSN","PNJ"]:
        p = os.path.join(DATA_DIR, "raw", "parquet", f"{t}_history.parquet")
        if not os.path.exists(p):
            p2 = os.path.join(DATA_DIR, "raw", "csv", f"{t}_history.csv")
            if not os.path.exists(p2): continue
            df = pd.read_csv(p2); df["time"] = pd.to_datetime(df["time"])
        else:
            df = pd.read_parquet(p, engine="pyarrow"); df["time"] = pd.to_datetime(df["time"])
        c = df["close"].copy()
        df["ema20"]    = c.ewm(span=20,adjust=False).mean()
        df["ema50"]    = c.ewm(span=50,adjust=False).mean()
        df["ema200"]   = c.ewm(span=200,adjust=False).mean()
        delta = c.diff()
        g = delta.where(delta > 0, 0).rolling(14).mean()
        l = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["rsi"]      = 100 - 100/(1+g/(l+1e-9))
        df["macd"]     = c.ewm(span=12,adjust=False).mean() - c.ewm(span=26,adjust=False).mean()
        df["macd_sig"] = df["macd"].ewm(span=9,adjust=False).mean()
        df["macd_hist"]= df["macd"] - df["macd_sig"]
        if "volume" in df.columns:
            df["vol_ma20"]  = df["volume"].rolling(20).mean()
            df["vol_ratio"] = df["volume"]/(df["vol_ma20"]+1)
        else:
            df["vol_ratio"] = 1.0
        all_dfs[t] = df.sort_values("time").reset_index(drop=True)
    print(f"  Loaded {len(all_dfs)} mã")

    # Build synthetic index
    print("\n[2/4] Xây Synthetic Market Index từ VNM + MBB...")
    idx_df = build_synthetic_index(DATA_DIR)

    # Test regime detection
    test_dates = ["2024-03-01", "2024-07-01", "2024-12-01", "2025-06-01"]
    print("  Test Regime:")
    for td in test_dates:
        r = detect_regime_synthetic(idx_df, td)
        print(f"    {td}: {r}")

    # Prioritized replay
    print("\n[3/4] Prioritized Experience Replay...")
    rl_res = run_prioritized_replay(all_dfs, idx_df)
    print(f"  Signals: {rl_res.get('signals',0):,} | Outcomes: {rl_res.get('outcomes',0):,} | Updates: {rl_res.get('updates',0)}")
    print(f"  Avg Reward: {rl_res.get('avg_reward',0):+.6f} | Pos: {rl_res.get('positive',0)} | Neg: {rl_res.get('negative',0)}")
    fw = rl_res.get("final_weights", {})
    print(f"  Weights: tech={fw.get('technical',0):.3f} sent={fw.get('sentiment',0):.3f} macro={fw.get('macro',0):.3f} risk={fw.get('risk',0):.3f}")

    # Portfolio v4
    print("\n[4/4] Backtest Portfolio v4...")
    pf = run_portfolio_v4(all_dfs, idx_df, fw)

    elapsed = round(time.time()-t0, 2)
    print("\n" + "═"*60)
    print("  KẾT QUẢ PORTFOLIO v4")
    print("═"*60)
    print(f"  NAV cuối kỳ : {pf['final_nav']:>12,.0f} VND")
    print(f"  Lợi nhuận   : {pf['profit']:>+12,.0f} VND  ({pf['return_pct']:+.2f}%)")
    print(f"  Mục tiêu    : {'✅ ĐẠT' if pf['goal'] else '❌ Chưa đạt'} 14,000,000 VND")
    print(f"  Sharpe      : {pf['sharpe']:.4f}")
    print(f"  MaxDrawdown : {pf['max_dd']:.2f}%")
    print(f"  Win Rate    : {pf['win_rate']:.1f}%  ({pf['wins']}/{pf['trades']} lệnh)")
    print(f"  Regime thay đổi: {pf['regime_changes']} lần")
    print()
    ts = pf["ticker_stats"]
    print(f"  {'Mã':<6} {'Lệnh':>5} {'Win':>5} {'Win%':>7} {'Lợi nhuận':>14}")
    print(f"  {'─'*6} {'─'*5} {'─'*5} {'─'*7} {'─'*14}")
    for tk in sorted(ts, key=lambda x: -ts[x]["profit"]):
        d = ts[tk]; wr = d["wins"]/d["trades"]*100 if d["trades"] > 0 else 0
        print(f"  {tk:<6} {d['trades']:>5} {d['wins']:>5} {wr:>6.1f}% {d['profit']:>+14,.0f}")
    print()
    print(f"  ⏱ {elapsed}s")
    print("═"*60)

    # Lưu JSON
    report = {
        "generated_at": datetime.now().isoformat(),
        "elapsed_sec": elapsed,
        "rl_training": rl_res,
        "portfolio_v4": pf,
    }
    out = os.path.join(JSON_DIR, "improve_pipeline_report.json")
    with open(out, "w", encoding="utf-8") as jf:
        json.dump(report, jf, indent=2, ensure_ascii=False, default=str)
    print(f"✓ Report: {out}")

if __name__ == "__main__":
    main()
