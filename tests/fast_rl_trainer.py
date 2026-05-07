"""
fast_rl_trainer.py — Giả lập thời gian nhanh để train RLHF
============================================================
Ý tưởng: "Experience Replay" từ dữ liệu lịch sử 2024-2025.

Thay vì chờ giao dịch thực (mất hàng tháng để tích lũy đủ mẫu),
script này REPLAY toàn bộ 2 năm dữ liệu lịch sử để:

  1. Phát hiện tín hiệu BUY/SELL từng ngày (như đang giao dịch thật)
  2. Ghi nhận vào RLHF FeedbackStore (giả lập signal_date thật)
  3. Sau mỗi N ngày, fill outcome (đã có giá thật → tính ngay)
  4. Chạy RLHF cycle → update weights
  5. Dùng weights mới cho giai đoạn tiếp theo
  → Trong vài giây, AI học được "2 năm kinh nghiệm"

Chạy: python tests/fast_rl_trainer.py
"""

import json, os, sys, time, sqlite3
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR     = os.path.join(BASE_DIR, "src")
DATA_DIR    = os.path.join(BASE_DIR, "data")
JSON_DIR    = os.path.join(DATA_DIR, "reports", "json")
TXT_OUT     = os.path.join(DATA_DIR, "reports", "fast_rl_training_report.txt")
JSON_OUT    = os.path.join(JSON_DIR,  "fast_rl_training_report.json")

sys.path.insert(0, SRC_DIR)

# ── Config ────────────────────────────────────────────────────────────────────
START_DATE   = "2024-01-01"
END_DATE     = "2025-12-31"
OUTCOME_DAYS = 10          # Sau 10 ngày giao dịch → xem kết quả thực
UPDATE_EVERY = 20          # Sau 20 ngày trading → chạy 1 chu kỳ RLHF update
MIN_SCORE_TO_SIGNAL = 0.40 # Chỉ ghi nhận tín hiệu đủ mạnh

UNIVERSE = [
    "VNM", "VCB", "FPT", "HPG", "MBB",
    "TCB", "ACB", "MWG", "SSI", "VHM",
    "BID", "CTG",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def log(msg, f=None):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if f: f.write(line + "\n"); f.flush()

def sep(title="", f=None):
    n = max(0, 56 - len(title))
    log((f"── {title} " + "─" * n) if title else "─" * 60, f)

def load_df(ticker):
    for ext, reader in [(".parquet", pd.read_parquet), (".csv", pd.read_csv)]:
        sub = "parquet" if ext == ".parquet" else "csv"
        p   = os.path.join(DATA_DIR, "raw", sub, f"{ticker}_history{ext}")
        if os.path.exists(p):
            kw = {"engine": "pyarrow"} if ext == ".parquet" else {}
            df = reader(p, **kw)
            df["time"] = pd.to_datetime(df["time"])
            return df.sort_values("time").reset_index(drop=True)
    return None

def add_indicators(df):
    c = df["close"].copy()
    df["ema20"]    = c.ewm(span=20, adjust=False).mean()
    df["ema50"]    = c.ewm(span=50, adjust=False).mean()
    df["ema200"]   = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    g = delta.where(delta > 0, 0).rolling(14).mean()
    l = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"]      = 100 - 100 / (1 + g / (l + 1e-9))
    df["macd"]     = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]= df["macd"] - df["macd_sig"]
    if "volume" in df.columns:
        df["vol_ma20"]  = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / (df["vol_ma20"] + 1)
    else:
        df["vol_ratio"] = 1.0
    return df

def score_row(row) -> tuple:
    """Trả về (score, agent_scores_dict, signal_str)"""
    score = 0.0
    try:
        if pd.isna(row["macd"]) or pd.isna(row["ema200"]):
            return 0.0, {}, "HOLD"
        c, e20, e50, e200 = row["close"], row["ema20"], row["ema50"], row["ema200"]
        rsi  = row["rsi"]
        macd, msig, mhst = row["macd"], row["macd_sig"], row["macd_hist"]
        vr   = row["vol_ratio"] if not pd.isna(row["vol_ratio"]) else 1.0

        tech_score = 0.0
        if c < e200 * 0.99:
            tech_score = -0.5
        else:
            if c > e20 > e50 > e200: tech_score += 0.30
            elif c > e20 > e50:      tech_score += 0.20
            elif c > e20:            tech_score += 0.10
            if c < e20 < e50:        tech_score -= 0.25
            if c < e50 < e200:       tech_score -= 0.45

        macd_score = 0.0
        if macd > msig and mhst > 0:  macd_score = 0.25
        elif macd > msig:             macd_score = 0.10
        elif macd < msig and mhst < 0:macd_score = -0.25
        elif macd < msig:             macd_score = -0.10

        rsi_score = 0.0
        if 45 < rsi < 65:   rsi_score = 0.15
        if 35 < rsi <= 45:  rsi_score = 0.10
        if rsi > 70:        rsi_score -= 0.20
        if rsi < 30:        rsi_score -= 0.20

        vol_score = 0.10 if vr > 1.5 else (-0.05 if vr < 0.5 else 0.0)

        # Agent scores (giả lập 4 agents)
        agent_scores = {
            "technical":  float(np.clip(tech_score + macd_score, -1, 1)),
            "sentiment":  0.0,   # Chưa có data thật → neutral
            "macro":      float(np.clip(rsi_score * 0.5, -1, 1)),
            "risk":       float(np.clip(vol_score + (0.1 if c > e200 else -0.1), -1, 1)),
        }

        score = tech_score + macd_score + rsi_score + vol_score
        score = float(np.clip(score, -1.0, 1.0))
    except Exception:
        return 0.0, {}, "HOLD"

    signal = "BUY" if score >= MIN_SCORE_TO_SIGNAL else "SELL" if score <= -MIN_SCORE_TO_SIGNAL else "HOLD"
    return score, agent_scores, signal

def get_actual_return(df, signal_date_str: str, outcome_days: int) -> float | None:
    """Tính return thực tế sau outcome_days ngày giao dịch."""
    try:
        mask0 = df["time"].dt.strftime("%Y-%m-%d") == signal_date_str
        if not mask0.any():
            return None
        idx0  = df[mask0].index[0]
        idx1  = idx0 + outcome_days
        if idx1 >= len(df):
            return None
        p0 = float(df.loc[idx0, "close"])
        p1 = float(df.loc[idx1, "close"])
        return (p1 - p0) / p0 * 100.0 if p0 > 0 else None
    except Exception:
        return None

def get_vnindex_return(vnindex_df, d0: str, outcome_days: int) -> float:
    try:
        mask  = vnindex_df["time"].dt.strftime("%Y-%m-%d") == d0
        if not mask.any(): return 0.0
        idx0  = vnindex_df[mask].index[0]
        idx1  = idx0 + outcome_days
        if idx1 >= len(vnindex_df): return 0.0
        p0 = float(vnindex_df.loc[idx0, "close"])
        p1 = float(vnindex_df.loc[idx1, "close"])
        return (p1 - p0) / p0 * 100.0 if p0 > 0 else 0.0
    except Exception:
        return 0.0

# ── RLHF Fast-Replay ─────────────────────────────────────────────────────────
def run_fast_rl_training(f=None):
    try:
        from rlhf_engine import FeedbackStore, WeightAdapter, RewardCalculator
    except ImportError as e:
        log(f"❌ Không load được rlhf_engine: {e}", f)
        return None

    # Load dữ liệu
    all_dfs    = {}
    for t in UNIVERSE:
        df = load_df(t)
        if df is None: continue
        df = add_indicators(df)
        all_dfs[t] = df

    vnindex_df = load_df("VNINDEX")
    if vnindex_df is None:
        log("⚠  Không có VNINDEX — dùng 0 cho alpha", f)

    if not all_dfs:
        log("❌ Không có dữ liệu nào!", f)
        return None

    # Lấy danh sách ngày giao dịch trong khoảng 2024-2025
    sample_df = next(iter(all_dfs.values()))
    dates = sample_df[
        (sample_df["time"] >= START_DATE) & (sample_df["time"] <= END_DATE)
    ]["time"].dt.strftime("%Y-%m-%d").tolist()

    log(f"  Tổng ngày giao dịch: {len(dates)} phiên", f)
    log(f"  Mã theo dõi: {len(all_dfs)} ({', '.join(sorted(all_dfs.keys()))})", f)

    store   = FeedbackStore()
    adapter = WeightAdapter.load()   # Load weights hiện tại nếu có

    # ── Replay loop ──────────────────────────────────────────────────────────
    signals_recorded = 0
    outcomes_filled  = 0
    weight_updates   = 0
    pending: list[dict] = []   # {id, ticker, date, df_ref}

    weight_history: list[dict] = []
    prev_weights = dict(adapter.weights)

    log("", f)
    log(f"  Bắt đầu replay 2024-2025... (mỗi dấu = 20 phiên)", f)

    for day_idx, date_str in enumerate(dates):

        # 1. Phát tín hiệu cho mỗi mã hôm nay
        for ticker, df in all_dfs.items():
            mask = df["time"].dt.strftime("%Y-%m-%d") == date_str
            if not mask.any(): continue
            row = df[mask].iloc[0]
            score, agent_scores, signal = score_row(row)

            if signal != "HOLD":
                # Tính forecast_return (đơn giản: score * 5% như proxy)
                forecast_ret = float(score * 5.0)
                confidence   = min(abs(score), 1.0)
                row_id = store.record_signal(
                    ticker=ticker,
                    signal_date=date_str,
                    signal=signal,
                    forecast_return_pct=forecast_ret,
                    confidence=confidence,
                    agent_scores=agent_scores,
                )
                pending.append({
                    "id": row_id, "ticker": ticker,
                    "date": date_str, "signal": signal,
                    "agent_scores": agent_scores,
                })
                signals_recorded += 1

        # 2. Fill outcomes cho tín hiệu đủ OUTCOME_DAYS ngày trước
        still_pending = []
        for p in pending:
            d0   = p["date"]
            td   = (pd.Timestamp(date_str) - pd.Timestamp(d0)).days
            if td < OUTCOME_DAYS:
                still_pending.append(p); continue

            df_ref = all_dfs.get(p["ticker"])
            if df_ref is None:
                still_pending.append(p); continue

            actual = get_actual_return(df_ref, d0, OUTCOME_DAYS)
            if actual is None:
                still_pending.append(p); continue

            vn_ret = get_vnindex_return(vnindex_df, d0, OUTCOME_DAYS) if vnindex_df is not None else 0.0
            store.update_outcome(p["id"], actual_return_pct=actual, vnindex_return_pct=vn_ret)
            outcomes_filled += 1
        pending = still_pending

        # 3. RLHF weight update mỗi UPDATE_EVERY ngày
        if (day_idx + 1) % UPDATE_EVERY == 0:
            rh = store.get_recent_rewards("ALL", lookback_days=999)
            rewarded = [r for r in rh if r.get("reward") is not None]
            if len(rewarded) >= 10:
                adapter.adapt_from_history(rewarded)
                adapter.save()
                weight_updates += 1
                diff = {k: round(adapter.weights[k] - prev_weights.get(k, 0), 4)
                        for k in adapter.weights}
                weight_history.append({
                    "day": date_str, "weights": dict(adapter.weights), "delta": diff
                })
                prev_weights = dict(adapter.weights)
            if (day_idx + 1) % 100 == 0:
                print(".", end="", flush=True)

    print()  # newline sau dots

    # Lưu weights cuối
    adapter.save()

    # ── Tính thống kê ─────────────────────────────────────────────────────────
    all_rewards = store.get_recent_rewards("ALL", lookback_days=999)
    r_with_outcome = [r for r in all_rewards if r.get("reward") is not None]
    pos_r = [r["reward"] for r in r_with_outcome if r["reward"] > 0]
    neg_r = [r["reward"] for r in r_with_outcome if r["reward"] < 0]

    # Per-ticker accuracy
    ticker_acc: dict = defaultdict(lambda: {"total": 0, "correct": 0, "reward_sum": 0.0})
    for r in r_with_outcome:
        tk = r.get("ticker") or "?"
        # Lấy ticker từ pending history — gần đúng từ agent_scores key
        ticker_acc[tk]["total"] += 1
        if r["reward"] > 0:
            ticker_acc[tk]["correct"] += 1
        ticker_acc[tk]["reward_sum"] += r["reward"]

    return {
        "signals_recorded": signals_recorded,
        "outcomes_filled":  outcomes_filled,
        "weight_updates":   weight_updates,
        "final_weights":    adapter.weights,
        "weight_history":   weight_history,
        "total_rewarded":   len(r_with_outcome),
        "positive_rewards": len(pos_r),
        "negative_rewards": len(neg_r),
        "avg_reward":       round(float(np.mean([r["reward"] for r in r_with_outcome])), 6)
                            if r_with_outcome else 0.0,
        "ticker_accuracy":  {k: dict(v) for k, v in ticker_acc.items()},
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(JSON_DIR, exist_ok=True)
    t0 = time.time()

    with open(TXT_OUT, "w", encoding="utf-8") as f:
        log("═" * 60, f)
        log("  STOCK-AI — FAST RL TRAINER (Experience Replay)", f)
        log(f"  Chạy lúc    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f)
        log(f"  Replay      : {START_DATE} → {END_DATE}", f)
        log(f"  Outcome sau : {OUTCOME_DAYS} ngày giao dịch", f)
        log(f"  Update RLHF : mỗi {UPDATE_EVERY} phiên", f)
        log(f"  Ngưỡng tín hiệu: |score| ≥ {MIN_SCORE_TO_SIGNAL}", f)
        log("═" * 60, f)
        log("", f)
        log("  🎯 Mục tiêu: Dùng dữ liệu lịch sử 2024-2025 để cho AI", f)
        log("     học nhanh — thay vì chờ tháng/năm từ giao dịch thực.", f)
        log("     Kỹ thuật: Experience Replay (giống DQN trong game AI)", f)
        log("", f)

        sep("BẮT ĐẦU REPLAY", f)
        res = run_fast_rl_training(f)

        if res is None:
            log("❌ Training thất bại!", f)
            return

        elapsed = round(time.time() - t0, 2)

        sep("KẾT QUẢ TRAINING", f)
        log(f"  Tín hiệu ghi nhận  : {res['signals_recorded']:>8,}", f)
        log(f"  Outcomes đã fill   : {res['outcomes_filled']:>8,}", f)
        log(f"  RLHF updates       : {res['weight_updates']:>8,}", f)
        log(f"  Tổng mẫu có reward : {res['total_rewarded']:>8,}", f)
        log(f"  Reward dương       : {res['positive_rewards']:>8,}  (tín hiệu đúng)", f)
        log(f"  Reward âm          : {res['negative_rewards']:>8,}  (tín hiệu sai)", f)
        log(f"  Avg Reward         : {res['avg_reward']:>+8.6f}", f)
        log("", f)

        sep("WEIGHTS SAU TRAINING", f)
        fw = res["final_weights"]
        log(f"  technical  : {fw.get('technical',0):.4f}", f)
        log(f"  sentiment  : {fw.get('sentiment',0):.4f}", f)
        log(f"  macro      : {fw.get('macro',0):.4f}", f)
        log(f"  risk       : {fw.get('risk',0):.4f}", f)
        log("", f)

        sep("LỊCH SỬ THAY ĐỔI WEIGHTS", f)
        for wh in res["weight_history"][:10]:
            tech  = wh["weights"].get("technical", 0)
            sent  = wh["weights"].get("sentiment", 0)
            macro = wh["weights"].get("macro", 0)
            risk  = wh["weights"].get("risk", 0)
            dt    = wh["delta"]
            dt_str= f"tech{dt.get('technical',0):+.3f} sent{dt.get('sentiment',0):+.3f}"
            log(f"  {wh['day']}: tech={tech:.3f} sent={sent:.3f} macro={macro:.3f} risk={risk:.3f} | Δ {dt_str}", f)
        if len(res["weight_history"]) > 10:
            log(f"  ... ({len(res['weight_history'])-10} updates nữa)", f)
        log("", f)

        sep("PHÂN TÍCH", f)
        aw = res["avg_reward"]
        if aw > 0.05:
            log("  ✅ AI học được — avg reward dương: hướng đầu tư đúng", f)
        elif aw > 0:
            log("  ⚠  AI học yếu — avg reward dương nhưng gần 0", f)
        else:
            log("  ❌ AI chưa học được — avg reward âm", f)
            log("     → Thị trường 2024-2025 có nhiều giai đoạn sideways/bear", f)
            log("     → Tín hiệu kỹ thuật EMA/MACD nhiễu cao trong pha này", f)

        log("", f)
        log("  💡 Gợi ý cải tiến Experience Replay:", f)
        log("  1. Thêm Market Regime vào agent_scores để RLHF phân biệt pha", f)
        log("  2. Dùng Prioritized Replay: học nhiều hơn từ lệnh lỗ lớn", f)
        log("  3. Tăng OUTCOME_DAYS lên 20-30 để signal có thời gian phát huy", f)
        log("  4. Chạy lại backtest_portfolio sau training để đo cải thiện", f)
        log("  5. Schedule: chạy fast_rl_trainer mỗi tuần với data mới nhất", f)

        log("", f)
        log(f"  ⏱ Thời gian thực thi: {elapsed}s (replay 2 năm!)", f)
        log(f"  📄 TXT : {TXT_OUT}", f)
        log(f"  📊 JSON: {JSON_OUT}", f)
        log("═" * 60, f)

    # Lưu JSON
    with open(JSON_OUT, "w", encoding="utf-8") as jf:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "config": {
                "start_date":   START_DATE, "end_date": END_DATE,
                "outcome_days": OUTCOME_DAYS, "update_every": UPDATE_EVERY,
                "min_score":    MIN_SCORE_TO_SIGNAL, "universe": UNIVERSE,
            },
            "result":         res,
            "elapsed_sec":    elapsed,
        }, jf, indent=2, ensure_ascii=False, default=str)

    print(f"\n✓ Hoàn thành! Weights đã lưu vào: {JSON_DIR}/rlhf_weights.json")
    print(f"  → Chạy backtest_portfolio_2024.py để đo hiệu quả sau training")


if __name__ == "__main__":
    main()
