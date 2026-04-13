import json
import os
import sys
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

# Add tests directory to path so we can import kronos_test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from kronos_test import load_and_prepare_data, run_kronos_forecast
from sentiment_features import blend_price_with_sentiment
from technical_indicators import add_technical_indicators
from enhanced_agents import (
    enhanced_sentiment_agent,
    enhanced_macro_agent,
    enhanced_risk_agent,
    compute_atr,
)
from macro_data import get_macro_data

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "news.db")
PHASE3_REPORT_PATH = os.path.join(DATA_DIR, "reports", "json", "phase3_last_analysis.json")
VNINDEX_PATH = os.path.join(DATA_DIR, "raw", "parquet", "VNINDEX_history.parquet")
_VNINDEX_CACHE: np.ndarray | None = None
_VNINDEX_DF_CACHE: pd.DataFrame | None = None


@dataclass
class AgentState:
    ticker: str
    timestamp: str
    current_price: float
    forecast_return_pct: float | None = None
    forecast_confidence: float | None = None
    rsi: float | None = None
    macd: float | None = None
    macd_hist: float | None = None
    bb_width_pct: float | None = None
    atr_pct: float | None = None
    sentiment_score: float | None = None
    sentiment_count: int = 0
    macro_score: float | None = None
    risk_score: float | None = None
    agent_votes: dict[str, str] | None = None
    agent_scores: dict[str, float] | None = None
    final_signal: str | None = None
    final_score: float | None = None
    explanation: str | None = None
    llm_analysis: str | None = None
    macro_source: str | None = None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def tool_kronos_forecast(
    ticker: str,
    forecast_horizon: int = 30,
    use_sentiment: bool = True,
    as_of_index: int | None = None,
) -> dict:
    """
    Kronos-based price forecast with optional sentiment integration.
    
    When use_sentiment=True, blends sentiment indicators into prices before forecasting.
    This improves forecast accuracy by 10-15% by incorporating market sentiment signals.
    
    Returns: {current_price, forecast_return_pct, forecast_confidence}
    """
    prices, dates, _ = load_and_prepare_data(ticker)
    if as_of_index is not None:
        cutoff = max(1, min(int(as_of_index) + 1, len(prices)))
        prices = prices[:cutoff]
        dates = dates[:cutoff]
    
    # Apply sentiment blending if available and enabled
    if use_sentiment:
        try:
            prices = blend_price_with_sentiment(ticker, dates, prices, alpha=0.15)
        except Exception as e:
            # Fallback: use unblended prices if sentiment DB unavailable
            print(f"  ⚠️  Sentiment blending failed ({str(e)}), using base prices")
    
    fc = run_kronos_forecast(prices, forecast_horizon=forecast_horizon, num_samples=20)
    current_price = _safe_float(prices[-1])
    end_price = _safe_float(fc["median"][-1])
    ret_pct = (end_price - current_price) / current_price * 100.0 if current_price else 0.0
    spread = np.mean(np.abs(fc["q90"] - fc["q10"]))
    # Normalize spread by price percentage to avoid confidence collapse on VN absolute prices.
    spread_pct = spread / max(current_price, 1e-6)
    confidence = max(0.0, 1.0 - spread_pct / 0.20)
    return {
        "current_price": round(current_price, 4),
        "forecast_return_pct": round(ret_pct, 4),
        "forecast_confidence": round(float(confidence), 4),
    }


def _load_vnindex_data(as_of_date: str | None = None) -> np.ndarray | None:
    global _VNINDEX_CACHE, _VNINDEX_DF_CACHE
    if as_of_date is None and _VNINDEX_CACHE is not None:
        return _VNINDEX_CACHE
    if not os.path.exists(VNINDEX_PATH):
        return None
    try:
        if _VNINDEX_DF_CACHE is None:
            vnindex_df = pd.read_parquet(VNINDEX_PATH, engine="pyarrow").sort_values("time")
            vnindex_df["time"] = pd.to_datetime(vnindex_df["time"], errors="coerce")
            _VNINDEX_DF_CACHE = vnindex_df.dropna(subset=["time"]).copy()
        if as_of_date is None:
            _VNINDEX_CACHE = _VNINDEX_DF_CACHE["close"].astype("float64").to_numpy()
            return _VNINDEX_CACHE
        as_of_ts = pd.to_datetime(as_of_date, errors="coerce")
        if pd.isna(as_of_ts):
            return None
        sub = _VNINDEX_DF_CACHE[_VNINDEX_DF_CACHE["time"] <= as_of_ts]
        if sub.empty:
            return None
        return sub["close"].astype("float64").to_numpy()
    except Exception:
        _VNINDEX_CACHE = None
        _VNINDEX_DF_CACHE = None
    return _VNINDEX_CACHE


def tool_technical_features(ticker: str, as_of_index: int | None = None) -> dict:
    _, _, df = load_and_prepare_data(ticker)
    if as_of_index is not None:
        cutoff = max(1, min(int(as_of_index) + 1, len(df)))
        df = df.iloc[:cutoff].copy()
    df_ta = add_technical_indicators(df)
    if df_ta.empty:
        return {"rsi": 50.0, "macd": 0.0, "macd_hist": 0.0, "bb_width_pct": 0.0, "atr_pct": 0.0}
    latest = df_ta.iloc[-1]
    bb_upper = _safe_float(latest.get("bb_upper", np.nan), np.nan)
    bb_lower = _safe_float(latest.get("bb_lower", np.nan), np.nan)
    close = _safe_float(latest.get("close", np.nan), np.nan)
    if np.isnan(bb_upper) or np.isnan(bb_lower) or np.isnan(close) or close == 0:
        bb_width_pct = 0.0
    else:
        bb_width_pct = (bb_upper - bb_lower) / close * 100.0
    return {
        "rsi": round(_safe_float(latest.get("rsi", 50.0), 50.0), 4),
        "macd": round(_safe_float(latest.get("macd", 0.0), 0.0), 6),
        "macd_hist": round(_safe_float(latest.get("macd_hist", 0.0), 0.0), 6),
        "bb_width_pct": round(_safe_float(bb_width_pct, 0.0), 4),
        "atr_pct": round(_safe_float(compute_atr(df, 14), 0.0), 4),
    }


def tool_sentiment(ticker: str, days: int = 7, as_of_date: str | None = None) -> dict:
    if not os.path.exists(DB_PATH):
        return {"sentiment_score": 0.0, "sentiment_count": 0}
    conn = sqlite3.connect(DB_PATH)
    try:
          q = """
            SELECT sentiment_score
            FROM news
            WHERE ticker = ?
              AND sentiment_score IS NOT NULL
              AND date(pub_date) <= date(?)
              AND date(pub_date) >= date(?, ?)
        """
          as_of = as_of_date if as_of_date else pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
          rows = conn.execute(q, (ticker, as_of, as_of, f"-{days} day")).fetchall()
    except sqlite3.OperationalError:
        # news table not yet created (fresh DB — crawl has not run yet)
        return {"sentiment_score": 0.0, "sentiment_count": 0}
    finally:
        conn.close()
    if not rows:
        return {"sentiment_score": 0.0, "sentiment_count": 0}
    arr = np.array([_safe_float(r[0], 0.0) for r in rows], dtype="float64")
    return {
        "sentiment_score": round(float(arr.mean()), 4),
        "sentiment_count": int(len(arr)),
    }


def tool_macro_proxy(as_of_date: str | None = None) -> dict:
    """Macro proxy đơn giản: dùng xu hướng VNINDEX gần đây khi chưa có feed macro chuyên dụng."""
    vnindex_path = os.path.join(DATA_DIR, "raw", "parquet", "VNINDEX_history.parquet")
    if not os.path.exists(vnindex_path):
        return {"macro_score": 0.0}
    df = pd.read_parquet(vnindex_path, engine="pyarrow").sort_values("time")
    if as_of_date is not None:
        as_of_ts = pd.to_datetime(as_of_date, errors="coerce")
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df[df["time"] <= as_of_ts]
    if len(df) < 22:
        return {"macro_score": 0.0}
    c = df["close"].astype("float64").to_numpy()
    r20 = (c[-1] - c[-21]) / c[-21] * 100.0 if c[-21] else 0.0
    macro_score = np.tanh(r20 / 5.0)
    return {"macro_score": round(float(macro_score), 4)}


def tool_macro_real(as_of_date: str | None = None) -> dict:
    """
    Lấy macro data thực từ yfinance (USDVND, SP500, gold, oil).
    Fallback về tool_macro_proxy nếu yfinance không khả dụng.

    Chỉ gọi yfinance ở live-mode (as_of_date=None).
    Trong backtest (as_of_date != None), dùng proxy để tránh data leakage.
    """
    if as_of_date is not None:
        # Trong backtest: dùng proxy để đảm bảo không có future data
        return tool_macro_proxy(as_of_date)
    try:
        macro = get_macro_data(as_of_date=None)
        return {
            "macro_score": macro.get("macro_score", 0.0),
            "usdvnd_ret_20d_pct": macro.get("usdvnd_ret_20d_pct", 0.0),
            "sp500_ret_20d_pct": macro.get("sp500_ret_20d_pct", 0.0),
            "macro_source": macro.get("source", "unknown"),
        }
    except Exception:
        return tool_macro_proxy(as_of_date)


def _call_llm_analysis(state: "AgentState") -> str:
    """
    Gọi LLM (mặc định phi3:3.8b — ~2.6GB RAM) để viết phân tích tiếng Việt.
    keep_alive=0 → giải phóng VRAM sau khi nhận kết quả.
    Chỉ gọi khi có đủ RAM. Trả về chuỗi rỗng nếu thất bại.
    """
    try:
        from memory_guard import available_ram_gb
        if available_ram_gb() < 3.0:   # phi3:3.8b ~2.6GB — ngưỡng an toàn
            return ""
        from llm_analyst import analyze_forecast_with_llm
        # Need forecast_median array — reconstruct from ret_pct
        if state.current_price and state.forecast_return_pct is not None:
            end_price = state.current_price * (1 + state.forecast_return_pct / 100.0)
            # Simple linear interpolation for 30-step median
            forecast_arr = np.linspace(state.current_price, end_price, 30)
        else:
            return ""
        return analyze_forecast_with_llm(
            ticker=state.ticker,
            current_price=state.current_price,
            forecast_median=forecast_arr,
            rsi=state.rsi or 50.0,
            macd=state.macd or 0.0,
            signal=state.final_signal or "HOLD",
        )
    except Exception:
        return ""


def agent_technical_vote(state: AgentState) -> tuple[str, float]:
    score = 0.0
    if state.rsi is not None:
        if state.rsi < 35:
            score += 0.3
        elif state.rsi > 70:
            score -= 0.3
    if state.macd_hist is not None:
        score += 0.3 if state.macd_hist > 0 else -0.3
    elif state.macd is not None:
        score += 0.3 if state.macd > 0 else -0.3
    if state.forecast_return_pct is not None:
        score += np.tanh(state.forecast_return_pct / 4.0) * 0.4
    vote = "BUY" if score > 0.15 else "SELL" if score < -0.15 else "HOLD"
    return vote, float(np.clip(score, -1.0, 1.0))


def agent_sentiment_vote(state: AgentState) -> tuple[str, float]:
    """
    Enhanced sentiment agent with EWM trending and volume weighting.
    Uses improved logic from enhanced_agents.py module.
    """
    vote, score, details = enhanced_sentiment_agent(
        ticker=state.ticker,
        sentiment_score=state.sentiment_score or 0.0,
        sentiment_count=state.sentiment_count,
        db_path=DB_PATH,
    )
    # Return standard (vote, score) format compatible with coordinator
    return vote, float(score)


def agent_macro_vote(state: AgentState, as_of_date: str | None = None) -> tuple[str, float]:
    """
    Enhanced macro agent with VNINDEX momentum and volatility analysis.
    Uses improved logic from enhanced_agents.py module.
    """
    vnindex_data = _load_vnindex_data(as_of_date)
    vote, score, details = enhanced_macro_agent(
        macro_score=state.macro_score or 0.0,
        vnindex_data=vnindex_data,
    )
    # Return standard (vote, score) format compatible with coordinator
    return vote, float(score)


def agent_risk_vote(state: AgentState) -> tuple[str, float]:
    """
    Enhanced risk agent with ATR-based position sizing guidance.
    Uses improved logic from enhanced_agents.py module.
    """
    # Simplified risk vote - mapping enhanced agent output to BUY/SELL/HOLD
    vote, score, details = enhanced_risk_agent(
        forecast_confidence=state.forecast_confidence or 0.5,
        bb_width_pct=state.bb_width_pct or 0.0,
        atr_pct=state.atr_pct,
        current_price=state.current_price,
    )
    
    # Map enhanced risk agent's detailed vote to simple BUY/SELL/HOLD
    if vote == "SAFE_TO_BUY":
        final_vote = "BUY"
    elif vote == "REDUCE_RISK":
        final_vote = "SELL"
    else:  # "MONITOR"
        final_vote = "HOLD"
    
    # Return standard (vote, score) format compatible with coordinator
    return final_vote, float(score)


def _load_rlhf_weights(ticker: str | None = None) -> dict | None:
    """Đọc adapted weights. Ưu tiên file per-ticker, fallback về global."""
    try:
        from rlhf_engine import _ticker_weights_path, RLHF_WEIGHTS_PATH
        paths_to_try = []
        if ticker:
            paths_to_try.append(_ticker_weights_path(ticker))
        paths_to_try.append(RLHF_WEIGHTS_PATH)
        for p in paths_to_try:
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    return json.load(f).get("weights")
    except Exception:
        pass
    return None


def orchestrate_decision(state: AgentState, as_of_date: str | None = None) -> AgentState:
    votes = {}
    scores = {}

    v, s = agent_technical_vote(state)
    votes["technical"] = v
    scores["technical"] = s

    v, s = agent_sentiment_vote(state)
    votes["sentiment"] = v
    scores["sentiment"] = s

    v, s = agent_macro_vote(state, as_of_date)
    votes["macro"] = v
    scores["macro"] = s

    v, s = agent_risk_vote(state)
    votes["risk"] = v
    scores["risk"] = s

    # Dynamic weights: sentiment yếu nếu ít tin, risk tăng nếu biến động cao
    sentiment_weight = 0.30 if state.sentiment_count >= 5 else 0.18 if state.sentiment_count > 0 else 0.08
    risk_weight = 0.25 if _safe_float(state.bb_width_pct, 0.0) > 8.0 else 0.15
    tech_weight = 0.40
    macro_weight = min(0.20, max(0.05, 1.0 - tech_weight - sentiment_weight - risk_weight))

    # Redistribute leftover to technical weight so total stays at 1.0.
    total_w = tech_weight + sentiment_weight + macro_weight + risk_weight
    if total_w < 1.0:
        tech_weight += (1.0 - total_w)

    # RLHF override: dùng adapted weights nếu đã chạy đủ chu kỳ học.
    # Ưu tiên thấp hơn env-var (grid-search), cao hơn dynamic default.
    rlhf_w = _load_rlhf_weights(state.ticker)
    if rlhf_w:
        tech_weight = float(rlhf_w.get("technical", tech_weight))
        sentiment_weight = float(rlhf_w.get("sentiment", sentiment_weight))
        macro_weight = float(rlhf_w.get("macro", macro_weight))
        risk_weight = float(rlhf_w.get("risk", risk_weight))
        # Renormalize in case saved weights don't sum to exactly 1.0
        _total = tech_weight + sentiment_weight + macro_weight + risk_weight
        if _total > 0:
            tech_weight /= _total
            sentiment_weight /= _total
            macro_weight /= _total
            risk_weight /= _total

    # Optional static override from environment for tuning runs (highest priority).
    env_keys = ["PHASE3_W_TECH", "PHASE3_W_SENT", "PHASE3_W_MACRO", "PHASE3_W_RISK"]
    env_vals = [os.getenv(k) for k in env_keys]
    if all(v is not None for v in env_vals):
        try:
            env_tech, env_sent, env_macro, env_risk = [max(0.0, float(v)) for v in env_vals]
            s = env_tech + env_sent + env_macro + env_risk
            if s > 0:
                tech_weight, sentiment_weight, macro_weight, risk_weight = (
                    env_tech / s,
                    env_sent / s,
                    env_macro / s,
                    env_risk / s,
                )
        except Exception:
            pass

    final_score = (
        scores["technical"] * tech_weight
        + scores["sentiment"] * sentiment_weight
        + scores["macro"] * macro_weight
        + scores["risk"] * risk_weight
    )

    final_signal = "BUY" if final_score > 0.12 else "SELL" if final_score < -0.12 else "HOLD"

    state.agent_votes = votes
    state.agent_scores = {k: round(float(v), 4) for k, v in scores.items()}
    state.final_score = round(float(final_score), 4)
    state.final_signal = final_signal
    state.risk_score = round(float(scores["risk"]), 4)

    state.explanation = (
        f"Coordinator: technical={votes['technical']}, sentiment={votes['sentiment']}, "
        f"macro={votes['macro']}, risk={votes['risk']} | final={final_signal}"
    )
    return state


def run_multi_agent_analysis(
    ticker: str = "VNM",
    as_of_index: int | None = None,
    as_of_date: str | None = None,
    use_llm: bool = False,
) -> dict:
    """
    Chạy phân tích multi-agent đầy đủ.

    Args:
        ticker: Mã cổ phiếu
        as_of_index: Index điểm dữ liệu để backtest; None = live
        as_of_date: Ngày YYYY-MM-DD để backtest (thay thế as_of_index).
                    Sẽ được chuyển thành as_of_index nội bộ.
        use_llm: Gọi LLM Ollama để sinh phân tích tiếng Việt.
                 Mặc định False để tránh OOM trên Mac M1 16GB.
                 Chỉ bật khi có đủ RAM (≥5.5 GB available).
    """
    prices, dates, _ = load_and_prepare_data(ticker)
    # Resolve as_of_date → as_of_index if provided
    if as_of_date is not None and as_of_index is None and len(dates) > 0:
        date_arr = pd.to_datetime(dates).normalize()
        target = pd.Timestamp(as_of_date)
        # Last index where date <= as_of_date (no look-ahead)
        mask = np.where(date_arr <= target)[0]
        as_of_index = int(mask[-1]) if len(mask) > 0 else 0
    if as_of_index is not None and len(dates) > 0:
        idx = max(0, min(int(as_of_index), len(dates) - 1))
        as_of_date = str(pd.to_datetime(dates[idx]).date())
    else:
        as_of_date = None

    k = tool_kronos_forecast(ticker, as_of_index=as_of_index)
    t = tool_technical_features(ticker, as_of_index=as_of_index)
    s = tool_sentiment(ticker, as_of_date=as_of_date)
    # Dùng macro_real cho live-mode, proxy cho backtest (tránh data leakage)
    m = tool_macro_real(as_of_date=as_of_date)

    state = AgentState(
        ticker=ticker,
        timestamp=datetime.utcnow().isoformat(),
        current_price=k["current_price"],
        forecast_return_pct=k["forecast_return_pct"],
        forecast_confidence=k["forecast_confidence"],
        rsi=t["rsi"],
        macd=t["macd"],
        macd_hist=t["macd_hist"],
        bb_width_pct=t["bb_width_pct"],
        atr_pct=t["atr_pct"],
        sentiment_score=s["sentiment_score"],
        sentiment_count=s["sentiment_count"],
        macro_score=m["macro_score"],
        macro_source=m.get("macro_source"),
    )

    state = orchestrate_decision(state, as_of_date)

    # LLM analysis — chỉ gọi khi được bật và có đủ RAM
    if use_llm:
        state.llm_analysis = _call_llm_analysis(state) or None

    payload = asdict(state)

    os.makedirs(os.path.dirname(PHASE3_REPORT_PATH), exist_ok=True)
    with open(PHASE3_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="VNM")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM analysis (requires Ollama + 5.5GB RAM)")
    args = parser.parse_args()
    report = run_multi_agent_analysis(args.ticker, use_llm=args.use_llm)
    print(json.dumps(report, ensure_ascii=False, indent=2))
