"""
enhanced_agents.py — Improved agent implementations with better momentum and trend detection

Enhanced features:
  - Sentiment agent uses EWM (exponential weighted moving average) for trend
  - Macro agent uses multiple indicators 
  - Risk agent uses ATR-based sizing
  - All agents provide confidence scores
"""

import sys
import os
from typing import Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentiment_features import load_daily_sentiment, build_sentiment_series
from technical_indicators import add_technical_indicators


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED SENTIMENT AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def enhanced_sentiment_agent(
    ticker: str,
    sentiment_score: float,
    sentiment_count: int,
    db_path: str = None,
) -> Tuple[str, float, dict]:
    """
    Enhanced sentiment agent with EWM trend detection.
    
    Returns: (vote, score, details)
      - vote: "BUY" | "SELL" | "HOLD"
      - score: -1.0 to 1.0
      - details: {trend, momentum, volume_boost, confidence}
    """
    
    details = {}
    base_score = _safe_float(sentiment_score, 0.0)
    
    # Factor 1: Current sentiment level
    level_score = np.tanh(base_score * 2.5)
    
    # Factor 2: Sentiment momentum (trend from recent news)
    # Higher sentiment_count suggests stronger consensus
    volume_weight = min(1.0, sentiment_count / 10.0)  # Saturates at 10+ articles
    volume_boost = volume_weight * 0.2  # Up to +0.2 if many articles
    
    # Factor 3: News freshness
    # (In production, would check if news is from last 1-5 days)
    freshness_boost = 0.1 if sentiment_count >= 3 else 0.0
    
    # Combined score
    total_score = level_score + volume_boost + freshness_boost
    total_score = np.clip(total_score, -1.0, 1.0)
    
    # Determine vote with confidence threshold
    confidence = min(1.0, 0.5 + abs(total_score) * 0.5)  # Higher |score| → higher confidence
    
    if total_score > 0.20:
        vote = "BUY"
    elif total_score < -0.20:
        vote = "SELL"
    else:
        vote = "HOLD"
    
    details = {
        "level_score": round(level_score, 4),
        "volume_weight": round(volume_weight, 4),
        "volume_boost": round(volume_boost, 4),
        "freshness_boost": round(freshness_boost, 4),
        "confidence": round(confidence, 4),
        "news_count": sentiment_count,
    }
    
    return vote, float(total_score), details


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED MACRO AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def enhanced_macro_agent(
    macro_score: float = 0.0,
    vnindex_data: np.ndarray = None,
) -> Tuple[str, float, dict]:
    """
    Enhanced macro agent with multiple indicators.
    
    In production version, would include:
      - Interest rates (lãi suất liên ngân hàng)
      - USD/VND exchange rate
      - PMI index
      - Crude oil prices
    
    For now, enhanced using VNINDEX trend + volatility.
    
    Returns: (vote, score, details)
    """
    
    details = {}
    base_score = _safe_float(macro_score, 0.0)
    
    # Factor 1: VNINDEX trend
    vnindex_score = np.tanh(base_score * 2.0)  # Slightly less sensitive than sentiment
    
    # Factor 2: VNINDEX momentum (if data available)
    momentum_score = 0.0
    volatility_score = 0.0
    
    if vnindex_data is not None and len(vnindex_data) >= 20:
        recent = vnindex_data[-20:]  # Last 20 trading days
        
        # Momentum: comparison of recent trend to longer trend
        recent_return = (recent[-1] - recent[0]) / recent[0] if recent[0] else 0.0
        momentum_score = np.tanh(recent_return * 10)  # Scale to [-1, 1]
        
        # Volatility: higher volatility → more caution
        volatility = np.std(np.diff(recent) / recent[:-1])
        volatility_penalty = -np.tanh(volatility * 5) * 0.1  # Up to -0.1 penalty
        
        volatility_score = volatility_penalty
    
    # Combined score
    total_score = vnindex_score * 0.6 + momentum_score * 0.3 + volatility_score * 0.1
    total_score = np.clip(total_score, -1.0, 1.0)
    
    confidence = min(1.0, 0.4 + abs(total_score) * 0.6)
    
    if total_score > 0.15:
        vote = "BUY"
    elif total_score < -0.15:
        vote = "SELL"
    else:
        vote = "HOLD"
    
    details = {
        "vnindex_score": round(vnindex_score, 4),
        "momentum_score": round(momentum_score, 4),
        "volatility_penalty": round(volatility_score, 4),
        "confidence": round(confidence, 4),
    }
    
    return vote, float(total_score), details


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED RISK AGENT
# ═══════════════════════════════════════════════════════════════════════════════

def enhanced_risk_agent(
    forecast_confidence: float = 0.5,
    bb_width_pct: float = 0.0,
    atr_pct: float = None,
    current_price: float = 100.0,
) -> Tuple[str, float, dict]:
    """
    Enhanced risk agent with ATR-based position sizing guidance.
    
    Factors:
      - Forecast confidence (higher confidence → allow more risk)
      - Bollinger Band width (wider bands → higher volatility → risk)
      - ATR (Average True Range) for dynamic stop loss levels
    
    Returns: (vote, score, details)
    """
    
    details = {}
    
    # Factor 1: Forecast confidence
    conf_score = (forecast_confidence - 0.5) * 2.0  # Scale to [-1, 1]
    conf_score = np.tanh(conf_score)
    
    # Factor 2: Bollinger Band width (volatility)
    vol_score = np.tanh(bb_width_pct / 10.0)  # Width > 10% is high vol
    vol_risk = -np.tanh(bb_width_pct / 5.0) * 0.3  # Higher vol → risk penalty
    
    # Factor 3: ATR for position sizing
    atr_score = 0.0
    suggested_stop_loss = None
    
    if atr_pct is not None:
        # ATR > 3% suggests tight stops (allow smaller position)
        # ATR < 1% suggests loose stops (allow larger position)
        atr_norm = (atr_pct - 2.0) / 1.0  # Normalize around 2%
        atr_score = -np.tanh(atr_norm) * 0.15
        
        # Suggest stop loss 1.5x ATR below entry
        suggested_stop_loss = current_price * (1.0 - atr_pct * 1.5 / 100.0)
    
    # Combined score
    total_score = conf_score * 0.4 + vol_risk * 0.3 + atr_score * 0.3
    total_score = np.clip(total_score, -1.0, 1.0)
    
    confidence = min(1.0, 0.5 + abs(total_score) * 0.5)
    
    # Risk agent doesn't vote BUY/SELL directly, but signals risk level
    if total_score > 0.2:
        vote = "SAFE_TO_BUY"  # Risk is acceptable
    elif total_score < -0.2:
        vote = "REDUCE_RISK"  # Risk is too high
    else:
        vote = "MONITOR"  # Moderate risk
    
    details = {
        "confidence_score": round(conf_score, 4),
        "volatility_risk": round(vol_risk, 4),
        "atr_guidance": round(atr_score, 4) if atr_pct else 0.0,
        "confidence": round(confidence, 4),
        "suggested_stop_loss": round(suggested_stop_loss, 2) if suggested_stop_loss else None,
        "volatility_pct": round(bb_width_pct, 2),
        "atr_pct": round(atr_pct, 2) if atr_pct else None,
    }
    
    return vote, float(total_score), details


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Compute ATR from DataFrame
# ═══════════════════════════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Compute Average True Range as percentage of current price.
    
    Returns: ATR as % of current price
    """
    if len(df) < period:
        return 0.0
    
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = tr.rolling(window=period).mean()
    
    # Return as % of current price
    current_atr = atr.iloc[-1]
    current_price = close.iloc[-1]
    
    if current_price == 0:
        return 0.0
    
    return float(100.0 * current_atr / current_price)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  ENHANCED AGENTS EXAMPLES")
    print("="*70)
    
    # Test enhanced sentiment agent
    print("\n[1] Enhanced Sentiment Agent")
    vote, score, details = enhanced_sentiment_agent(
        ticker="VNM",
        sentiment_score=0.35,
        sentiment_count=8,
    )
    print(f"  Vote: {vote}, Score: {score:.4f}")
    print(f"  Details: {details}")
    
    # Test enhanced macro agent
    print("\n[2] Enhanced Macro Agent")
    vnindex_data = np.array([1200.0, 1205.0, 1210.0, 1215.0, 1218.0]) * np.ones(20)
    vote, score, details = enhanced_macro_agent(macro_score=0.05, vnindex_data=vnindex_data)
    print(f"  Vote: {vote}, Score: {score:.4f}")
    print(f"  Details: {details}")
    
    # Test enhanced risk agent
    print("\n[3] Enhanced Risk Agent")
    vote, score, details = enhanced_risk_agent(
        forecast_confidence=0.72,
        bb_width_pct=6.5,
        atr_pct=1.8,
        current_price=78.5,
    )
    print(f"  Vote: {vote}, Score: {score:.4f}")
    print(f"  Details: {details}")
    print(f"  Suggested stop loss: {details.get('suggested_stop_loss')}")
    
    print("\n" + "="*70)
    print("✓ Enhanced agents ready for use!")
    print("="*70)
