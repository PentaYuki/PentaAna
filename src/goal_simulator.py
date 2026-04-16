"""
goal_simulator.py — Goal-Oriented Strategy Simulation 

Fast-Mode backtesting engine that simulates buying/selling with realistic constraints:
- Lot sizing (multiples of 100 for HOSE)
- Transaction fees (0.2% per trade)
- Records the first date the target profit was reached, but continues until the end.
- Outputs equity curve and drawdown curve for UI rendering.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _load_price_df(ticker: str) -> pd.DataFrame:
    pq_path = os.path.join(DATA_DIR, "raw", "parquet", f"{ticker}_history.parquet")
    csv_path = os.path.join(DATA_DIR, "raw", "csv", f"{ticker}_history.csv")
    if os.path.exists(pq_path):
        df = pd.read_parquet(pq_path, engine="pyarrow")
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Missing data for {ticker}")
    
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-calculate basic indicators for Fast-Mode trend following policy."""
    if "close" not in df.columns:
        return df
    
    close = df["close"]
    
    # EMA 20 & 50
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    
    # RSI 14
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    
    return df


def generate_signal(row: pd.Series) -> str:
    """
    Fast-Mode policy rule:
    - BUY when MACD > MACD Signal AND Close > EMA 20
    - SELL when MACD < MACD Signal OR Close < EMA 50
    """
    try:
        macd, macd_sig = row["macd"], row["macd_signal"]
        close, ema20, ema50 = row["close"], row["ema_20"], row["ema_50"]
        rsi = row["rsi_14"]
        
        if pd.isna(macd) or pd.isna(macd_sig) or pd.isna(ema20):
            return "HOLD"
            
        if macd > macd_sig and close > ema20 and rsi < 70:
            return "BUY"
        elif macd < macd_sig or close < ema50:
            return "SELL"
    except Exception:
        pass
        
    return "HOLD"


def simulate_goal_oriented(
    ticker: str,
    initial_capital: float,
    target_profit: float,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    fee_pct: float = 0.2,
    lot_size: int = 100
) -> dict:
    """
    Mô phỏng chiến lược (Fast-Mode).
    - Tính đúng block tròn lô (mặc định 100 cổ/lô).
    - Phí giao dịch mặc định 0.2% mỗi vòng.
    - Không ngắt sớm, ghi nhận target_hit_date để phản ánh Drawdown đầy đủ phía sau.
    """
    df = _load_price_df(ticker)
    df = calculate_indicators(df)
    
    # Lọc range thời gian
    mask = (df["time"] >= start_date) & (df["time"] <= end_date)
    df = df[mask].reset_index(drop=True)
    
    if len(df) < 20:
        raise ValueError("Không đủ dữ liệu cho khoảng thời gian này.")
        
    cash = initial_capital
    shares = 0
    
    target_nav = initial_capital + target_profit
    target_hit_date = None
    
    equity_curve = []
    trade_logs = []
    
    entry_price = 0
    entry_date = None
    
    fee_rate = fee_pct / 100.0
    
    peak_nav = initial_capital
    max_drawdown = 0.0

    for i in range(len(df)):
        row = df.iloc[i]
        date_str = str(row["time"])[:10]
        price = float(row["close"])
        
        # Determine Signal
        signal = generate_signal(row)
        
        # Execute orders
        if signal == "BUY" and cash > 0 and shares == 0:
            # Calculate max possible shares considering fee
            cash_available = cash
            max_shares = int(cash_available / (price * (1 + fee_rate)))
            
            # Apply lot size (floor to nearest 100)
            actual_shares = (max_shares // lot_size) * lot_size
            
            if actual_shares > 0:
                cost = actual_shares * price
                fee = cost * fee_rate
                cash -= (cost + fee)
                shares = actual_shares
                entry_price = price
                entry_date = date_str
                
                trade_logs.append({
                    "action": "BUY",
                    "date": date_str,
                    "price": price,
                    "shares": actual_shares,
                    "fee": fee,
                    "cash_after": cash,
                    "nav": cash + shares * price
                })

        elif signal == "SELL" and shares > 0:
            proceeds = shares * price
            fee = proceeds * fee_rate
            cash += (proceeds - fee)
            
            profit = proceeds - fee - (shares * entry_price * (1 + fee_rate))
            profit_pct = (price - entry_price) / entry_price * 100
            
            trade_logs.append({
                "action": "SELL",
                "date": date_str,
                "price": price,
                "shares": shares,
                "fee": fee,
                "profit": profit,
                "profit_pct": profit_pct,
                "cash_after": cash,
                "nav": cash
            })
            
            shares = 0
            entry_price = 0
            entry_date = None

        # Tracking NAV & Drawdown
        current_nav = cash + shares * price
        
        if current_nav > peak_nav:
            peak_nav = current_nav
        
        dd = (peak_nav - current_nav) / peak_nav * 100.0
        if dd > max_drawdown:
            max_drawdown = dd
            
        equity_curve.append({
            "date": date_str,
            "nav": round(current_nav, 2),
            "drawdown": round(dd, 2)
        })
        
        # Check target goal
        if target_hit_date is None and current_nav >= target_nav:
            target_hit_date = date_str

    # End of period: Liquidate remaining position
    if shares > 0:
        last_price = float(df.iloc[-1]["close"])
        date_str = str(df.iloc[-1]["time"])[:10]
        
        proceeds = shares * last_price
        fee = proceeds * fee_rate
        cash += (proceeds - fee)
        
        profit = proceeds - fee - (shares * entry_price * (1 + fee_rate))
        profit_pct = (last_price - entry_price) / entry_price * 100
        
        trade_logs.append({
            "action": "SELL (LIQUIDATE)",
            "date": date_str,
            "price": last_price,
            "shares": shares,
            "fee": fee,
            "profit": profit,
            "profit_pct": profit_pct,
            "cash_after": cash,
            "nav": cash
        })
        
        equity_curve[-1]["nav"] = round(cash, 2)
    
    final_nav = equity_curve[-1]["nav"] if equity_curve else initial_capital
    total_return_pct = (final_nav - initial_capital) / initial_capital * 100

    return {
        "ticker": ticker,
        "initial_capital": initial_capital,
        "target_profit": target_profit,
        "target_nav": target_nav,
        "final_nav": round(final_nav, 2),
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "target_hit_date": target_hit_date,
        "equity_curve": equity_curve,
        "trade_logs": trade_logs,
        "total_trades": len([t for t in trade_logs if t["action"].startswith("SELL")])
    }

if __name__ == "__main__":
    # Smoke test
    res = simulate_goal_oriented("VNM", 6_000_000, 3_000_000, "2023-01-01", "2024-04-10")
    print(f"Final NAV: {res['final_nav']:,.0f}")
    print(f"Max DD: {res['max_drawdown_pct']}%")
    print(f"Target Hit: {res['target_hit_date']}")
    print(f"Trades: {res['total_trades']}")
