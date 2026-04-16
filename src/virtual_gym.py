"""
virtual_gym.py — Môi trường Gym V2 (Action Liên tục, Không rò rỉ chỉ báo)
Tối ưu hóa bởi chuyên gia tài chính để đảm bảo tính thực tế trong quản trị rủi ro.
"""

import os
import math
import random
import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    class gym: Env = object
    class spaces: Box = None; Discrete = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_data_for_gym_v2(ticker: str) -> pd.DataFrame:
    pq_path = os.path.join(DATA_DIR, "raw", "parquet", f"{ticker}_history.parquet")
    csv_path = os.path.join(DATA_DIR, "raw", "csv", f"{ticker}_history.csv")
    if os.path.exists(pq_path):
        df = pd.read_parquet(pq_path, engine="pyarrow")
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        dates = pd.date_range("2023-01-01", periods=250, freq="B")
        df = pd.DataFrame({"time": dates, "close": np.random.normal(100, 2, size=250).cumsum()})
    
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df

class VirtualStockEnv(gym.Env):
    """
    Gym V2: 
    - Action: Box(0, 1) -> Target Weight (Tỉ trọng cổ phiếu mục tiêu)
    - Observation: [P/EMA, MACD, RSI, Sentiment, Cash_Ratio, Share_Ratio, Profit, Price_Diff, Volatility]
    - Reward: Log Return based.
    - Chaos Engine: Recalculates indicators on mutated price.
    """
    
    def __init__(self, ticker="VNM", initial_capital=9_000_000, target_profit=6_000_000):
        super(VirtualStockEnv, self).__init__()
        
        self.ticker = ticker
        self.initial_capital = float(initial_capital)
        self.target_nav = initial_capital + target_profit
        
        self.df_base = load_data_for_gym_v2(ticker)
        self.max_steps = len(self.df_base) - 1
        
        # Action: Target Weight [0.0, 1.0]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Obs: 9 variables (Added Volatility)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        self.reset()

    def _recalc_indicators(self):
        """Tính lại toàn bộ chỉ báo trên cột 'price_sim' để tránh rò rỉ tương lai."""
        close = self.df["price_sim"]
        
        self.df["ema20"] = close.ewm(span=20, adjust=False).mean()
        self.df["macd"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        self.df["rsi"] = 100 - (100 / (1 + rs))
        
        # Volatility (Rolling Std of returns)
        returns = close.pct_change()
        self.df["volatility"] = returns.rolling(window=20).std()
        
        # Sentiment base
        if "base_sentiment" not in self.df.columns:
            self.df["base_sentiment"] = np.random.normal(0.1, 0.15, size=len(self.df)).clip(-1, 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares = 0
        
        # Clone data for simulation
        self.df = self.df_base.copy()
        self.df["price_sim"] = self.df["close"].copy()
        self._recalc_indicators() # Initial indicators
        self.df.fillna(method="bfill", inplace=True)
        
        self.chaos_active = False
        self.chaos_decay_remaining = 0
        self.sentiment_override = None
        
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        
        price = row["price_sim"]
        ema20 = row["ema20"]
        macd = row["macd"]
        rsi = row["rsi"]
        vol = row["volatility"] if not pd.isna(row["volatility"]) else 0.0
        
        sentiment = self.sentiment_override if self.sentiment_override is not None else row["base_sentiment"]
        
        portfolio_val = self.shares * price
        nav = self.cash + portfolio_val
        
        obs = np.array([
            price / ema20 if ema20 > 0 else 1.0,
            macd / price if price > 0 else 0.0,
            rsi / 100.0,
            sentiment,
            self.cash / nav,
            portfolio_val / nav,
            (nav - self.initial_capital) / self.initial_capital,
            row["price_sim"] / self.df.iloc[max(0, self.current_step-1)]["price_sim"] - 1,
            vol * 10.0 # Scaling vol for net
        ], dtype=np.float32)
        
        return obs

    def step(self, action):
        # Action is target weight [0, 1]
        target_weight = float(action[0])
        
        # ─── Chaos Engine ───
        if not self.chaos_active and 20 < self.current_step < self.max_steps - 20:
            if random.random() < 0.03: # 3% chance daily for Black Swan
                self.chaos_active = True
                self.sentiment_override = random.uniform(-0.95, -0.70)
                self.chaos_decay_remaining = random.randint(3, 7)
                decay_rate = random.uniform(-0.02, -0.06)
                
                # Bẻ gãy tương lai ngay lập tức để chỉ báo MACD/RSI phản ứng
                self.df.loc[self.current_step + 1:, "price_sim"] *= (1 + decay_rate)
                self._recalc_indicators()
        
        if self.chaos_active:
            if self.chaos_decay_remaining > 0:
                self.chaos_decay_remaining -= 1
            else:
                self.chaos_active = False
                self.sentiment_override = None

        row = self.df.iloc[self.current_step]
        price = float(row["price_sim"])
        prev_nav = self.cash + self.shares * price
        
        # ─── Execute Target Weight ───
        target_value = prev_nav * target_weight
        current_value = self.shares * price
        diff_value = target_value - current_value
        
        fee_rate = 0.002
        
        if diff_value > 0: # BUY
            # Cần mua thêm lượng VND = diff_value
            can_buy_vnd = min(self.cash, diff_value)
            shares_to_buy = int(can_buy_vnd / (price * (1 + fee_rate)) / 100) * 100
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                fee = cost * fee_rate
                self.cash -= (cost + fee)
                self.shares += shares_to_buy
        elif diff_value < 0: # SELL
            # Cần bán bớt lượng VND = abs(diff_value)
            shares_to_sell = int(abs(diff_value) / price / 100) * 100
            shares_to_sell = min(self.shares, shares_to_sell)
            if shares_to_sell > 0:
                proceeds = shares_to_sell * price
                fee = proceeds * fee_rate
                self.cash += (proceeds - fee)
                self.shares -= shares_to_sell

        self.current_step += 1
        
        # New State
        row_next = self.df.iloc[self.current_step]
        next_price = float(row_next["price_sim"])
        current_nav = self.cash + self.shares * next_price
        
        # ─── Reward Function ───
        # Log Return
        daily_ret = (current_nav / prev_nav)
        reward = math.log(max(1e-6, daily_ret)) * 20.0 # Multiplier to bring it to a visible scale
        
        # Heavy Penalty for ignoring bad news
        if self.chaos_active and target_weight > 0.5 and self.sentiment_override < -0.7:
            reward -= 5.0 # Phạt vì quá hung hãn khi tin cực xấu
            
        # Target Bonus
        if current_nav >= self.target_nav:
            reward += 10.0
            
        # Bankrupcy
        done = self.current_step >= self.max_steps
        if current_nav < self.initial_capital * 0.5:
            reward -= 20.0
            done = True
            
        return self._get_obs(), reward, done, False, {"nav": current_nav, "shares": self.shares, "chaos": self.chaos_active}

if __name__ == "__main__":
    env = VirtualStockEnv()
    obs, _ = env.reset()
    print(f"Initial Obs: {obs}")
    for _ in range(30):
        # Random weight action
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        if info["chaos"]:
            print(f"!!! CHAOS DETECTED - Sentiment: {info['sentiment']:.2f}")
        if done:
            break
    print(f"Final NAV: {info['nav']:,.0f} VND")
    print("Smoke test passed.")
