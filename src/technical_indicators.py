import pandas as pd
import pandas_ta_classic as ta

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm các chỉ số kỹ thuật vào DataFrame giá.
    
    Chỉ số bao gồm:
    - Trend: SMA20, SMA50, EMA12, EMA26
    - Momentum: RSI, MACD, Stochastic
    - Volatility: Bollinger Bands, ATR
    - Volume: OBV, Volume SMA
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    # --- Trend ---
    df['sma_20']   = ta.sma(df['close'], length=20)
    df['sma_50']   = ta.sma(df['close'], length=50)
    df['ema_12']   = ta.ema(df['close'], length=12)
    df['ema_26']   = ta.ema(df['close'], length=26)
    
    # --- Momentum ---
    df['rsi']      = ta.rsi(df['close'], length=14)
    macd           = ta.macd(df['close'])
    df['macd']     = macd['MACD_12_26_9']
    df['macd_sig'] = macd['MACDs_12_26_9']
    df['macd_hist']= macd['MACDh_12_26_9']
    
    # --- Volatility ---
    bb             = ta.bbands(df['close'], length=20)
    df['bb_upper'] = bb['BBU_20_2.0']
    df['bb_lower'] = bb['BBL_20_2.0']
    df['bb_mid']   = bb['BBM_20_2.0']
    df['atr']      = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # --- Volume ---
    df['obv']      = ta.obv(df['close'], df['volume'])
    df['vol_sma']  = ta.sma(df['volume'], length=20)
    
    # --- Tín hiệu đơn giản ---
    df['golden_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought']= (df['rsi'] > 70).astype(int)
    
    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    import pandas as pd
    import os
    
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "parquet", "VNM_history.parquet"))
    if os.path.exists(data_path):
        df = pd.read_parquet(data_path, engine='pyarrow')
        df_with_ta = add_technical_indicators(df)
        print(df_with_ta[['close', 'sma_20', 'rsi', 'macd']].tail(10))
        out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "analyzed", "with_indicators", "VNM_with_indicators.parquet"))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_with_ta.to_parquet(out_path, index=False, engine='pyarrow')
    else:
        print(f"Không tìm thấy file: {data_path}")
