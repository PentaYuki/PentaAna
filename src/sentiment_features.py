import os
import sqlite3

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "news.db")


def load_daily_sentiment(db_path: str = DB_PATH) -> dict:
    if not os.path.exists(db_path):
        return {}
    conn = sqlite3.connect(db_path)
    try:
        query = """
            SELECT ticker, pub_date, AVG(sentiment_score) AS avg_score
            FROM news
            WHERE sentiment_score IS NOT NULL
              AND ticker IS NOT NULL
              AND pub_date IS NOT NULL
            GROUP BY ticker, pub_date
        """
        df = pd.read_sql_query(query, conn)
    except Exception:
        # news table not yet created (fresh DB with only rlhf_feedback)
        return {}
    finally:
        conn.close()

    if df.empty:
        return {}

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["pub_date"])

    out = {}
    for ticker, grp in df.groupby("ticker"):
        out[ticker] = dict(zip(grp["pub_date"], grp["avg_score"].astype(float)))
    return out


def build_sentiment_series(ticker: str, dates, sentiment_lookup: dict) -> np.ndarray:
    ticker = str(ticker).upper()
    per_day = sentiment_lookup.get(ticker, {})
    if not per_day:
        return np.zeros(len(dates), dtype="float32")

    idx = pd.to_datetime(dates, errors="coerce").strftime("%Y-%m-%d")
    raw = np.array([float(per_day.get(d, 0.0)) for d in idx], dtype="float32")
    smoothed = pd.Series(raw).ewm(span=5, adjust=False).mean().to_numpy(dtype="float32")
    return smoothed


def blend_price_with_sentiment(ticker: str, dates, prices: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    lookup = load_daily_sentiment(DB_PATH)
    sent = build_sentiment_series(ticker, dates, lookup)
    std = float(np.std(sent))
    if std < 1e-6:
        return prices.astype("float32")
    z = (sent - float(np.mean(sent))) / std
    z = np.clip(z, -3.0, 3.0)
    adjusted = prices.astype("float32") * (1.0 + alpha * z * 0.1)
    return adjusted.astype("float32")
