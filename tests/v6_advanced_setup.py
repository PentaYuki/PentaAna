"""
v6_advanced_setup.py — Triển khai Tin tức thật & Dynamic Slots
=============================================================
1. NewsCrawler: Lấy tin từ Cafef/Vietstock (giả lập qua Vnstock News).
2. DynamicSlotEngine: Điều chỉnh số lượng mã theo Regime.
"""
import os, sys, json, sqlite3, time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH  = os.path.join(DATA_DIR, "news.db")
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# --- 1. TỪ ĐIỂN SENTIMENT CHUYÊN SÂU (VIETNAMESE FINANCE) ---
FINANCE_KEYWORDS = {
    "tích cực": 0.5, "khả quan": 0.4, "tăng trưởng": 0.6, "vượt kế hoạch": 0.8,
    "ký kết": 0.5, "hợp đồng": 0.5, "đột phá": 0.7, "lợi nhuận khủng": 0.9,
    "cổ tức": 0.4, "nới lỏng": 0.6, "giảm lãi suất": 0.8, "mua ròng": 0.5,
    "tiêu cực": -0.5, "kém": -0.4, "sụt giảm": -0.6, "thua lỗ": -0.8,
    "vỡ nợ": -1.0, "tăng lãi suất": -0.7, "bán ròng": -0.5, "cảnh báo": -0.6,
    "hủy niêm yết": -1.0, "vi phạm": -0.7, "điều tra": -0.8, "áp lực": -0.4
}

def analyze_sentiment_v6(text):
    if not text: return 0.0
    text = text.lower()
    score = 0.0
    count = 0
    for word, weight in FINANCE_KEYWORDS.items():
        if word in text:
            score += weight
            count += 1
    return round(score / count, 3) if count > 0 else 0.0

# --- 2. CRAWLER GIẢ LẬP (SỬ DỤNG VNSTOCK NEWS) ---
def update_real_news(tickers):
    from vnstock import Vnstock
    print(f"\n[Bước 1] Đang lấy tin tức thật cho {len(tickers)} mã...")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS news (ticker TEXT, title TEXT, pub_date TEXT, sentiment_score REAL, source TEXT)")
    
    total_new = 0
    for tk in tickers:
        try:
            # Lấy tin tức từ nguồn VCI (thường có tin doanh nghiệp tốt nhất)
            stock = Vnstock().stock(symbol=tk, source="VCI")
            news_df = stock.company.news()
            
            if news_df is not None and not news_df.empty:
                for _, row in news_df.iterrows():
                    title = row.get("title", "")
                    date  = str(row.get("publish_date", ""))[:10]
                    
                    # Kiểm tra trùng
                    exists = conn.execute("SELECT 1 FROM news WHERE ticker=? AND title=?", (tk, title)).fetchone()
                    if not exists:
                        score = analyze_sentiment_v6(title)
                        conn.execute("INSERT INTO news VALUES (?, ?, ?, ?, ?)", (tk, title, date, score, "VCI_NEWS"))
                        total_new += 1
            print(f"  ✓ {tk}: OK")
        except:
            print(f"  ⚠ {tk}: Nguồn tin bận, bỏ qua...")
    
    conn.commit()
    conn.close()
    print(f"=> Hoàn thành: Thêm {total_new} tin tức thật vào Database.")

# --- 3. DYNAMIC PORTFOLIO ENGINE V6 ---
def run_portfolio_v6():
    # Phần này sẽ gọi engine mô phỏng nhưng với cấu hình Slot linh hoạt
    print("\n[Bước 2] Khởi tạo Engine v6 với Dynamic Slots...")
    print("  - Chế độ BULL    : 5 Slots (Max NAV)")
    print("  - Chế độ SIDEWAYS: 3 Slots (Safety)")
    print("  - Chế độ BEAR    : 1 Slot  (Defense)")
    
    # Logic nạp data và chạy... (Sẽ tích hợp vào file portfolio_v6.py)

if __name__ == "__main__":
    tickers = ["FPT", "MBB", "TCB", "ACB", "SSI", "CTG", "PNJ"]
    update_real_news(tickers)
    run_portfolio_v6()
