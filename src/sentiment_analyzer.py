import os
import sqlite3
import torch
from transformers import pipeline

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "data", "news.db")

# FinBERT model cho tiếng Anh (Reuters/Yahoo Finance)
FINBERT_MODEL = "ProsusAI/finbert"

# PhoBERT fine-tuned cho Vietnamese sentiment (vinai/phobert-base backbone)
PHOBERT_MODEL = "wonrax/phobert-base-vietnamese-sentiment"

def load_phobert_model():
    print(f"Đang nạp mô hình {PHOBERT_MODEL} vào RAM...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = pipeline(
        "text-classification",
        model=PHOBERT_MODEL,
        device=device,
        top_k=1,
    )
    return pipe


def load_finbert_model():
    """FinBERT — phân tích cảm xúc tài chính tiếng Anh (Reuters, Yahoo Finance)."""
    print("Đang nạp mô hình FinBERT (ProsusAI/finbert) vào RAM...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = pipeline(
        "text-classification",
        model=FINBERT_MODEL,
        device=device,
        top_k=1,
    )
    return pipe


def map_score(label: str, score: float) -> float:
    """Ánh xạ nhãn → điểm Float (-1.0 đến 1.0).
    wonrax/phobert-base-vietnamese-sentiment: NEG=0, NEU=1, POS=2
    ProsusAI/finbert: negative / neutral / positive
    """
    label = label.lower()
    if label in ('pos', 'positive', '2'):
        return round(score, 2)
    elif label in ('neg', 'negative', '0'):
        return round(-score, 2)
    # neutral / '1'
    return 0.0


def _score_english_articles(conn: sqlite3.Connection, limit: int = 100):
    """Dùng FinBERT chấm điểm bài tiếng Anh (ticker=GLOBAL)."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title FROM news WHERE sentiment_score IS NULL AND ticker = 'GLOBAL' LIMIT ?",
        (limit,)
    )
    articles = cursor.fetchall()
    if not articles:
        return 0

    model = load_finbert_model()
    scored = 0
    for art_id, title in articles:
        if not title or len(title.strip()) < 5:
            cursor.execute("UPDATE news SET sentiment_score = 0.0 WHERE id = ?", (art_id,))
            continue
        try:
            res = model(title[:512], truncation=True)[0]
            if isinstance(res, list):
                res = res[0]
            score_val = map_score(res["label"], res["score"])
            cursor.execute("UPDATE news SET sentiment_score = ? WHERE id = ?", (score_val, art_id))
            scored += 1
        except Exception as e:
            print(f"✗ FinBERT lỗi bài {art_id}: {e}")
            cursor.execute("UPDATE news SET sentiment_score = 0.0 WHERE id = ?", (art_id,))

    conn.commit()
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print(f"✓ FinBERT: {scored} bài EN được chấm điểm")
    return scored

def batch_score_news():
    print("Bắt đầu chấm điểm Sentiment hàng loạt (Batch Scoring)...")
    if not os.path.exists(DB_PATH):
        print("Lỗi: Không tìm thấy Cơ sở dữ liệu SQLite news.db. Hãy chạy news_crawler.py trước.")
        return
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Chấm điểm bài tiếng Anh (FinBERT) trước — nhẹ hơn
    en_scored = _score_english_articles(conn)

    # 2. Chấm tiếng Việt (PhoBERT) — lấy tối đa 200 bài chưa có điểm, bỏ GLOBAL
    cursor.execute(
        "SELECT id, content FROM news WHERE sentiment_score IS NULL AND ticker != 'GLOBAL' LIMIT 200"
    )
    articles = cursor.fetchall()
    
    if not articles:
        print(f"✓ Không có bài VI mới cần chấm điểm. EN: {en_scored} bài đã xử lý.")
        conn.close()
        return

    model = load_phobert_model()
    print(f"Tiến hành tính điểm {len(articles)} bài báo tiếng Việt...")
    
    for art_id, content in articles:
        if not content or len(content.strip()) < 10:
            cursor.execute("UPDATE news SET sentiment_score = ? WHERE id = ?", (0.0, art_id))
            continue
            
        text_chunk = content[:1000] 
        try:
            res = model(text_chunk, truncation=True, max_length=256)[0]
            score_val = map_score(res['label'], res['score'])
            cursor.execute("UPDATE news SET sentiment_score = ? WHERE id = ?", (score_val, art_id))
        except Exception as e:
            print(f"✗ Lỗi chấm điểm bài {art_id}: {e}")
            cursor.execute("UPDATE news SET sentiment_score = ? WHERE id = ?", (0.0, art_id))
            
    conn.commit()
    conn.close()
    
    # Dọn dẹp RAM khẩn cấp cho MAC M1
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print(f"✓ Hoàn tất chấm điểm {len(articles)} bài. Đã giải phóng RAM!")

if __name__ == "__main__":
    batch_score_news()
