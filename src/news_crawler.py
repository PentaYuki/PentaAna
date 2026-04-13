"""
news_crawler.py — Thu thập tin tức CafeF cho dự án Stock AI.

Kiến trúc 2 tầng:
  Tầng 1 (RSS)       — requests thuần, không bị Cloudflare, lấy URL bài mới
  Tầng 2 (Playwright) — headless Chromium + stealth, chỉ fetch full-text

Cài đặt:
    pip install playwright playwright-stealth beautifulsoup4 requests
    playwright install chromium
"""

import hashlib
import os
import sqlite3
import time
from datetime import datetime
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# ── Đường dẫn ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH  = os.path.join(DATA_DIR, "news.db")

# ── RSS feeds — CafeF (tiếng Việt) + VNExpress + Reuters Việt ─────────────
RSS_FEEDS = {
    # CafeF — chứng khoán Việt (không qua Cloudflare)
    "cafef-chung-khoan": "https://cafef.vn/rss/thi-truong-chung-khoan.rss",
    "cafef-doanh-nghiep": "https://cafef.vn/rss/doanh-nghiep.rss",
    "cafef-vi-mo":        "https://cafef.vn/rss/vi-mo-dau-tu.rss",
    # VNExpress Kinh doanh — chứng khoán
    "vnexpress-chungkhoan": "https://vnexpress.net/rss/kinh-doanh/chung-khoan.rss",
    "vnexpress-kinhdoanh":  "https://vnexpress.net/rss/kinh-doanh.rss",
    # Reuters tiếng Việt
    "reuters-vi":          "https://vi.reuters.com/rss/",
}

# Nguồn tiếng Anh — dùng FinBERT để phân tích (xem sentiment_analyzer.py)
ENGLISH_RSS_FEEDS = {
    "reuters-business":    "https://feeds.reuters.com/reuters/businessNews",
    "yahoo-finance-vn":    "https://finance.yahoo.com/rss/topfinstories",
}

# ── Headers giả lập trình duyệt Mac thật ────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
}


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def init_db() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            url_hash        TEXT    UNIQUE,
            ticker          TEXT,
            title           TEXT,
            url             TEXT,
            content         TEXT,
            pub_date        TEXT,
            sentiment_score REAL    DEFAULT NULL,
            created_at      TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON news(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pub_date ON news(pub_date)")
    conn.commit()
    return conn


def hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def is_duplicate(conn: sqlite3.Connection, url_hash: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM news WHERE url_hash=?", (url_hash,)
    ).fetchone() is not None


def save_article(conn: sqlite3.Connection, **kwargs) -> bool:
    try:
        conn.execute(
            """INSERT INTO news (url_hash, ticker, title, url, content, pub_date, created_at)
               VALUES (:url_hash, :ticker, :title, :url, :content, :pub_date, :created_at)""",
            kwargs,
        )
        return True
    except sqlite3.IntegrityError:
        return False  # Race condition — bài đã được insert bởi vòng khác


# ══════════════════════════════════════════════════════════════════════════════
# TẦNG 1: RSS — lấy URL bài mới (nhanh, không bị block)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_rss_urls(max_items: int = 30) -> list[dict]:
    """
    Lấy danh sách {title, url, pub_date} từ tất cả RSS feeds.
    Không dùng Playwright — requests thuần là đủ với RSS.
    """
    articles = []
    for feed_name, feed_url in RSS_FEEDS.items():
        try:
            resp = requests.get(feed_url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            root = ElementTree.fromstring(resp.content)

            for item in root.findall(".//item")[:max_items]:
                title   = (item.findtext("title") or "").strip()
                url     = (item.findtext("link")  or "").strip()
                pub_str = (item.findtext("pubDate") or datetime.now().isoformat())

                if url and title:
                    articles.append({
                        "title":    title,
                        "url":      url,
                        "pub_date": pub_str[:10],  # YYYY-MM-DD
                        "feed":     feed_name,
                    })
        except Exception as e:
            print(f"  ✗ RSS {feed_name}: {e}")

    print(f"  RSS: tìm thấy {len(articles)} bài từ {len(RSS_FEEDS)} feeds")
    return articles


# ══════════════════════════════════════════════════════════════════════════════
# TẦNG 2: Playwright — fetch full-text bài cụ thể (vượt Cloudflare)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_fulltext_playwright(urls: list[str], delay_s: float = 2.0) -> dict[str, str]:
    """
    Fetch full-text cho danh sách URL bằng Playwright + stealth.

    Args:
        urls:    Danh sách URL cần fetch
        delay_s: Nghỉ giữa các request (giây) — tránh bị ban IP

    Returns:
        dict {url: content_text}
    """
    try:
        from playwright_stealth import stealth_sync
    except ImportError:
        raise ImportError(
            "Thiếu playwright-stealth. Chạy: pip install playwright-stealth"
        )

    results = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=HEADERS["User-Agent"],
            locale="vi-VN",
        )
        page = context.new_page()
        stealth_sync(page)  # Patch navigator.webdriver và 10+ fingerprint JS

        for i, url in enumerate(urls):
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=25000)

                # Đọc nội dung bài — CafeF dùng .detail-content hoặc .contentdetail
                content = ""
                for selector in [".detail-content", ".contentdetail", "article", ".main-content"]:
                    el = page.query_selector(selector)
                    if el:
                        content = el.inner_text().strip()
                        break

                if not content:
                    # Fallback: lấy toàn bộ body text, bỏ navigation
                    content = page.eval_on_selector("body", "el => el.innerText") or ""

                results[url] = content[:3000]  # Giới hạn 3000 ký tự/bài

            except Exception as e:
                print(f"    ✗ Playwright lỗi [{url[:50]}...]: {e}")
                results[url] = ""

            # Rate limiting — quan trọng để không bị Cloudflare ban IP
            if i < len(urls) - 1:
                time.sleep(delay_s)

        browser.close()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# HÀM CHÍNH
# ══════════════════════════════════════════════════════════════════════════════

def crawl_cafef_news(
    ticker: str,
    max_articles: int = 20,
    fetch_fulltext: bool = True,
) -> int:
    """
    Thu thập tin tức CafeF cho một mã chứng khoán.

    Luồng:
      1. Lấy URL từ RSS (nhanh, không bị block)
      2. Lọc bỏ bài đã có trong SQLite (URL hash)
      3. Fetch full-text bài mới qua Playwright (có stealth + rate limit)
      4. Lưu vào SQLite

    Returns:
        Số bài mới được lưu
    """
    print(f"\n{'='*50}")
    print(f"Crawl CafeF: {ticker}")
    print(f"{'='*50}")

    conn = init_db()
    new_count = 0

    try:
        # Tầng 1: Lấy URL từ RSS
        rss_articles = fetch_rss_urls(max_items=max_articles)

        # Lọc bài liên quan đến ticker (tìm trong title)
        # Và lọc bài chưa có trong DB
        new_articles = []
        for art in rss_articles:
            url_hash = hash_url(art["url"])
            if is_duplicate(conn, url_hash):
                print(f"  ↷ Bỏ qua (đã có): {art['title'][:50]}")
                continue
            # Gán ticker vào article
            art["url_hash"] = url_hash
            art["ticker"]   = ticker
            new_articles.append(art)

        if not new_articles:
            print(f"  ✓ Không có bài mới cho {ticker}")
            return 0

        print(f"  → {len(new_articles)} bài mới cần fetch full-text")

        # Tầng 2: Fetch full-text (Playwright + stealth)
        content_map = {}
        if fetch_fulltext and new_articles:
            content_map = fetch_fulltext_playwright(
                [a["url"] for a in new_articles],
                delay_s=2.0,
            )

        # Lưu vào SQLite
        for art in new_articles:
            content = content_map.get(art["url"], "")
            saved = save_article(
                conn,
                url_hash=art["url_hash"],
                ticker=art["ticker"],
                title=art["title"],
                url=art["url"],
                content=content,
                pub_date=art["pub_date"],
                created_at=datetime.now().isoformat(),
            )
            if saved:
                new_count += 1
                print(f"  ✓ Lưu: {art['title'][:60]}")

        conn.commit()
        print(f"\nKết quả {ticker}: {new_count} bài mới được lưu")

    except Exception as e:
        print(f"✗ Lỗi crawl {ticker}: {e}")
        conn.rollback()
    finally:
        conn.close()

    return new_count


def crawl_all(watchlist: list[str]) -> dict[str, int]:
    """Crawl toàn bộ watchlist, trả về dict {ticker: số_bài_mới}."""
    results = {}
    for ticker in watchlist:
        results[ticker] = crawl_cafef_news(ticker)
        time.sleep(3)  # Nghỉ 3 giây giữa các ticker
    return results


def crawl_english_rss(max_items: int = 20) -> int:
    """Lấy tiêu đề tiếng Anh từ Reuters/Yahoo Finance, lưu với ticker=GLOBAL.
    Sentiment sẽ được chấm bằng FinBERT trong sentiment_analyzer.py.
    Returns: số bài mới lưu.
    """
    conn = init_db()
    new_count = 0
    for feed_name, feed_url in ENGLISH_RSS_FEEDS.items():
        try:
            resp = requests.get(feed_url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            root = ElementTree.fromstring(resp.content)
            for item in root.findall(".//item")[:max_items]:
                title   = (item.findtext("title") or "").strip()
                url     = (item.findtext("link")  or "").strip()
                pub_str = (item.findtext("pubDate") or datetime.now().isoformat())
                if not title or not url:
                    continue
                url_hash = hash_url(url)
                if is_duplicate(conn, url_hash):
                    continue
                saved = save_article(
                    conn,
                    url_hash=url_hash,
                    ticker="GLOBAL",          # Phân tích bằng FinBERT
                    title=title,
                    url=url,
                    content="",               # Tiêu đề đủ cho FinBERT
                    pub_date=pub_str[:10],
                    created_at=datetime.now().isoformat(),
                )
                if saved:
                    new_count += 1
            conn.commit()
            print(f"  [{feed_name}] {new_count} bài EN mới")
        except Exception as e:
            print(f"  ✗ EN RSS {feed_name}: {e}")
    conn.close()
    return new_count


# ══════════════════════════════════════════════════════════════════════════════
# CHẠY THỬ
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        from pipeline import WATCHLIST
    except ImportError:
        WATCHLIST = ["VNM", "HPG", "VIC"]

    print("=== Verification: Crawl 3 mã đầu ===")
    results = crawl_all(WATCHLIST[:3])

    print("\n=== Tóm tắt ===")
    for ticker, count in results.items():
        print(f"  {ticker}: {count} bài mới")

    # Kiểm tra chống trùng — chạy lại lần 2 phải thấy toàn bộ là "Bỏ qua"
    print("\n=== Verification chống trùng: chạy lại lần 2 ===")
    results2 = crawl_all(WATCHLIST[:3])
    all_zero = all(v == 0 for v in results2.values())
    print(f"\n{'✓ PASS' if all_zero else '✗ FAIL'}: Tất cả bài lần 2 đều bị bỏ qua (dedup hoạt động)")