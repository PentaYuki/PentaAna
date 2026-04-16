from vnstock import Vnstock
import pandas as pd
import os
from datetime import datetime, timedelta

DATA_DIR    = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
RAW_PQ_DIR  = os.path.join(DATA_DIR, "raw", "parquet")
RAW_CSV_DIR = os.path.join(DATA_DIR, "raw", "csv")


PRICE_SOURCES = ["KBS", "VCI", "TCBS", "SSI"]

def get_stock_history(ticker: str, years: int = 5) -> pd.DataFrame:
    """
    Lấy dữ liệu lịch sử giá cổ phiếu.
    
    Args:
        ticker: Mã cổ phiếu (VD: 'VNM', 'HPG', 'VIC')
        years: Số năm lịch sử cần lấy
    
    Returns:
        DataFrame với cột: time, open, high, low, close, volume
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')

    last_err = None
    for source in PRICE_SOURCES:
        try:
            stock = Vnstock().stock(symbol=ticker, source=source)
            df = stock.quote.history(start=start_date, end=end_date, interval='1D')
            if df is None or df.empty:
                raise ValueError(f"Nguồn {source} trả về rỗng")

            df['ticker'] = ticker
            os.makedirs(RAW_PQ_DIR, exist_ok=True)
            os.makedirs(RAW_CSV_DIR, exist_ok=True)
            pq_path = f"{RAW_PQ_DIR}/{ticker}_history.parquet"
            csv_path = f"{RAW_CSV_DIR}/{ticker}_history.csv"
            df.to_parquet(pq_path, index=False, engine='pyarrow')
            df.to_csv(csv_path, index=False)
            print(f"✓ Đã lưu {len(df)} phiên giao dịch của {ticker} (source={source})")
            return df
        except Exception as e:
            last_err = e
            print(f"⚠ Nguồn {source} lỗi cho {ticker}: {e}")

    raise RuntimeError(f"Không tải được dữ liệu cho {ticker} từ mọi nguồn: {last_err}")


def get_market_overview() -> pd.DataFrame:
    """Lấy danh sách toàn bộ cổ phiếu trên HOSE và HNX."""
    stock = Vnstock().stock(symbol='VNM', source='VCI')
    listing = stock.listing.all_symbols(exchange='HOSE')
    listing.to_parquet(f"{RAW_PQ_DIR}/market_listing.parquet", index=False, engine='pyarrow')
    return listing


def get_vnindex_history(years: int = 5) -> pd.DataFrame:
    """Lấy dữ liệu VN-Index."""
    return get_stock_history('VNINDEX', years)


def batch_download(tickers: list, years: int = 5):
    """Download nhiều mã cùng lúc."""
    results = {}
    for ticker in tickers:
        try:
            df = get_stock_history(ticker, years)
            results[ticker] = df
        except Exception as e:
            print(f"✗ Lỗi {ticker}: {e}")
    return results


if __name__ == "__main__":
    # Danh sách blue-chip VN
    blue_chips = [
        'VNM', 'HPG', 'VIC', 'VHM', 'MSN',
        'GAS', 'VCB', 'BID', 'CTG', 'MBB',
        'TCB', 'ACB', 'FPT', 'MWG', 'PNJ'
    ]
    
    print("=== Bắt đầu tải dữ liệu chứng khoán Việt Nam ===")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RAW_PQ_DIR, exist_ok=True)
    
    # Tải VN-Index
    vnindex = get_vnindex_history(years=5)
    
    # Tải blue-chip
    data = batch_download(blue_chips, years=5)
    print(f"\\n✓ Hoàn thành: {len(data)}/{len(blue_chips)} mã")
