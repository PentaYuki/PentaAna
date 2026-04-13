import schedule
import time
import logging
import os
import sys

# Thêm thư mục hiện tại vào sys.path để import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collector import batch_download, get_vnindex_history
from technical_indicators import add_technical_indicators

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
log_file = os.path.join(base_dir, "logs", "pipeline.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

WATCHLIST = ['VNM', 'HPG', 'VIC', 'FPT', 'MWG', 'VCB', 'TCB', 'ACB']

def run_daily_pipeline():
    logging.info("=== Bắt đầu pipeline hàng ngày ===")
    
    # 1. Cập nhật dữ liệu
    data = batch_download(WATCHLIST, years=1)
    get_vnindex_history(years=1)
    
    # 2. Tính toán chỉ số kỹ thuật
    for ticker, df in data.items():
        df_ta = add_technical_indicators(df)
        out_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            "data",
            "analyzed",
            "indicators",
            f"{ticker}_indicators.parquet",
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_ta.to_parquet(out_path, index=False, engine='pyarrow')
    
    logging.info(f"✓ Hoàn thành pipeline: {len(data)} mã")

# Chạy lúc 15:30 mỗi ngày (sau giờ đóng cửa HoSE 15:00)
schedule.every().day.at("15:30").do(run_daily_pipeline)

if __name__ == "__main__":
    print("Pipeline đang chạy. Nhấn Ctrl+C để dừng.")
    try:
        run_daily_pipeline()  # Chạy ngay lập tức lần đầu
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\\nĐã dừng pipeline.")
