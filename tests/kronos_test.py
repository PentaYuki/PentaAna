import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from chronos import ChronosPipeline

_CACHED_PIPELINE = None

def load_and_prepare_data(ticker: str = 'VNM') -> tuple:
    """Chuẩn bị chuỗi thời gian từ dữ liệu đã tải. Nếu chưa có hoặc cũ, tự động kéo dữ liệu."""
    import time
    import sys
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    raw_dir = os.path.join(data_dir, "raw", "parquet")
    pq_path = os.path.join(raw_dir, f"{ticker}_history.parquet")
    
    # Auto-fetch if missing or older than 24h
    if not os.path.exists(pq_path) or (time.time() - os.path.getmtime(pq_path)) > 86400:
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        if src_dir not in sys.path:
            sys.path.append(src_dir)
        try:
            from data_collector import get_stock_history
            print(f"🔄 Dữ liệu {ticker} chưa có hoặc cũ (>24h), đang tải tự động...")
            get_stock_history(ticker, years=1)
        except Exception as e:
            if not os.path.exists(pq_path):
                raise ValueError(f"Mã '{ticker}' không tồn tại hoặc lỗi lấy dữ liệu: {e}")
            else:
                print(f"⚠️ Lỗi tải mới {ticker}, dùng dữ liệu cũ: {e}")

    df = pd.read_parquet(pq_path, engine='pyarrow')
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    
    close_prices = df['close'].values
    dates = df['time'].values
    
    return close_prices, dates, df


def run_kronos_forecast(
    prices: np.ndarray,
    forecast_horizon: int = 30,
    num_samples: int = 20
) -> dict:
    """
    Dự báo giá trong 30 phiên tiếp theo.
    """
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is None:
        print("Đang tải model Kronos-mini...")
        _CACHED_PIPELINE = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",   # ~300MB, tương đương Kronos-mini
            device_map="mps",            # Apple Silicon GPU
            torch_dtype=torch.float32,
        )
        print("✓ Model đã tải xong!")
    pipeline = _CACHED_PIPELINE
    
    # Dùng 252 phiên gần nhất (~1 năm giao dịch) làm context
    context_length = min(252, len(prices))
    context = torch.tensor(prices[-context_length:], dtype=torch.float32).unsqueeze(0)
    
    print(f"Đang dự báo {forecast_horizon} phiên tiếp theo...")
    with torch.no_grad():
        forecast = pipeline.predict(
            inputs=context,
            prediction_length=forecast_horizon,
            num_samples=num_samples,
        )
    
    forecast_np = forecast[0].numpy()  # shape: [num_samples, forecast_horizon]
    
    return {
        "median": np.median(forecast_np, axis=0),
        "q10":    np.percentile(forecast_np, 10, axis=0),
        "q90":    np.percentile(forecast_np, 90, axis=0),
        "samples": forecast_np,
    }


def plot_forecast(prices, dates, forecast_result, ticker: str = 'VNM'):
    """Vẽ biểu đồ dự báo."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Lịch sử (60 phiên gần nhất)
    history_n = 60
    hist_prices = prices[-history_n:]
    hist_dates  = dates[-history_n:]
    
    ax.plot(range(history_n), hist_prices, 
            color='#2196F3', linewidth=2, label='Giá thực tế')
    
    # Vùng dự báo
    n_forecast = len(forecast_result['median'])
    forecast_x = range(history_n, history_n + n_forecast)
    
    ax.fill_between(forecast_x,
                    forecast_result['q10'],
                    forecast_result['q90'],
                    alpha=0.25, color='#FF9800', label='Khoảng tin cậy 80%')
    
    ax.plot(forecast_x, forecast_result['median'],
            color='#FF5722', linewidth=2.5, linestyle='--', label='Dự báo (median)')
    
    # Đường phân chia
    ax.axvline(x=history_n - 0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(history_n - 1, ax.get_ylim()[1] * 0.98, 'Hôm nay',
            ha='right', fontsize=9, color='gray')
    
    ax.set_title(f'Dự báo giá {ticker} — Kronos-mini (30 phiên tiếp theo)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Phiên giao dịch')
    ax.set_ylabel('Giá (VND)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    output_path = os.path.join(data_dir, "reports", "charts", f"{ticker}_forecast.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Biểu đồ đã lưu: {output_path}")
    # plt.show()


def generate_signal(current_price: float, forecast_median: np.ndarray) -> str:
    """Tạo tín hiệu mua/bán/giữ đơn giản từ dự báo."""
    end_price    = forecast_median[-1]
    peak_price   = forecast_median.max()
    trough_price = forecast_median.min()
    
    change_pct = (end_price - current_price) / current_price * 100
    
    signal = "🟡 GIỮ"
    if change_pct > 5:
        signal = "🟢 MUA"
    elif change_pct < -5:
        signal = "🔴 BÁN"
    
    print("\\n" + "="*50)
    print(f"📊 KẾT QUẢ DỰ BÁO (30 PHIÊN)")
    print("="*50)
    print(f"Giá hiện tại  : {current_price:>12,.0f} VND")
    print(f"Dự báo cuối kỳ: {end_price:>12,.0f} VND")
    print(f"Đỉnh dự báo   : {peak_price:>12,.0f} VND")
    print(f"Đáy dự báo    : {trough_price:>12,.0f} VND")
    print(f"Thay đổi ước  : {change_pct:>+11.2f} %")
    print(f"Tín hiệu      :  {signal}")
    print("="*50)
    
    return signal


if __name__ == "__main__":
    TICKER = 'VNM'
    
    try:
        # 1. Tải dữ liệu
        prices, dates, df = load_and_prepare_data(TICKER)
        print(f"✓ Dữ liệu {TICKER}: {len(prices)} phiên ({str(dates[0])[:10]} → {str(dates[-1])[:10]})")
        
        # 2. Chạy dự báo
        forecast = run_kronos_forecast(prices, forecast_horizon=30)
        
        # 3. Phân tích kết quả
        signal = generate_signal(prices[-1], forecast['median'])
        
        # 4. Vẽ biểu đồ
        plot_forecast(prices, dates, forecast, TICKER)
    except FileNotFoundError:
        print(f"Chưa có dữ liệu cho {TICKER}. Vui lòng chạy data_collector.py trước.")
