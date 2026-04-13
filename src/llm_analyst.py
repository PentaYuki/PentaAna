import requests as _requests
import numpy as np

# Dùng phi3:3.8b thay vì llama3:8b — giảm RAM 4.7GB → 2.6GB
# keep_alive=0 → Ollama giải phóng VRAM ngay sau khi trả lời
_OLLAMA_BASE = "http://localhost:11434"
_DEFAULT_MODEL = "phi3:3.8b"   # fallback: "llama3:8b" nếu phi3 chưa pull


def _ollama_generate(prompt: str, model: str = _DEFAULT_MODEL) -> str:
    """Gọi Ollama REST API với keep_alive=0 để giải phóng RAM ngay sau khi nhận kết quả."""
    try:
        resp = _requests.post(
            f"{_OLLAMA_BASE}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": 0,       # Giải phóng VRAM sau mỗi lần gọi
                "options": {"num_predict": 300},
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except _requests.exceptions.ConnectionError:
        return "[Ollama không chạy — khởi động bằng: ollama serve]"
    except Exception as e:
        return f"[LLM error: {e}]"


def analyze_forecast_with_llm(
    ticker: str,
    current_price: float,
    forecast_median: np.ndarray,
    rsi: float,
    macd: float,
    signal: str,
    model: str = _DEFAULT_MODEL,
) -> str:
    """
    Dùng phi3:3.8b (hoặc model nhẹ hơn) để viết phân tích từ dữ liệu kỹ thuật.
    Giải phóng VRAM ngay sau khi nhận kết quả (keep_alive=0).
    """
    change_pct = (forecast_median[-1] - current_price) / current_price * 100
    
    prompt = f"""Bạn là chuyên gia phân tích chứng khoán Việt Nam.
Dưới đây là dữ liệu kỹ thuật của cổ phiếu {ticker}:

- Giá hiện tại: {current_price:,.0f} VND
- Dự báo sau 30 phiên: {forecast_median[-1]:,.0f} VND ({change_pct:+.1f}%)
- RSI (14): {rsi:.1f} {'(Quá bán - RSI < 30)' if rsi < 30 else '(Quá mua - RSI > 70)' if rsi > 70 else '(Trung tính)'}
- MACD: {macd:.2f} {'(Tích cực)' if macd > 0 else '(Tiêu cực)'}
- Tín hiệu tổng hợp: {signal}

Hãy viết phân tích ngắn gọn (3-4 câu) bằng tiếng Việt, tập trung vào:
1. Xu hướng dự báo
2. Các điểm rủi ro và cơ hội
3. Khuyến nghị cho nhà đầu tư ngắn hạn

Lưu ý: Đây là hỗ trợ phân tích, không phải tư vấn đầu tư chính thức."""

    return _ollama_generate(prompt, model=model)


if __name__ == "__main__":
    import os
    import pandas as pd
    from kronos_test import load_and_prepare_data, run_kronos_forecast, generate_signal
    
    TICKER = "VNM"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tech_path = os.path.join(base_dir, "data", "analyzed", "with_indicators", f"{TICKER}_with_indicators.parquet")
    
    if not os.path.exists(tech_path):
        print(f"Lỗi: Chưa có file {tech_path}. Vui lòng chạy technical_indicators.py trước.")
        exit(1)
        
    df_ta = pd.read_parquet(tech_path, engine='pyarrow')
    latest = df_ta.iloc[-1]
    current_price = latest['close']
    rsi = latest['rsi']
    macd = latest['macd']
    
    prices, dates, df_hist = load_and_prepare_data(TICKER)
    forecast = run_kronos_forecast(prices, forecast_horizon=30)
    signal = generate_signal(current_price, forecast['median'])
    
    print("\\nBắt đầu gọi AI (llama3) phân tích dựa trên dữ liệu thực tế...")
    analysis = analyze_forecast_with_llm(
        ticker=TICKER,
        current_price=current_price,
        forecast_median=forecast['median'],
        rsi=rsi,
        macd=macd,
        signal=signal
    )
    print("\\n📝 PHÂN TÍCH AI:")
    print(analysis)
