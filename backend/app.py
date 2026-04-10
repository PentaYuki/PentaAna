from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import json
from services.analyzer import MarketAnalyzer

app = FastAPI()

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
base_dir = os.path.dirname(os.path.abspath(__file__))
calendar_path = os.path.join(base_dir, "data/calendar.json")
analyzer = MarketAnalyzer(calendar_path)

# LLM Configuration (Ollama)
OLLAMA_URL = "http://localhost:11434/api/chat"

class AnalysisRequest(BaseModel):
    topic: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze_market(req: AnalysisRequest):
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")
    
    # 0. Resolve Topic Info (Search step)
    info = analyzer.resolve_topic_info(topic)
    
    # 1. Gather Data (Live vs Search Proxy)
    if info["steam_id"]:
        steam_data = analyzer.get_live_steam_data(info["steam_id"])
    else:
        # Fallback simulation or web-search mock-up
        steam_data = {"daily_velocity": 0, "7d_growth": 0, "estimated_wishlists": 0, "current_ccu": 0}

    if info["mobile_id"]:
        mobile_data = analyzer.get_live_mobile_data(info["mobile_id"])
    else:
        mobile_data = {"current_rank": "N/A", "estimated_daily_installs": 0, "arpu_proxy": 0}
        
    reddit_data = analyzer.get_reddit_data(topic)
    twitch_data = analyzer.get_twitch_data(topic)
    
    # 2. Financial/Health Score
    score_metrics = {**steam_data, **mobile_data, "category": info["category"]}
    health_score = analyzer.calculate_health_score(score_metrics)
    
    # 2b. Strategic Scenarios (New)
    scenarios = analyzer.calculate_strategic_scenarios(score_metrics, reddit_data, twitch_data)
    
    # 3. Correlation Analysis
    correlations = analyzer.correlate_with_calendar([])
    
    # 4. LLM Synthesis (Ollama - Strategic Intelligence)
    prompt = f"""
    Hãy đóng vai trò là Cố vấn Chiến lược Cấp cao (Strategic Consultant). Phân tích trí tuệ thị trường cho: {topic}
    
    DỮ LIỆU ĐỊNH LƯỢNG:
    - Loại hình: {info['category']}
    - CCU/Lượt tải Hiện tại: {steam_data.get('current_ccu', 0)}
    - Ước tính Followers (Steam): {scenarios['followers_proxy']}
    - Điểm Hype (Social Media Pulse): {scenarios['hype_score']}/100
    - Trần Thị trường (Market Ceiling): {scenarios['market_ceiling']}
    - Chỉ số Sức khỏe: {health_score['total']}
    
    CẢNH BÁO THẤT BẠI (Failure Patterns):
    {json.dumps(scenarios['failure_warnings'], ensure_ascii=False)}
    
    NHIỆM VỤ:
    Cung cấp một "Bản đồ Giải pháp Tương lai" súc tích (mỗi kịch bản 1 câu hành động):
    1. KỊCH BẢN ĐỘT PHÁ (Growth): Làm sao để đạt tới Trần thị trường {scenarios['market_ceiling']}?
    2. KỊCH BẢN AN TOÀN (Stable): Phương án tối ưu dựa trên Sức khỏe {health_score['total']}.
    3. KỊCH BẢN PHÒNG THỦ (Defense): Tránh các "Failure Patterns" đã được cảnh báo.
    
    YÊU CẦU:
    - Trả lời bằng TIẾNG VIỆT, văn phong sắc bén, có tính thực thi cao.
    - Không lặp lại ý giữa các kịch bản. Đưa ra giải pháp thực tế.
    """
    
    llm_response = "Ollama hiện không khả dụng. Vui lòng kiểm tra lại dịch vụ."
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": "llama3:8b",
            "messages": [
                {"role": "system", "content": "Bạn là Cố vấn Chiến lược PentaAna. Bạn đưa ra giải pháp định lượng, không nói lý thuyết sáo rỗng."},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }, timeout=90)
        
        if resp.status_code == 200:
            llm_response = resp.json()['message']['content']
    except Exception as e:
        print(f"Ollama Error: {e}")

    return {
        "topic": topic,
        "category": info["category"],
        "steam": steam_data,
        "mobile": mobile_data,
        "reddit": reddit_data,
        "twitch": twitch_data,
        "health": health_score,
        "scenarios": scenarios,
        "correlations": correlations,
        "synthesis": llm_response
    }

@app.post("/analyze-concept")
async def analyze_concept(req: Request):
    data = await req.json()
    genre = data.get("genre", "Unknown")
    platform = data.get("platform", "PC")
    description = data.get("description", "")
    
    # Simulate data for a new concept based on genre averages
    # (In a real product, we would fetch similar genre performance)
    metrics = {
        "current_ccu": 0,
        "7d_growth": 15.0, # Assumed initial hype
        "estimated_wishlists": 120000,
        "estimated_daily_installs": 15000 if "Mobile" in platform else 0,
        "arpu_proxy": 1.6 if "Mobile" in platform else 0,
        "category": "PC Game" if "PC" in platform else "Mobile Game"
    }

    steam_data = {
        "daily_velocity": round(metrics["estimated_wishlists"] * 0.012),
        "7d_growth": metrics["7d_growth"],
        "estimated_wishlists": metrics["estimated_wishlists"],
        "current_ccu": metrics["current_ccu"]
    }
    mobile_data = {
        "current_rank": 60 if "Mobile" in platform else "N/A",
        "estimated_daily_installs": metrics["estimated_daily_installs"],
        "arpu_proxy": metrics["arpu_proxy"]
    }
    reddit_data = {
        "mentions": 0,
        "engagement": 0,
        "status": "CONCEPT MODE"
    }
    twitch_data = {
        "viewers": 0,
        "channels": 0,
        "status": "CONCEPT MODE"
    }
    
    health_score = analyzer.calculate_health_score(metrics)
    scenarios = analyzer.calculate_strategic_scenarios(metrics, reddit_data, twitch_data)
    
    prompt = f"""
    Đánh giá ý tưởng game mới: {genre} trên {platform}
    Mô tả: {description}
    
    DỮ LIỆU ĐỐI SOÁT (Benchmarks):
    - Trần thị trường dự kiến: {scenarios['market_ceiling']}
    - Những lỗi thường gặp ở thể loại này (Failure Patterns): {json.dumps(scenarios['failure_warnings'], ensure_ascii=False)}
    
    NHIỆM VỤ:
    Hãy đóng vai trò là 'Red Team' Analyst để phản biện ý tưởng này.
    1. CẢNH BÁO RỦI RO: Ý tưởng này dễ thất bại ở điểm nào (dựa trên Failure Patterns)?
    2. ĐIỂM SÁNG: Đâu là tiềm năng để đạt tới con số {scenarios['market_ceiling']}?
    3. HÀNH ĐỘNG TIẾP THEO: 1 giải pháp cụ thể để hiện thực hóa ý tưởng.
    
    YÊU CẦU: TIẾNG VIỆT, ngắn gọn, quyết đoán.
    """
    
    llm_response = "Sẵn sàng phân tích ý tưởng..."
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": "llama3:8b",
            "messages": [
                {"role": "system", "content": "Bạn là chuyên gia thẩm định dự án của PentaAna."},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }, timeout=90)
        
        if resp.status_code == 200:
            llm_response = resp.json()['message']['content']
    except Exception as e:
        print(f"Ollama Error: {e}")

    return {
        "topic": f"CONCEPT: {genre}",
        "category": metrics["category"],
        "steam": steam_data,
        "mobile": mobile_data,
        "reddit": reddit_data,
        "twitch": twitch_data,
        "health": health_score,
        "scenarios": scenarios,
        "synthesis": llm_response,
        "correlations": analyzer.correlate_with_calendar([])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
