# PentaAna Pro - Market Intelligence Terminal 🚀

[Tải bản Tiếng Việt bên dưới](#tiếng-việt)

**PentaAna Pro** is a high-fidelity, data-driven "Market Intelligence Terminal" designed with a premium, cyber-cyan aesthetic. It integrates real-time and proxy market data to deliver strategic modeling and actionable insights natively powered by local LLMs (Llama-3-8B).

## 🌟 Key Features

1. **Dual Mode Interface**:
   - **SEARCH Mode**: Analyzes current or upcoming games/products in the market by pulling data across platforms.
   - **CONCEPT Mode (PRO Mode)**: Evaluates a new game concept or idea using a Red-Team analysis methodology to identify potential failure patterns and growth opportunities.

2. **Data Integration layer**:
   - **Steam / Mobile Proxies**: Estimates wishlists, daily velocity, and revenue potential.
   - **Social Intelligence**: Pulls real Reddit discussion data to calculate hype and community engagement. Integrates Twitch viewer proxies.

3. **Intelligent Pipelines**:
   - **TimesFM 2.5 Forecasting**: Simulates a 14-day engagement horizon and confidence band.
   - **Red-Team Logic / Calendar Correlation**: Aligns trend data to an industry calendar to filter signals from noise.
   - **Llama-3-8B Local Synthesis**: Runs locally over Ollama to provide a "3-Card Strategy" (Growth, Stable, Defense) focusing on strictly quantitative and actionable steps.

## 🛠 Tech Stack

### Frontend
- **Framework**: React / Vite
- **Styling**: Vanilla CSS (High-Performance Glassmorphism, Neon Cyan Accents)
- **Data Vis**: Recharts (Dynamic monotone area plotting)
- **Icons**: Lucide React

### Backend
- **Framework**: FastAPI (Python)
- **Intelligence**: Local LLM via Ollama (`llama3:8b`)
- **Data Scrapers**: Unauthenticated JSON APIs (e.g., Reddit Search API, Steam API)

## ⚡ Deployment & Usage

### 1. Backend Setup
1. Make sure Python 3.9+ is installed.
2. Install dependencies: `pip install fastapi uvicorn requests`
3. Make sure you have Ollama installed and have pulled the Llama 3 model: `ollama run llama3:8b`
4. Run the server:
   ```bash
   cd backend
   python app.py
   ```
   *The backend will be live on http://127.0.0.1:8000*

### 2. Frontend Setup
1. Ensure Node.js 18+ is installed.
2. Install dependencies: `npm install`
3. Run the Vite development server:
   ```bash
   npm run dev
   ```
   *The frontend will be live on http://127.0.0.1:5173*

---

<br>

<h1 id="tiếng-việt">PentaAna Pro - Market Intelligence Terminal 🚀</h1>

**PentaAna Pro** là hệ thống "Thiết bị đầu cuối phân tích thị trường" (Market Intelligence Terminal) với giao diện mang phong cách cyber-cyan chuyên nghiệp. Hệ thống được thiết kế để tích hợp dữ liệu thời gian thực, lập mô hình kịch bản chiến lược và cung cấp các quyết định thực thi với sức mạnh từ mô hình ngôn ngữ cục bộ Llama-3-8B.

## 🌟 Tính Năng Cốt Lõi

1. **Giao diện đa chế độ (Dual Mode)**:
   - **Chế độ TÌM KIẾM (SEARCH)**: Truy vấn dữ liệu thật của các trò chơi / sản phẩm hiện tại, dự báo độ nóng trên thị trường.
   - **Chế độ LÊN Ý TƯỞNG (CONCEPT / PRO)**: Thẩm định kịch bản và ý tưởng game mới dựa trên logic "Phản biện đỏ" (Red-Team), đưa ra Cảnh báo thất bại (Failure patterns). 

2. **Tích hợp lượng dữ liệu đa nguồn**:
   - **Steam & Mobile**: Ước tính wishlist, tốc độ tăng trưởng hàng ngày và biểu đồ giả lập ARPU cho mobile.
   - **Phân tích mạng xã hội**: Lấy dữ liệu thảo luận **thật** qua API của Reddit để phân tích các tương tác và sự nhắc tới (mentions). Tích hợp số liệu mô phỏng từ Twitch.

3. **Quy trình kết xuất thông minh (Pipeline)**:
   - **Dự báo TimesFM 2.5**: Phỏng diễn chu kỳ tương tác (engagement) 14 ngày tới.
   - **Red-Team Logic & Đối soát Lịch trình**: Đối chiếu thông số xu hướng với "Industry Calendar" để tạo kịch bản thực tế.
   - **Tổng hợp (Synthesis) qua Llama-3-8B**: Triển khai tại máy chạy trực tiếp (qua Ollama), lập "Chiến lược 3-Lá" (Đột phá - Ổn định - Phòng thủ).

## 🛠 Ngăn Xếp Công Nghệ (Tech Stack)

### Frontend
- **Framework**: React / Vite
- **Giao diện**: Vanilla CSS cao cấp (Thiết kế giả kính mờ Glassmorphism, hiệu ứng lưới Tech Grid)
- **Biểu đồ**: Recharts
- **Icon**: Lucide React

### Backend
- **Framework**: FastAPI (Python)
- **Mô hình Trí tuệ**: Ollama (`llama3:8b`)
- **Thu thập dữ liệu (Scraper)**: Kết nối JSON API bảo mật (vd: Reddit Search API, Steam API).

## ⚡ Hướng Dẫn Cài Đặt

### 1. Phía Backend
1. Yêu cầu Python 3.9+.
2. Cài các thư viện: `pip install fastapi uvicorn requests`
3. Chắc chắn hệ thống đã cài đặt Ollama: chạy lệnh `ollama run llama3:8b`
4. Chạy máy chủ:
   ```bash
   cd backend
   python app.py
   ```
   *Máy chủ hiển thị tại http://127.0.0.1:8000*

### 2. Phía Frontend
1. Yêu cầu Node.js 18+.
2. Cài đặt các gói phụ thuộc: `npm install`
3. Khởi chạy dự án:
   ```bash
   npm run dev
   ```
   *Giao diện người dùng sẽ vận hành tại http://127.0.0.1:5173*
