# 📖 Hướng dẫn cách hoạt động của Market Intelligence Terminal

Chương trình này là một **Hệ thống Thu thập Thông tin Thị trường Real-time** được xây dựng hoàn toàn cục bộ, không phụ thuộc dịch vụ bên thứ 3.

---

## 🔧 Kiến trúc tổng thể

| Lớp | Công nghệ | Chức năng |
|-----|-----------|-----------|
| 🎨 Frontend | React + Recharts | Giao diện Terminal UI, Hiển thị đồ thị & dữ liệu |
| 🔌 Proxy Sourcing | SteamDB + Mobile Rank Algorithm | Thu thập dữ liệu thực tế về thị trường, tốc độ wishlist, lượt tải |
| 📊 Dự báo | Google TimesFM 2.5 | Mô hình dự báo xu hướng 14 ngày dựa trên dữ liệu lịch sử |
| 🧠 Tổng hợp | Llama-3-8B | Mô hình LLM chạy cục bộ để tổng hợp dữ liệu thành insight hữu ích |
| 📅 Tương quan | Industry Calendar | Phân tích sự kiện tương quan với xu hướng thị trường |

---

## ⚡ Luồng hoạt động chi tiết

### 1. **Giai đoạn Khởi tạo**
✅ Khi người dùng nhập chủ đề và nhấn `RUN ANALYSIS`:
- Reset toàn bộ trạng thái dữ liệu cũ
- Gọi API Backend chạy trên `http://localhost:8000/analyze`
- Bắt đầu 4 giai đoạn xử lý tuần tự

---

### 2. **Giai đoạn 1: SCRAPING (Thu thập dữ liệu)**
> ✅ **STEAMDB / MOBILE PROXY**
```
📌 Công việc thực hiện:
├─ Truy xuất tốc độ wishlist thực tế từ SteamDB
├─ Ước tính lượt tải hàng ngày bằng thuật toán Rank Proxy
├─ Quét 8 nguồn dữ liệu đồng thời:
│  • Reddit • X/Twitter • YouTube • HackerNews
│  • Polymarket • TikTok • Instagram • Bluesky
└─ Tính toán ARPU (Average Revenue Per User) ước tính
```

✅ Mục tiêu: Thu thập tất cả tín hiệu thô từ thị trường

---

### 3. **Giai đoạn 2: FORECASTING (Dự báo xu hướng)**
> ✅ **TIMESFM 2.5**
```
📌 Công việc thực hiện:
├─ Lấy 30 ngày dữ liệu lịch sử đã thu thập
├─ Chạy mô hình TimesFM của Google để dự báo 14 ngày tới
├─ Tính toán dải tin cậy (Confidence Band)
└─ Tạo dự báo xu hướng cảm xúc cộng đồng
```

✅ Mô hình này là mô hình chuyên dụng cho chuỗi thời gian, chính xác hơn rất nhiều so với các thuật toán thông thường.

---

### 4. **Giai đoạn 3: CORRELATION (Tương quan sự kiện)**
> ✅ **INDUSTRY CALENDAR**
```
📌 Công việc thực hiện:
├─ So sánh xu hướng hiện tại với lịch sử sự kiện ngành
├─ Tìm ra các sự kiện có tương quan cao với độ lệch hiện tại
└─ Gửi cảnh báo nếu phát hiện mẫu hình giống các sự kiện trong quá khứ
```

---

### 5. **Giai đoạn 4: SYNTHESIZING (Tổng hợp Insight)**
> ✅ **LLAMA-3-8B (chạy cục bộ)**
```
📌 Công việc thực hiện:
├─ Nhận tất cả dữ liệu thô từ 3 giai đoạn trước
├─ Dùng LLM chạy hoàn toàn cục bộ trên máy người dùng
├─ Tổng hợp thành các insight có thể hành động được
├─ Tính toán Health Score tổng thể (0-100 điểm)
└─ Phân loại trạng thái: SAFE / STABLE / DANGER
```

✅ Quan trọng: Tất cả xử lý AI diễn ra **hoàn toàn cục bộ**, không có dữ liệu nào được gửi ra ngoài máy người dùng.

---

## 🎯 Các thành phần giao diện

| Vùng giao diện | Chức năng |
|----------------|-----------|
| 📊 Bảng chỉ số trên cùng | Tổng hợp các chỉ số tổng quan: Tổng tín hiệu, Cảm xúc trung bình, Số nguồn hoạt động, Khoảng thời gian dự báo |
| 🟢 Bảng trái | Signal Intelligence: Dữ liệu chi tiết Steam, ước tính tải, tỷ lệ cược Polymarket |
| 🔵 Bảng giữa | Đồ thị xu hướng: 30 ngày lịch sử + 14 ngày dự báo + Cảm xúc cộng đồng |
| 🟡 Bảng phải | Insight tổng hợp từ Llama-3, Điểm sức khỏe thị trường, Cảnh báo tương quan |

---

## 🛡️ Đặc điểm kỹ thuật đặc biệt

✅ **Không có theo dõi**: Toàn bộ chương trình chạy cục bộ
✅ **Không cần API Key**: Không phụ thuộc bất kỳ dịch vụ bên ngoài nào
✅ **Real-time**: Dữ liệu được cập nhật liên tục
✅ **Offline capable**: Có thể chạy hoàn toàn không có internet sau khi đã cài đặt mô hình
✅ **Open source**: Toàn bộ mã nguồn được hiển thị và có thể kiểm tra

---

## 🚀 Khởi chạy chương trình

1.  Chạy Backend FastAPI: `cd backend && python app.py`
2.  Chạy Frontend Vite: `npm run dev`
3.  Truy cập trình duyệt: `http://localhost:5173`

---

### ⚠️ Lưu ý quan trọng:
> Chương trình này được thiết kế cho mục đích nghiên cứu thị trường. Không phải là lời khuyên đầu tư. Tất cả dữ liệu là ước tính và có thể sai lệch.