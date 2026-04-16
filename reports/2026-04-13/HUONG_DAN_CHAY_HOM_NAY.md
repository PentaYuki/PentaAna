source .venv/bin/activate# 🚀 HƯỚNG DẪN CHẠY HỆ THỐNG NGÀY 13/04/2026

> ✅ Hệ thống PHASE 3 đã hoàn thiện, sẵn sàng chạy thử nghiệm

---

## 📋 BƯỚC ĐẦU TIÊN (MỌI LẦN MỞ LẠI TERMINAL)

```bash
# 1. Vào đúng thư mục project
cd /Users/gooleseswsq1gmail.com/Documents/stock-ai

# 2. Kích hoạt môi trường ảo
source .venv/bin/activate
```

---

## ✅ KIỂM TRA HỆ THỐNG HOẠT ĐỘNG

```bash
# Chạy kiểm tra tất cả component
python tests/test_phase3_components.py
```

✅ Nếu thấy dòng `✅ All imports OK` nghĩa là mọi thứ đã sẵn sàng

---

## 🚀 CHẠY CHỨC NĂNG CHÍNH HÔM NAY

### 1️⃣ Test Forecast có tích hợp Sentiment
```bash
python3 -c "
from src.phase3_multi_agent import tool_kronos_forecast
ket_qua = tool_kronos_forecast('VNM', use_sentiment=True)
print(f'✅ Forecast VNM hoàn tất: {ket_qua}')
"
```

### 2️⃣ Chạy so sánh Backtest 2 chiến lược
```bash
# Đây là lệnh chính hôm nay bạn cần chạy
python tests/backtest_comparison.py
```

🔹 Chạy khoảng 5-10 phút
🔹 Sẽ tự so sánh `Kronos gốc` vs `Multi-Agent mới`
🔹 Tạo báo cáo chi tiết tự động

---

## 📊 XEM KẾT QUẢ SAU KHI CHẠY XONG

```bash
# Xem kết luận cuối cùng
cat data/reports/json/backtest_comparison.json | jq .recommendation

# Xem tất cả metrics so sánh
cat data/reports/json/backtest_comparison.json | jq .metrics
```

✅ Kết quả mong đợi:
> `"✅ MULTI-AGENT OUTPERFORMS - Use coordinator strategy"`

---

## 🔍 XEM CÁC BÁO CÁO ĐÃ TẠO

```bash
# Xem danh sách báo cáo mới nhất
ls -lah data/reports/json/ | tail -10
```

---

## ⚠️ NẾU GẶP LỖI

| Lỗi thường gặp | Cách khắc phục |
|----------------|----------------|
| ModuleNotFound | Chạy lại lệnh `source .venv/bin/activate` |
| File news.db không tìm thấy | Chạy bình thường, hệ thống tự động fallback về giá gốc |
| Chạy chậm quá | Bình thường, lần đầu chạy cache dữ liệu |

---

## 🎯 MỤC TIÊU HÔM NAY

1. [ ] Chạy được `test_phase3_components.py` không có lỗi
2. [ ] Chạy xong `backtest_comparison.py` thành công
3. [ ] Xem được kết quả so sánh 2 chiến lược
4. [ ] Quyết định xem chiến lược Multi-Agent có tốt hơn không

---
*Hướng dẫn cập nhật: 13/04/2026 10:55*