# 📘 Tài liệu Hướng dẫn Giảng dạy AI — PentaAna Stock Intelligence

> **Phiên bản:** 2.0 · **Cập nhật:** 2026-04-15 · **Hệ thống:** Phase 4 Self-Evolving MLOps

---

## Tổng quan: AI học như thế nào?

Hệ thống **PentaAna** được thiết kế để **không đứng yên**. Nó học tập liên tục từ **4 nguồn dữ liệu** khác nhau:

```
┌──────────────────────────────────────────────────────────┐
│                   VÒNG TRÒNG HỌC TẬP                    │
│                                                          │
│   Thị trường ──→ Backtest ──→ RLHF Reward                │
│       ↑               │            │                     │
│   Dữ liệu mới         ↓            ↓                     │
│       ↑          Drift Check → Fine-tuning               │
│       └────────── Trọng số mới (Agent Weights) ──────────┘
└──────────────────────────────────────────────────────────┘
```

| Nguồn học | Cơ chế | Tần suất |
|-----------|--------|----------|
| **Bạn** (RLHF) | Chấm sao 1-5 trên Dashboard | Bất kỳ lúc nào |
| **Thị trường** | So sánh dự báo vs. giá thực | Tự động mỗi 30 ngày |
| **Lịch sử** | Walk-forward backtest | Mỗi tuần (Thứ 7) |
| **Model** | LoRA Fine-tuning (Kronos) | Khi phát hiện Drift |

---

## 1. Dạy AI qua Phản hồi của Bạn (RLHF — Nhanh nhất)

Đây là **cách mạnh nhất và nhanh nhất** để điều chỉnh "nhận thức" của AI. Chỉ mất 5 giây cho mỗi tín hiệu.

### Khi nào nên chấm điểm?

| Tình huống | Hành động | Ý nghĩa |
|-----------|-----------|---------|
| AI phát BUY đúng, giá tăng | Chấm **5 sao** ★★★★★ | "Tuyệt! Giữ phong cách này" |
| AI phát SELL đúng, giá giảm | Chấm **5 sao** ★★★★★ | "Phân tích rủi ro chuẩn xác" |
| AI phát BUY nhưng giá đi ngang | Chấm **3 sao** ★★★☆☆ | "Trung tính, không tốt không xấu" |
| AI phát BUY nhưng giá giảm mạnh | Chấm **1 sao** ★☆☆☆☆ | "Sai lầm nghiêm trọng" |
| AI phát HOLD trong khi thị trường biến động | Chấm **2 sao** | "Quá thụ động" |

### Cách chấm điểm trên Dashboard:

```
1. Mở http://localhost:5173
2. Chạy phân tích bất kỳ mã nào (VD: VCB, FPT)
3. Kéo xuống phần "◈ LỊCH SỬ TÍN HIỆU RLHF"
4. Tìm tín hiệu cần đánh giá
5. Click vào số sao (★) tương ứng trong cột cuối cùng
```

### Công thức tính Reward sau khi bạn chấm điểm:

```
Reward = (market_return × signal_direction × confidence) × 0.7
       + (user_rating - 3) / 2 × 0.5

clip(Reward, -2.0, +2.0)
```

- **5 sao** → user_component = +1.0 → tăng reward tối đa
- **1 sao** → user_component = -1.0 → giảm reward (phạt)
- **3 sao** → user_component = 0.0 → trung tính

### File lưu trữ:
- Feedback: `data/news.db` (bảng `rlhf_feedback`)
- Trọng số cập nhật: `data/reports/json/rlhf_weights.json`
- Trọng số theo mã: `data/reports/json/rlhf_weights_{TICKER}.json`

---

## 2. Dạy AI qua Kết quả Thị trường (Auto-Outcome)

AI tự động đối chiếu dự báo của mình với **giá thực tế** sau 30 ngày giao dịch.

### Cơ chế hoạt động:

```
Ngày 1:   AI phát tín hiệu BUY cho VCB, dự báo return +5%
          → Ghi vào DB: signal="BUY", forecast_return=5.0, confidence=0.72

Ngày 31:  Hệ thống tự động lấy giá thực tế
          actual_return = (giá_ngày31 - giá_ngày1) / giá_ngày1 × 100

Alpha-adjusted reward:
          alpha_return = actual_return - VNIndex_return
          reward = alpha_return × sign(forecast) × confidence
```

> [!NOTE]
> **Alpha-based reward** nghĩa là AI được thưởng khi **outperform thị trường**, không chỉ khi giá tăng. Nếu thị trường tăng 3% mà cổ phiếu tăng 2%, AI bị phạt vì dự báo kém hơn index.

### Kích hoạt thủ công:
```bash
python src/rlhf_engine.py
```

---

## 3. Dạy AI qua Backtest Hàng tuần (Tự kiểm tra)

AI "soi gương" toàn bộ lịch sử giao dịch mỗi tuần.

### Walk-forward Backtest là gì?

```
Tuần 1:  Train trên dữ liệu 2024-01 → 2024-06
         Test trên 2024-07 → 2024-09  [Window 1]

Tuần 2:  Train trên dữ liệu 2024-04 → 2024-09
         Test trên 2024-10 → 2024-12  [Window 2]

...cứ trượt dần như vậy, không "nhìn tương lai"
```

### Kết quả backtest ảnh hưởng đến AI như thế nào?

| Chỉ số | Ngưỡng | Hành động |
|--------|--------|----------|
| **Sharpe Ratio** < 0.3 | Hiệu suất thấp | Trigger drift detection |
| **MAE** (Mean Abs Error) giá > 15% | Dự báo sai | Ghi penalty reward |
| **Win Rate** < 40% | Tỉ lệ thắng thấp | Giảm confidence weights |
| **PSI** (drift score) > 0.2 | Phân phối thay đổi | Trigger fine-tuning |

### Chạy thủ công:
```bash
python src/weekly_backtest_scheduler.py
```

### Xem kết quả:
```bash
cat data/reports/json/backtest_report.json
```

---

## 4. Dạy AI qua Fine-tuning LoRA (Học sâu — Mạnh nhất)

Cập nhật chính "bản năng" dự báo của model Kronos.

### LoRA (Low-Rank Adaptation) là gì?

Thay vì huấn luyện lại toàn bộ model (tốn hàng GB RAM), LoRA chỉ thêm các **adapter nhỏ** vào các layer quan trọng:

```
Model gốc (đóng băng):  [──────── 125M params ────────]
LoRA adapters (học):              [A] × [B]
                                   ↑      ↑
                              rank=4   nhỏ gọn
```

**RAM cần thiết:** ~4GB (M1 Mac hoàn toàn chạy được)

### Khi nào nên fine-tune?

1. PSI drift > 0.2 (phân phối dữ liệu thay đổi)
2. Mỗi 4-6 tuần sau khi có đủ dữ liệu mới
3. Sau sự kiện thị trường lớn (KRX ra mắt, thay đổi biên độ...)

### Fine-tune qua Dashboard:

```
1. Mở Dashboard → Kéo xuống "FINE-TUNE HYPERPARAMETERS"
2. Điều chỉnh tham số:
   - Epochs: 3-5 (đủ học mà không overfit)
   - Learning Rate: 0.0001 (an toàn cho M1)
   - Batch Size: 2-4 (tùy RAM)
   - Context Length: 64-128
   - Sentiment Alpha: 0.15-0.3
3. Nhấn "▶ BẮT ĐẦU FINE-TUNE"
4. Theo dõi progress bar
```

### Fine-tune qua command line:
```bash
python src/kronos_trainer.py
```

---

## 5. Cơ chế Cập nhật Trọng số Agent (EWM)

Đây là "bộ não phân phối" của PentaAna — quyết định agent nào được tin tưởng nhất.

### 4 Agent và ý nghĩa:

| Agent | Vai trò | EWM Alpha | Tốc độ thích nghi |
|-------|---------|-----------|------------------|
| **Technical** | RSI, MACD, BB, ATR | 0.08 | Chậm (lagging) |
| **Sentiment** | Tin tức, cảm xúc | 0.15 | Nhanh (reactive) |
| **Macro** | VNIndex, lãi suất, USD | 0.05 | Rất chậm (trend) |
| **Risk** | Volatility, drawdown | 0.10 | Trung bình |

### Cơ chế cập nhật EWM:

```python
new_weight = old_weight + alpha × reward × contribution
```

Trong đó:
- `alpha`: tốc độ học của từng agent (xem bảng trên)
- `reward`: kết quả thực tế (+/-2.0)
- `contribution`: agent này đóng góp bao nhiêu vào tín hiệu sai/đúng

### Guardrails (Bảo vệ):

```
MIN_WEIGHT = 5%   → Không agent nào bị "tắt hoàn toàn"
MAX_WEIGHT = 60%  → Không agent nào "độc tài"
Sau mỗi update: renormalize để tổng = 100%
```

**Ví dụ thực tế:**
```
Tuần 1 (thị trường biến động mạnh, tin tức quan trọng):
  Technical: 38%  Sentiment: 30%  Macro: 18%  Risk: 14%

Tuần 4 (thị trường trending, ít tin tức):
  Technical: 45%  Sentiment: 20%  Macro: 22%  Risk: 13%
```

---

## 6. Quy trình Hàng tuần — "Teaching Routine"

Để AI luôn thông minh và cập nhật nhất, thực hiện theo lịch sau:

### 📅 Thứ 7 (Cuối tuần — ~20 phút):

```bash
# Chạy toàn bộ train-and-evolve pipeline
./train_and_evolve.command
```

Script này tự động:
1. ✅ Cập nhật dữ liệu giá và tin tức mới nhất
2. ✅ Chạy walk-forward backtest (kiểm tra 30 ngày qua)
3. ✅ Fill kết quả actual_return cho tín hiệu đã đủ 30 ngày
4. ✅ Chạy RLHF cycle cho từng ticker trong danh sách
5. ✅ Kiểm tra PSI drift — nếu drift → trigger fine-tuning
6. ✅ Lưu log vào `mlops_log.json`

### 📅 Chủ nhật (Chấm điểm — ~10 phút):

```
1. Mở Dashboard: http://localhost:5173
2. Chạy phân tích các mã quan tâm (VCB, FPT, VNM...)
3. Kéo xuống "LỊCH SỬ TÍN HIỆU RLHF"
4. Chấm điểm 5-10 tín hiệu gần nhất
5. Đặc biệt chấm điểm các tín hiệu quan trọng (đã có kết quả rõ ràng)
```

### 📅 Thứ 2 (Kiểm tra — ~5 phút):

```
1. Mở Dashboard
2. Kiểm tra phần "TRỌNG SỐ AGENT (RLHF)" — weights có cập nhật không?
3. Chạy phân tích mã quan tâm → AI đã "học" từ feedback chưa?
```

---

## 7. Troubleshooting — Khi AI "học lệch"

### Vấn đề: AI cứ phát BUY dù thị trường đang giảm

**Nguyên nhân:** Sentiment agent (tin tức) đang có trọng số quá cao trong giai đoạn thị trường đi ngược sentiment.

**Giải pháp:**
```
1. Chấm 1-2 sao cho các tín hiệu BUY sai của tuần vừa qua
2. Đợi RLHF cycle chạy (tự động) → Technical agent sẽ được tăng trọng số
3. Hoặc: xem file rlhf_weights.json, kiểm tra sentiment weight
```

### Vấn đề: AI quá thụ động (toàn HOLD)

**Nguyên nhân:** Risk agent có trọng số quá cao sau nhiều lần phạt.

**Giải pháp:**
```
1. Chấm 4-5 sao cho các tín hiệu HOLD trong giai đoạn thị trường tăng mạnh
2. Kích hoạt lại bằng cách fine-tune với sentiment_alpha cao hơn (0.3)
```

### Vấn đề: Backtest báo Sharpe âm

**Nguyên nhân:** Model đang overfit dữ liệu cũ, không thích nghi với thị trường mới.

**Giải pháp:**
```bash
python src/kronos_trainer.py  # Fine-tune với dữ liệu mới nhất
```

---

## 8. Cấu trúc File Quan trọng

```
data/
├── news.db                          # SQLite: tín hiệu RLHF + feedback
├── raw/parquet/
│   ├── VCB_history.parquet          # Lịch sử giá từng mã
│   └── VNINDEX_history.parquet      # VNINDEX để tính alpha
└── reports/json/
    ├── rlhf_weights.json            # Trọng số agent toàn cục
    ├── rlhf_weights_VCB.json        # Trọng số agent riêng cho VCB
    ├── backtest_report.json         # Kết quả backtest tuần này
    ├── mlops_log.json               # Log PSI drift + retrain events
    └── finetune_status.json         # Tiến trình fine-tuning hiện tại
```

---

## 9. Tóm tắt Nhanh — Command Reference

```bash
# Hệ thống hàng ngày
./run.command                        # Khởi động toàn bộ hệ thống

# Dạy AI (hàng tuần)
./train_and_evolve.command           # Train + evolve tổng hợp

# Từng module riêng lẻ
python src/weekly_backtest_scheduler.py   # Chỉ chạy backtest
python src/kronos_trainer.py             # Chỉ fine-tune model
python src/rlhf_engine.py                # Chỉ chạy RLHF cycle
python src/mlops_pipeline.py             # Chỉ kiểm tra drift
```

---

> [!NOTE]
> **Guardrails:** Hệ thống có cơ chế bảo vệ tự động. Agent không bao giờ bị giảm dưới **5%** hoặc tăng quá **60%**. Reward bị clip trong khoảng **[-2.0, +2.0]** để tránh update quá đột ngột.

> [!TIP]
> **Mẹo dạy AI hiệu quả:** Tập trung chấm điểm các tín hiệu có **confidence cao** (>70%) vì chúng ảnh hưởng nhiều hơn đến weight update. Tín hiệu confidence thấp (<40%) ít ảnh hưởng dù chấm sao nào.

> [!IMPORTANT]
> **Không nên fine-tune quá thường xuyên.** Fine-tune mỗi 4-6 tuần là đủ. Fine-tune quá nhiều có thể làm model "quên" các pattern dài hạn.
