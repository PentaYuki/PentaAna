#!/bin/bash
# ==============================================================================
# train_and_evolve.command — PentaAna Self-Evolving MLOps Pipeline
# Double-click trên macOS hoặc chạy: ./train_and_evolve.command
# ==============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

LOG_DIR="$DIR/data/reports"
LOG_FILE="$LOG_DIR/train_and_evolve_$(date '+%Y%m%d_%H%M%S').log"

# Tee ra màn hình + file log
exec > >(tee -i "$LOG_FILE") 2>&1

echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║        PentaAna — TRAIN & EVOLVE PIPELINE v2.0               ║${NC}"
echo -e "${BOLD}${CYAN}║        Thời gian: $(date '+%Y-%m-%d %H:%M:%S')                      ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Kích hoạt virtualenv ──────────────────────────────────────────────────────
if [ -d "$DIR/.venv" ]; then
    echo -e "${GREEN}✅ Kích hoạt môi trường ảo (.venv)...${NC}"
    source "$DIR/.venv/bin/activate"
elif [ -d "$DIR/venv" ]; then
    echo -e "${GREEN}✅ Kích hoạt môi trường ảo (venv)...${NC}"
    source "$DIR/venv/bin/activate"
else
    echo -e "${YELLOW}⚠️  Không tìm thấy venv — dùng Python hệ thống${NC}"
fi

export PYTHONPATH="$DIR/src:$PYTHONPATH"

# ── Tạo thư mục cần thiết ────────────────────────────────────────────────────
mkdir -p "$DIR/data/reports/json"
mkdir -p "$DIR/data/raw/parquet"
mkdir -p "$DIR/data/raw/csv"
mkdir -p "$DIR/data/analyzed/indicators"
mkdir -p "$DIR/logs"

# ── Danh sách ticker cần xử lý ───────────────────────────────────────────────
# Có thể truyền qua argument: ./train_and_evolve.command VCB FPT HPG
if [ "$#" -gt 0 ]; then
    TICKERS=("$@")
else
    TICKERS=("VNM" "VCB" "FPT" "HPG" "TCB" "MWG" "ACB")
fi

echo -e "${BOLD}📋 Danh sách ticker:${NC} ${TICKERS[*]}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: Cập nhật dữ liệu thị trường mới nhất
# ══════════════════════════════════════════════════════════════════════════════
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}[1/5] 📡 CẬP NHẬT DỮ LIỆU GIÁ VÀ TIN TỨC${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

STEP1_OK=true
for TICKER in "${TICKERS[@]}"; do
    echo -ne "  → Tải dữ liệu ${TICKER}... "
    if python -c "
import sys; sys.path.insert(0,'$DIR/src')
try:
    from data_collector import collect_ticker_data
    collect_ticker_data('${TICKER}')
    print('OK')
except Exception as e:
    print(f'SKIP ({e})')
" 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${YELLOW}⚠ Bỏ qua (không ảnh hưởng)${NC}"
    fi
done

# Cập nhật tin tức nếu có news_crawler
echo -ne "  → Cập nhật tin tức... "
python -c "
import sys; sys.path.insert(0,'$DIR/src')
try:
    from news_crawler import crawl_news
    crawl_news()
    print('OK')
except Exception as e:
    print(f'SKIP ({e})')
" 2>/dev/null || echo -e "${YELLOW}⚠ Bỏ qua${NC}"

sleep 1

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: Walk-forward Backtest hàng tuần
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}[2/5] 📊 CHẠY WALK-FORWARD BACKTEST${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -ne "  → Chạy weekly backtest scheduler... "
if python src/weekly_backtest_scheduler.py > "$LOG_DIR/backtest_$(date '+%Y%m%d').log" 2>&1; then
    echo -e "${GREEN}✅ Hoàn thành${NC}"
    # Trích xuất điểm chính
    SHARPE=$(python -c "
import json,os
p='$DIR/data/reports/json/backtest_report.json'
if os.path.exists(p):
    d=json.load(open(p))
    sr=d.get('summary',{}).get('sharpe_ratio')
    print(f'{sr:.3f}' if sr else '—')
else:
    print('—')
" 2>/dev/null)
    echo -e "  📈 Sharpe Ratio: ${BOLD}${SHARPE}${NC}"
else
    echo -e "${YELLOW}⚠ Backtest gặp lỗi (xem log)${NC}"
    tail -5 "$LOG_DIR/backtest_$(date '+%Y%m%d').log"
fi

sleep 1

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3: Fill Actual Outcomes + Chạy RLHF Cycle
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}[3/5] 🧠 RLHF CYCLE — CẬP NHẬT TRỌNG SỐ AGENT${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -ne "  → Fill kết quả thực tế (outcomes)... "
FILLED=$(python -c "
import sys; sys.path.insert(0,'$DIR/src')
from rlhf_engine import FeedbackStore, fill_pending_outcomes
store = FeedbackStore()
n = fill_pending_outcomes(store, outcome_delay_days=30)
print(n)
" 2>/dev/null || echo "0")
echo -e "${GREEN}✅ ${FILLED} tín hiệu được cập nhật${NC}"

echo ""
echo -e "  → Chạy RLHF cycle cho từng ticker:"
for TICKER in "${TICKERS[@]}"; do
    echo -ne "     ${TICKER}: "
    RESULT=$(python -c "
import sys, json; sys.path.insert(0,'$DIR/src')
from rlhf_engine import run_rlhf_cycle
r = run_rlhf_cycle('${TICKER}', outcome_delay_days=30, min_samples=5)
weights = r.get('adapted_weights', {})
tech = weights.get('technical', 0)
sent = weights.get('sentiment', 0)
macro = weights.get('macro', 0)
risk = weights.get('risk', 0)
skipped = r.get('skipped', False)
n = r.get('rewards_processed', 0)
if skipped:
    print(f'SKIP ({r.get(\"reason\",\"\")})')
else:
    print(f'OK | {n} rewards | Tech={tech:.0%} Sent={sent:.0%} Macro={macro:.0%} Risk={risk:.0%}')
" 2>/dev/null || echo "ERR")
    if echo "$RESULT" | grep -q "OK"; then
        echo -e "${GREEN}✅ $RESULT${NC}"
    elif echo "$RESULT" | grep -q "SKIP"; then
        echo -e "${YELLOW}⏭  $RESULT${NC}"
    else
        echo -e "${RED}❌ $RESULT${NC}"
    fi
done

sleep 1

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 4: PSI Drift Detection
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}[4/5] 🔍 DRIFT DETECTION — PSI CHECK${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

DRIFT_DETECTED=false
echo -ne "  → Chạy MLOps drift pipeline... "
if python -c "
import sys; sys.path.insert(0,'$DIR/src')
try:
    from mlops_pipeline import MLOpsPipeline
    pipeline = MLOpsPipeline()
    pipeline.run_scheduled_check()
    print('OK')
except Exception as e:
    print(f'SKIP ({e})')
" 2>/dev/null | grep -q "OK"; then
    echo -e "${GREEN}✅ Hoàn thành${NC}"

    # Kiểm tra có drift không
    DRIFT_INFO=$(python -c "
import json, os
p = '$DIR/data/reports/json/mlops_log.json'
if not os.path.exists(p): print('NO_FILE'); exit()
entries = json.load(open(p))
if not entries: print('NO_DATA'); exit()
# Lấy entry mới nhất
last = entries[-1]
results = last.get('results', [])
drifted = [r for r in results if r.get('drift_detected')]
if drifted:
    print('DRIFT:' + ','.join([r['ticker'] for r in drifted]))
else:
    print('CLEAN')
" 2>/dev/null || echo "UNKNOWN")

    if echo "$DRIFT_INFO" | grep -q "DRIFT:"; then
        DRIFT_TICKERS=$(echo "$DRIFT_INFO" | sed 's/DRIFT://')
        echo -e "  ${YELLOW}⚠️  Phát hiện DRIFT cho: ${BOLD}${DRIFT_TICKERS}${NC}"
        DRIFT_DETECTED=true
    else
        echo -e "  ${GREEN}✅ Không có drift — model ổn định${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Bỏ qua (module chưa sẵn sàng)${NC}"
fi

sleep 1

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 5: Fine-tuning nếu phát hiện Drift
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}[5/5] 🚀 FINE-TUNING (LoRA)${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ "$DRIFT_DETECTED" = true ]; then
    echo -e "  ${YELLOW}→ Phát hiện drift — bắt đầu fine-tuning Kronos...${NC}"
    echo -e "  ${YELLOW}  (Quá trình này có thể mất 10-30 phút trên M1)${NC}"

    if python -c "
import sys; sys.path.insert(0,'$DIR/src')
from kronos_trainer import finetune_kronos
finetune_kronos(
    epochs=3,
    context_len=64,
    batch_size=2,
    learning_rate=1e-4,
    use_sentiment=True,
    sentiment_alpha=0.15,
    status_path='$DIR/data/reports/json/finetune_status.json',
)
print('DONE')
" 2>&1 | tail -20; then
        echo -e "  ${GREEN}✅ Fine-tuning hoàn thành${NC}"
    else
        echo -e "  ${RED}❌ Fine-tuning thất bại (xem log)${NC}"
    fi
else
    echo -e "  ${GREEN}✅ Không cần fine-tuning — model vẫn tốt${NC}"
    echo -e "  ${YELLOW}  Tip: Chạy fine-tune thủ công bằng: python src/kronos_trainer.py${NC}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# TÓM TẮT
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${GREEN}✅ TRAIN & EVOLVE PIPELINE HOÀN THÀNH!${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  📁 Log đầy đủ: ${BOLD}$LOG_FILE${NC}"
echo ""
echo -e "  📋 Bước tiếp theo:"
echo -e "    1. Mở Dashboard: ${CYAN}http://localhost:5173${NC}"
echo -e "    2. Chấm điểm các tín hiệu trong tuần (RLHF Rating)"
echo -e "    3. Kiểm tra TRỌNG SỐ AGENT đã cập nhật chưa"
echo ""
echo -e "  ⏰ Hoàn thành lúc: $(date '+%H:%M:%S %d/%m/%Y')"
echo ""

# Giữ terminal mở trên macOS khi double-click
if [[ "$0" == *".command" ]]; then
    echo -e "${YELLOW}Nhấn Enter để đóng terminal...${NC}"
    read -r
fi
