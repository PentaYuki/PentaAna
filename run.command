#!/bin/bash
# ==============================================================================
# Stock-AI Startup Script for macOS — v2.1
# Fix: Kill theo cổng thay vì chỉ theo PID file → không bao giờ bị chiếm cổng
# ==============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║   🚀 Stock-AI — Khởi động Hệ thống              ║${NC}"
echo -e "${BOLD}${CYAN}║   $(date '+%Y-%m-%d %H:%M:%S')                           ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ── Hàm kill theo cổng (fix chiếm cổng) ──────────────────────────────────────
kill_port() {
    local PORT=$1
    local NAME=$2
    # lsof -ti:<port> trả về danh sách PID đang dùng port này
    local PIDS
    PIDS=$(lsof -ti:"$PORT" 2>/dev/null)
    if [ -n "$PIDS" ]; then
        echo -e "  ${YELLOW}⚠️  Cổng $PORT đang bị chiếm bởi:${NC} $PIDS (${NAME})"
        echo "$PIDS" | xargs kill -9 2>/dev/null
        sleep 0.8
        # Xác nhận đã giải phóng
        local REMAINING
        REMAINING=$(lsof -ti:"$PORT" 2>/dev/null)
        if [ -z "$REMAINING" ]; then
            echo -e "  ${GREEN}✓ Cổng $PORT đã được giải phóng${NC}"
        else
            echo -e "  ${RED}❌ Cổng $PORT vẫn còn bị giữ — thử lần 2...${NC}"
            echo "$REMAINING" | xargs kill -9 2>/dev/null
            sleep 1
        fi
    else
        echo -e "  ${GREEN}✓ Cổng $PORT: trống${NC}"
    fi
}

# ── Bước 0: Giải phóng tất cả cổng trước khi khởi động ──────────────────────
echo -e "${BOLD}[0/3] 🧹 Dọn dẹp cổng cũ...${NC}"
kill_port 8088 "Python Backend"
kill_port 5173 "Vite React"
kill_port 5174 "Vite React (fallback)"

# Dọn PID file nếu còn sót
PID_FILE="$DIR/.pids"
if [ -f "$PID_FILE" ]; then
    echo -e "  ${YELLOW}→ Xóa .pids file cũ...${NC}"
    while read -r OLD_PID; do
        kill -9 "$OLD_PID" 2>/dev/null
    done < "$PID_FILE"
    rm -f "$PID_FILE"
fi

# Kill mọi process web_dashboard hoặc pipeline_manager còn sót
pkill -9 -f "web_dashboard.py" 2>/dev/null
pkill -9 -f "pipeline_manager.py" 2>/dev/null
sleep 0.5
echo ""

# ── Kiểm tra Virtual Environment ─────────────────────────────────────────────
if [ -d "$DIR/.venv" ]; then
    echo -e "${GREEN}✅ Kích hoạt môi trường ảo (.venv)...${NC}"
    source "$DIR/.venv/bin/activate"
elif [ -d "$DIR/venv" ]; then
    echo -e "${GREEN}✅ Kích hoạt môi trường ảo (venv)...${NC}"
    source "$DIR/venv/bin/activate"
else
    echo -e "${YELLOW}⚠️  Không tìm thấy môi trường ảo — dùng Python hệ thống${NC}"
fi

# ── Tạo thư mục cần thiết ────────────────────────────────────────────────────
mkdir -p data/reports/json data/raw/parquet data/raw/csv
mkdir -p data/analyzed/indicators data/analyzed/with_indicators logs
echo -e "${GREEN}✅ Thư mục dữ liệu sẵn sàng.${NC}"
echo ""

# Export PYTHONPATH
export PYTHONPATH="$DIR/src:$PYTHONPATH"

# ── [1] Pipeline Manager ─────────────────────────────────────────────────────
echo -e "${BOLD}[1/3] 🔧 Pipeline Manager (background scheduler)${NC}"
nohup python src/pipeline_manager.py > data/reports/pipeline_manager.log 2>&1 &
PIPELINE_PID=$!
echo -e "  → PID: ${BOLD}$PIPELINE_PID${NC}"

sleep 4
if kill -0 "$PIPELINE_PID" 2>/dev/null; then
    echo -e "  ${GREEN}✅ Pipeline Manager: ĐANG CHẠY (PID $PIPELINE_PID)${NC}"
else
    echo -e "  ${YELLOW}⚠️  Pipeline Manager crash — 10 dòng log cuối:${NC}"
    tail -10 data/reports/pipeline_manager.log | sed 's/^/     /'
    echo -e "  ${YELLOW}Hệ thống tiếp tục không có background scheduler.${NC}"
    PIPELINE_PID=""
fi
echo ""

# ── [2] Python Backend (port 8088) ───────────────────────────────────────────
echo -e "${BOLD}[2/3] 🌐 Python Backend (cổng 8088)${NC}"
nohup python src/web_dashboard.py > data/reports/web_dashboard.log 2>&1 &
BACKEND_PID=$!
echo -e "  → PID: ${BOLD}$BACKEND_PID${NC}"

echo -e "  ⏳ Chờ backend khởi động..."
for i in 1 2 3 4 5 6 7 8; do
    sleep 1
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "  ${RED}❌ Backend crashed sớm! Log cuối:${NC}"
        tail -15 data/reports/web_dashboard.log | sed 's/^/     /'
        BACKEND_PID=""
        break
    fi
    # Kiểm tra cổng đã mở chưa
    if lsof -ti:8088 >/dev/null 2>&1; then
        echo -e "  ${GREEN}✅ Backend: READY tại http://localhost:8088 (PID $BACKEND_PID, ${i}s)${NC}"
        break
    fi
    echo -ne "  ."
done
echo ""

# ── [3] React Dashboard (Vite, port 5173) ────────────────────────────────────
VITE_PID=""
echo -e "${BOLD}[3/3] 🖥️  React Dashboard (Vite, cổng 5173)${NC}"
if [ -d "$DIR/dashboard" ] && [ -f "$DIR/dashboard/package.json" ]; then
    cd "$DIR/dashboard"
    npm run dev > "$DIR/data/reports/vite.log" 2>&1 &
    VITE_PID=$!
    echo -e "  → PID: ${BOLD}$VITE_PID${NC}"
    cd "$DIR"

    # Chờ Vite lên (tối đa 12 giây)
    echo -e "  ⏳ Chờ Vite khởi động..."
    VITE_READY=false
    for i in $(seq 1 12); do
        sleep 1
        if lsof -ti:5173 >/dev/null 2>&1 || lsof -ti:5174 >/dev/null 2>&1; then
            VITE_READY=true
            break
        fi
        if ! kill -0 "$VITE_PID" 2>/dev/null; then
            echo -e "  ${RED}❌ Vite crash! Log cuối:${NC}"
            tail -10 "$DIR/data/reports/vite.log" | sed 's/^/     /'
            VITE_PID=""
            break
        fi
    done

    if [ "$VITE_READY" = true ]; then
        ACTUAL_VITE_PORT=5173
        lsof -ti:5174 >/dev/null 2>&1 && ACTUAL_VITE_PORT=5174
        echo -e "  ${GREEN}✅ React Dashboard: READY tại http://localhost:${ACTUAL_VITE_PORT} (PID $VITE_PID)${NC}"
    elif [ -n "$VITE_PID" ]; then
        echo -e "  ${YELLOW}⚠️  Vite chưa ready hoàn toàn — thử mở http://localhost:5173${NC}"
    fi
else
    echo -e "  ${YELLOW}⚠️  Không tìm thấy thư mục dashboard — bỏ qua Vite.${NC}"
fi
echo ""

# ── Lưu PIDs ─────────────────────────────────────────────────────────────────
{
    [ -n "$PIPELINE_PID" ] && echo "$PIPELINE_PID"
    [ -n "$BACKEND_PID"  ] && echo "$BACKEND_PID"
    [ -n "$VITE_PID"     ] && echo "$VITE_PID"
} > "$PID_FILE"

# ── Summary ───────────────────────────────────────────────────────────────────
echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  🌐 Backend API  :  http://localhost:8088${NC}"
echo -e "${BOLD}  🖥️  Dashboard   :  http://localhost:5173${NC}"
echo ""
echo -e "  📋 Logs:"
echo -e "     Pipeline : ${CYAN}data/reports/pipeline_manager.log${NC}"
echo -e "     Backend  : ${CYAN}data/reports/web_dashboard.log${NC}"
echo -e "     Vite     : ${CYAN}data/reports/vite.log${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  Nhấn Ctrl+C để dừng tất cả dịch vụ.${NC}"
echo ""

# ── Cleanup khi Ctrl+C / terminal đóng ───────────────────────────────────────
cleanup() {
    echo ""
    echo -e "${RED}🛑 Đang dừng các dịch vụ...${NC}"

    # Kill theo PID trước
    [ -n "$VITE_PID"     ] && kill "$VITE_PID"     2>/dev/null && echo -e "  ${YELLOW}→ Đã dừng Vite (PID $VITE_PID)${NC}"
    [ -n "$BACKEND_PID"  ] && kill "$BACKEND_PID"  2>/dev/null && echo -e "  ${YELLOW}→ Đã dừng Backend (PID $BACKEND_PID)${NC}"
    [ -n "$PIPELINE_PID" ] && kill "$PIPELINE_PID" 2>/dev/null && echo -e "  ${YELLOW}→ Đã dừng Pipeline (PID $PIPELINE_PID)${NC}"

    sleep 1

    # Đảm bảo cổng sạch (fallback kill-by-port)
    local LEFTOVER_8088
    LEFTOVER_8088=$(lsof -ti:8088 2>/dev/null)
    [ -n "$LEFTOVER_8088" ] && echo "$LEFTOVER_8088" | xargs kill -9 2>/dev/null && echo -e "  ${RED}→ Force-kill cổng 8088${NC}"

    local LEFTOVER_5173
    LEFTOVER_5173=$(lsof -ti:5173 2>/dev/null)
    [ -n "$LEFTOVER_5173" ] && echo "$LEFTOVER_5173" | xargs kill -9 2>/dev/null && echo -e "  ${RED}→ Force-kill cổng 5173${NC}"

    rm -f "$PID_FILE"
    echo -e "${GREEN}✅ Tất cả dịch vụ đã dừng sạch.${NC}"
    exit 0
}
trap cleanup INT TERM EXIT

# ── Watch loop: tự restart pipeline_manager nếu crash ────────────────────────
echo -e "${CYAN}📡 Giám sát hệ thống... (Ctrl+C để dừng)${NC}"
RESTART_COUNT=0
while true; do
    sleep 10

    # Tự động restart pipeline_manager nếu crash
    if [ -n "$PIPELINE_PID" ] && ! kill -0 "$PIPELINE_PID" 2>/dev/null; then
        RESTART_COUNT=$((RESTART_COUNT + 1))
        echo -e "${YELLOW}⚠️  [$(date '+%H:%M:%S')] Pipeline Manager crash (lần $RESTART_COUNT). Đang restart...${NC}"
        nohup python src/pipeline_manager.py >> data/reports/pipeline_manager.log 2>&1 &
        PIPELINE_PID=$!
        echo -e "  ${GREEN}→ Restart PID: $PIPELINE_PID${NC}"
        {
            [ -n "$PIPELINE_PID" ] && echo "$PIPELINE_PID"
            [ -n "$BACKEND_PID"  ] && kill -0 "$BACKEND_PID" 2>/dev/null && echo "$BACKEND_PID"
            [ -n "$VITE_PID"     ] && kill -0 "$VITE_PID"    2>/dev/null && echo "$VITE_PID"
        } > "$PID_FILE"
    fi

    # Cảnh báo nếu backend bị down
    if [ -n "$BACKEND_PID" ] && ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "${RED}❌ [$(date '+%H:%M:%S')] Backend :8088 đã chết! Khởi động lại...${NC}"
        nohup python src/web_dashboard.py >> data/reports/web_dashboard.log 2>&1 &
        BACKEND_PID=$!
        sleep 3
        if kill -0 "$BACKEND_PID" 2>/dev/null; then
            echo -e "${GREEN}✅ Backend đã restart (PID $BACKEND_PID)${NC}"
        fi
        {
            [ -n "$PIPELINE_PID" ] && kill -0 "$PIPELINE_PID" 2>/dev/null && echo "$PIPELINE_PID"
            [ -n "$BACKEND_PID"  ] && echo "$BACKEND_PID"
            [ -n "$VITE_PID"     ] && kill -0 "$VITE_PID"    2>/dev/null && echo "$VITE_PID"
        } > "$PID_FILE"
    fi
done
