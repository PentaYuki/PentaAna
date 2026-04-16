#!/bin/bash
# ==============================================================================
# Hướng dẫn & Tự động cấu hình Crontab cho macOS / Linux
# ==============================================================================

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "==================================================="
echo "⏳ Thiết lập Cron Jobs cho Stock-AI"
echo "==================================================="

# Chuyển tới thư mục làm việc để tạo chuỗi cron hoàn chỉnh
CRON_1="0 0 * * * cd $DIR && .venv/bin/python src/pipeline_manager.py >> $DIR/data/reports/pipeline_manager.log 2>&1"
CRON_2="0 7 * * 1 cd $DIR && .venv/bin/python src/weekly_backtest_scheduler.py --run-now >> $DIR/data/reports/backtest_scheduler.log 2>&1"

# Lấy crontab hiện có
(crontab -l 2>/dev/null; echo ""; echo "### STOCK-AI CRON JOBS ###"; echo "$CRON_1"; echo "$CRON_2") | awk '!x[$0]++' > mycron

crontab mycron
rm mycron

echo "✅ Đã thêm các cấu hình sau vào Crontab của bạn:"
echo "---------------------------------------------------"
crontab -l | grep "STOCK-AI" -A 2
echo "---------------------------------------------------"
echo "Lưu ý: Mặc định cron yêu cầu máy tính của bạn phải MỞ trong khung giờ đó."
echo "Bạn có thể kiểm tra các log hoạt động nằm trong mục: data/reports/"
