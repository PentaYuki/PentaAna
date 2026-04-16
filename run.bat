@echo off
TITLE Stock-AI Dashboard & Pipeline
echo ===================================================
echo 🚀 Khoi dong He thong Stock-AI...
echo ===================================================

:: Chuyển về thư mục chứa file batch
cd /d "%~dp0"

:: Kích hoạt môi trường ảo
if exist ".venv\Scripts\activate.bat" (
    echo [OK] Dang kich hoat moi truong ao (.venv)...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [OK] Dang kich hoat moi truong ao (venv)...
    call venv\Scripts\activate.bat
) else (
    echo [CẢNH BÁO] Khong tim thay moi truong ao (venv/.venv^)!
)

:: Khởi chạy Pipeline Manager dưới chế độ nền (Dùng pythonw để chạy ngầm)
echo [OK] Dang khoi dong Pipeline Manager...
start /b pythonw src\pipeline_manager.py

:: Khởi chạy Web Dashboard
echo [OK] Dang khoi dong Web Dashboard...
echo 👉 Truy cap http://localhost:8088 de xem trang tinh trang he thong.
echo ===================================================
python src\web_dashboard.py

pause
