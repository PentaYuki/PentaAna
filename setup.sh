#!/bin/bash
# Thiết lập môi trường stock-ai

echo "Cài đặt các gói hệ thống qua brew (yêu cầu brew đã cài)..."
brew install python@3.11 git wget

cd ~/stock-ai
echo "Tạo virtualenv .venv..."
python3.11 -m venv .venv
source .venv/bin/activate

echo "Cài đặt các thư viện Python..."
pip install --upgrade pip
pip install ollama
pip install mlx mlx-lm
pip install numpy pandas scipy scikit-learn matplotlib seaborn plotly torch torchvision transformers huggingface-hub jupyter notebook tqdm rich schedule
pip install vnstock pandas-ta ta-lib-binary
pip install git+https://github.com/amazon-science/chronos-forecasting.git

echo "Lưu requirements..."
pip freeze > requirements.txt
echo "Hoàn tất cài đặt môi trường!"
