import pandas as pd

# Đọc file dữ liệu bạn đã lưu
import os
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "VNM_history.csv")
df = pd.read_csv(data_path)
print("=== 5 dòng đầu tiên ===")
print(df.head())
print("\n=== Tên các cột ===")
print(df.columns.tolist())
print("\n=== Kiểu dữ liệu ===")
print(df.dtypes)
print("\n=== Thông tin tổng quan ===")
df.info()