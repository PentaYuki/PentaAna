from datetime import datetime
print("Today:", datetime.now().strftime("%Y-%m-%d, %A"))
print("April 13:", datetime(2026, 4, 13).strftime("%Y-%m-%d, %A"))
from vnstock import Vnstock
try:
    df = Vnstock().stock(symbol="VNM", source="VCI").quote.history(start='2026-04-01', end='2026-04-15', interval='1D')
    print(df.tail())
except Exception as e:
    print(e)
