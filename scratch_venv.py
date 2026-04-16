import sys
from datetime import datetime
print("Today:", datetime.now().strftime("%Y-%m-%d, %A"))
print("April 13:", datetime(2026, 4, 13).strftime("%Y-%m-%d, %A"))
try:
    from vnstock import Vnstock
    df = Vnstock().stock(symbol="VNM", source="VCI").quote.history(start='2026-04-01', end='2026-04-15', interval='1D')
    print("VCI source data:")
    print(df.tail(3))
except Exception as e:
    print("VCI error:", e)

try:
    from vnstock import Vnstock
    df2 = Vnstock().stock(symbol="VNM", source="TCBS").quote.history(start='2026-04-01', end='2026-04-15', interval='1D')
    print("TCBS source data:")
    print(df2.tail(3))
except Exception as e:
    print("TCBS error:", e)

