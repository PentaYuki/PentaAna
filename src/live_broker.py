"""
live_broker.py — Module giao dịch thực cho Stock-AI (VNDirect / SSI / Giả lập).

Kiến trúc:
  ┌─────────────────────────────────────────────────────────────────┐
  │  phase4_orchestrator.run_phase4()                               │
  │      └── LiveTradingEngine.run_signal()                         │
  │              ├── RiskManager (ATR stop, trailing stop, sizing)   │
  │              ├── VNDirectBroker / SSIBroker / PaperBroker        │
  │              └── PositionTracker (SQLite, realtime update)       │
  └─────────────────────────────────────────────────────────────────┘

Cài đặt:
    pip install requests cryptography vnstock

Cấu hình môi trường (.env hoặc biến môi trường):
    BROKER=vndirect           # vndirect | ssi | paper
    VNDIRECT_ACCOUNT=...      # Số tài khoản VNDirect
    VNDIRECT_CONSUMER_ID=...  # Consumer ID từ VNDirect Developer
    VNDIRECT_PRIVATE_KEY=...  # RSA private key PEM (base64 or path)
    SSI_CLIENT_ID=...
    SSI_CLIENT_SECRET=...
    MAX_POSITION_VND=50000000 # Tối đa 50 triệu/lệnh
    MAX_DAILY_ORDERS=5
    PAPER_CAPITAL=500000000   # 500 triệu VND giả lập
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ─── Đường dẫn ───────────────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH  = os.path.join(DATA_DIR, "news.db")     # Dùng chung DB với RLHF/news
POSITIONS_PATH = os.path.join(DATA_DIR, "reports", "json", "live_positions.json")

os.makedirs(os.path.dirname(POSITIONS_PATH), exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS & DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

class OrderStatus(str, Enum):
    PENDING   = "PENDING"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED  = "REJECTED"
    PARTIAL   = "PARTIAL"


class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    ticker:     str
    side:       OrderSide
    quantity:   int
    price:      float               # Giá đặt lệnh (VND)
    order_type: str = "LO"          # LO=Limit, MP=Market
    status:     OrderStatus = OrderStatus.PENDING
    order_id:   Optional[str] = None
    filled_qty: int = 0
    filled_price: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: Optional[str] = None
    error_msg:  Optional[str] = None
    slippage_pct: float = 0.0      # Slippage thực tế sau khi khớp

    def net_value(self) -> float:
        """Giá trị lệnh đã khớp (VND)."""
        return self.filled_qty * self.filled_price

    def is_done(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)


@dataclass
class Position:
    ticker:          str
    quantity:        int
    entry_price:     float          # Giá mua trung bình (VND)
    entry_date:      str
    atr_pct:         float = 2.0    # ATR % tại thời điểm mở lệnh
    stop_loss:       float = 0.0    # Stop loss tuyệt đối (VND)
    trailing_stop:   float = 0.0    # Trailing stop cao nhất đạt được
    highest_price:   float = 0.0    # Giá cao nhất kể từ khi mua
    rlhf_row_id:     Optional[int] = None   # ID trong bảng rlhf_feedback
    agent_signal_id: Optional[str] = None

    def unrealized_pct(self, current_price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price * 100.0

    def update_trailing(self, current_price: float, atr_multiplier: float = 2.0):
        """Cập nhật trailing stop khi giá lên cao hơn."""
        if current_price > self.highest_price:
            self.highest_price = current_price
            # Trailing stop = giá cao nhất - (ATR_multiplier × ATR)
            new_trail = current_price * (1.0 - atr_multiplier * self.atr_pct / 100.0)
            self.trailing_stop = max(self.trailing_stop, new_trail)

    def should_stop_out(self, current_price: float) -> tuple[bool, str]:
        """Kiểm tra có cần cắt lỗ không. Trả về (True/False, lý do)."""
        if self.stop_loss > 0 and current_price <= self.stop_loss:
            return True, f"STOP_LOSS ({current_price:.0f} ≤ {self.stop_loss:.0f})"
        if self.trailing_stop > 0 and current_price <= self.trailing_stop:
            return True, f"TRAILING_STOP ({current_price:.0f} ≤ {self.trailing_stop:.0f})"
        return False, ""


# ══════════════════════════════════════════════════════════════════════════════
# RISK MANAGER — Tích hợp ATR từ enhanced_risk_agent
# ══════════════════════════════════════════════════════════════════════════════

class RiskManager:
    """
    Quản lý rủi ro realtime:
    - Tính kích thước lệnh từ % rủi ro tài khoản (Kelly-lite)
    - Thiết lập stop loss ATR (1.5× ATR) và trailing stop (2× ATR)
    - Kiểm tra giới hạn: max lệnh/ngày, max drawdown danh mục
    - Tính slippage kỳ vọng theo thanh khoản

    Tham số (có thể override qua constructor hoặc env):
      risk_per_trade_pct : % vốn tối đa chấp nhận mất mỗi lệnh (mặc định 1.5%)
      max_position_vnd   : Tối đa VND cho 1 lệnh
      max_daily_orders   : Số lệnh tối đa/ngày
      atr_stop_mult      : Hệ số ATR cho stop loss ban đầu (mặc định 1.5)
      atr_trail_mult     : Hệ số ATR cho trailing stop (mặc định 2.0)
      max_slippage_pct   : Từ chối lệnh nếu slippage vượt ngưỡng (mặc định 0.5%)
    """

    def __init__(
        self,
        account_value_vnd:  float = 500_000_000,
        risk_per_trade_pct: float = 1.5,
        max_position_vnd:   float = None,
        max_daily_orders:   int   = 5,
        atr_stop_mult:      float = 1.5,
        atr_trail_mult:     float = 2.0,
        max_slippage_pct:   float = 0.5,
    ):
        self.account_value       = account_value_vnd
        self.risk_per_trade_pct  = risk_per_trade_pct
        self.max_position_vnd    = max_position_vnd or float(
            os.getenv("MAX_POSITION_VND", 50_000_000)
        )
        self.max_daily_orders    = max_daily_orders
        self.atr_stop_mult       = atr_stop_mult
        self.atr_trail_mult      = atr_trail_mult
        self.max_slippage_pct    = max_slippage_pct

        self._daily_order_count  = 0
        self._daily_reset_date   = datetime.now().date()

    def _reset_daily_if_needed(self):
        today = datetime.now().date()
        if today != self._daily_reset_date:
            self._daily_order_count = 0
            self._daily_reset_date  = today

    def check_daily_limit(self) -> tuple[bool, str]:
        self._reset_daily_if_needed()
        if self._daily_order_count >= self.max_daily_orders:
            return False, f"Đã đủ {self.max_daily_orders} lệnh hôm nay"
        return True, ""

    def compute_position_size(
        self,
        price:    float,
        atr_pct:  float,
        confidence: float = 0.6,
    ) -> int:
        """
        Tính số lượng cổ phiếu tối ưu theo ATR-based position sizing.

        Công thức:
          risk_amount = account_value × risk_per_trade_pct / 100
          stop_distance = price × atr_pct × atr_stop_mult / 100
          shares = risk_amount / stop_distance

        Confidence scaling: giảm size khi confidence thấp.
        Limit: không vượt max_position_vnd.

        Returns:
            Số lượng cổ phiếu (làm tròn xuống 100, tối thiểu 100)
        """
        if price <= 0 or atr_pct <= 0:
            return 0

        risk_amount    = self.account_value * self.risk_per_trade_pct / 100.0
        stop_distance  = price * (atr_pct * self.atr_stop_mult / 100.0)

        if stop_distance < 1.0:
            return 0

        raw_shares = risk_amount / stop_distance

        # Confidence scaling: full size chỉ khi confidence ≥ 0.8
        conf_factor = min(1.0, max(0.3, (confidence - 0.5) / 0.3))
        raw_shares *= conf_factor

        # Giới hạn theo max_position_vnd
        max_by_value = self.max_position_vnd / price
        shares = min(raw_shares, max_by_value)

        # HOSE lot size = 100 cổ phiếu
        shares = int(shares // 100) * 100
        return max(100, shares)

    def compute_stop_levels(
        self,
        entry_price: float,
        atr_pct:     float,
    ) -> tuple[float, float]:
        """
        Trả về (stop_loss, trailing_stop_initial).
        - stop_loss     = entry × (1 - atr_stop_mult × atr_pct)
        - trailing_stop = entry × (1 - atr_trail_mult × atr_pct)
        """
        stop_loss     = entry_price * (1.0 - self.atr_stop_mult  * atr_pct / 100.0)
        trailing_stop = entry_price * (1.0 - self.atr_trail_mult * atr_pct / 100.0)
        return round(stop_loss, 0), round(trailing_stop, 0)

    def estimate_slippage(
        self,
        price:         float,
        quantity:      int,
        avg_vol_20d:   float = 1_000_000,   # Khối lượng TB 20 phiên
    ) -> float:
        """
        Ước lượng slippage % theo thanh khoản.
        Lệnh lớn hơn 1% avg volume → slippage tăng.

        Returns: slippage % (dương = tốt hơn, âm = tệ hơn cho buyer)
        """
        order_value  = price * quantity
        market_value = price * avg_vol_20d
        if market_value <= 0:
            return 0.3  # Default 0.3% nếu không có dữ liệu volume

        impact_ratio = order_value / market_value
        # Market impact model: slippage ~ sqrt(impact_ratio) × 0.5%
        slippage = min(0.5 * (impact_ratio ** 0.5) * 100.0, 1.5)
        return round(slippage, 4)

    def approve_order(
        self,
        price:    float,
        quantity: int,
        atr_pct:  float,
        avg_vol:  float = 1_000_000,
    ) -> tuple[bool, str]:
        """
        Kiểm tra toàn bộ điều kiện trước khi gửi lệnh.
        Returns: (approved, reason)
        """
        ok, msg = self.check_daily_limit()
        if not ok:
            return False, msg

        if price <= 0 or quantity < 100:
            return False, f"Thông số không hợp lệ: price={price}, qty={quantity}"

        order_value = price * quantity
        if order_value > self.max_position_vnd:
            return False, (
                f"Giá trị lệnh {order_value:,.0f} VND > max {self.max_position_vnd:,.0f} VND"
            )

        slippage = self.estimate_slippage(price, quantity, avg_vol)
        if slippage > self.max_slippage_pct:
            return False, (
                f"Slippage ước tính {slippage:.2f}% > ngưỡng {self.max_slippage_pct}%"
            )

        self._daily_order_count += 1
        return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# POSITION TRACKER — SQLite + JSON
# ══════════════════════════════════════════════════════════════════════════════

class PositionTracker:
    """
    Theo dõi vị thế mở trong SQLite (bảng live_positions).
    Thread-safe với lock.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock   = threading.Lock()
        self._init_table()

    def _init_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS live_positions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker          TEXT    NOT NULL UNIQUE,
                    quantity        INTEGER NOT NULL,
                    entry_price     REAL    NOT NULL,
                    entry_date      TEXT    NOT NULL,
                    atr_pct         REAL    DEFAULT 2.0,
                    stop_loss       REAL    DEFAULT 0.0,
                    trailing_stop   REAL    DEFAULT 0.0,
                    highest_price   REAL    DEFAULT 0.0,
                    rlhf_row_id     INTEGER,
                    status          TEXT    DEFAULT 'OPEN',
                    opened_at       TEXT    DEFAULT (datetime('now')),
                    closed_at       TEXT,
                    close_price     REAL,
                    realized_pct    REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS live_orders (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker      TEXT,
                    side        TEXT,
                    quantity    INTEGER,
                    price       REAL,
                    order_type  TEXT,
                    status      TEXT,
                    order_id    TEXT,
                    filled_qty  INTEGER DEFAULT 0,
                    filled_price REAL   DEFAULT 0,
                    slippage_pct REAL   DEFAULT 0,
                    error_msg   TEXT,
                    created_at  TEXT,
                    updated_at  TEXT
                )
            """)
            conn.commit()

    def open_position(self, pos: Position):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO live_positions
                    (ticker, quantity, entry_price, entry_date, atr_pct,
                     stop_loss, trailing_stop, highest_price, rlhf_row_id, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
                """, (
                    pos.ticker, pos.quantity, pos.entry_price, pos.entry_date,
                    pos.atr_pct, pos.stop_loss, pos.trailing_stop,
                    pos.highest_price or pos.entry_price, pos.rlhf_row_id,
                ))
                conn.commit()
        logger.info(f"[Position] Mở vị thế {pos.ticker}: {pos.quantity} cổ @ {pos.entry_price:,.0f} VND")

    def update_trailing(self, ticker: str, current_price: float, atr_multiplier: float = 2.0):
        """Cập nhật trailing stop khi giá lên."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT highest_price, atr_pct, trailing_stop FROM live_positions "
                    "WHERE ticker=? AND status='OPEN'",
                    (ticker,)
                ).fetchone()
                if row is None:
                    return
                highest, atr_pct, current_trail = row
                new_highest = max(current_price, highest or 0.0)
                new_trail   = new_highest * (1.0 - atr_multiplier * atr_pct / 100.0)
                new_trail   = max(current_trail or 0.0, new_trail)
                conn.execute(
                    "UPDATE live_positions SET highest_price=?, trailing_stop=? "
                    "WHERE ticker=? AND status='OPEN'",
                    (new_highest, new_trail, ticker)
                )
                conn.commit()

    def get_open_position(self, ticker: str) -> Optional[Position]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT ticker, quantity, entry_price, entry_date, atr_pct, "
                "stop_loss, trailing_stop, highest_price, rlhf_row_id "
                "FROM live_positions WHERE ticker=? AND status='OPEN'",
                (ticker,)
            ).fetchone()
        if row is None:
            return None
        return Position(
            ticker=row[0], quantity=row[1], entry_price=row[2],
            entry_date=row[3], atr_pct=row[4], stop_loss=row[5],
            trailing_stop=row[6], highest_price=row[7], rlhf_row_id=row[8],
        )

    def get_all_open(self) -> list[Position]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT ticker, quantity, entry_price, entry_date, atr_pct, "
                "stop_loss, trailing_stop, highest_price, rlhf_row_id "
                "FROM live_positions WHERE status='OPEN'"
            ).fetchall()
        return [
            Position(
                ticker=r[0], quantity=r[1], entry_price=r[2],
                entry_date=r[3], atr_pct=r[4], stop_loss=r[5],
                trailing_stop=r[6], highest_price=r[7], rlhf_row_id=r[8],
            )
            for r in rows
        ]

    def close_position(self, ticker: str, close_price: float, reason: str = "SIGNAL"):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT entry_price FROM live_positions WHERE ticker=? AND status='OPEN'",
                    (ticker,)
                ).fetchone()
                realized_pct = 0.0
                if row:
                    ep = row[0]
                    realized_pct = (close_price - ep) / ep * 100.0 if ep > 0 else 0.0
                conn.execute(
                    "UPDATE live_positions SET status='CLOSED', closed_at=?, "
                    "close_price=?, realized_pct=? WHERE ticker=? AND status='OPEN'",
                    (datetime.now(timezone.utc).isoformat(), close_price, realized_pct, ticker)
                )
                conn.commit()
            logger.info(
                f"[Position] Đóng {ticker} @ {close_price:,.0f} "
                f"({realized_pct:+.2f}%) — {reason}"
            )
            return realized_pct

    def save_order(self, order: Order) -> int:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    INSERT INTO live_orders
                    (ticker, side, quantity, price, order_type, status,
                     order_id, filled_qty, filled_price, slippage_pct, error_msg, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    order.ticker, order.side.value, order.quantity, order.price,
                    order.order_type, order.status.value, order.order_id,
                    order.filled_qty, order.filled_price, order.slippage_pct,
                    order.error_msg, order.created_at,
                ))
                conn.commit()
                return cur.lastrowid


# ══════════════════════════════════════════════════════════════════════════════
# BASE BROKER — Interface chung cho tất cả broker
# ══════════════════════════════════════════════════════════════════════════════

class BaseBroker(ABC):
    """Abstract interface. Tất cả broker phải implement 3 method này."""

    @abstractmethod
    def get_price(self, ticker: str) -> float:
        """Lấy giá thị trường hiện tại (VND). 0.0 nếu lỗi."""

    @abstractmethod
    def send_order(self, order: Order) -> Order:
        """Gửi lệnh. Cập nhật order.status, order.order_id trước khi return."""

    @abstractmethod
    def get_order_status(self, order_id: str, ticker: str) -> OrderStatus:
        """Kiểm tra trạng thái lệnh đã gửi."""

    def cancel_order(self, order_id: str, ticker: str) -> bool:
        """Hủy lệnh. Mặc định trả False (chưa hỗ trợ)."""
        return False


# ══════════════════════════════════════════════════════════════════════════════
# VNDIRECT BROKER
# ══════════════════════════════════════════════════════════════════════════════

class VNDirectBroker(BaseBroker):
    """
    Kết nối VNDirect qua REST API chính thức.

    Xác thực: HMAC-SHA256 (consumer_id + timestamp + private_key).
    Tài liệu: https://developers.vndirect.com.vn/

    Cài đặt môi trường:
        VNDIRECT_ACCOUNT=...          # Số tài khoản (VD: 000C123456)
        VNDIRECT_CONSUMER_ID=...      # Consumer ID từ developer portal
        VNDIRECT_PRIVATE_KEY=...      # RSA private key (PEM, base64)
        VNDIRECT_ENV=prod             # prod | sandbox
    """

    SANDBOX_URL = "https://sandbox.vndirect.com.vn/services/v3.1"
    PROD_URL    = "https://api.vndirect.com.vn/v4"

    def __init__(
        self,
        account:      Optional[str] = None,
        consumer_id:  Optional[str] = None,
        private_key:  Optional[str] = None,
        env:          str = "sandbox",
    ):
        self.account     = account     or os.getenv("VNDIRECT_ACCOUNT", "")
        self.consumer_id = consumer_id or os.getenv("VNDIRECT_CONSUMER_ID", "")
        self.private_key = private_key or os.getenv("VNDIRECT_PRIVATE_KEY", "")
        self.env         = env
        self.base_url    = self.PROD_URL if env == "prod" else self.SANDBOX_URL
        self._token: Optional[str]   = None
        self._token_exp: float       = 0.0

        if not all([self.account, self.consumer_id, self.private_key]):
            logger.warning(
                "[VNDirect] Thiếu thông tin xác thực. "
                "Set VNDIRECT_ACCOUNT, VNDIRECT_CONSUMER_ID, VNDIRECT_PRIVATE_KEY"
            )

    def _sign(self, payload: str) -> str:
        """HMAC-SHA256 signature với private key."""
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding

            key_bytes = base64.b64decode(self.private_key)
            private   = serialization.load_pem_private_key(key_bytes, password=None)
            signature = private.sign(payload.encode(), padding.PKCS1v15(), hashes.SHA256())
            return base64.b64encode(signature).decode()
        except Exception as e:
            logger.error(f"[VNDirect] Lỗi ký xác thực: {e}")
            return ""

    def _get_token(self) -> str:
        """Lấy hoặc làm mới access token."""
        if self._token and time.time() < self._token_exp:
            return self._token

        ts      = str(int(time.time() * 1000))
        payload = f"{self.consumer_id}:{ts}"
        sig     = self._sign(payload)

        try:
            resp = requests.post(
                f"{self.base_url}/authenticate",
                json={
                    "consumerID":  self.consumer_id,
                    "timestamp":   ts,
                    "signature":   sig,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token     = data.get("data", {}).get("accessToken", "")
            self._token_exp = time.time() + 3600 - 60   # Token thường 1 giờ
            return self._token
        except Exception as e:
            logger.error(f"[VNDirect] Lấy token thất bại: {e}")
            return ""

    def _headers(self) -> dict:
        token = self._get_token()
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "x-api-key":     self.consumer_id,
        }

    def get_price(self, ticker: str) -> float:
        """Lấy giá trần/sàn/TC từ VNDirect market data."""
        try:
            resp = requests.get(
                f"{self.base_url}/market-depth/{ticker}",
                headers=self._headers(),
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            # Dùng lastPrice, fallback về matchedPrice
            price = data.get("lastPrice") or data.get("matchedPrice") or 0.0
            return float(price) * 1000  # VNDirect trả đơn vị nghìn đồng
        except Exception as e:
            logger.error(f"[VNDirect] get_price({ticker}): {e}")
            return 0.0

    def send_order(self, order: Order) -> Order:
        """
        Gửi lệnh qua VNDirect API.

        Lưu ý quan trọng:
        - Giá VNDirect API = VND / 1000 (làm tròn 100 đồng)
        - Quantity phải là bội số của 100 (lô HOSE)
        - LO: limit order, MP: market order (ATC/ATO)
        """
        try:
            payload = {
                "account":    self.account,
                "side":       order.side.value,
                "symbol":     order.ticker,
                "quantity":   order.quantity,
                "price":      round(order.price / 1000, 2),  # Chuyển sang nghìn đồng
                "orderType":  order.order_type,
            }
            resp = requests.post(
                f"{self.base_url}/orders",
                headers=self._headers(),
                json=payload,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            order.order_id = str(data.get("orderID", ""))
            order.status   = OrderStatus.PENDING
            logger.info(
                f"[VNDirect] Lệnh gửi thành công: "
                f"{order.side.value} {order.ticker} x{order.quantity} @ {order.price:,.0f}"
            )
        except Exception as e:
            order.status    = OrderStatus.REJECTED
            order.error_msg = str(e)
            logger.error(f"[VNDirect] Gửi lệnh thất bại: {e}")

        order.updated_at = datetime.now(timezone.utc).isoformat()
        return order

    def get_order_status(self, order_id: str, ticker: str) -> OrderStatus:
        try:
            resp = requests.get(
                f"{self.base_url}/orders/{order_id}",
                headers=self._headers(),
                timeout=8,
            )
            resp.raise_for_status()
            data   = resp.json().get("data", {})
            status = data.get("orderStatus", "").upper()
            mapping = {
                "FILLED":    OrderStatus.FILLED,
                "CANCELLED": OrderStatus.CANCELLED,
                "REJECTED":  OrderStatus.REJECTED,
                "PARTIAL":   OrderStatus.PARTIAL,
            }
            return mapping.get(status, OrderStatus.PENDING)
        except Exception as e:
            logger.error(f"[VNDirect] get_order_status({order_id}): {e}")
            return OrderStatus.PENDING

    def cancel_order(self, order_id: str, ticker: str) -> bool:
        try:
            resp = requests.delete(
                f"{self.base_url}/orders/{order_id}",
                headers=self._headers(),
                timeout=8,
            )
            return resp.status_code in (200, 204)
        except Exception as e:
            logger.error(f"[VNDirect] cancel_order({order_id}): {e}")
            return False


# ══════════════════════════════════════════════════════════════════════════════
# SSI BROKER (Fast Connect API)
# ══════════════════════════════════════════════════════════════════════════════

class SSIBroker(BaseBroker):
    """
    Kết nối SSI qua Fast Connect API (OAuth2 + WebSocket).
    Phù hợp nếu bạn có tài khoản SSI.

    Tài liệu: https://fc.ssi.com.vn/

    Cài đặt:
        SSI_CLIENT_ID=...
        SSI_CLIENT_SECRET=...
        SSI_ACCOUNT=...
        SSI_PIN=...
    """

    AUTH_URL  = "https://fc-api.ssi.com.vn/api/v2/Oauth/AccessToken"
    ORDER_URL = "https://fc-api.ssi.com.vn/api/v2/Order"
    PRICE_URL = "https://fc-data.ssi.com.vn/api/v2/Market"

    def __init__(self):
        self.client_id     = os.getenv("SSI_CLIENT_ID", "")
        self.client_secret = os.getenv("SSI_CLIENT_SECRET", "")
        self.account       = os.getenv("SSI_ACCOUNT", "")
        self.pin           = os.getenv("SSI_PIN", "")
        self._token:       Optional[str] = None
        self._token_exp:   float = 0.0

    def _get_token(self) -> str:
        if self._token and time.time() < self._token_exp:
            return self._token
        try:
            resp = requests.post(
                self.AUTH_URL,
                json={
                    "client_id":     self.client_id,
                    "client_secret": self.client_secret,
                    "grant_type":    "client_credentials",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token     = data.get("data", {}).get("access_token", "")
            self._token_exp = time.time() + int(data.get("data", {}).get("expires_in", 3600)) - 60
        except Exception as e:
            logger.error(f"[SSI] Lấy token thất bại: {e}")
        return self._token or ""

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type":  "application/json",
        }

    def get_price(self, ticker: str) -> float:
        try:
            resp = requests.get(
                f"{self.PRICE_URL}/Securities",
                headers=self._headers(),
                params={"symbol": ticker, "pageIndex": 1, "pageSize": 1},
                timeout=8,
            )
            resp.raise_for_status()
            items = resp.json().get("data", {}).get("list", [])
            if items:
                return float(items[0].get("LastPrice", 0.0)) * 1000
        except Exception as e:
            logger.error(f"[SSI] get_price({ticker}): {e}")
        return 0.0

    def send_order(self, order: Order) -> Order:
        try:
            resp = requests.post(
                f"{self.ORDER_URL}/NewOrder",
                headers=self._headers(),
                json={
                    "instrumentID":    order.ticker,
                    "market":          "VN",
                    "buySell":         order.side.value[0],   # B / S
                    "orderType":       order.order_type,
                    "channelID":       "WTS",
                    "account":         self.account,
                    "orderQty":        order.quantity,
                    "price":           round(order.price / 1000, 2),
                    "stopOrder":       False,
                    "lossOrder":       False,
                    "quickOrder":      False,
                    "inputTime":       str(int(time.time() * 1000)),
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            order.order_id = str(data.get("orderID", ""))
            order.status   = OrderStatus.PENDING
        except Exception as e:
            order.status    = OrderStatus.REJECTED
            order.error_msg = str(e)
            logger.error(f"[SSI] send_order({order.ticker}): {e}")
        order.updated_at = datetime.now(timezone.utc).isoformat()
        return order

    def get_order_status(self, order_id: str, ticker: str) -> OrderStatus:
        try:
            resp = requests.get(
                f"{self.ORDER_URL}/OrderHistory",
                headers=self._headers(),
                params={"account": self.account, "orderID": order_id},
                timeout=8,
            )
            resp.raise_for_status()
            items = resp.json().get("data", {}).get("orderList", [])
            if items:
                s = items[0].get("orderStatus", "").upper()
                return {
                    "F": OrderStatus.FILLED,
                    "C": OrderStatus.CANCELLED,
                    "R": OrderStatus.REJECTED,
                    "P": OrderStatus.PARTIAL,
                }.get(s, OrderStatus.PENDING)
        except Exception as e:
            logger.error(f"[SSI] get_order_status({order_id}): {e}")
        return OrderStatus.PENDING


# ══════════════════════════════════════════════════════════════════════════════
# PAPER BROKER — Giả lập giao dịch (không cần broker thật)
# ══════════════════════════════════════════════════════════════════════════════

class PaperBroker(BaseBroker):
    """
    Paper trading broker: giả lập khớp lệnh từ giá parquet local.
    Slippage = 0.2% mặc định (tương đương phí mua HOSE thực tế).

    Dùng để kiểm tra logic trước khi đấu nối broker thật.
    Cũng dùng trong test suite.
    """

    def __init__(
        self,
        capital:       float = 500_000_000,
        slippage_pct:  float = 0.2,
    ):
        self.capital      = capital
        self.slippage_pct = slippage_pct
        self._cash        = capital
        self._filled: dict[str, Order] = {}

    def get_price(self, ticker: str) -> float:
        """Lấy giá close mới nhất từ parquet local."""
        import pandas as pd
        paths = [
            os.path.join(DATA_DIR, "analyzed", "with_indicators", f"{ticker}_with_indicators.parquet"),
            os.path.join(DATA_DIR, "raw", "parquet", f"{ticker}_history.parquet"),
        ]
        for p in paths:
            if os.path.exists(p):
                try:
                    df = pd.read_parquet(p, engine="pyarrow").sort_values("time")
                    price = float(df["close"].iloc[-1])
                    # Parquet lưu giá VND/1000 (vnstock convention)
                    return price * 1000 if price < 1000 else price
                except Exception:
                    continue
        logger.warning(f"[Paper] Không có dữ liệu giá cho {ticker}")
        return 0.0

    def send_order(self, order: Order) -> Order:
        """Giả lập khớp lệnh ngay lập tức với slippage cố định."""
        market_price = self.get_price(order.ticker)
        if market_price <= 0:
            order.status    = OrderStatus.REJECTED
            order.error_msg = "Không có dữ liệu giá"
            return order

        # Áp slippage: mua trả thêm, bán nhận bớt
        slip = self.slippage_pct / 100.0
        if order.side == OrderSide.BUY:
            fill_price = market_price * (1.0 + slip)
            cost = fill_price * order.quantity
            if cost > self._cash:
                order.status    = OrderStatus.REJECTED
                order.error_msg = f"Không đủ tiền: cần {cost:,.0f}, có {self._cash:,.0f}"
                return order
            self._cash -= cost
        else:
            fill_price = market_price * (1.0 - slip)

        order.order_id   = f"PAPER-{int(time.time() * 1000)}"
        order.status      = OrderStatus.FILLED
        order.filled_qty  = order.quantity
        order.filled_price = fill_price
        order.slippage_pct = abs(fill_price - market_price) / market_price * 100.0
        order.updated_at   = datetime.now(timezone.utc).isoformat()

        self._filled[order.order_id] = order
        logger.info(
            f"[Paper] ✓ {order.side.value} {order.ticker} x{order.quantity} "
            f"@ {fill_price:,.0f} VND (slip={order.slippage_pct:.2f}%)"
        )
        return order

    def get_order_status(self, order_id: str, ticker: str) -> OrderStatus:
        order = self._filled.get(order_id)
        return order.status if order else OrderStatus.PENDING

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def portfolio_value(self) -> float:
        return self._cash  # Simplified — không tính open positions


# ══════════════════════════════════════════════════════════════════════════════
# LIVE TRADING ENGINE — Điểm tích hợp với phase4_orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class LiveTradingEngine:
    """
    Engine giao dịch thực: nhận signal từ phase4, quản lý vị thế + stop loss.

    Luồng cho mỗi ticker mỗi phiên:
      1. Lấy giá thị trường hiện tại
      2. Cập nhật trailing stop cho vị thế đang mở
      3. Kiểm tra stop-out (stop loss / trailing stop)
      4. Nếu chưa có vị thế: chạy multi-agent → BUY nếu signal đồng ý
      5. Nếu đang có vị thế: chạy multi-agent → SELL nếu tín hiệu đảo chiều

    Tích hợp với:
      - phase4_orchestrator.run_phase4() để lấy signal
      - rlhf_engine.FeedbackStore để ghi nhận kết quả thực tế
      - mlops_pipeline.MarketDriftDetector để skip giao dịch khi drift cao

    Sử dụng:
        engine = LiveTradingEngine(broker=PaperBroker())
        report = engine.run_signal("VNM")    # Gọi mỗi phiên sau 15:30
    """

    def __init__(
        self,
        broker:           Optional[BaseBroker] = None,
        risk_manager:     Optional[RiskManager] = None,
        position_tracker: Optional[PositionTracker] = None,
        account_value:    float = 500_000_000,
        skip_on_drift:    bool  = True,
        min_confidence:   float = 0.60,
    ):
        # Auto-detect broker từ env nếu không truyền vào
        if broker is None:
            broker_type = os.getenv("BROKER", "paper").lower()
            if broker_type == "vndirect":
                broker = VNDirectBroker()
            elif broker_type == "ssi":
                broker = SSIBroker()
            else:
                broker = PaperBroker(capital=float(os.getenv("PAPER_CAPITAL", account_value)))

        self.broker           = broker
        self.risk             = risk_manager or RiskManager(account_value_vnd=account_value)
        self.tracker          = position_tracker or PositionTracker()
        self.skip_on_drift    = skip_on_drift
        self.min_confidence   = min_confidence

    # ── Bước 1: Cập nhật trailing stop ──────────────────────────────────────

    def _update_trailing_stops(self, ticker: str, current_price: float):
        self.tracker.update_trailing(ticker, current_price, atr_multiplier=2.0)

    # ── Bước 2: Kiểm tra stop-out ────────────────────────────────────────────

    def _check_stop_out(self, ticker: str, current_price: float) -> Optional[str]:
        """Trả về lý do nếu cần đóng vị thế, None nếu không."""
        pos = self.tracker.get_open_position(ticker)
        if pos is None:
            return None
        triggered, reason = pos.should_stop_out(current_price)
        return reason if triggered else None

    # ── Bước 3: Gửi lệnh BUY ─────────────────────────────────────────────────

    def _execute_buy(
        self,
        ticker:      str,
        price:       float,
        atr_pct:     float,
        confidence:  float,
        rlhf_row_id: Optional[int] = None,
    ) -> Optional[Order]:
        """
        Thực hiện lệnh mua sau khi đã qua RiskManager.
        Thiết lập stop loss và trailing stop ngay sau khi khớp.
        """
        qty = self.risk.compute_position_size(price, atr_pct, confidence)
        approved, reason = self.risk.approve_order(price, qty, atr_pct)

        if not approved:
            logger.warning(f"[Engine] BUY {ticker} bị từ chối: {reason}")
            return None

        order = Order(
            ticker     = ticker,
            side       = OrderSide.BUY,
            quantity   = qty,
            price      = price,
            order_type = "LO",
        )

        order = self.broker.send_order(order)
        self.tracker.save_order(order)

        if order.status in (OrderStatus.FILLED, OrderStatus.PENDING):
            fill_price = order.filled_price or price
            stop_loss, trailing_stop = self.risk.compute_stop_levels(fill_price, atr_pct)
            pos = Position(
                ticker        = ticker,
                quantity      = qty,
                entry_price   = fill_price,
                entry_date    = datetime.now().strftime("%Y-%m-%d"),
                atr_pct       = atr_pct,
                stop_loss     = stop_loss,
                trailing_stop = trailing_stop,
                highest_price = fill_price,
                rlhf_row_id   = rlhf_row_id,
            )
            self.tracker.open_position(pos)
            logger.info(
                f"[Engine] ✓ BUY {ticker}: qty={qty}, "
                f"stop_loss={stop_loss:,.0f}, trailing={trailing_stop:,.0f}"
            )
        return order

    # ── Bước 4: Gửi lệnh SELL ────────────────────────────────────────────────

    def _execute_sell(
        self,
        ticker:  str,
        price:   float,
        reason:  str = "SIGNAL",
    ) -> Optional[Order]:
        """Đóng vị thế hiện tại."""
        pos = self.tracker.get_open_position(ticker)
        if pos is None:
            logger.debug(f"[Engine] Không có vị thế mở cho {ticker}")
            return None

        order = Order(
            ticker     = ticker,
            side       = OrderSide.SELL,
            quantity   = pos.quantity,
            price      = price,
            order_type = "LO",
        )
        order = self.broker.send_order(order)
        self.tracker.save_order(order)

        if order.status in (OrderStatus.FILLED, OrderStatus.PENDING):
            fill_price   = order.filled_price or price
            realized_pct = self.tracker.close_position(ticker, fill_price, reason)

            # Feed actual return vào RLHF
            if pos.rlhf_row_id:
                try:
                    from rlhf_engine import FeedbackStore
                    store = FeedbackStore()
                    store.update_outcome(pos.rlhf_row_id, actual_return_pct=realized_pct)
                    logger.info(
                        f"[RLHF] Cập nhật outcome {pos.rlhf_row_id}: "
                        f"{realized_pct:+.2f}%"
                    )
                except Exception as e:
                    logger.warning(f"[RLHF] Không cập nhật được outcome: {e}")

        return order

    # ── Điểm tích hợp chính ──────────────────────────────────────────────────

    def run_signal(
        self,
        ticker:  str,
        use_llm: bool = False,
    ) -> dict:
        """
        Chạy một chu kỳ giao dịch đầy đủ cho một ticker.

        Gọi sau mỗi phiên giao dịch (VD: 15:35 mỗi ngày qua pipeline_manager).
        Tự động:
          - Cập nhật trailing stop nếu đang giữ vị thế
          - Stop-out nếu giá chạm stop loss
          - Mua khi signal BUY + confidence cao
          - Bán khi signal SELL hoặc đạt profit target

        Args:
            ticker:  Mã cổ phiếu
            use_llm: Bật LLM phân tích (cần Ollama + RAM)

        Returns:
            dict {ticker, action, signal, price, order, drift_info, ...}
        """
        report = {
            "ticker":    ticker,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action":    "NONE",
            "signal":    None,
            "price":     0.0,
            "order":     None,
            "drift_info": None,
            "error":     None,
        }

        try:
            # 1. Lấy giá thị trường
            current_price = self.broker.get_price(ticker)
            if current_price <= 0:
                # Fallback về giá từ parquet
                from data_collector import get_stock_history
                import pandas as pd
                df = pd.read_parquet(
                    os.path.join(DATA_DIR, "raw", "parquet", f"{ticker}_history.parquet"),
                    engine="pyarrow"
                )
                current_price = float(df["close"].iloc[-1])
                if current_price < 1000:
                    current_price *= 1000

            report["price"] = current_price

            # 2. Kiểm tra drift — bỏ qua giao dịch nếu drift cao
            if self.skip_on_drift:
                try:
                    from mlops_pipeline import MarketDriftDetector
                    detector = MarketDriftDetector(ticker=ticker)
                    drift    = detector.check_drift()
                    report["drift_info"] = drift
                    if drift.get("drift_detected"):
                        logger.warning(
                            f"[Engine] {ticker} drift detected ({drift['drift_reason']}) "
                            f"— bỏ qua giao dịch phiên này"
                        )
                        report["action"] = "SKIP_DRIFT"
                        report["signal"] = "HOLD"
                        # Vẫn cập nhật trailing stop để bảo vệ vị thế đang mở
                        self._update_trailing_stops(ticker, current_price)
                        stop_reason = self._check_stop_out(ticker, current_price)
                        if stop_reason:
                            order = self._execute_sell(ticker, current_price, stop_reason)
                            report["action"] = f"STOP_OUT (drift + {stop_reason})"
                            report["order"]  = asdict(order) if order else None
                        return report
                except Exception as drift_err:
                    logger.debug(f"[Engine] Drift check lỗi: {drift_err}")

            # 3. Cập nhật trailing stop cho vị thế đang mở
            self._update_trailing_stops(ticker, current_price)

            # 4. Kiểm tra stop-out TRƯỚC KHI chạy signal
            stop_reason = self._check_stop_out(ticker, current_price)
            if stop_reason:
                order = self._execute_sell(ticker, current_price, stop_reason)
                report["action"] = f"STOP_OUT ({stop_reason})"
                report["order"]  = asdict(order) if order else None
                return report

            # 5. Chạy phase4 để lấy signal
            from phase4_orchestrator import run_phase4
            phase4_report = run_phase4(
                ticker  = ticker,
                mode    = "analysis_only",
                use_llm = use_llm,
            )
            analysis = phase4_report.get("analysis", {})
            if not analysis:
                report["error"] = "Không có kết quả phân tích từ phase4"
                return report

            signal     = analysis.get("final_signal", "HOLD")
            confidence = float(analysis.get("forecast_confidence") or 0.0)
            atr_pct    = float(analysis.get("atr_pct") or 2.0)
            rlhf_row   = phase4_report.get("steps", {}).get("record_signal", {}).get("row_id")

            report["signal"]     = signal
            report["confidence"] = confidence

            existing_pos = self.tracker.get_open_position(ticker)

            # 6a. Chưa có vị thế — xem xét MUA
            if existing_pos is None:
                if signal == "BUY" and confidence >= self.min_confidence:
                    order = self._execute_buy(
                        ticker      = ticker,
                        price       = current_price,
                        atr_pct     = atr_pct,
                        confidence  = confidence,
                        rlhf_row_id = rlhf_row,
                    )
                    report["action"] = "BUY" if order and order.status != OrderStatus.REJECTED else "BUY_REJECTED"
                    report["order"]  = asdict(order) if order else None
                else:
                    report["action"] = "HOLD_NO_POSITION"

            # 6b. Đang có vị thế — xem xét BÁN
            else:
                should_sell = (
                    signal == "SELL"
                    or (signal == "HOLD" and confidence < 0.40)  # Confidence sụt → thoát
                )
                if should_sell:
                    order = self._execute_sell(ticker, current_price, f"SIGNAL_{signal}")
                    report["action"] = "SELL"
                    report["order"]  = asdict(order) if order else None
                else:
                    unrealized = existing_pos.unrealized_pct(current_price)
                    report["action"]       = "HOLD_POSITION"
                    report["unrealized_pct"] = round(unrealized, 2)

        except Exception as e:
            logger.error(f"[Engine] run_signal({ticker}) lỗi: {e}", exc_info=True)
            report["error"] = str(e)

        return report

    def scan_stops_all_positions(self) -> list[dict]:
        """
        Quét stop loss / trailing stop toàn bộ vị thế đang mở.
        Gọi mỗi 5 phút trong giờ giao dịch (9:00–15:00).

        Returns:
            List các vị thế đã bị stop out.
        """
        stopped = []
        for pos in self.tracker.get_all_open():
            price = self.broker.get_price(pos.ticker)
            if price <= 0:
                continue
            self.tracker.update_trailing(pos.ticker, price)
            triggered, reason = pos.should_stop_out(price)
            if triggered:
                order = self._execute_sell(pos.ticker, price, reason)
                stopped.append({
                    "ticker": pos.ticker,
                    "price":  price,
                    "reason": reason,
                    "order":  asdict(order) if order else None,
                })
                logger.warning(f"[StopScan] {pos.ticker} STOPPED @ {price:,.0f}: {reason}")
        return stopped


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY — Tạo engine từ env
# ══════════════════════════════════════════════════════════════════════════════

def create_engine(
    account_value:  float = None,
    skip_on_drift:  bool  = True,
    min_confidence: float = 0.60,
) -> LiveTradingEngine:
    """
    Factory function: tạo LiveTradingEngine từ biến môi trường.

    Thiết lập .env:
        BROKER=paper              # paper | vndirect | ssi
        PAPER_CAPITAL=500000000
        MAX_POSITION_VND=50000000
        MAX_DAILY_ORDERS=5
    """
    capital = account_value or float(os.getenv("PAPER_CAPITAL", 500_000_000))
    return LiveTradingEngine(
        account_value  = capital,
        skip_on_drift  = skip_on_drift,
        min_confidence = min_confidence,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TÍCH HỢP VỚI pipeline_manager.py
# ══════════════════════════════════════════════════════════════════════════════

def job_live_trading(watchlist: list[str] = None, use_llm: bool = False):
    """
    Job định kỳ: chạy sau 15:35 mỗi ngày giao dịch (sau khi có dữ liệu EOD).
    Thêm vào pipeline_manager.py:

        from live_broker import job_live_trading, job_scan_stops
        scheduler.add_job(job_live_trading, 'cron',
                          day_of_week='mon-fri', hour=15, minute=35)
        scheduler.add_job(job_scan_stops,   'cron',
                          day_of_week='mon-fri', hour='9-14', minute='*/5')
    """
    if watchlist is None:
        try:
            from pipeline import WATCHLIST
        except ImportError:
            WATCHLIST = ["VNM", "HPG", "VCB", "FPT", "TCB"]
        watchlist = WATCHLIST

    engine = create_engine()
    results = []
    for ticker in watchlist:
        try:
            r = engine.run_signal(ticker, use_llm=use_llm)
            results.append(r)
            logger.info(
                f"[LiveTrading] {ticker}: action={r['action']}, "
                f"signal={r['signal']}, price={r['price']:,.0f}"
            )
        except Exception as e:
            logger.error(f"[LiveTrading] {ticker} lỗi: {e}")

    # Lưu báo cáo phiên
    report_path = os.path.join(DATA_DIR, "reports", "json", "live_session_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "session_date": datetime.now().strftime("%Y-%m-%d"),
            "results": results,
        }, f, indent=2, ensure_ascii=False, default=str)

    return results


def job_scan_stops(watchlist: list[str] = None):
    """
    Job 5 phút: quét stop loss trong giờ giao dịch.
    Bảo vệ danh mục khỏi sụt mạnh trong phiên.
    """
    engine = create_engine()
    stopped = engine.scan_stops_all_positions()
    if stopped:
        logger.warning(f"[StopScan] {len(stopped)} vị thế bị stop out: {[s['ticker'] for s in stopped]}")
    return stopped


# ══════════════════════════════════════════════════════════════════════════════
# CHẠY THỬ
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print("\n" + "═"*60)
    print("  LIVE BROKER — Paper Trading Test")
    print("═"*60)

    # Test với Paper Broker (an toàn, không cần API key)
    engine = LiveTradingEngine(
        broker          = PaperBroker(capital=500_000_000),
        account_value   = 500_000_000,
        min_confidence  = 0.55,
        skip_on_drift   = True,
    )

    # Test RiskManager
    print("\n[1] RiskManager")
    rm = RiskManager(account_value_vnd=500_000_000)
    qty = rm.compute_position_size(price=75_000, atr_pct=2.5, confidence=0.72)
    sl, ts = rm.compute_stop_levels(entry_price=75_000, atr_pct=2.5)
    slip = rm.estimate_slippage(price=75_000, quantity=qty, avg_vol_20d=800_000)
    print(f"  Qty: {qty:,} cổ | Stop Loss: {sl:,.0f} | Trailing: {ts:,.0f} | Slip: {slip:.3f}%")

    # Test Paper get_price
    print("\n[2] PaperBroker.get_price")
    broker = PaperBroker()
    for t in ["VNM", "HPG", "VCB"]:
        price = broker.get_price(t)
        print(f"  {t}: {price:,.0f} VND" if price > 0 else f"  {t}: (chưa có dữ liệu local)")

    # Test PositionTracker
    print("\n[3] PositionTracker (SQLite)")
    tracker = PositionTracker()
    test_pos = Position(
        ticker="VNM_TEST", quantity=500, entry_price=75_000,
        entry_date="2026-04-13", atr_pct=2.5,
        stop_loss=71_812, trailing_stop=71_250, highest_price=75_000,
    )
    tracker.open_position(test_pos)
    tracker.update_trailing("VNM_TEST", current_price=78_000)
    pos = tracker.get_open_position("VNM_TEST")
    if pos:
        print(f"  Trailing stop sau khi giá lên 78k: {pos.trailing_stop:,.0f}")
        unrealized = pos.unrealized_pct(78_000)
        print(f"  Unrealized P&L: {unrealized:+.2f}%")

    # Test run_signal (full pipeline, cần data)
    print("\n[4] LiveTradingEngine.run_signal (VNM)")
    try:
        report = engine.run_signal("VNM")
        print(f"  Action: {report['action']}")
        print(f"  Signal: {report['signal']}")
        print(f"  Price:  {report['price']:,.0f} VND" if report["price"] else "  Price: N/A")
        if report.get("order"):
            o = report["order"]
            print(f"  Order:  {o.get('side')} x{o.get('quantity')} @ {o.get('filled_price', 0):,.0f}")
    except Exception as e:
        print(f"  ⚠ run_signal lỗi (cần dữ liệu): {e}")

    print("\n✓ Kiểm tra xong. Xem log để biết chi tiết.")
    print("═"*60)