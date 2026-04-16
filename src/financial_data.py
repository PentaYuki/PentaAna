"""
financial_data.py — Thu thập chỉ tiêu tài chính cơ bản cho cổ phiếu VN.

Bao gồm:
  - Định giá: P/E, P/B, EPS, Vốn hóa, Giá sổ sách, SLCP lưu hành
  - Hiệu quả hoạt động ngân hàng: NIM, YEA, CoF, CIR, ROE, ROA
  - Chất lượng tài sản: Tỷ lệ nợ xấu (NPL), Dự phòng/Nợ xấu
  - An toàn vốn: LDR, CASA, Tlệ TS/VCSH
  - Tăng trưởng: Thu nhập lãi thuần, Cho vay KH, Tiền gửi KH, Tổng TS

Nguồn: vnstock (TCBS) → yfinance (fallback)
Cache: data/reports/json/financials_{ticker}.json (TTL 6 giờ)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).parent.parent.resolve()
CACHE_DIR = BASE_DIR / "data" / "reports" / "json"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL = 6 * 3600   # 6 giờ

# ─── Nhóm ngân hàng VN ───────────────────────────────────────────────────────
BANKING_TICKERS = {
    "VCB", "BID", "CTG", "MBB", "TCB", "ACB", "VPB", "STB",
    "HDB", "TPB", "MSB", "OCB", "SHB", "LPB", "NAB", "VIB",
}


# ══════════════════════════════════════════════════════════════════════════════
# CACHE
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"financials_{ticker.upper()}.json"


def _load_cache(ticker: str) -> dict | None:
    p = _cache_path(ticker)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        fetched_at = data.get("_fetched_at", 0)
        if time.time() - fetched_at < CACHE_TTL:
            return data
    except Exception:
        pass
    return None


def _save_cache(ticker: str, data: dict):
    data["_fetched_at"] = time.time()
    try:
        with open(_cache_path(ticker), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logger.warning("[financial_data] Không lưu được cache: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# VNSTOCK — nguồn chính
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_via_vnstock(ticker: str) -> dict:
    """
    Thu thập chỉ tiêu tài chính qua vnstock (TCBS source).
    Trả về dict chuẩn hóa. Có thể raise Exception nếu lỗi.
    """
    from vnstock import Vnstock
    stock = Vnstock().stock(symbol=ticker, source="TCBS")

    result: dict[str, Any] = {"ticker": ticker, "source": "vnstock/TCBS"}

    # ── Giá & Định giá ────────────────────────────────────────────────────────
    try:
        overview = stock.company.overview()
        if isinstance(overview, pd.DataFrame) and not overview.empty:
            row = overview.iloc[0]
            result["exchange"]       = str(row.get("exchange", "HOSE"))
            result["industry"]       = str(row.get("industryName", ""))
            result["company_name"]   = str(row.get("companyName", ticker))
            result["short_name"]     = str(row.get("shortName", ticker))
    except Exception as e:
        logger.debug("[financial_data] overview lỗi %s: %s", ticker, e)

    # ── Chỉ tiêu tài chính (Financial Ratios) ────────────────────────────────
    try:
        ratios_df = stock.finance.ratio(period="quarter", lang="en", dropna=True)
        # TCBS trả về nhiều cột – lấy quý gần nhất
        if isinstance(ratios_df, pd.DataFrame) and not ratios_df.empty:
            ratios_df = ratios_df.sort_index(ascending=False)
            latest = ratios_df.iloc[0]
            prev   = ratios_df.iloc[1] if len(ratios_df) > 1 else latest

            def _pct_yoy(col: str) -> float | None:
                """Tăng trưởng YoY % so với 4 quý trước (nếu có 5+ hàng)."""
                try:
                    if len(ratios_df) > 4:
                        v_now  = float(ratios_df.iloc[0][col])
                        v_prev = float(ratios_df.iloc[4][col])
                        if v_prev and v_prev != 0:
                            return round((v_now - v_prev) / abs(v_prev) * 100, 2)
                except Exception:
                    pass
                return None

            # Định giá
            result["pe"]         = _sf(latest.get("priceToEarning"))
            result["pb"]         = _sf(latest.get("priceToBook"))
            result["eps"]        = _sf(latest.get("eps"))
            result["book_value"] = _sf(latest.get("bookValuePerShare"))
            result["market_cap"] = _sf(latest.get("marketCap"))       # tỷ VND
            result["shares_outstanding"] = _sf(latest.get("sharesOutstanding"))  # triệu cp

            # Hiệu quả — ngân hàng
            result["nim"]        = _sf(latest.get("netInterestMargin"))   # %
            result["roe"]        = _sf(latest.get("roe"))                 # %
            result["roa"]        = _sf(latest.get("roa"))                 # %
            result["cir"]        = _sf(latest.get("costToIncome"))        # %
            result["net_profit_margin"] = _sf(latest.get("netProfitMargin"))  # %

            # Chất lượng tài sản
            result["npl_ratio"]  = _sf(latest.get("badDebtRatio"))       # %
            result["coverage"]   = _sf(latest.get("provisionCoverage"))  # %
            result["casa_ratio"] = _sf(latest.get("casaRatio"))          # %
            result["ldr"]        = _sf(latest.get("loanToDeposit"))      # %

            # An toàn vốn
            result["leverage"]   = _sf(latest.get("totalAssetsToEquity"))

            # Tăng trưởng YoY
            result["yoy_nii"]         = _pct_yoy("netInterestIncome")
            result["yoy_fee"]         = _pct_yoy("netFeeIncome")
            result["yoy_pat"]         = _pct_yoy("netProfit")
            result["yoy_total_assets"]= _pct_yoy("totalAssets")
            result["yoy_loans"]       = _pct_yoy("customerLoans")
            result["yoy_deposits"]    = _pct_yoy("customerDeposits")

            # Chi phí vốn / YEA (nếu TCBS cung cấp)
            result["yea"]           = _sf(latest.get("yieldOnEarningAssets"))
            result["cof"]           = _sf(latest.get("costOfFunds"))

            result["period"]     = str(ratios_df.index[0]) if hasattr(ratios_df.index, "__iter__") else ""
    except Exception as e:
        logger.debug("[financial_data] ratios lỗi %s: %s", ticker, e)

    # ── Income Statement — tổng thu nhập lãi thuần, lợi nhuận ───────────────
    try:
        income_df = stock.finance.income_statement(period="quarter", lang="en", dropna=True)
        if isinstance(income_df, pd.DataFrame) and not income_df.empty:
            income_df = income_df.sort_index(ascending=False)
            row = income_df.iloc[0]
            result["net_interest_income"] = _sf(row.get("netInterestIncome"))  # tỷ
            result["net_fee_income"]      = _sf(row.get("netFeeAndCommissionIncome"))
            result["operating_income"]    = _sf(row.get("totalOperatingIncome"))
            result["net_profit"]          = _sf(row.get("netProfit"))
            result["pat_parent"]          = _sf(row.get("netProfitParentCompany"))
    except Exception as e:
        logger.debug("[financial_data] income_statement lỗi %s: %s", ticker, e)

    # ── Balance Sheet ────────────────────────────────────────────────────────
    try:
        bs_df = stock.finance.balance_sheet(period="quarter", lang="en", dropna=True)
        if isinstance(bs_df, pd.DataFrame) and not bs_df.empty:
            bs_df = bs_df.sort_index(ascending=False)
            row = bs_df.iloc[0]
            result["total_assets"]     = _sf(row.get("totalAssets"))       # tỷ
            result["customer_loans"]   = _sf(row.get("customerLoans"))
            result["customer_deposits"]= _sf(row.get("customerDeposits"))
            result["equity"]           = _sf(row.get("equity"))
            result["bad_debt"]         = _sf(row.get("badDebt"))
            result["provisions"]       = _sf(row.get("provisions"))
    except Exception as e:
        logger.debug("[financial_data] balance_sheet lỗi %s: %s", ticker, e)

    result["fetched_at"] = datetime.now(timezone.utc).isoformat()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# YFINANCE — fallback
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_via_yfinance(ticker: str) -> dict:
    """Fallback: lấy ratios từ yfinance (.VN suffix)."""
    import yfinance as yf
    sym = f"{ticker}.VN"
    info = yf.Ticker(sym).info
    if not info:
        return {"ticker": ticker, "source": "yfinance", "error": "empty info"}
    result = {
        "ticker":          ticker,
        "source":          "yfinance",
        "pe":              _sf(info.get("trailingPE")),
        "pb":              _sf(info.get("priceToBook")),
        "eps":             _sf(info.get("trailingEps")),
        "market_cap":      _sf(info.get("marketCap")),
        "book_value":      _sf(info.get("bookValue")),
        "roe":             _sf(info.get("returnOnEquity")) and round(info.get("returnOnEquity", 0) * 100, 2),
        "roa":             _sf(info.get("returnOnAssets")) and round(info.get("returnOnAssets", 0) * 100, 2),
        "shares_outstanding": _sf(info.get("sharesOutstanding")),
        "fetched_at":      datetime.now(timezone.utc).isoformat(),
    }
    return result


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_financial_data(ticker: str, force_refresh: bool = False) -> dict:
    """
    Lấy toàn bộ chỉ tiêu tài chính cơ bản cho một mã cổ phiếu.

    Args:
        ticker: Mã cổ phiếu (VD: "VCB", "HPG")
        force_refresh: Bỏ qua cache, lấy mới từ API

    Returns:
        dict với đầy đủ chỉ tiêu. Luôn trả về dict (không raise).
    """
    ticker = ticker.upper()

    if not force_refresh:
        cached = _load_cache(ticker)
        if cached:
            logger.debug("[financial_data] Dùng cache: %s", ticker)
            return cached

    # Thử vnstock trước
    data: dict = {}
    try:
        data = _fetch_via_vnstock(ticker)
        logger.info("[financial_data] ✅ vnstock OK: %s", ticker)
    except Exception as e1:
        logger.warning("[financial_data] vnstock thất bại %s: %s → thử yfinance", ticker, e1)
        try:
            data = _fetch_via_yfinance(ticker)
            logger.info("[financial_data] ✅ yfinance OK: %s", ticker)
        except Exception as e2:
            logger.error("[financial_data] Cả 2 nguồn thất bại %s: vnstock=%s, yf=%s", ticker, e1, e2)
            data = {"ticker": ticker, "source": "none", "error": str(e2)}

    _save_cache(ticker, data)
    return data


def get_financial_summary(ticker: str) -> dict:
    """
    Trả về một dict gọn gàng để hiển thị trên dashboard.
    Tất cả giá trị đều là float hoặc None.
    """
    d = get_financial_data(ticker)
    is_bank = ticker.upper() in BANKING_TICKERS

    summary = {
        # ── Thông tin cơ bản ────────────────────────────────────────────────
        "ticker":       ticker.upper(),
        "company_name": d.get("company_name", ticker),
        "exchange":     d.get("exchange", "HOSE"),
        "industry":     d.get("industry", ""),
        "source":       d.get("source", ""),
        "period":       d.get("period", ""),

        # ── Định giá ────────────────────────────────────────────────────────
        "pe":           d.get("pe"),
        "pb":           d.get("pb"),
        "eps":          d.get("eps"),
        "book_value":   d.get("book_value"),
        "market_cap":   d.get("market_cap"),
        "shares_outstanding": d.get("shares_outstanding"),

        # ── Hiệu quả ────────────────────────────────────────────────────────
        "roe":          d.get("roe"),
        "roa":          d.get("roa"),
        "net_profit_margin": d.get("net_profit_margin"),

        # ── Chỉ tiêu ngân hàng ──────────────────────────────────────────────
        "nim":          d.get("nim") if is_bank else None,
        "yea":          d.get("yea") if is_bank else None,
        "cof":          d.get("cof") if is_bank else None,
        "cir":          d.get("cir"),
        "npl_ratio":    d.get("npl_ratio") if is_bank else None,
        "coverage":     d.get("coverage") if is_bank else None,
        "casa_ratio":   d.get("casa_ratio") if is_bank else None,
        "ldr":          d.get("ldr") if is_bank else None,
        "leverage":     d.get("leverage"),

        # ── Tăng trưởng YoY ────────────────────────────────────────────────
        "yoy_nii":          d.get("yoy_nii"),
        "yoy_fee":          d.get("yoy_fee"),
        "yoy_pat":          d.get("yoy_pat"),
        "yoy_total_assets": d.get("yoy_total_assets"),
        "yoy_loans":        d.get("yoy_loans"),
        "yoy_deposits":     d.get("yoy_deposits"),
    }
    return summary


def score_fundamentals(ticker: str) -> dict:
    """
    Chấm điểm cơ bản từ -1.0 đến +1.0 để tích hợp vào AI agents.

    Logic:
      - ROE cao (>15%) → điểm dương
      - NPL thấp (<1%) → điểm dương (ngân hàng)
      - P/E thấp (<12) → định giá hấp dẫn
      - NIM cao (>3%) → biên lãi tốt
      - Coverage cao (>200%) → dự phòng tốt
      - Tăng trưởng lợi nhuận YoY dương → điểm cộng

    Returns:
        {"fundamental_score": float [-1,1], "signals": dict, "rating": str}
    """
    d = get_financial_summary(ticker)
    score = 0.0
    signals: dict[str, str] = {}
    is_bank = ticker.upper() in BANKING_TICKERS

    # ROE
    roe = d.get("roe")
    if roe is not None:
        if roe > 20:
            score += 0.3; signals["roe"] = f"Xuất sắc ({roe:.1f}%)"
        elif roe > 15:
            score += 0.2; signals["roe"] = f"Tốt ({roe:.1f}%)"
        elif roe > 10:
            score += 0.1; signals["roe"] = f"Trung bình ({roe:.1f}%)"
        else:
            score -= 0.1; signals["roe"] = f"Yếu ({roe:.1f}%)"

    # P/E định giá
    pe = d.get("pe")
    if pe and pe > 0:
        if pe < 10:
            score += 0.25; signals["pe"] = f"Rất hấp dẫn (P/E={pe:.1f})"
        elif pe < 15:
            score += 0.15; signals["pe"] = f"Hấp dẫn (P/E={pe:.1f})"
        elif pe < 20:
            score += 0.05; signals["pe"] = f"Hợp lý (P/E={pe:.1f})"
        else:
            score -= 0.1;  signals["pe"] = f"Đắt (P/E={pe:.1f})"

    if is_bank:
        # NIM
        nim = d.get("nim")
        if nim is not None:
            if nim > 4:
                score += 0.2; signals["nim"] = f"NIM xuất sắc ({nim:.2f}%)"
            elif nim > 3:
                score += 0.1; signals["nim"] = f"NIM tốt ({nim:.2f}%)"
            elif nim > 2:
                score += 0.0; signals["nim"] = f"NIM trung bình ({nim:.2f}%)"
            else:
                score -= 0.1; signals["nim"] = f"NIM yếu ({nim:.2f}%)"

        # NPL
        npl = d.get("npl_ratio")
        if npl is not None:
            if npl < 0.5:
                score += 0.2; signals["npl"] = f"Chất lượng TS xuất sắc (NPL={npl:.2f}%)"
            elif npl < 1.0:
                score += 0.1; signals["npl"] = f"Chất lượng TS tốt (NPL={npl:.2f}%)"
            elif npl < 2.0:
                score -= 0.1; signals["npl"] = f"NPL cần theo dõi ({npl:.2f}%)"
            else:
                score -= 0.25; signals["npl"] = f"NPL cao rủi ro ({npl:.2f}%)"

        # Coverage
        cov = d.get("coverage")
        if cov is not None:
            if cov > 200:
                score += 0.15; signals["coverage"] = f"Dự phòng vững ({cov:.0f}%)"
            elif cov > 100:
                score += 0.05; signals["coverage"] = f"Dự phòng đủ ({cov:.0f}%)"
            else:
                score -= 0.1; signals["coverage"] = f"Dự phòng thấp ({cov:.0f}%)"

    # Tăng trưởng lợi nhuận
    yoy_pat = d.get("yoy_pat")
    if yoy_pat is not None:
        if yoy_pat > 20:
            score += 0.15; signals["growth"] = f"Tăng trưởng mạnh (+{yoy_pat:.1f}% YoY)"
        elif yoy_pat > 5:
            score += 0.05; signals["growth"] = f"Tăng trưởng ổn (+{yoy_pat:.1f}% YoY)"
        elif yoy_pat < -10:
            score -= 0.15; signals["growth"] = f"Suy giảm ({yoy_pat:.1f}% YoY)"

    score = round(max(-1.0, min(1.0, score)), 4)

    if score > 0.4:
        rating = "Rất hấp dẫn"
    elif score > 0.2:
        rating = "Hấp dẫn"
    elif score > -0.1:
        rating = "Trung bình"
    elif score > -0.3:
        rating = "Kém hấp dẫn"
    else:
        rating = "Không hấp dẫn"

    return {
        "fundamental_score": score,
        "rating": rating,
        "signals": signals,
        "data": d,
    }


# ── helper ───────────────────────────────────────────────────────────────────
def _sf(v: Any) -> float | None:
    """Safe float conversion."""
    try:
        f = float(v)
        return None if (f != f) else round(f, 4)   # NaN check
    except Exception:
        return None


# ── CLI test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "VCB"
    print(f"\n{'='*60}")
    print(f"  Chỉ tiêu tài chính: {ticker}")
    print(f"{'='*60}")
    summary = get_financial_summary(ticker)
    for k, v in summary.items():
        if v is not None:
            print(f"  {k:30s}: {v}")
    print()
    scoring = score_fundamentals(ticker)
    print(f"  Điểm cơ bản: {scoring['fundamental_score']:+.3f} → {scoring['rating']}")
    for k, v in scoring["signals"].items():
        print(f"  [{k}] {v}")
