"""
system_test.py — Kiểm thử toàn hệ thống Stock-AI Phase 2.5

Chạy lần lượt từng module, ghi số liệu ra:
  data/reports/json/system_test_report.json  — Kết quả kiểm thử chi tiết
  data/reports/charts/kronos_loss_curve.png   — Biểu đồ training loss (từ trainer)
  data/reports/charts/forecast_comparison.png — So sánh forecast trước / sau fine-tune
  data/reports/charts/VNM_forecast.png        — Forecast chart sau fine-tune
"""

import json
import os
import sqlite3
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch

# Đưa thư mục src vào path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR     = os.path.join(BASE_DIR, "data")
RAW_PQ_DIR   = os.path.join(DATA_DIR, "raw", "parquet")
CHARTS_DIR   = os.path.join(DATA_DIR, "reports", "charts")
REPORT_PATH  = os.path.join(DATA_DIR, "reports", "json", "system_test_report.json")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ram_gb() -> float:
    return psutil.virtual_memory().available / (1024 ** 3)


def _section(title: str):
    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)


def _ok(msg: str):
    print(f"  ✓ {msg}")


def _fail(msg: str):
    print(f"  ✗ {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_memory_guard() -> dict:
    _section("TEST 1 — memory_guard.py")
    result = {"module": "memory_guard", "passed": False, "details": {}}
    try:
        from memory_guard import available_ram_gb, ram_is_safe_for_training

        ram = available_ram_gb()
        safe = ram_is_safe_for_training(required_gb=7.0)
        result["details"] = {
            "available_ram_gb": round(ram, 2),
            "safe_for_training": safe,
        }
        result["passed"] = True
        _ok(f"RAM khả dụng: {ram:.2f} GB | Đủ điều kiện training: {safe}")
    except Exception as e:
        result["error"] = str(e)
        _fail(str(e))
    return result


def test_csv_to_parquet() -> dict:
    _section("TEST 2 — CSV → Parquet Conversion")
    result = {"module": "convert_csv_to_parquet", "passed": False, "details": {}}
    try:
        from kronos_trainer import convert_csv_to_parquet

        n = convert_csv_to_parquet()
        parquet_files = [f for f in os.listdir(RAW_PQ_DIR) if f.endswith("_history.parquet")]
        result["details"] = {
            "converted_count": n,
            "parquet_files": parquet_files,
        }
        result["passed"] = n > 0
        _ok(f"{n} file CSV → Parquet. Tổng file parquet: {len(parquet_files)}")
    except Exception as e:
        result["error"] = str(e)
        _fail(str(e))
    return result


def test_news_db() -> dict:
    _section("TEST 3 — NewsDB (SQLite)")
    result = {"module": "news_db", "passed": False, "details": {}}
    db_path = os.path.join(DATA_DIR, "news.db")
    try:
        if not os.path.exists(db_path):
            result["error"] = "news.db không tồn tại — chạy news_crawler.py trước"
            _fail(result["error"])
            return result
        conn = sqlite3.connect(db_path)
        total_news = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
        scored     = conn.execute("SELECT COUNT(*) FROM news WHERE sentiment_score IS NOT NULL").fetchone()[0]
        unscored   = total_news - scored
        recent     = conn.execute(
            "SELECT title, pub_date FROM news ORDER BY created_at DESC LIMIT 3"
        ).fetchall()
        conn.close()
        result["details"] = {
            "total_articles"  : total_news,
            "scored_articles" : scored,
            "unscored_articles": unscored,
            "recent_titles"   : [r[0][:60] + "..." if len(r[0]) > 60 else r[0] for r in recent],
        }
        # DB pass nếu schema tồn tại và query chạy được (kể cả 0 bài)
        result["passed"] = True
        _ok(f"Tổng bài báo: {total_news} | Đã chấm điểm: {scored} | Chưa chấm: {unscored}")
        for title, date in recent:
            print(f"    [{date}] {title[:70]}")
    except Exception as e:
        result["error"] = str(e)
        _fail(str(e))
    return result


def test_technical_indicators() -> dict:
    _section("TEST 4 — Technical Indicators")
    result = {"module": "technical_indicators", "passed": False, "details": {}}
    try:
        from technical_indicators import add_technical_indicators

        vnm_path = os.path.join(RAW_PQ_DIR, "VNM_history.parquet")
        if not os.path.exists(vnm_path):
            result["error"] = "VNM_history.parquet chưa tồn tại"
            _fail(result["error"])
            return result
        df = pd.read_parquet(vnm_path, engine="pyarrow")
        df_ta = add_technical_indicators(df)
        expected_cols = ["sma_20", "rsi", "macd", "bb_upper", "atr", "obv"]
        present = [c for c in expected_cols if c in df_ta.columns]
        result["details"] = {
            "base_rows"  : len(df),
            "after_rows" : len(df_ta),
            "indicators" : present,
            "latest_rsi" : round(float(df_ta["rsi"].iloc[-1]), 2),
            "latest_macd": round(float(df_ta["macd"].iloc[-1]), 4),
        }
        result["passed"] = len(present) == len(expected_cols)
        _ok(f"VNM: {len(df_ta)} phiên sau dropna | RSI={result['details']['latest_rsi']} "
            f"| MACD={result['details']['latest_macd']}")
    except Exception as e:
        result["error"] = str(e)
        _fail(str(e))
    return result


def test_kronos_inference_before() -> dict:
    """Chạy inference với base model (chưa fine-tune) và ghi MAE."""
    _section("TEST 5 — Kronos Inference (Base Model)")
    result = {"module": "kronos_inference_base", "passed": False, "details": {}}
    try:
        from chronos import ChronosPipeline

        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map="mps" if torch.backends.mps.is_available() else "cpu",
            dtype=torch.float32,
        )
        df = pd.read_parquet(os.path.join(RAW_PQ_DIR, "VNM_history.parquet"), engine="pyarrow")
        prices = df["close"].values.astype("float32")

        # Holdout: 64 phiên cuối (không dùng trong training)
        HOLDOUT = 64
        CONTEXT = 128
        context  = torch.tensor(prices[-(HOLDOUT + CONTEXT):-HOLDOUT],
                                dtype=torch.float32).unsqueeze(0)
        actuals  = prices[-HOLDOUT:]  # 64 phiên cuối là holdout thực tế

        t0 = time.time()
        with torch.no_grad():
            forecast = pipeline.predict(context, prediction_length=64, num_samples=20)
        inference_ms = round((time.time() - t0) * 1000, 1)

        forecast_np = forecast[0].numpy()
        median_fc   = np.median(forecast_np, axis=0)
        mae         = float(np.mean(np.abs(median_fc - actuals)))
        mae_pct     = mae / float(np.mean(np.abs(actuals))) * 100

        result["details"] = {
            "inference_ms"   : inference_ms,
            "mae_vnd"        : round(mae, 2),
            "mae_pct"        : round(mae_pct, 2),
            "forecast_median": median_fc.tolist(),
            "actuals"        : actuals.tolist(),
        }
        result["passed"] = True
        _ok(f"Inference: {inference_ms}ms | MAE: {mae:.2f} VND ({mae_pct:.2f}%)")

        # Lưu forecast để vẽ so sánh sau
        np.save(os.path.join(DATA_DIR, "_tmp_base_forecast.npy"), median_fc)
        np.save(os.path.join(DATA_DIR, "_tmp_actuals.npy"), actuals)
        np.save(os.path.join(DATA_DIR, "_tmp_context.npy"), prices[-(HOLDOUT + CONTEXT):-HOLDOUT])

        # Giải phóng RAM
        del pipeline
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        _ok(f"RAM sau inference: {_ram_gb():.1f} GB")
    except Exception as e:
        result["error"] = str(e)
        _fail(str(e))
    return result


def test_kronos_finetune() -> dict:
    """Chạy fine-tune LoRA với cấu hình ổn định để baseline có tính tái lập."""
    _section("TEST 6 — Kronos LoRA Fine-Tune (Phase 2.5)")
    result = {"module": "kronos_finetune", "passed": False, "details": {}}
    try:
        from kronos_trainer import finetune_kronos

        metrics = finetune_kronos(
            epochs=3,
            context_len=128,
            batch_size=2,
            learning_rate=5e-5,
            use_sentiment=True,
            sentiment_alpha=0.2,
            max_samples_ticker=30,
            seed=42,
            deterministic=True,
        )
        if metrics is None:
            result["error"] = "finetune_kronos trả về None — kiểm tra dataset."
            _fail(result["error"])
            return result

        result["details"] = metrics
        result["passed"]  = True
        _ok(f"Fine-tune OK | Loss: {metrics['epoch_losses'][0]:.4f} → {metrics['final_loss']:.4f}")
        _ok(f"MAE after: {metrics.get('holdout_mae_after', {}).get('mae')} VND")
    except Exception as e:
        result["error"] = str(e)
        _fail(str(e))
    return result


def test_kronos_inference_after() -> dict:
    """Chạy inference với model đã fine-tune và ghi MAE."""
    _section("TEST 7 — Kronos Inference (After Fine-Tune)")
    result = {"module": "kronos_inference_finetuned", "passed": False, "details": {}}
    try:
        from chronos import ChronosPipeline
        from peft import PeftModel

        CHECKPOINT_DIR = os.path.join(DATA_DIR, "models", "kronos_checkpoints")
        if not os.path.exists(CHECKPOINT_DIR):
            result["error"] = "Checkpoint chưa tồn tại — chạy fine-tune trước"
            _fail(result["error"])
            return result

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map="cpu",
            dtype=torch.float32,
        )
        # Gắn LoRA adapter
        pipeline.model.model = PeftModel.from_pretrained(
            pipeline.model.model, CHECKPOINT_DIR
        ).to(device)
        pipeline.model.model.eval()
        # Di chuyển embedding và config sang device
        pipeline.model.to(device)

        df = pd.read_parquet(os.path.join(RAW_PQ_DIR, "VNM_history.parquet"), engine="pyarrow")
        prices = df["close"].values.astype("float32")
        HOLDOUT = 64
        CONTEXT = 128
        context = torch.tensor(prices[-(HOLDOUT + CONTEXT):-HOLDOUT],
                               dtype=torch.float32).unsqueeze(0)
        actuals = prices[-HOLDOUT:]  # 64 phiên cuối là holdout thực tế

        t0 = time.time()
        with torch.no_grad():
            forecast = pipeline.predict(context, prediction_length=64, num_samples=20)
        inference_ms = round((time.time() - t0) * 1000, 1)

        forecast_np  = forecast[0].numpy()
        median_fc    = np.median(forecast_np, axis=0)
        mae          = float(np.mean(np.abs(median_fc - actuals)))
        mae_pct      = mae / float(np.mean(np.abs(actuals))) * 100

        result["details"] = {
            "inference_ms"  : inference_ms,
            "mae_vnd"       : round(mae, 2),
            "mae_pct"       : round(mae_pct, 2),
            "forecast_median": median_fc.tolist(),
        }
        result["passed"] = True
        _ok(f"Inference: {inference_ms}ms | MAE: {mae:.2f} VND ({mae_pct:.2f}%)")

        # Lưu forecast sau fine-tune để vẽ so sánh
        np.save(os.path.join(DATA_DIR, "_tmp_ft_forecast.npy"), median_fc)

        del pipeline
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as e:
        result["error"] = str(e)
        _fail(str(e))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_forecast_comparison():
    """Biểu đồ so sánh forecast Base vs Fine-Tuned vs Actual."""
    _section("CHART — Forecast Comparison")
    base_path = os.path.join(DATA_DIR, "_tmp_base_forecast.npy")
    ft_path   = os.path.join(DATA_DIR, "_tmp_ft_forecast.npy")
    act_path  = os.path.join(DATA_DIR, "_tmp_actuals.npy")
    ctx_path  = os.path.join(DATA_DIR, "_tmp_context.npy")

    missing = [p for p in [base_path, ft_path, act_path, ctx_path] if not os.path.exists(p)]
    if missing:
        _fail(f"Thiếu dữ liệu tạm: {[os.path.basename(m) for m in missing]}")
        return

    base_fc  = np.load(base_path)
    ft_fc    = np.load(ft_path)
    actuals  = np.load(act_path)
    context  = np.load(ctx_path)

    pred_len = len(base_fc)
    ctx_len  = len(context)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("VNM — Forecast So Sánh: Base vs Fine-Tuned", fontsize=14, fontweight="bold")

    for ax, (label, fc, color) in zip(
        axes,
        [("Base (Chưa Fine-Tune)", base_fc, "#FF5722"),
         ("Sau Fine-Tune (LoRA)", ft_fc,   "#4CAF50")],
    ):
        # Context (lịch sử)
        ax.plot(range(ctx_len), context, color="#2196F3", linewidth=1.5, label="Lịch sử")
        # Actual (holdout)
        ax.plot(range(ctx_len, ctx_len + pred_len), actuals,
                color="#1565C0", linewidth=2, linestyle="--", alpha=0.8, label="Thực tế (holdout)")
        # Forecast
        ax.plot(range(ctx_len, ctx_len + pred_len), fc,
                color=color, linewidth=2.5, label=label)

        mae     = np.mean(np.abs(fc - actuals))
        mae_pct = mae / np.mean(np.abs(actuals)) * 100
        ax.axvline(x=ctx_len - 0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.set_title(f"{label}\nMAE = {mae:.2f} VND ({mae_pct:.2f}%)", fontsize=11)
        ax.set_xlabel("Phiên giao dịch")
        ax.set_ylabel("Giá đóng cửa (×1000 VND)")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(CHARTS_DIR, "forecast_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    _ok(f"Biểu đồ so sánh → {out}")

    # Dọn file tạm
    for p in [base_path, ft_path, act_path, ctx_path]:
        try:
            os.remove(p)
        except OSError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: list[dict]):
    _section("TỔNG KẾT KIỂM THỬ HỆ THỐNG")
    passed = sum(1 for r in results if r.get("passed"))
    total  = len(results)
    print(f"\n  Kết quả: {passed}/{total} test PASSED\n")
    for r in results:
        status = "✓ PASS" if r.get("passed") else "✗ FAIL"
        err    = f"  → {r.get('error', '')}" if not r.get("passed") else ""
        print(f"  [{status}]  {r['module']}{err}")

    # So sánh MAE
    base_result = next((r for r in results if r["module"] == "kronos_inference_base"), None)
    ft_result   = next((r for r in results if r["module"] == "kronos_inference_finetuned"), None)
    if base_result and ft_result and base_result.get("passed") and ft_result.get("passed"):
        mae_before = base_result["details"]["mae_vnd"]
        mae_after  = ft_result["details"]["mae_vnd"]
        improvement = (mae_before - mae_after) / mae_before * 100
        sign = "▼" if improvement > 0 else "▲"
        print(f"\n  MAE VNM (64 phiên holdout):")
        print(f"    Base model : {mae_before:.2f} VND")
        print(f"    Fine-tuned : {mae_after:.2f} VND  {sign} {abs(improvement):.1f}%")

    # Output files
    print(f"\n  Báo cáo chi tiết  : {REPORT_PATH}")
    print(f"  Biểu đồ loss      : {os.path.join(CHARTS_DIR, 'kronos_loss_curve.png')}")
    print(f"  Biểu đồ so sánh  : {os.path.join(CHARTS_DIR, 'forecast_comparison.png')}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 60)
    print("  STOCK-AI — HỆ THỐNG KIỂM THỬ TOÀN DIỆN (Phase 2.5)")
    print("█" * 60)
    print(f"  RAM khả dụng: {_ram_gb():.1f} GB")
    print(f"  MPS: {torch.backends.mps.is_available()}")
    print(f"  Thời gian: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    t_global = time.time()
    all_results = []

    # Chạy các test tuần tự (quản lý RAM)
    all_results.append(test_memory_guard())
    all_results.append(test_csv_to_parquet())
    all_results.append(test_news_db())
    all_results.append(test_technical_indicators())
    all_results.append(test_kronos_inference_before())
    all_results.append(test_kronos_finetune())
    all_results.append(test_kronos_inference_after())

    # Vẽ biểu đồ so sánh
    plot_forecast_comparison()

    # In tổng kết
    print_summary(all_results)

    # Lưu JSON report
    report = {
        "timestamp"    : pd.Timestamp.now().isoformat(),
        "total_time_sec": round(time.time() - t_global, 1),
        "system"       : {
            "ram_available_gb": round(_ram_gb(), 2),
            "mps_available"   : torch.backends.mps.is_available(),
            "python_version"  : sys.version.split()[0],
        },
        "test_results" : all_results,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "reports", "json"), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Báo cáo JSON đã lưu: {REPORT_PATH}\n")


if __name__ == "__main__":
    main()
