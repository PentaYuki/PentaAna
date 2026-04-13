"""
kronos_trainer.py — Fine-tune Chronos T5 Small trên dữ liệu chứng khoán Việt Nam.

Kiến trúc:
  Base model : amazon/chronos-t5-small (46M params)
  Adapter    : LoRA PEFT (r=8, α=32) → chỉ train ~1.8% params
  Dataset    : Sliding window 128 context → 64 forecast từ đa mã CP
  Backend    : MPS (Apple Silicon) hoặc CPU fallback

Kết quả:
  data/models/kronos_checkpoints/    — LoRA adapter weights
  data/reports/json/kronos_metrics.json — Loss + metrics từng epoch
  data/reports/charts/kronos_loss_curve.png — Biểu đồ training loss
"""

import json
import os
import random
import sqlite3
import time

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend cho headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from chronos import ChronosPipeline
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **kw):  # fallback khi chưa cài tqdm
        return it

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR       = os.path.join(BASE_DIR, "data")
RAW_CSV_DIR    = os.path.join(DATA_DIR, "raw", "csv")
RAW_PQ_DIR     = os.path.join(DATA_DIR, "raw", "parquet")
CHECKPOINT_DIR = os.path.join(DATA_DIR, "models", "kronos_checkpoints")
CHARTS_DIR     = os.path.join(DATA_DIR, "reports", "charts")
METRICS_PATH   = os.path.join(DATA_DIR, "reports", "json", "kronos_metrics.json")
DB_PATH        = os.path.join(DATA_DIR, "news.db")
STATUS_PATH    = os.path.join(DATA_DIR, "reports", "json", "finetune_status.json")

# ── Hyper-parameters ───────────────────────────────────────────────────────
CONTEXT_LEN         = 128   # phiên context đưa vào encoder
PRED_LEN            = 64    # phiên dự báo (cố định bởi config Chronos-small)
BATCH_SIZE          = 2     # nhỏ để an toàn RAM M1
EPOCHS              = 5
LR                  = 1e-4
MAX_SAMPLES_TICKER  = 100   # giới hạn mẫu/mã để kiểm soát tiêu thụ RAM

# Tất cả mã có lịch sử CSV (bỏ VNINDEX — index, không phải cổ phiếu)
TICKERS = ["ACB", "BID", "FPT", "GAS", "HPG", "MBB", "MSN", "MWG", "PNJ",
           "TCB", "VCB", "VHM", "VNM"]


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class StockSlidingWindowDataset(Dataset):
    """
    Sliding window trên đa biến (close + volume + RSI + MACD) của đa mã cổ phiếu.
    Chronos nhận chuỗi 1-D nên các features được chuẩn hoá rồi blend vào close.
    Mỗi sample: (context [CONTEXT_LEN], forecast [PRED_LEN])

    Multi-variate blending:
        blended = close_norm + w_vol * volume_norm + w_rsi * rsi_norm + w_macd * macd_norm
    Weights nhỏ (0.05-0.10) để không che mất cấu trúc giá.
    """
    # Các cột kỹ thuật ưu tiên dùng nếu có trong parquet với_indicators
    EXTRA_FEATURES = ["volume", "rsi", "macd"]
    FEAT_WEIGHTS   = {"volume": 0.05, "rsi": 0.08, "macd": 0.07}

    def __init__(self, tickers, data_dir, context_len=CONTEXT_LEN,
                 pred_len=PRED_LEN, step=2, holdout_last=64,
                 max_per_ticker=MAX_SAMPLES_TICKER,
                 use_sentiment=False,
                 sentiment_alpha=0.15,
                 sentiment_db_path=DB_PATH,
                 use_multivariate=True):
        self.samples = []
        total_tickers = 0
        sentiment_lookup = load_daily_sentiment(sentiment_db_path) if use_sentiment else {}
        for ticker in tickers:
            df = load_history_df(ticker, data_dir)
            if df is None:
                continue
            prices = df["close"].values.astype("float32")

            # ── Blending multi-variate features ───────────────────────────────
            if use_multivariate:
                prices = _blend_multivariate(ticker, df, prices, self.EXTRA_FEATURES, self.FEAT_WEIGHTS)

            if use_sentiment:
                prices = blend_price_with_sentiment(
                    ticker=ticker,
                    dates=df["time"],
                    prices=prices,
                    sentiment_lookup=sentiment_lookup,
                    alpha=sentiment_alpha,
                )

            # Giữ lại holdout_last phiên cuối để đánh giá (không train)
            prices_train = prices[: len(prices) - holdout_last]
            max_start = len(prices_train) - context_len - pred_len
            if max_start < 0:
                continue
            ticker_samples = []
            for start in range(0, max_start + 1, step):
                ctx = prices_train[start : start + context_len]
                fct = prices_train[start + context_len : start + context_len + pred_len]
                ticker_samples.append((ctx, fct))
            if max_per_ticker and len(ticker_samples) > max_per_ticker:
                import random
                random.shuffle(ticker_samples)
                ticker_samples = ticker_samples[:max_per_ticker]
            self.samples.extend(ticker_samples)
            n_windows = len(ticker_samples)
            if n_windows > 0:
                print(f"  {ticker}: {n_windows} windows (train trên {len(prices_train)} phiên)")
                total_tickers += 1
        print(f"[DATA] {total_tickers} mã → {len(self.samples)} training samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctx, fct = self.samples[idx]
        return (torch.tensor(ctx, dtype=torch.float32),
                torch.tensor(fct, dtype=torch.float32))


def load_history_df(ticker: str, data_dir: str = DATA_DIR):
    """Ưu tiên Parquet (collector hiện tại), fallback CSV để tương thích ngược.
    Cố gắng load file with_indicators trước (có RSI, MACD, volume)."""
    ind_path = os.path.join(data_dir, "analyzed", "with_indicators", f"{ticker}_with_indicators.parquet")
    pq_path  = os.path.join(data_dir, "raw", "parquet", f"{ticker}_history.parquet")
    csv_path = os.path.join(data_dir, "raw", "csv", f"{ticker}_history.csv")

    for path in (ind_path, pq_path):
        if os.path.exists(path):
            return pd.read_parquet(path, engine="pyarrow").sort_values("time")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path).sort_values("time")
    return None


def _blend_multivariate(
    ticker: str,
    df: pd.DataFrame,
    prices: np.ndarray,
    features: list[str],
    weights: dict[str, float],
) -> np.ndarray:
    """Blend các feature kỹ thuật (volume, RSI, MACD) vào chuỗi giá.
    Mỗi feature được chuẩn hoá z-score rồi nhân hệ số nhỏ trước khi cộng vào close_norm.
    Giữ nguyên đơn vị giá để Chronos có thể học được.
    """
    result = prices.copy()
    for feat in features:
        if feat not in df.columns:
            continue
        raw = df[feat].values.astype("float32")
        std = float(np.nanstd(raw))
        if std < 1e-8:
            continue
        mean = float(np.nanmean(raw))
        z = np.clip((raw - mean) / std, -3.0, 3.0)
        # Đổ vào prices dưới dạng perturbation nhỏ
        result = result * (1.0 + weights.get(feat, 0.05) * z * 0.1)
    return result.astype("float32")


def load_daily_sentiment(db_path: str) -> dict:
    """Load sentiment trung bình theo ngày/ticker từ news.db."""
    if not os.path.exists(db_path):
        return {}
    conn = sqlite3.connect(db_path)
    try:
        q = """
            SELECT ticker, pub_date, AVG(sentiment_score) AS avg_score
            FROM news
            WHERE sentiment_score IS NOT NULL
              AND ticker IS NOT NULL
              AND pub_date IS NOT NULL
            GROUP BY ticker, pub_date
        """
        df = pd.read_sql_query(q, conn)
    except Exception:
        # news table not yet created
        return {}
    finally:
        conn.close()
    if df.empty:
        return {}
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["pub_date"])
    lookup = {}
    for ticker, grp in df.groupby("ticker"):
        lookup[ticker] = dict(zip(grp["pub_date"], grp["avg_score"].astype(float)))
    return lookup


def blend_price_with_sentiment(
    ticker: str,
    dates,
    prices: np.ndarray,
    sentiment_lookup: dict,
    alpha: float = 0.15,
) -> np.ndarray:
    """
    Trộn sentiment vào chuỗi giá để mô hình học được tín hiệu cảm xúc thị trường.
    adjusted_price = close * (1 + alpha * zscore(sentiment_ema) * 0.1)
    """
    ticker = str(ticker).upper()
    per_day = sentiment_lookup.get(ticker, {})
    if not per_day:
        return prices

    date_idx = pd.to_datetime(dates, errors="coerce").strftime("%Y-%m-%d")
    raw_sent = np.array([float(per_day.get(d, 0.0)) for d in date_idx], dtype="float32")
    sent_smooth = pd.Series(raw_sent).ewm(span=5, adjust=False).mean().to_numpy(dtype="float32")

    std = float(np.std(sent_smooth))
    if std < 1e-6:
        return prices
    z = (sent_smooth - float(np.mean(sent_smooth))) / std
    z = np.clip(z, -3.0, 3.0)
    adjusted = prices * (1.0 + alpha * z * 0.1)
    return adjusted.astype("float32")


def write_status(payload: dict, path: str = STATUS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def set_global_seed(seed: int = 42):
    """Best-effort deterministic setup (MPS has some nondeterministic ops)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def convert_csv_to_parquet() -> int:
    """Convert tất cả *_history.csv sang Parquet để dùng cho inference."""
    print("\n[CONVERT] CSV → Parquet...")
    os.makedirs(RAW_PQ_DIR, exist_ok=True)
    if not os.path.exists(RAW_CSV_DIR):
        print("  • Không thấy thư mục CSV, bỏ qua convert (dùng Parquet trực tiếp).")
        return 0

    converted = 0
    for fname in sorted(os.listdir(RAW_CSV_DIR)):
        if not fname.endswith("_history.csv"):
            continue
        csv_path = os.path.join(RAW_CSV_DIR, fname)
        pq_path  = os.path.join(RAW_PQ_DIR, fname.replace(".csv", ".parquet"))
        df = pd.read_csv(csv_path)
        df.to_parquet(pq_path, index=False, engine="pyarrow")
        converted += 1
        print(f"  ✓ {fname} → {os.path.basename(pq_path)} ({len(df)} rows)")
    print(f"[CONVERT] Chuyển xong {converted} file.\n")
    return converted


def compute_holdout_mae(pipeline, ticker="VNM", holdout_last=64, context_len=CONTEXT_LEN) -> dict:
    """
    Tính MAE trên tập holdout (holdout_last phiên cuối không tham gia training).
    Trả về dict {'mae': float, 'mae_pct': float}.
    """
    df = load_history_df(ticker, DATA_DIR)
    if df is None:
        return {}
    prices = df["close"].values.astype("float32")

    context   = torch.tensor(prices[-(holdout_last + context_len):-holdout_last],
                             dtype=torch.float32).unsqueeze(0)
    actuals   = prices[-holdout_last:]  # 64 phiên cuối = holdout thực tế

    with torch.no_grad():
        forecast = pipeline.predict(context, prediction_length=PRED_LEN, num_samples=20)
    forecast_median = np.median(forecast[0].numpy(), axis=0)

    mae     = float(np.mean(np.abs(forecast_median - actuals)))
    mae_pct = float(mae / np.mean(np.abs(actuals)) * 100)
    return {"mae": round(mae, 4), "mae_pct": round(mae_pct, 4)}


def plot_loss_curve(epoch_losses: list, batch_losses: list, out_path: str):
    """Vẽ biểu đồ training loss 2 panel: epoch-level + batch-level."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Kronos LoRA Fine-Tune — Training Loss", fontsize=14, fontweight="bold")

    # — Panel 1: Loss theo epoch —
    ax = axes[0]
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses,
            "o-", color="#2196F3", linewidth=2.5, markersize=8, label="Avg Loss / epoch")
    ax.fill_between(range(1, len(epoch_losses) + 1), epoch_losses, alpha=0.12, color="#2196F3")
    for i, v in enumerate(epoch_losses):
        ax.annotate(f"{v:.4f}", (i + 1, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color="#1565C0")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax.set_title("Loss theo Epoch", fontsize=12)
    ax.set_xticks(range(1, len(epoch_losses) + 1))
    ax.grid(True, alpha=0.3)
    ax.legend()

    # — Panel 2: Loss theo batch (tất cả epoch) —
    ax2 = axes[1]
    ax2.plot(batch_losses, color="#FF5722", linewidth=0.7, alpha=0.6, label="Loss / batch")
    window = max(1, len(batch_losses) // 15)
    if len(batch_losses) >= window:
        smooth = np.convolve(batch_losses, np.ones(window) / window, mode="valid")
        ax2.plot(range(window - 1, len(batch_losses)), smooth,
                 color="#B71C1C", linewidth=2.2, label=f"MA({window})")
    ax2.set_xlabel("Batch (toàn bộ epoch)", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.set_title("Loss theo Batch", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[CHART] Biểu đồ loss → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINER
# ══════════════════════════════════════════════════════════════════════════════

def finetune_kronos(
    epochs: int = EPOCHS,
    context_len: int = CONTEXT_LEN,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LR,
    use_sentiment: bool = True,
    sentiment_alpha: float = 0.15,
    max_samples_ticker: int = MAX_SAMPLES_TICKER,
    tickers: list[str] | None = None,
    status_path: str = STATUS_PATH,
    seed: int = 42,
    deterministic: bool = True,
) -> dict | None:
    print("\n" + "=" * 60)
    print("=== Fine-Tune Kronos LoRA PEFT — Phase 2.5 ===")
    print("=" * 60)

    if tickers is None:
        tickers = TICKERS

    if deterministic:
        set_global_seed(seed)

    write_status({
        "stage": "starting",
        "progress": 0,
        "message": "Khoi tao fine-tune",
        "use_sentiment": use_sentiment,
        "updated_at": pd.Timestamp.now().isoformat(),
    }, status_path)

    # 0. Convert CSV → Parquet (đảm bảo inference hoạt động)
    convert_csv_to_parquet()

    # 1. Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[DEVICE] {device.upper()}\n")

    # 2. Tải base model
    print("[MODEL] Tải amazon/chronos-t5-small...")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",       # load lên CPU trước
        dtype=torch.float32,
    )
    t5_model  = pipeline.model.model   # T5ForConditionalGeneration (46M params)
    tokenizer = pipeline.tokenizer     # MeanScaleUniformBins
    base_params = sum(p.numel() for p in t5_model.parameters())
    print(f"  Base params: {base_params/1e6:.2f}M")

    # 3. Áp dụng LoRA
    print("[LORA] Áp dụng LoRA Adapter (r=8, α=32)...")
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"],  # Query + Value trong T5 SelfAttention
    )
    peft_model = get_peft_model(t5_model, lora_cfg)
    trainable, total = peft_model.get_nb_trainable_parameters()
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    peft_model = peft_model.to(device)

    # 4. Dataset
    print("\n[DATA] Xây dựng sliding window dataset...")
    dataset = StockSlidingWindowDataset(
        tickers=tickers,
        data_dir=DATA_DIR,
        context_len=context_len,
        pred_len=PRED_LEN,
        max_per_ticker=max_samples_ticker,
        use_sentiment=use_sentiment,
        sentiment_alpha=sentiment_alpha,
    )
    if len(dataset) == 0:
        print("✗ Dataset trống — kiểm tra lại file CSV.")
        return None
    data_gen = torch.Generator()
    data_gen.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=data_gen,
    )
    print(f"  Batches/epoch: {len(loader)}")

    # 5. Optimizer
    optimizer = AdamW(peft_model.parameters(), lr=learning_rate, weight_decay=0.01)

    # 6. Training loop
    print(f"\n[TRAIN] {epochs} epochs | lr={learning_rate} | batch={batch_size} | device={device}")
    print("-" * 50)
    epoch_losses: list[float] = []
    batch_losses: list[float] = []
    t_start = time.time()

    peft_model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        n_batches    = 0
        batch_iter = _tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False,
                           dynamic_ncols=True)
        for batch_idx, (ctx_batch, fct_batch) in enumerate(batch_iter, start=1):
            # Tokenize: context + forecast trước khi chuyển sang device
            # (tokenizer chạy trên CPU; scale phải ở CPU khi gọi label_input_transform)
            input_ids, attn_mask, scale = tokenizer.context_input_transform(ctx_batch)
            label_ids, _              = tokenizer.label_input_transform(fct_batch, scale)

            # Sau khi tokenize xong mới chuyển sang device
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            label_ids = label_ids.to(device)

            # Forward
            out  = peft_model(input_ids=input_ids, attention_mask=attn_mask, labels=label_ids)
            loss = out.loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            batch_losses.append(loss.item())
            n_batches    += 1
            batch_iter.set_postfix(loss=f"{loss.item():.4f}")

            if batch_idx % 50 == 0 or batch_idx == len(loader):
                total_batches = max(1, epochs * len(loader))
                done_batches = (epoch * len(loader)) + batch_idx
                progress = int(done_batches / total_batches * 100)
                write_status({
                    "stage": "training",
                    "progress": progress,
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "batch": batch_idx,
                    "batches_per_epoch": len(loader),
                    "loss": round(float(loss.item()), 6),
                    "updated_at": pd.Timestamp.now().isoformat(),
                }, status_path)

        avg = running_loss / n_batches if n_batches else 0.0
        epoch_losses.append(avg)
        elapsed = time.time() - t_start
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg:.4f} | batches={n_batches} | {elapsed:.0f}s")

    total_time = round(time.time() - t_start, 1)
    print(f"\n[TRAIN] Hoàn thành {epochs} epochs trong {total_time}s")

    # 7. Đánh giá holdout MAE (trước fine-tune pipeline đã tải sẵn)
    print("\n[EVAL] Tính MAE holdout (VNM, 64 phiên cuối)...")
    # Khôi phục base model để eval "before"
    peft_model.eval()
    # Gắn lại peft_model vào pipeline để predict
    pipeline.model.model = peft_model
    mae_after = compute_holdout_mae(pipeline, ticker="VNM", context_len=context_len)
    print(f"  MAE after  fine-tune: {mae_after.get('mae')} VND  ({mae_after.get('mae_pct')}%)")

    # 8. Lưu checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "reports", "json"), exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    peft_model.save_pretrained(CHECKPOINT_DIR)
    print(f"\n[SAVE] LoRA adapter → {CHECKPOINT_DIR}")

    # 9. Metrics JSON
    metrics = {
        "model"            : "amazon/chronos-t5-small + LoRA",
        "device"           : device,
        "epochs"           : epochs,
        "batch_size"       : batch_size,
        "learning_rate"    : learning_rate,
        "lora_r"           : 8,
        "lora_alpha"       : 32,
        "context_len"      : context_len,
        "use_sentiment"    : use_sentiment,
        "sentiment_alpha"  : sentiment_alpha,
        "pred_len"         : PRED_LEN,
        "trainable_params" : trainable,
        "total_params"     : total,
        "pct_trainable"    : round(100 * trainable / total, 4),
        "train_samples"    : len(dataset),
        "batches_per_epoch": len(loader),
        "epoch_losses"     : [round(l, 6) for l in epoch_losses],
        "final_loss"       : round(epoch_losses[-1], 6),
        "total_time_sec"   : total_time,
        "holdout_mae_after": mae_after,
        "seed"             : seed,
        "deterministic"    : deterministic,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] Metrics → {METRICS_PATH}")

    write_status({
        "stage": "completed",
        "progress": 100,
        "message": "Fine-tune hoan tat",
        "metrics_path": METRICS_PATH,
        "checkpoint_dir": CHECKPOINT_DIR,
        "final_loss": metrics["final_loss"],
        "mae_after": metrics.get("holdout_mae_after", {}),
        "updated_at": pd.Timestamp.now().isoformat(),
    }, status_path)

    # 10. Biểu đồ loss
    loss_chart = os.path.join(CHARTS_DIR, "kronos_loss_curve.png")
    plot_loss_curve(epoch_losses, batch_losses, loss_chart)

    # Tóm tắt
    print("\n" + "=" * 60)
    print("TỔNG KẾT FINE-TUNE")
    print("=" * 60)
    print(f"  Loss ban đầu : {epoch_losses[0]:.4f}  →  cuối: {epoch_losses[-1]:.4f}")
    drop = (epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100 if epoch_losses[0] else 0
    print(f"  Giảm loss    : {drop:.1f}%")
    print(f"  MAE holdout  : {mae_after.get('mae')} VND ({mae_after.get('mae_pct')}%)")
    print(f"  Checkpoint   : {CHECKPOINT_DIR}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    finetune_kronos()
