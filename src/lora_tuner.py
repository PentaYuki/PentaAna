import itertools
import json
import os

from kronos_trainer import finetune_kronos

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_PATH = os.path.join(DATA_DIR, "reports", "json", "lora_tuning_results.json")
SHORT_OUT_PATH = os.path.join(DATA_DIR, "reports", "json", "lora_tuning_short_results.json")


def run_grid_search():
    contexts = [96, 128, 160]
    batch_sizes = [1, 2]
    epochs_list = [3, 5]
    lrs = [5e-5, 1e-4]

    combos = list(itertools.product(contexts, batch_sizes, epochs_list, lrs))
    results = []

    for i, (context_len, batch_size, epochs, lr) in enumerate(combos, start=1):
        print("\n" + "=" * 70)
        print(f"[{i}/{len(combos)}] context={context_len}, batch={batch_size}, epochs={epochs}, lr={lr}")
        print("=" * 70)

        try:
            metrics = finetune_kronos(
                epochs=epochs,
                context_len=context_len,
                batch_size=batch_size,
                learning_rate=lr,
                use_sentiment=True,
                sentiment_alpha=0.15,
                status_path=os.path.join(DATA_DIR, "reports", "json", f"finetune_status_trial_{i}.json"),
            )
            if metrics is None:
                raise RuntimeError("Trainer trả về None")

            mae = float(metrics.get("holdout_mae_after", {}).get("mae", 1e9))
            item = {
                "context_len": context_len,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": lr,
                "final_loss": metrics.get("final_loss"),
                "holdout_mae": mae,
                "metrics": metrics,
            }
            results.append(item)
        except Exception as e:
            results.append(
                {
                    "context_len": context_len,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "learning_rate": lr,
                    "error": str(e),
                }
            )

    valid = [r for r in results if "holdout_mae" in r]
    best = min(valid, key=lambda x: x["holdout_mae"]) if valid else None

    payload = {
        "n_trials": len(combos),
        "best": best,
        "results": results,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n=== TUNING DONE ===")
    if best:
        print(f"Best MAE: {best['holdout_mae']}")
        print(
            "Best params: "
            f"context={best['context_len']}, batch={best['batch_size']}, "
            f"epochs={best['epochs']}, lr={best['learning_rate']}"
        )
    print(f"Saved: {OUT_PATH}")


def run_short_batch(trials: int = 5):
    """Run a quick 3-5 trial tuning batch for fast feedback."""
    candidates = [
        {"context_len": 96, "batch_size": 2, "epochs": 1, "learning_rate": 1e-4, "sentiment_alpha": 0.10},
        {"context_len": 128, "batch_size": 2, "epochs": 1, "learning_rate": 1e-4, "sentiment_alpha": 0.15},
        {"context_len": 128, "batch_size": 2, "epochs": 1, "learning_rate": 5e-5, "sentiment_alpha": 0.20},
        {"context_len": 160, "batch_size": 2, "epochs": 1, "learning_rate": 1e-4, "sentiment_alpha": 0.12},
        {"context_len": 96, "batch_size": 1, "epochs": 1, "learning_rate": 8e-5, "sentiment_alpha": 0.18},
    ]
    trials = max(1, min(trials, len(candidates)))

    selected = candidates[:trials]
    results = []

    for i, cfg in enumerate(selected, start=1):
        print("\n" + "=" * 70)
        print(
            f"[SHORT {i}/{trials}] "
            f"context={cfg['context_len']}, batch={cfg['batch_size']}, epochs={cfg['epochs']}, "
            f"lr={cfg['learning_rate']}, alpha={cfg['sentiment_alpha']}"
        )
        print("=" * 70)

        try:
            metrics = finetune_kronos(
                epochs=cfg["epochs"],
                context_len=cfg["context_len"],
                batch_size=cfg["batch_size"],
                learning_rate=cfg["learning_rate"],
                use_sentiment=True,
                sentiment_alpha=cfg["sentiment_alpha"],
                max_samples_ticker=30,
                status_path=os.path.join(DATA_DIR, "reports", "json", f"finetune_status_short_trial_{i}.json"),
            )
            if metrics is None:
                raise RuntimeError("Trainer trả về None")

            mae = float(metrics.get("holdout_mae_after", {}).get("mae", 1e9))
            item = {
                **cfg,
                "final_loss": metrics.get("final_loss"),
                "holdout_mae": mae,
                "metrics": metrics,
            }
            results.append(item)
        except Exception as e:
            results.append({**cfg, "error": str(e)})

    valid = [r for r in results if "holdout_mae" in r]
    best = min(valid, key=lambda x: x["holdout_mae"]) if valid else None

    payload = {
        "mode": "short_batch",
        "n_trials": trials,
        "best": best,
        "results": results,
    }

    os.makedirs(os.path.dirname(SHORT_OUT_PATH), exist_ok=True)
    with open(SHORT_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n=== SHORT TUNING DONE ===")
    if best:
        print(f"Best MAE: {best['holdout_mae']}")
        print(
            "Best params: "
            f"context={best['context_len']}, batch={best['batch_size']}, epochs={best['epochs']}, "
            f"lr={best['learning_rate']}, alpha={best['sentiment_alpha']}"
        )
    print(f"Saved: {SHORT_OUT_PATH}")

    return payload


if __name__ == "__main__":
    run_grid_search()
