from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


VAL_PRED_PATH = Path("ml/logs/reports/stage3_rubert_tiny2_patched/val_predictions.csv")
OUTPUT_DIR = Path("ml/logs/reports/stage3_threshold_calibration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PRECISION = 0.92


def main():
    if not VAL_PRED_PATH.exists():
        raise FileNotFoundError(f"Missing file: {VAL_PRED_PATH}")

    df = pd.read_csv(VAL_PRED_PATH)

    required_cols = {"label", "pred_label", "pred_confidence"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["is_correct"] = (df["label"] == df["pred_label"]).astype(int)

    thresholds = np.round(np.arange(0.05, 1.00, 0.01), 2)

    rows = []
    for thr in thresholds:
        subset = df[df["pred_confidence"] >= thr].copy()
        coverage = len(subset) / len(df)

        if len(subset) == 0:
            precision = 0.0
        else:
            precision = subset["is_correct"].mean()

        rows.append({
            "threshold": float(thr),
            "coverage": float(coverage),
            "precision": float(precision),
            "selected_count": int(len(subset)),
            "total_count": int(len(df)),
        })

    metrics_df = pd.DataFrame(rows)

    valid = metrics_df[metrics_df["precision"] >= TARGET_PRECISION].copy()

    if len(valid) > 0:
        # выбираем максимальное coverage при precision >= target
        best_row = valid.sort_values(
            by=["coverage", "threshold"],
            ascending=[False, True]
        ).iloc[0]
        chosen_threshold = float(best_row["threshold"])
    else:
        # fallback: выбираем threshold с максимальной precision
        best_row = metrics_df.sort_values(
            by=["precision", "coverage"],
            ascending=[False, False]
        ).iloc[0]
        chosen_threshold = float(best_row["threshold"])

    # save csv
    metrics_path = OUTPUT_DIR / "threshold_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8")

    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df["threshold"], metrics_df["precision"], label="precision")
    plt.plot(metrics_df["threshold"], metrics_df["coverage"], label="coverage")
    plt.axhline(TARGET_PRECISION, color="red", linestyle="--", label=f"target precision={TARGET_PRECISION}")
    plt.axvline(chosen_threshold, color="green", linestyle="--", label=f"chosen threshold={chosen_threshold}")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Metric value")
    plt.title("Threshold calibration on validation set")
    plt.legend()
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "threshold_calibration.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    summary = {
        "target_precision": TARGET_PRECISION,
        "chosen_threshold": chosen_threshold,
        "chosen_row": {
            "threshold": float(best_row["threshold"]),
            "coverage": float(best_row["coverage"]),
            "precision": float(best_row["precision"]),
            "selected_count": int(best_row["selected_count"]),
            "total_count": int(best_row["total_count"]),
        },
        "val_pred_path": str(VAL_PRED_PATH),
        "metrics_csv": str(metrics_path),
        "plot_path": str(plot_path),
    }

    summary_path = OUTPUT_DIR / "chosen_threshold.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] Threshold calibration finished")
    print(f"[INFO] Target precision: {TARGET_PRECISION}")
    print(f"[INFO] Chosen threshold: {chosen_threshold:.2f}")
    print(f"[INFO] Chosen precision: {best_row['precision']:.4f}")
    print(f"[INFO] Chosen coverage: {best_row['coverage']:.4f}")
    print(f"[INFO] Selected count: {int(best_row['selected_count'])} / {int(best_row['total_count'])}")

    print(f"\n[OK] Saved metrics: {metrics_path}")
    print(f"[OK] Saved plot: {plot_path}")
    print(f"[OK] Saved chosen threshold: {summary_path}")


if __name__ == "__main__":
    main()