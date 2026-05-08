import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PRED_PATH = Path("ml/logs/reports/stage4_retrieval_ablation/dense_predictions.csv")
OUTPUT_DIR = Path("ml/logs/reports/stage4_retrieval_thresholds")

TARGET_PRECISION_UPPER = 0.98  # Порог, чтобы не врать (98% точность)
TARGET_RECALL_LOWER = 0.99     # Порог, чтобы не потерять правильные ответы


def main():
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Missing file: {PRED_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PRED_PATH)

    # score лежит в колонке top1_score
    if "top1_score" not in df.columns:
        raise ValueError(f"Missing top1_score in {df.columns.tolist()}")

    df["is_correct"] = df["hit_at_1"]

    # Распределение правильных и неправильных
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x="top1_score", hue="is_correct", common_norm=False, fill=True)
    plt.title("Distribution of Retrieval Scores (Hit vs Miss)")
    plot_dist_path = OUTPUT_DIR / "score_distribution.png"
    plt.savefig(plot_dist_path, dpi=150)
    plt.close()

    # Калибровка Upper Bound (Максимизация Precision)
    thresholds = np.linspace(df["top1_score"].min(), df["top1_score"].max(), 100)

    upper_bound = None
    for thr in thresholds:
        subset = df[df["top1_score"] >= thr]
        if len(subset) == 0:
            continue
        precision = subset["is_correct"].mean()
        if precision >= TARGET_PRECISION_UPPER:
            upper_bound = float(thr)
            break

    if upper_bound is None:
        upper_bound = float(df["top1_score"].quantile(0.75))

    # Калибровка Lower Bound (Отсечение мусора, сохранение Recall)
    lower_bound = None
    for thr in reversed(thresholds):
        subset = df[df["top1_score"] >= thr]
        if len(subset) == 0:
            continue
        recall = subset["is_correct"].sum() / df["is_correct"].sum()
        if recall >= TARGET_RECALL_LOWER:
            lower_bound = float(thr)
            break

    if lower_bound is None:
        lower_bound = float(df["top1_score"].quantile(0.10))

    # Убедимся, что lower < upper
    if lower_bound >= upper_bound:
        lower_bound = upper_bound - 0.02

    # Сохраняем
    summary = {
        "upper_bound": round(upper_bound, 4),
        "lower_bound": round(lower_bound, 4),
        "target_precision_upper": TARGET_PRECISION_UPPER,
        "target_recall_lower": TARGET_RECALL_LOWER,
        "metrics_on_train_set": {
            "total_queries": int(len(df)),
            "direct_answer_queries (>= upper)": int((df["top1_score"] >= upper_bound).sum()),
            "clarify_queries (lower <= x < upper)": int(((df["top1_score"] >= lower_bound) & (df["top1_score"] < upper_bound)).sum()),
            "fallback_queries (< lower)": int((df["top1_score"] < lower_bound).sum()),
        }
    }

    summary_path = OUTPUT_DIR / "retrieval_thresholds.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[INFO] Retrieval Thresholds Calibrated")
    print(f"  Upper Bound (Direct Answer): >= {upper_bound:.4f}")
    print(f"  Lower Bound (Clarify):       >= {lower_bound:.4f}")
    print(f"  Fallback (Drop):             <  {lower_bound:.4f}")

    print("\n[INFO] Routing simulation on FAQ eval:")
    print(json.dumps(summary["metrics_on_train_set"], indent=2))
    print(f"\n[OK] Saved to {summary_path}")


if __name__ == "__main__":
    main()