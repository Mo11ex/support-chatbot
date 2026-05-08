import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.evaluation.evaluation import (
    classification_report_metrics,
    accuracy,
    build_confusion_matrix,
    precision_at_k,
    recall_at_k,
    mrr
)


def main():
    # --- Classification metrics ---
    y_true = ["a", "b", "a", "c", "b", "a"]
    y_pred = ["a", "b", "c", "c", "b", "a"]

    report = classification_report_metrics(y_true, y_pred)
    cm = build_confusion_matrix(y_true, y_pred, labels=["a", "b", "c"])
    acc = accuracy(y_true, y_pred)

    print("=== Classification Metrics ===")
    print(f"accuracy:    {acc:.4f}")
    print(f"macro_f1:    {report['macro avg']['f1-score']:.4f}")
    print(f"weighted_f1: {report['weighted avg']['f1-score']:.4f}")
    print(f"confusion_matrix:\n{cm}")

    # --- Retrieval metrics ---
    all_relevant = [["doc1"], ["doc2"], ["doc3"]]
    all_predicted = [
        ["doc1", "doc5", "doc9"],
        ["doc7", "doc2", "doc8"],
        ["doc4", "doc6", "doc3"],
    ]

    print("\n=== Retrieval Metrics ===")
    print(f"precision@1: {precision_at_k(all_relevant, all_predicted, k=1):.4f}")
    print(f"precision@3: {precision_at_k(all_relevant, all_predicted, k=3):.4f}")
    print(f"recall@1:    {recall_at_k(all_relevant, all_predicted, k=1):.4f}")
    print(f"recall@3:    {recall_at_k(all_relevant, all_predicted, k=3):.4f}")
    print(f"mrr@3:       {mrr(all_relevant, all_predicted, k=3):.4f}")

    print("\n[OK] All evaluation functions work correctly.")


if __name__ == "__main__":
    main()