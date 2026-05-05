from pathlib import Path
import json
import pandas as pd


TEST_PATH = Path("ml/logs/reports/stage3_rubert_base_classifier/test_predictions.csv")
HELDOUT_PATH = Path("ml/logs/reports/stage3_rubert_base_classifier/heldout_predictions.csv")

OUTPUT_DIR = Path("ml/logs/reports/stage3_strong_error_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOW_CONF_THRESHOLD = 0.80
SHORT_QUERY_MAX_WORDS = 2


def detect_text_column(df):
    for c in ["text_original", "text"]:
        if c in df.columns:
            return c
    raise ValueError(f"No text column found. Columns: {df.columns.tolist()}")


def add_text_features(df, text_col):
    df = df.copy()
    df["text_for_analysis"] = df[text_col].astype(str)
    df["num_words"] = df["text_for_analysis"].str.split().apply(len)
    return df


def confusion_pairs(df):
    wrong = df[df["label"] != df["pred_label"]].copy()
    if wrong.empty:
        return pd.DataFrame(columns=["label", "pred_label", "count"])
    return (
        wrong.groupby(["label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def save_df(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def main():
    test_df = pd.read_csv(TEST_PATH)
    heldout_df = pd.read_csv(HELDOUT_PATH)

    test_df = add_text_features(test_df, detect_text_column(test_df))
    heldout_df = add_text_features(heldout_df, detect_text_column(heldout_df))

    test_errors = test_df[test_df["label"] != test_df["pred_label"]].copy()
    heldout_errors = heldout_df[heldout_df["label"] != heldout_df["pred_label"]].copy()

    test_confusions = confusion_pairs(test_df)
    heldout_confusions = confusion_pairs(heldout_df)

    heldout_short_errors = heldout_errors[heldout_errors["num_words"] <= SHORT_QUERY_MAX_WORDS]
    heldout_low_conf_all = heldout_df[heldout_df["pred_confidence"] < LOW_CONF_THRESHOLD]
    heldout_low_conf_errors = heldout_errors[heldout_errors["pred_confidence"] < LOW_CONF_THRESHOLD]

    heldout_high_conf_wrong = heldout_errors.sort_values("pred_confidence", ascending=False).head(30)

    heldout_error_by_label = (
        heldout_errors.groupby("label").size()
        .reset_index(name="error_count")
        .sort_values("error_count", ascending=False)
    )

    save_df(test_errors, OUTPUT_DIR / "test_errors.csv")
    save_df(heldout_errors, OUTPUT_DIR / "heldout_errors.csv")
    save_df(test_confusions, OUTPUT_DIR / "test_confusion_pairs.csv")
    save_df(heldout_confusions, OUTPUT_DIR / "heldout_confusion_pairs.csv")
    save_df(heldout_high_conf_wrong, OUTPUT_DIR / "heldout_top30_high_conf_wrong.csv")
    save_df(heldout_error_by_label, OUTPUT_DIR / "heldout_error_by_true_label.csv")

    summary = {
        "test_total": int(len(test_df)),
        "heldout_total": int(len(heldout_df)),
        "test_errors": int(len(test_errors)),
        "heldout_errors": int(len(heldout_errors)),
        "test_error_rate": round(len(test_errors) / len(test_df), 4),
        "heldout_error_rate": round(len(heldout_errors) / len(heldout_df), 4),
        "heldout_short_errors": int(len(heldout_short_errors)),
        "heldout_low_conf_all": int(len(heldout_low_conf_all)),
        "heldout_low_conf_errors": int(len(heldout_low_conf_errors)),
    }

    with open(OUTPUT_DIR / "strong_error_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # === COMPARISON WITH BASELINE ===
    baseline_summary_path = Path("ml/logs/reports/stage3_baseline_error_analysis/baseline_error_summary.json")
    if baseline_summary_path.exists():
        with open(baseline_summary_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)

        print("\n[INFO] === COMPARISON: baseline vs strong ===")
        print(f"{'Metric':<35} {'Baseline':>10} {'Strong':>10} {'Delta':>10}")
        print("-" * 65)

        comparisons = [
            ("test_errors", baseline["test_errors_count"], summary["test_errors"]),
            ("heldout_errors", baseline["heldout_errors_count"], summary["heldout_errors"]),
            ("test_error_rate", baseline["test_error_rate"], summary["test_error_rate"]),
            ("heldout_error_rate", baseline["heldout_error_rate"], summary["heldout_error_rate"]),
            ("heldout_short_errors", baseline["heldout_short_errors_count"], summary["heldout_short_errors"]),
            ("heldout_low_conf_all", baseline["heldout_low_conf_all_count"], summary["heldout_low_conf_all"]),
            ("heldout_low_conf_errors", baseline["heldout_low_conf_errors_count"], summary["heldout_low_conf_errors"]),
        ]

        for name, b_val, s_val in comparisons:
            delta = s_val - b_val
            sign = "+" if delta > 0 else ""
            print(f"{name:<35} {b_val:>10} {s_val:>10} {sign}{delta:>9}")
    else:
        print("[WARN] Baseline error summary not found, skipping comparison.")

    print("\n[INFO] Strong model error summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n[INFO] Top heldout confusion pairs (strong):")
    print(heldout_confusions.head(10).to_string(index=False))

    print("\n[INFO] Heldout errors by true label (strong):")
    print(heldout_error_by_label.to_string(index=False))

    print(f"\n[OK] Reports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()