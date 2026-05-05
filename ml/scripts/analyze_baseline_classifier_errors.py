from pathlib import Path
import json
import pandas as pd


TEST_PATH = Path("ml/logs/reports/stage2_baseline_classifier/test_predictions.csv")
HELDOUT_PATH = Path("ml/logs/reports/stage2_baseline_classifier/heldout_predictions.csv")

OUTPUT_DIR = Path("ml/logs/reports/stage3_baseline_error_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


LOW_CONF_THRESHOLD = 0.80
SHORT_QUERY_MAX_WORDS = 2


def detect_text_column(df: pd.DataFrame) -> str:
    if "text_original" in df.columns:
        return "text_original"
    if "text" in df.columns:
        return "text"
    raise ValueError(f"Cannot detect text column in dataframe. Columns: {df.columns.tolist()}")


def add_text_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df = df.copy()
    df["text_for_analysis"] = df[text_col].astype(str)
    df["num_chars"] = df["text_for_analysis"].str.len()
    df["num_words"] = df["text_for_analysis"].str.split().apply(len)
    return df


def confusion_pairs(df: pd.DataFrame) -> pd.DataFrame:
    wrong = df[df["label"] != df["pred_label"]].copy()
    if wrong.empty:
        return pd.DataFrame(columns=["label", "pred_label", "count"])
    pairs = (
        wrong.groupby(["label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return pairs


def save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def main():
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TEST_PATH}")
    if not HELDOUT_PATH.exists():
        raise FileNotFoundError(f"Missing file: {HELDOUT_PATH}")

    test_df = pd.read_csv(TEST_PATH)
    heldout_df = pd.read_csv(HELDOUT_PATH)

    test_text_col = detect_text_column(test_df)
    heldout_text_col = detect_text_column(heldout_df)

    test_df = add_text_features(test_df, test_text_col)
    heldout_df = add_text_features(heldout_df, heldout_text_col)

    # Errors
    test_errors = test_df[test_df["label"] != test_df["pred_label"]].copy()
    heldout_errors = heldout_df[heldout_df["label"] != heldout_df["pred_label"]].copy()

    # Sort by confidence ascending for most uncertain wrong predictions
    test_errors_low_conf = test_errors.sort_values("pred_confidence", ascending=True).reset_index(drop=True)
    heldout_errors_low_conf = heldout_errors.sort_values("pred_confidence", ascending=True).reset_index(drop=True)

    # High-confidence wrong predictions: dangerous errors
    test_errors_high_conf = test_errors.sort_values("pred_confidence", ascending=False).reset_index(drop=True)
    heldout_errors_high_conf = heldout_errors.sort_values("pred_confidence", ascending=False).reset_index(drop=True)

    # Confusion pairs
    test_confusions = confusion_pairs(test_df)
    heldout_confusions = confusion_pairs(heldout_df)

    # Short query errors
    test_short_errors = test_errors[test_errors["num_words"] <= SHORT_QUERY_MAX_WORDS].copy()
    heldout_short_errors = heldout_errors[heldout_errors["num_words"] <= SHORT_QUERY_MAX_WORDS].copy()

    # Low-confidence all predictions
    test_low_conf_all = test_df[test_df["pred_confidence"] < LOW_CONF_THRESHOLD].copy()
    heldout_low_conf_all = heldout_df[heldout_df["pred_confidence"] < LOW_CONF_THRESHOLD].copy()

    # Low-confidence errors only
    test_low_conf_errors = test_errors[test_errors["pred_confidence"] < LOW_CONF_THRESHOLD].copy()
    heldout_low_conf_errors = heldout_errors[heldout_errors["pred_confidence"] < LOW_CONF_THRESHOLD].copy()

    # Per-class error counts on heldout
    heldout_error_by_true_label = (
        heldout_errors.groupby("label")
        .size()
        .reset_index(name="error_count")
        .sort_values("error_count", ascending=False)
        .reset_index(drop=True)
    )

    # Save CSVs
    save_df(test_errors, OUTPUT_DIR / "test_errors.csv")
    save_df(heldout_errors, OUTPUT_DIR / "heldout_errors.csv")

    save_df(test_confusions, OUTPUT_DIR / "test_confusion_pairs.csv")
    save_df(heldout_confusions, OUTPUT_DIR / "heldout_confusion_pairs.csv")

    save_df(test_short_errors, OUTPUT_DIR / "test_short_query_errors.csv")
    save_df(heldout_short_errors, OUTPUT_DIR / "heldout_short_query_errors.csv")

    save_df(test_low_conf_all, OUTPUT_DIR / "test_low_confidence_predictions.csv")
    save_df(heldout_low_conf_all, OUTPUT_DIR / "heldout_low_confidence_predictions.csv")

    save_df(test_low_conf_errors, OUTPUT_DIR / "test_low_confidence_errors.csv")
    save_df(heldout_low_conf_errors, OUTPUT_DIR / "heldout_low_confidence_errors.csv")

    save_df(test_errors_high_conf.head(30), OUTPUT_DIR / "test_top30_high_confidence_wrong.csv")
    save_df(heldout_errors_high_conf.head(30), OUTPUT_DIR / "heldout_top30_high_confidence_wrong.csv")

    save_df(test_errors_low_conf.head(30), OUTPUT_DIR / "test_top30_low_confidence_wrong.csv")
    save_df(heldout_errors_low_conf.head(30), OUTPUT_DIR / "heldout_top30_low_confidence_wrong.csv")

    save_df(heldout_error_by_true_label, OUTPUT_DIR / "heldout_error_by_true_label.csv")

    summary = {
        "test_total_rows": int(len(test_df)),
        "heldout_total_rows": int(len(heldout_df)),
        "test_errors_count": int(len(test_errors)),
        "heldout_errors_count": int(len(heldout_errors)),
        "test_error_rate": float(len(test_errors) / len(test_df)),
        "heldout_error_rate": float(len(heldout_errors) / len(heldout_df)),
        "low_conf_threshold": LOW_CONF_THRESHOLD,
        "short_query_max_words": SHORT_QUERY_MAX_WORDS,
        "test_low_conf_all_count": int(len(test_low_conf_all)),
        "heldout_low_conf_all_count": int(len(heldout_low_conf_all)),
        "test_low_conf_errors_count": int(len(test_low_conf_errors)),
        "heldout_low_conf_errors_count": int(len(heldout_low_conf_errors)),
        "test_short_errors_count": int(len(test_short_errors)),
        "heldout_short_errors_count": int(len(heldout_short_errors)),
        "top_test_confusions": test_confusions.head(10).to_dict(orient="records"),
        "top_heldout_confusions": heldout_confusions.head(10).to_dict(orient="records"),
        "heldout_error_by_true_label_top10": heldout_error_by_true_label.head(10).to_dict(orient="records"),
    }

    with open(OUTPUT_DIR / "baseline_error_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Console report
    print("[INFO] Baseline error analysis summary")
    print(f"  test total rows: {len(test_df)}")
    print(f"  heldout total rows: {len(heldout_df)}")
    print(f"  test errors: {len(test_errors)}")
    print(f"  heldout errors: {len(heldout_errors)}")
    print(f"  test error rate: {len(test_errors) / len(test_df):.4f}")
    print(f"  heldout error rate: {len(heldout_errors) / len(heldout_df):.4f}")

    print("\n[INFO] Short-query errors")
    print(f"  test short errors (<= {SHORT_QUERY_MAX_WORDS} words): {len(test_short_errors)}")
    print(f"  heldout short errors (<= {SHORT_QUERY_MAX_WORDS} words): {len(heldout_short_errors)}")

    print("\n[INFO] Low-confidence predictions")
    print(f"  test low-confidence predictions (< {LOW_CONF_THRESHOLD}): {len(test_low_conf_all)}")
    print(f"  heldout low-confidence predictions (< {LOW_CONF_THRESHOLD}): {len(heldout_low_conf_all)}")
    print(f"  test low-confidence errors: {len(test_low_conf_errors)}")
    print(f"  heldout low-confidence errors: {len(heldout_low_conf_errors)}")

    print("\n[INFO] Top test confusion pairs:")
    print(test_confusions.head(10).to_string(index=False))

    print("\n[INFO] Top heldout confusion pairs:")
    print(heldout_confusions.head(10).to_string(index=False))

    print("\n[INFO] Top heldout true-label error counts:")
    print(heldout_error_by_true_label.head(10).to_string(index=False))

    print(f"\n[OK] Reports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()