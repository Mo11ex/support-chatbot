from pathlib import Path
import json
import pandas as pd

INPUT_PATH = Path("ml/data/processed/full_dataset_stage1_base.csv")
REVIEW_DIR = Path("ml/data/interim/review")
REPORT_PATH = Path("ml/logs/reports/stage1_exact_duplicates_summary.json")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    # 1. Exact duplicates by text_original + label
    exact_original_mask = df.duplicated(subset=["text_original", "label"], keep=False)
    exact_original = df[exact_original_mask].copy()
    exact_original = exact_original.sort_values(["label", "text_original", "sample_id"])

    # 2. Exact duplicates by text_normalized + label
    exact_normalized_mask = df.duplicated(subset=["text_normalized", "label"], keep=False)
    exact_normalized = df[exact_normalized_mask].copy()
    exact_normalized = exact_normalized.sort_values(["label", "text_normalized", "sample_id"])

    # 3. Same normalized text with different labels
    conflicting_groups = (
        df.groupby("text_normalized")["label"]
        .nunique()
        .reset_index(name="label_count")
    )
    conflicting_texts = conflicting_groups[conflicting_groups["label_count"] > 1]["text_normalized"]

    conflicting = df[df["text_normalized"].isin(conflicting_texts)].copy()
    conflicting = conflicting.sort_values(["text_normalized", "label", "sample_id"])

    # Save outputs
    exact_original_path = REVIEW_DIR / "exact_duplicates_original.csv"
    exact_normalized_path = REVIEW_DIR / "exact_duplicates_normalized.csv"
    conflicting_path = REVIEW_DIR / "conflicting_labels_same_text.csv"

    exact_original.to_csv(exact_original_path, index=False, encoding="utf-8")
    exact_normalized.to_csv(exact_normalized_path, index=False, encoding="utf-8")
    conflicting.to_csv(conflicting_path, index=False, encoding="utf-8")

    # Summary
    summary = {
        "input_path": str(INPUT_PATH),
        "total_rows": int(len(df)),
        "exact_duplicates_original_rows": int(len(exact_original)),
        "exact_duplicates_original_groups": int(
            exact_original.groupby(["text_original", "label"]).ngroups if len(exact_original) > 0 else 0
        ),
        "exact_duplicates_normalized_rows": int(len(exact_normalized)),
        "exact_duplicates_normalized_groups": int(
            exact_normalized.groupby(["text_normalized", "label"]).ngroups if len(exact_normalized) > 0 else 0
        ),
        "conflicting_rows": int(len(conflicting)),
        "conflicting_text_count": int(conflicting["text_normalized"].nunique()) if len(conflicting) > 0 else 0,
        "saved_files": {
            "exact_original": str(exact_original_path),
            "exact_normalized": str(exact_normalized_path),
            "conflicting_labels": str(conflicting_path),
        }
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Console report
    print(f"[INFO] Input dataset: {INPUT_PATH}")
    print(f"[INFO] Total rows: {len(df)}")

    print("\n[INFO] Exact duplicates by (text_original, label):")
    print(f"  rows:   {len(exact_original)}")
    print(f"  groups: {summary['exact_duplicates_original_groups']}")

    print("\n[INFO] Exact duplicates by (text_normalized, label):")
    print(f"  rows:   {len(exact_normalized)}")
    print(f"  groups: {summary['exact_duplicates_normalized_groups']}")

    print("\n[INFO] Same text_normalized with different labels:")
    print(f"  rows:   {len(conflicting)}")
    print(f"  texts:  {summary['conflicting_text_count']}")

    print("\n[OK] Saved files:")
    print(f"  - {exact_original_path}")
    print(f"  - {exact_normalized_path}")
    print(f"  - {conflicting_path}")
    print(f"  - {REPORT_PATH}")

    # Preview
    if len(conflicting) > 0:
        print("\n[INFO] Preview conflicting labels:")
        preview = conflicting[["sample_id", "text_normalized", "label"]].head(10)
        print(preview.to_string(index=False))
    else:
        print("\n[INFO] No conflicting normalized texts with different labels found.")


if __name__ == "__main__":
    main()