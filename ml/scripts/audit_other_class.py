from pathlib import Path
import json
import re
import pandas as pd

INPUT_PATH = Path("ml/data/processed/full_dataset_stage1_base.csv")
OUTPUT_CSV = Path("ml/data/interim/review/other_class_audit.csv")
OUTPUT_JSON = Path("ml/logs/reports/stage1_other_class_summary.json")


def detect_other_subtype(text: str) -> str:
    text = str(text).lower()

    operator_patterns = [
        r"\bоператор\b",
        r"\bчеловек\b",
        r"\bагент\b",
        r"связа",
        r"поговор",
        r"\bассистент\b",
        r"кем[- ]?нибудь",
    ]

    feedback_patterns = [
        r"отзыв",
        r"жалоб",
        r"предложен",
        r"\bиде[яиюе]\b",
        r"похвал",
        r"пожалова",
        r"фидбек",
    ]

    legal_patterns = [
        r"иск",
        r"претензи",
        r"суд",
        r"потребительск",
    ]

    if any(re.search(p, text) for p in operator_patterns):
        return "operator_contact"
    if any(re.search(p, text) for p in feedback_patterns):
        return "feedback_complaint"
    if any(re.search(p, text) for p in legal_patterns):
        return "legal_claim"
    return "unclassified"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    other_df = df[df["label"] == "other"].copy()

    if other_df.empty:
        print("[WARN] No rows with label='other' found.")
        return

    other_df["other_subtype_rule"] = other_df["text_normalized"].map(detect_other_subtype)

    other_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    summary = {
        "total_other_rows": int(len(other_df)),
        "subtype_distribution": other_df["other_subtype_rule"].value_counts().to_dict(),
        "output_csv": str(OUTPUT_CSV)
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Total rows in class 'other': {len(other_df)}")
    print("\n[INFO] Rule-based subtype distribution:")
    print(other_df["other_subtype_rule"].value_counts())

    print(f"\n[OK] Saved audit CSV: {OUTPUT_CSV}")
    print(f"[OK] Saved summary JSON: {OUTPUT_JSON}")

    print("\n[INFO] Sample examples by subtype:")
    for subtype in other_df["other_subtype_rule"].value_counts().index:
        print(f"\n--- {subtype} ---")
        sample = other_df[other_df["other_subtype_rule"] == subtype][
            ["sample_id", "text_original"]
        ].head(5)
        print(sample.to_string(index=False))


if __name__ == "__main__":
    main()