from pathlib import Path
import pandas as pd

INPUT_PATH = Path("ml/data/interim/review/cross_class_suspicious_pairs.csv")
OUTPUT_PATH = Path("ml/data/interim/review/manual_borderline_review.csv")


TARGET_PAIRS = [
    ("payment_refund", "return_exchange", 25),
    ("general_info", "product_info", 25),
]


def canonical_pair(row):
    ids = sorted([str(row["sample_id_1"]), str(row["sample_id_2"])])
    return f"{ids[0]}__{ids[1]}"


def matches_pair(label_1, label_2, a, b):
    return (label_1 == a and label_2 == b) or (label_1 == b and label_2 == a)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    # Убираем зеркальные дубликаты одной и той же пары
    df["pair_key"] = df.apply(canonical_pair, axis=1)
    df = df.sort_values("cosine_similarity", ascending=False)
    df = df.drop_duplicates(subset=["pair_key"]).reset_index(drop=True)

    selected_parts = []

    for label_a, label_b, limit in TARGET_PAIRS:
        part = df[
            df.apply(lambda row: matches_pair(row["label_1"], row["label_2"], label_a, label_b), axis=1)
        ].copy()

        part = part.head(limit).copy()
        part["target_pair"] = f"{label_a}__{label_b}"
        selected_parts.append(part)

        print(f"[INFO] Selected {len(part)} pairs for {label_a} <-> {label_b}")

    result = pd.concat(selected_parts, ignore_index=True)

    result = result.reset_index(drop=True)
    result.insert(0, "pair_id", [f"rev_{i:03d}" for i in range(1, len(result) + 1)])

    result["decision"] = ""
    result["target_label"] = ""
    result["comment"] = ""

    result = result[
        [
            "pair_id",
            "target_pair",
            "sample_id_1",
            "sample_id_2",
            "text_1",
            "text_2",
            "label_1",
            "label_2",
            "cosine_similarity",
            "decision",
            "target_label",
            "comment",
        ]
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\n[OK] Saved manual review file: {OUTPUT_PATH}")
    print(f"[INFO] Total review pairs: {len(result)}")

    print("\n[INFO] Distribution by target_pair:")
    print(result["target_pair"].value_counts())

    print("\n[INFO] Preview:")
    print(
        result[
            ["pair_id", "target_pair", "label_1", "label_2", "cosine_similarity"]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()