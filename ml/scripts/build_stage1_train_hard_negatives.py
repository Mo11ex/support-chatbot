from pathlib import Path
import json
import pandas as pd


TRAIN_PATH = Path("ml/data/splits/stage1/train_clean.csv")
VAL_PATH = Path("ml/data/splits/stage1/val.csv")
TEST_PATH = Path("ml/data/splits/stage1/test.csv")
HARD_NEG_PAIRS_PATH = Path("ml/data/interim/review/hard_negative_pairs_from_review.csv")

OUTPUT_HN_ONLY_PATH = Path("ml/data/splits/stage1/train_hard_negatives_only.csv")
OUTPUT_TRAIN_WITH_HN_PATH = Path("ml/data/splits/stage1/train_with_hard_negatives.csv")
OUTPUT_SUMMARY_PATH = Path("ml/logs/reports/stage1_train_hard_negatives_summary.json")


EXTRA_COLS = [
    "parent_sample_id",
    "hard_negative_pair",
    "hard_negative_pair_id",
    "hard_negative_pair_similarity",
]


def ensure_extra_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in EXTRA_COLS:
        if col not in df.columns:
            df[col] = ""
    return df


def main():
    for path in [TRAIN_PATH, VAL_PATH, TEST_PATH, HARD_NEG_PAIRS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    OUTPUT_HN_ONLY_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TRAIN_WITH_HN_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)
    hard_pairs = pd.read_csv(HARD_NEG_PAIRS_PATH)

    train = ensure_extra_cols(train)
    val = ensure_extra_cols(val)
    test = ensure_extra_cols(test)

    # На случай, если в будущем файл будет содержать не только hard_negative
    if "decision" in hard_pairs.columns:
        hard_pairs = hard_pairs[hard_pairs["decision"] == "hard_negative"].copy()

    train_ids = set(train["sample_id"])
    val_ids = set(val["sample_id"])
    test_ids = set(test["sample_id"])

    train_lookup = train.set_index("sample_id")

    rows = []
    skipped = {
        "from_val": 0,
        "from_test": 0,
        "missing_in_all_splits": 0,
    }

    for _, pair in hard_pairs.iterrows():
        for idx in [1, 2]:
            sid = pair[f"sample_id_{idx}"]

            if sid in train_ids:
                original = train_lookup.loc[sid].to_dict()

                new_row = dict(original)
                new_row["sample_id"] = f"hneg_{pair['pair_id']}_{idx}_{sid}"
                new_row["parent_sample_id"] = sid
                new_row["hard_negative_pair"] = pair["target_pair"]
                new_row["hard_negative_pair_id"] = pair["pair_id"]
                new_row["hard_negative_pair_similarity"] = pair["cosine_similarity"]

                new_row["is_hard_negative"] = True
                new_row["is_augmented"] = False

                # сохраняем provenance
                new_row["source"] = f"{original['source']}|hard_negative_manual"

                rows.append(new_row)

            elif sid in val_ids:
                skipped["from_val"] += 1
            elif sid in test_ids:
                skipped["from_test"] += 1
            else:
                skipped["missing_in_all_splits"] += 1

    hard_neg_raw = pd.DataFrame(rows)

    # Дедупликация:
    # если один и тот же sample участвует в нескольких hard-negative парах,
    # оставляем только самый сильный по similarity
    if not hard_neg_raw.empty:
        hard_neg_raw = hard_neg_raw.sort_values(
            by="hard_negative_pair_similarity",
            ascending=False
        )
        hard_neg_unique = hard_neg_raw.drop_duplicates(
            subset=["parent_sample_id"],
            keep="first"
        ).reset_index(drop=True)
    else:
        hard_neg_unique = hard_neg_raw.copy()

    # Расширяем исходный train служебными колонками
    train_for_merge = train.copy()

    # merged train
    train_with_hn = pd.concat(
        [train_for_merge, hard_neg_unique],
        ignore_index=True
    )

    # save
    hard_neg_unique.to_csv(OUTPUT_HN_ONLY_PATH, index=False, encoding="utf-8")
    train_with_hn.to_csv(OUTPUT_TRAIN_WITH_HN_PATH, index=False, encoding="utf-8")

    summary = {
        "train_rows_before": int(len(train)),
        "hard_negative_pairs_input": int(len(hard_pairs)),
        "hard_negative_raw_rows_before_dedup": int(len(hard_neg_raw)),
        "hard_negative_rows_after_dedup": int(len(hard_neg_unique)),
        "train_rows_after": int(len(train_with_hn)),
        "skipped": {k: int(v) for k, v in skipped.items()},
        "hard_negative_label_distribution": (
            hard_neg_unique["label"].value_counts().sort_index().to_dict()
            if not hard_neg_unique.empty else {}
        ),
        "output_hn_only_path": str(OUTPUT_HN_ONLY_PATH),
        "output_train_with_hn_path": str(OUTPUT_TRAIN_WITH_HN_PATH),
    }

    with open(OUTPUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Train rows before: {len(train)}")
    print(f"[INFO] Hard-negative pairs input: {len(hard_pairs)}")
    print(f"[INFO] Hard-negative raw rows before dedup: {len(hard_neg_raw)}")
    print(f"[INFO] Hard-negative rows after dedup: {len(hard_neg_unique)}")
    print(f"[INFO] Train rows after merge: {len(train_with_hn)}")

    print("\n[INFO] Skipped rows:")
    for k, v in skipped.items():
        print(f"  {k}: {v}")

    print("\n[INFO] Hard-negative label distribution:")
    if not hard_neg_unique.empty:
        print(hard_neg_unique["label"].value_counts().sort_index())
    else:
        print("No hard-negative rows were added.")

    print(f"\n[OK] Saved hard negatives only: {OUTPUT_HN_ONLY_PATH}")
    print(f"[OK] Saved train with hard negatives: {OUTPUT_TRAIN_WITH_HN_PATH}")
    print(f"[OK] Saved summary JSON: {OUTPUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()