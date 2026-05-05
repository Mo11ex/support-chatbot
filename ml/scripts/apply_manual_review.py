from pathlib import Path
import json
import pandas as pd


BASE_DATASET_PATH = Path("ml/data/processed/full_dataset_stage1_base.csv")
REVIEW_PATH = Path("ml/data/interim/review/manual_borderline_review.csv")

OUTPUT_DATASET_PATH = Path("ml/data/processed/full_dataset_stage1_clean.csv")
OUTPUT_HARD_NEG_PATH = Path("ml/data/interim/review/hard_negative_pairs_from_review.csv")
OUTPUT_SUMMARY_PATH = Path("ml/logs/reports/stage1_manual_review_apply_summary.json")


VALID_DECISIONS = {
    "keep",
    "relabel_1",
    "relabel_2",
    "drop_1",
    "drop_2",
    "drop_both",
    "hard_negative",
}


def main():
    if not BASE_DATASET_PATH.exists():
        raise FileNotFoundError(f"Base dataset not found: {BASE_DATASET_PATH}")
    if not REVIEW_PATH.exists():
        raise FileNotFoundError(f"Review file not found: {REVIEW_PATH}")

    OUTPUT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_HARD_NEG_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(BASE_DATASET_PATH)
    review = pd.read_csv(REVIEW_PATH)

    # normalize decision column
    review["decision"] = review["decision"].fillna("").astype(str).str.strip()
    review["target_label"] = review["target_label"].fillna("").astype(str).str.strip()

    invalid = review[~review["decision"].isin(VALID_DECISIONS)]
    if len(invalid) > 0:
        raise ValueError(
            f"Found invalid decisions in review file:\n"
            f"{invalid[['pair_id', 'decision']].to_string(index=False)}"
        )

    # track changes
    dropped_ids = set()
    relabel_map = {}
    hard_negative_rows = []

    review_stats = {
        "keep": 0,
        "relabel_1": 0,
        "relabel_2": 0,
        "drop_1": 0,
        "drop_2": 0,
        "drop_both": 0,
        "hard_negative": 0,
    }

    for _, row in review.iterrows():
        decision = row["decision"]
        review_stats[decision] += 1

        sid1 = row["sample_id_1"]
        sid2 = row["sample_id_2"]
        target_label = row["target_label"]

        if decision == "keep":
            continue

        elif decision == "relabel_1":
            if not target_label:
                raise ValueError(f"Missing target_label for pair {row['pair_id']} with decision relabel_1")
            relabel_map[sid1] = target_label

        elif decision == "relabel_2":
            if not target_label:
                raise ValueError(f"Missing target_label for pair {row['pair_id']} with decision relabel_2")
            relabel_map[sid2] = target_label

        elif decision == "drop_1":
            dropped_ids.add(sid1)

        elif decision == "drop_2":
            dropped_ids.add(sid2)

        elif decision == "drop_both":
            dropped_ids.add(sid1)
            dropped_ids.add(sid2)

        elif decision == "hard_negative":
            hard_negative_rows.append({
                "pair_id": row["pair_id"],
                "target_pair": row["target_pair"],
                "sample_id_1": sid1,
                "sample_id_2": sid2,
                "text_1": row["text_1"],
                "text_2": row["text_2"],
                "label_1": row["label_1"],
                "label_2": row["label_2"],
                "cosine_similarity": row["cosine_similarity"],
                "decision": decision,
                "comment": row.get("comment", "")
            })

    # apply relabeling
    df_clean = df.copy()
    df_clean["label_before_review"] = df_clean["label"]

    relabeled_count = 0
    for sid, new_label in relabel_map.items():
        mask = df_clean["sample_id"] == sid
        if mask.any():
            old_label = df_clean.loc[mask, "label"].iloc[0]
            if old_label != new_label:
                df_clean.loc[mask, "label"] = new_label
                relabeled_count += 1

    # apply drops
    before_drop = len(df_clean)
    df_clean = df_clean[~df_clean["sample_id"].isin(dropped_ids)].copy()
    dropped_count = before_drop - len(df_clean)

    # save hard-negative pair file
    hard_neg_df = pd.DataFrame(hard_negative_rows)
    hard_neg_df.to_csv(OUTPUT_HARD_NEG_PATH, index=False, encoding="utf-8")

    # save clean dataset
    df_clean.to_csv(OUTPUT_DATASET_PATH, index=False, encoding="utf-8")

    summary = {
        "base_dataset_path": str(BASE_DATASET_PATH),
        "review_path": str(REVIEW_PATH),
        "base_rows": int(len(df)),
        "reviewed_pairs": int(len(review)),
        "review_decision_counts": {k: int(v) for k, v in review_stats.items()},
        "relabeled_samples": int(relabeled_count),
        "dropped_unique_sample_ids": int(len(dropped_ids)),
        "dropped_rows_in_dataset": int(dropped_count),
        "hard_negative_pairs_saved": int(len(hard_neg_df)),
        "clean_dataset_rows": int(len(df_clean)),
        "output_dataset_path": str(OUTPUT_DATASET_PATH),
        "output_hard_negative_path": str(OUTPUT_HARD_NEG_PATH),
    }

    with open(OUTPUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # console report
    print(f"[INFO] Base dataset rows: {len(df)}")
    print(f"[INFO] Reviewed pairs: {len(review)}")

    print("\n[INFO] Review decision counts:")
    for k, v in review_stats.items():
        print(f"  {k}: {v}")

    print(f"\n[INFO] Relabeled samples: {relabeled_count}")
    print(f"[INFO] Unique dropped sample_ids: {len(dropped_ids)}")
    print(f"[INFO] Dropped rows in dataset: {dropped_count}")
    print(f"[INFO] Hard-negative pairs saved: {len(hard_neg_df)}")

    print(f"\n[OK] Saved clean dataset: {OUTPUT_DATASET_PATH}")
    print(f"[OK] Saved hard-negative pairs: {OUTPUT_HARD_NEG_PATH}")
    print(f"[OK] Saved summary JSON: {OUTPUT_SUMMARY_PATH}")

    print(f"\n[INFO] Clean dataset shape: {df_clean.shape}")
    print("\n[INFO] Clean label distribution:")
    print(df_clean["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()