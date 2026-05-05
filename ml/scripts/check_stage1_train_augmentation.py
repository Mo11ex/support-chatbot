from pathlib import Path
import json
import pandas as pd


TARGET_PER_CLASS = 500

TRAIN_WITH_HN_PATH = Path("ml/data/splits/stage1/train_with_hard_negatives.csv")
AUG_ONLY_PATH = Path("ml/data/splits/stage1/train_augmented_only.csv")
TRAIN_FINAL_PATH = Path("ml/data/splits/stage1/train_augmented.csv")
VAL_PATH = Path("ml/data/splits/stage1/val.csv")
TEST_PATH = Path("ml/data/splits/stage1/test.csv")

REPORT_PATH = Path("ml/logs/reports/stage1_train_augmentation_check.json")


def main():
    for path in [TRAIN_WITH_HN_PATH, AUG_ONLY_PATH, TRAIN_FINAL_PATH, VAL_PATH, TEST_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_with_hn = pd.read_csv(TRAIN_WITH_HN_PATH)
    aug_only = pd.read_csv(AUG_ONLY_PATH)
    train_final = pd.read_csv(TRAIN_FINAL_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)

    report = {
        "train_with_hn_rows": int(len(train_with_hn)),
        "aug_only_rows": int(len(aug_only)),
        "train_final_rows": int(len(train_final)),
        "all_aug_rows_flag_true": True,
        "all_aug_rows_have_parent": True,
        "all_aug_rows_not_hard_negative": True,
        "all_classes_reach_target": True,
        "merged_size_correct": True,
        "sample_id_unique": True,
        "val_clean": True,
        "test_clean": True,
        "status": "PASS"
    }

    if len(aug_only) > 0:
        if not aug_only["is_augmented"].all():
            report["all_aug_rows_flag_true"] = False
            report["status"] = "FAIL"

        if aug_only["parent_sample_id"].fillna("").astype(str).eq("").any():
            report["all_aug_rows_have_parent"] = False
            report["status"] = "FAIL"

        if aug_only["is_hard_negative"].any():
            report["all_aug_rows_not_hard_negative"] = False
            report["status"] = "FAIL"

    final_dist = train_final["label"].value_counts().sort_index()
    if (final_dist < TARGET_PER_CLASS).any():
        report["all_classes_reach_target"] = False
        report["status"] = "FAIL"

    if len(train_final) != len(train_with_hn) + len(aug_only):
        report["merged_size_correct"] = False
        report["status"] = "FAIL"

    if train_final["sample_id"].duplicated().any():
        report["sample_id_unique"] = False
        report["status"] = "FAIL"

    if val["is_augmented"].any() or val["is_hard_negative"].any():
        report["val_clean"] = False
        report["status"] = "FAIL"

    if test["is_augmented"].any() or test["is_hard_negative"].any():
        report["test_clean"] = False
        report["status"] = "FAIL"

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[INFO] train_with_hn_rows:", report["train_with_hn_rows"])
    print("[INFO] aug_only_rows:", report["aug_only_rows"])
    print("[INFO] train_final_rows:", report["train_final_rows"])

    print("\n[INFO] Final class distribution:")
    print(final_dist)

    print("\n[INFO] Checks:")
    print("  all_aug_rows_flag_true:", report["all_aug_rows_flag_true"])
    print("  all_aug_rows_have_parent:", report["all_aug_rows_have_parent"])
    print("  all_aug_rows_not_hard_negative:", report["all_aug_rows_not_hard_negative"])
    print("  all_classes_reach_target:", report["all_classes_reach_target"])
    print("  merged_size_correct:", report["merged_size_correct"])
    print("  sample_id_unique:", report["sample_id_unique"])
    print("  val_clean:", report["val_clean"])
    print("  test_clean:", report["test_clean"])

    print(f"\n[RESULT] {report['status']}")
    print(f"[INFO] Full report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()