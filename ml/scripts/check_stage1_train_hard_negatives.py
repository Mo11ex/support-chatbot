from pathlib import Path
import json
import pandas as pd


TRAIN_CLEAN_PATH = Path("ml/data/splits/stage1/train_clean.csv")
VAL_PATH = Path("ml/data/splits/stage1/val.csv")
TEST_PATH = Path("ml/data/splits/stage1/test.csv")
HN_ONLY_PATH = Path("ml/data/splits/stage1/train_hard_negatives_only.csv")
TRAIN_WITH_HN_PATH = Path("ml/data/splits/stage1/train_with_hard_negatives.csv")

REPORT_PATH = Path("ml/logs/reports/stage1_train_hard_negatives_check.json")


def main():
    for path in [TRAIN_CLEAN_PATH, VAL_PATH, TEST_PATH, HN_ONLY_PATH, TRAIN_WITH_HN_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_clean = pd.read_csv(TRAIN_CLEAN_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)
    hn_only = pd.read_csv(HN_ONLY_PATH)
    train_with_hn = pd.read_csv(TRAIN_WITH_HN_PATH)

    train_ids = set(train_clean["sample_id"])
    val_ids = set(val["sample_id"])
    test_ids = set(test["sample_id"])

    report = {
        "train_clean_rows": int(len(train_clean)),
        "hn_only_rows": int(len(hn_only)),
        "train_with_hn_rows": int(len(train_with_hn)),
        "all_hn_flag_true": True,
        "all_hn_augmented_false": True,
        "all_parents_from_train": True,
        "no_parents_from_val": True,
        "no_parents_from_test": True,
        "merged_size_correct": True,
        "sample_id_unique_in_merged": True,
        "status": "PASS"
    }

    if len(hn_only) > 0:
        if not hn_only["is_hard_negative"].all():
            report["all_hn_flag_true"] = False
            report["status"] = "FAIL"

        if hn_only["is_augmented"].any():
            report["all_hn_augmented_false"] = False
            report["status"] = "FAIL"

        parent_ids = set(hn_only["parent_sample_id"].dropna().astype(str))

        if not parent_ids.issubset(train_ids):
            report["all_parents_from_train"] = False
            report["status"] = "FAIL"

        if len(parent_ids.intersection(val_ids)) > 0:
            report["no_parents_from_val"] = False
            report["status"] = "FAIL"

        if len(parent_ids.intersection(test_ids)) > 0:
            report["no_parents_from_test"] = False
            report["status"] = "FAIL"

    expected_merged_rows = len(train_clean) + len(hn_only)
    if len(train_with_hn) != expected_merged_rows:
        report["merged_size_correct"] = False
        report["status"] = "FAIL"

    if train_with_hn["sample_id"].duplicated().any():
        report["sample_id_unique_in_merged"] = False
        report["status"] = "FAIL"

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[INFO] train_clean_rows:", report["train_clean_rows"])
    print("[INFO] hn_only_rows:", report["hn_only_rows"])
    print("[INFO] train_with_hn_rows:", report["train_with_hn_rows"])

    print("\n[INFO] Checks:")
    print("  all_hn_flag_true:", report["all_hn_flag_true"])
    print("  all_hn_augmented_false:", report["all_hn_augmented_false"])
    print("  all_parents_from_train:", report["all_parents_from_train"])
    print("  no_parents_from_val:", report["no_parents_from_val"])
    print("  no_parents_from_test:", report["no_parents_from_test"])
    print("  merged_size_correct:", report["merged_size_correct"])
    print("  sample_id_unique_in_merged:", report["sample_id_unique_in_merged"])

    print(f"\n[RESULT] {report['status']}")
    print(f"[INFO] Full report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()