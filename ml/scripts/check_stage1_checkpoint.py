from pathlib import Path
import json
import pandas as pd


FILES = {
    "stage1_base": Path("ml/data/processed/full_dataset_stage1_base.csv"),
    "stage1_clean": Path("ml/data/processed/full_dataset_stage1_clean.csv"),
    "manual_review": Path("ml/data/interim/review/manual_borderline_review.csv"),
    "hard_negative_pairs": Path("ml/data/interim/review/hard_negative_pairs_from_review.csv"),
    "train_clean": Path("ml/data/splits/stage1/train_clean.csv"),
    "val": Path("ml/data/splits/stage1/val.csv"),
    "test": Path("ml/data/splits/stage1/test.csv"),
    "train_with_hn": Path("ml/data/splits/stage1/train_with_hard_negatives.csv"),
    "train_augmented": Path("ml/data/splits/stage1/train_augmented.csv"),
    "notebook": Path("ml/notebooks/12_stage1_data_refinement.ipynb"),
    "clean_split_report": Path("ml/logs/reports/stage1_clean_splits_check.json"),
    "augmentation_report": Path("ml/logs/reports/stage1_train_augmentation_check.json"),
}

REPORT_PATH = Path("ml/logs/reports/stage1_checkpoint_report.json")


def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "files_exist": {},
        "manual_review_pairs": 0,
        "hard_negative_pairs": 0,
        "all_classes_500_plus": True,
        "val_clean": True,
        "test_clean": True,
        "status": "PASS",
        "notes": []
    }

    for name, path in FILES.items():
        exists = path.exists()
        report["files_exist"][name] = exists
        if not exists:
            report["status"] = "FAIL"

    if report["files_exist"]["manual_review"]:
        manual_review = pd.read_csv(FILES["manual_review"])
        report["manual_review_pairs"] = int(len(manual_review))
        if len(manual_review) < 50:
            report["status"] = "FAIL"

    if report["files_exist"]["hard_negative_pairs"]:
        hard_neg = pd.read_csv(FILES["hard_negative_pairs"])
        report["hard_negative_pairs"] = int(len(hard_neg))
        report["notes"].append(
            "Hard-negatives are included in pilot volume and planned for further expansion after classifier error analysis."
        )

    if report["files_exist"]["train_augmented"]:
        train_aug = pd.read_csv(FILES["train_augmented"])
        dist = train_aug["label"].value_counts()
        if (dist < 500).any():
            report["all_classes_500_plus"] = False
            report["status"] = "FAIL"

    if report["files_exist"]["val"]:
        val = pd.read_csv(FILES["val"])
        if val["is_augmented"].any() or val["is_hard_negative"].any():
            report["val_clean"] = False
            report["status"] = "FAIL"

    if report["files_exist"]["test"]:
        test = pd.read_csv(FILES["test"])
        if test["is_augmented"].any() or test["is_hard_negative"].any():
            report["test_clean"] = False
            report["status"] = "FAIL"

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[INFO] File existence:")
    for name, exists in report["files_exist"].items():
        print(f"  {name}: {exists}")

    print(f"\n[INFO] manual_review_pairs: {report['manual_review_pairs']}")
    print(f"[INFO] hard_negative_pairs: {report['hard_negative_pairs']}")
    print(f"[INFO] all_classes_500_plus: {report['all_classes_500_plus']}")
    print(f"[INFO] val_clean: {report['val_clean']}")
    print(f"[INFO] test_clean: {report['test_clean']}")

    if report["notes"]:
        print("\n[INFO] Notes:")
        for note in report["notes"]:
            print(f"  - {note}")

    print(f"\n[RESULT] STAGE 1 CHECKPOINT: {report['status']}")
    print(f"[INFO] Full report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()