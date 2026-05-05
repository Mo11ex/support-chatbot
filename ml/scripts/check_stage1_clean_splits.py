from pathlib import Path
import json
import pandas as pd


SPLIT_DIR = Path("ml/data/splits/stage1")
FILES = {
    "train_clean": SPLIT_DIR / "train_clean.csv",
    "val": SPLIT_DIR / "val.csv",
    "test": SPLIT_DIR / "test.csv",
}
REPORT_PATH = Path("ml/logs/reports/stage1_clean_splits_check.json")


def normalize_text(text):
    if pd.isna(text):
        return ""
    return " ".join(str(text).strip().lower().split())


def load_split(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_csv(path)
    df["text_norm_check"] = df["text_normalized"].map(normalize_text)
    return df


def intersection_count(df1, df2, cols):
    merged = df1[cols].drop_duplicates().merge(
        df2[cols].drop_duplicates(),
        on=cols,
        how="inner"
    )
    return len(merged)


def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = {name: load_split(path) for name, path in FILES.items()}

    report = {
        "sizes": {},
        "nulls": {},
        "all_classes_present": True,
        "augmented_flags_clean": True,
        "hard_negative_flags_clean": True,
        "intersections": {},
        "status": "PASS"
    }

    for name, df in data.items():
        report["sizes"][name] = int(len(df))
        report["nulls"][name] = {
            "text_original_nulls": int(df["text_original"].isna().sum()),
            "text_normalized_nulls": int(df["text_normalized"].isna().sum()),
            "label_nulls": int(df["label"].isna().sum()),
        }

        if df["is_augmented"].any():
            report["augmented_flags_clean"] = False
            report["status"] = "FAIL"

        if df["is_hard_negative"].any():
            report["hard_negative_flags_clean"] = False
            report["status"] = "FAIL"

        if any(v > 0 for v in report["nulls"][name].values()):
            report["status"] = "FAIL"

    # Проверка покрытия классов
    base_labels = sorted(data["train_clean"]["label"].unique().tolist())
    for name, df in data.items():
        labels = sorted(df["label"].unique().tolist())
        if labels != base_labels:
            report["all_classes_present"] = False
            report["status"] = "FAIL"

    pairs = [
        ("train_clean", "val"),
        ("train_clean", "test"),
        ("val", "test"),
    ]

    for a, b in pairs:
        exact = intersection_count(data[a], data[b], ["sample_id"])
        same_text = intersection_count(data[a], data[b], ["text_norm_check"])
        report["intersections"][f"{a}__{b}"] = {
            "same_sample_id": int(exact),
            "same_text_normalized": int(same_text),
        }
        if exact > 0 or same_text > 0:
            report["status"] = "FAIL"

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[INFO] Split sizes:")
    for k, v in report["sizes"].items():
        print(f"  {k}: {v}")

    print("\n[INFO] Null checks:")
    for k, v in report["nulls"].items():
        print(f"  {k}: {v}")

    print("\n[INFO] Intersections:")
    for k, v in report["intersections"].items():
        print(f"  {k}: {v}")

    print(f"\n[INFO] all_classes_present: {report['all_classes_present']}")
    print(f"[INFO] augmented_flags_clean: {report['augmented_flags_clean']}")
    print(f"[INFO] hard_negative_flags_clean: {report['hard_negative_flags_clean']}")

    print(f"\n[RESULT] {report['status']}")
    print(f"[INFO] Full report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()