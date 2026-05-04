from pathlib import Path
import json
import pandas as pd

SPLIT_DIR = Path("ml/data/splits")
LOG_DIR = Path("ml/logs")
FILES = {
    "train": SPLIT_DIR / "train.csv",
    "val": SPLIT_DIR / "val.csv",
    "test": SPLIT_DIR / "test.csv",
}


def normalize_text(text):
    if pd.isna(text):
        return ""
    return " ".join(str(text).strip().lower().split())


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_csv(path)
    df["text_norm"] = df["text"].map(normalize_text)
    return df


def get_distribution(df: pd.DataFrame) -> dict:
    return df["label"].value_counts(normalize=True).sort_index().to_dict()


def intersection_count(df1: pd.DataFrame, df2: pd.DataFrame, cols: list[str]) -> int:
    merged = df1[cols].drop_duplicates().merge(
        df2[cols].drop_duplicates(),
        on=cols,
        how="inner"
    )
    return len(merged)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    data = {name: load_split(path) for name, path in FILES.items()}

    report = {
        "sizes": {},
        "columns_match": True,
        "null_checks": {},
        "class_coverage": {},
        "distributions": {},
        "duplicates_within_split": {},
        "intersections_between_splits": {},
        "status": "PASS"
    }

    # 1. размеры
    total = 0
    for name, df in data.items():
        report["sizes"][name] = len(df)
        total += len(df)

    report["sizes"]["total"] = total

    # 2. одинаковые колонки
    base_cols = [c for c in data["train"].columns if c != "text_norm"]
    for name, df in data.items():
        cols = [c for c in df.columns if c != "text_norm"]
        if cols != base_cols:
            report["columns_match"] = False
            report["status"] = "FAIL"

    # 3. null checks
    for name, df in data.items():
        report["null_checks"][name] = {
            "text_nulls": int(df["text"].isna().sum()),
            "label_nulls": int(df["label"].isna().sum())
        }
        if report["null_checks"][name]["text_nulls"] > 0 or report["null_checks"][name]["label_nulls"] > 0:
            report["status"] = "FAIL"

    # 4. покрытие классов
    all_labels = sorted(data["train"]["label"].unique().tolist())
    for name, df in data.items():
        labels = sorted(df["label"].unique().tolist())
        report["class_coverage"][name] = labels
        if labels != all_labels:
            report["status"] = "FAIL"

    # 5. распределения
    for name, df in data.items():
        report["distributions"][name] = get_distribution(df)

    # 6. дубликаты внутри каждого сплита
    for name, df in data.items():
        exact_duplicates = int(df.duplicated(subset=["text", "label"]).sum())
        text_duplicates = int(df.duplicated(subset=["text_norm"]).sum())
        conflicting_labels = int(
            (df.groupby("text_norm")["label"].nunique() > 1).sum()
        )

        report["duplicates_within_split"][name] = {
            "exact_text_label_duplicates": exact_duplicates,
            "same_text_duplicates": text_duplicates,
            "same_text_with_different_labels": conflicting_labels
        }

    # 7. пересечения между сплитами
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        exact_pair_leak = intersection_count(data[a], data[b], ["text", "label"])
        text_leak = intersection_count(data[a], data[b], ["text_norm"])

        report["intersections_between_splits"][f"{a}__{b}"] = {
            "same_text_and_label": exact_pair_leak,
            "same_text": text_leak
        }

        if exact_pair_leak > 0 or text_leak > 0:
            report["status"] = "FAIL"

    # save
    out_path = LOG_DIR / "check_splits_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # print short report
    print("[INFO] Split sizes:")
    for k, v in report["sizes"].items():
        print(f"  {k}: {v}")

    print("\n[INFO] Null checks:")
    for split_name, checks in report["null_checks"].items():
        print(f"  {split_name}: {checks}")

    print("\n[INFO] Intersections between splits:")
    for pair, values in report["intersections_between_splits"].items():
        print(f"  {pair}: {values}")

    print(f"\n[RESULT] {report['status']}")
    print(f"[INFO] Full report saved to: {out_path}")


if __name__ == "__main__":
    main()