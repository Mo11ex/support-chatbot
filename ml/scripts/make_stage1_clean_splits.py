from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42

INPUT_PATH = Path("ml/data/processed/full_dataset_stage1_clean.csv")
OUTPUT_DIR = Path("ml/data/splits/stage1")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "sample_id",
        "text_original",
        "text_normalized",
        "label",
        "source",
        "is_synthetic",
        "is_augmented",
        "is_hard_negative",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # защита: clean dataset не должен содержать augmented/hard_negative rows
    if df["is_augmented"].any():
        raise ValueError("Clean dataset contains is_augmented=True rows")
    if df["is_hard_negative"].any():
        raise ValueError("Clean dataset contains is_hard_negative=True rows")

    # 15% test
    train_val, test = train_test_split(
        df,
        test_size=0.15,
        random_state=SEED,
        stratify=df["label"]
    )

    # из оставшихся 85% выделяем val = 15% от полного датасета
    val_relative_size = 0.15 / 0.85

    train, val = train_test_split(
        train_val,
        test_size=val_relative_size,
        random_state=SEED,
        stratify=train_val["label"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(OUTPUT_DIR / "train_clean.csv", index=False, encoding="utf-8")
    val.to_csv(OUTPUT_DIR / "val.csv", index=False, encoding="utf-8")
    test.to_csv(OUTPUT_DIR / "test.csv", index=False, encoding="utf-8")

    print(f"[OK] Saved Stage 1 clean splits to: {OUTPUT_DIR}")
    print(f"Train clean: {train.shape}")
    print(f"Val: {val.shape}")
    print(f"Test: {test.shape}")

    for name, part in [("train_clean", train), ("val", val), ("test", test)]:
        print(f"\n{name.upper()} distribution:")
        print(part["label"].value_counts().sort_index())
        print("\nNormalized distribution:")
        print(part["label"].value_counts(normalize=True).sort_index())

    print("\n[INFO] is_augmented distribution:")
    print("train_clean:", train["is_augmented"].value_counts().to_dict())
    print("val:", val["is_augmented"].value_counts().to_dict())
    print("test:", test["is_augmented"].value_counts().to_dict())

    print("\n[INFO] is_hard_negative distribution:")
    print("train_clean:", train["is_hard_negative"].value_counts().to_dict())
    print("val:", val["is_hard_negative"].value_counts().to_dict())
    print("test:", test["is_hard_negative"].value_counts().to_dict())


if __name__ == "__main__":
    main()