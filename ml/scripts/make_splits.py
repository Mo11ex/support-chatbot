from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

INPUT_PATH = Path("ml/data/processed/full_dataset.csv")
OUTPUT_DIR = Path("ml/data/splits")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    # 15% test
    train_val, test = train_test_split(
        df,
        test_size=0.15,
        random_state=SEED,
        stratify=df["label"]
    )

    # из оставшихся 85% выделяем val так, чтобы val = 15% от общего датасета
    val_relative_size = 0.15 / 0.85

    train, val = train_test_split(
        train_val,
        test_size=val_relative_size,
        random_state=SEED,
        stratify=train_val["label"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding="utf-8")
    val.to_csv(OUTPUT_DIR / "val.csv", index=False, encoding="utf-8")
    test.to_csv(OUTPUT_DIR / "test.csv", index=False, encoding="utf-8")

    print("[OK] Splits saved to:", OUTPUT_DIR)
    print("Train:", train.shape)
    print("Val:", val.shape)
    print("Test:", test.shape)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"\n{name.upper()} distribution:")
        print(part["label"].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()