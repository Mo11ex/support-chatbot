from pathlib import Path
import re
import pandas as pd


INPUT_PATH = Path("ml/data/processed/full_dataset.csv")
OUTPUT_PATH = Path("ml/data/processed/full_dataset_stage1_base.csv")


def normalize_text(text: str) -> str:
    """
    Базовая нормализация текста для Stage 1:
    - lower case
    - нормализация пробелов
    - сохранение эмодзи
    - без агрессивного удаления пунктуации
    """
    if pd.isna(text):
        return None

    text = str(text)

    # замена неразрывных пробелов и табов
    text = text.replace("\xa0", " ").replace("\t", " ")

    # trim
    text = text.strip()

    # lower
    text = text.lower()

    # нормализация множественных пробелов
    text = re.sub(r"\s+", " ", text)

    return text if text else None


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_columns = {"text", "raw_label", "label", "source", "is_synthetic"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = pd.DataFrame()

    out["sample_id"] = [f"stg1_{i:06d}" for i in range(1, len(df) + 1)]
    out["text_original"] = df["text"].astype(str)
    out["text_normalized"] = df["text"].map(normalize_text)

    out["raw_label"] = df["raw_label"].astype(str)
    out["label"] = df["label"].astype(str)
    out["source"] = df["source"].astype(str)
    out["is_synthetic"] = df["is_synthetic"].astype(bool)

    out["is_augmented"] = False
    out["is_hard_negative"] = False

    # удаляем пустые нормализованные тексты
    before_drop = len(out)
    out = out.dropna(subset=["text_normalized"]).reset_index(drop=True)
    dropped = before_drop - len(out)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"[OK] Saved Stage 1 base dataset to: {OUTPUT_PATH}")
    print(f"[INFO] Shape: {out.shape}")
    print(f"[INFO] Dropped empty normalized texts: {dropped}")

    print("\n[INFO] Columns:")
    print(out.columns.tolist())

    print("\n[INFO] Label distribution:")
    print(out["label"].value_counts().sort_index())

    print("\n[INFO] Source distribution:")
    print(out["source"].value_counts())

    print("\n[INFO] Synthetic distribution:")
    print(out["is_synthetic"].value_counts())

    print("\n[INFO] Augmented flag distribution:")
    print(out["is_augmented"].value_counts())

    print("\n[INFO] Hard negative flag distribution:")
    print(out["is_hard_negative"].value_counts())

    print("\n[INFO] Sample rows:")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    main()