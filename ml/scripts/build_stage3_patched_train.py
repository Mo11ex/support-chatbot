from pathlib import Path
import json
import pandas as pd


# Входные файлы
BASE_TRAIN = Path("ml/data/splits/stage1/train_with_hard_negatives.csv")
SHORT_QUERIES = Path("ml/data/interim/review/stage3_short_queries_manual.csv")
HARD_NEGATIVES = Path("ml/data/interim/review/stage3_hard_negatives_manual.csv")

# Выходные файлы
OUTPUT_DIR = Path("ml/data/splits/stage3")
OUTPUT_TRAIN = OUTPUT_DIR / "train_stage3_patched.csv"
OUTPUT_SUMMARY = Path("ml/logs/reports/stage3_patched_train_summary.json")

SEED = 42


def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sample_id", "parent_sample_id", "text_original", "text_normalized",
        "label", "source", "is_synthetic", "is_augmented", "is_hard_negative",
        "augmentation_method", "hard_negative_pair", "hard_negative_pair_id",
        "hard_negative_pair_similarity", "notes"
    ]
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df


def main():
    if not BASE_TRAIN.exists():
        raise FileNotFoundError(f"Missing file: {BASE_TRAIN}")
    if not SHORT_QUERIES.exists():
        raise FileNotFoundError(f"Missing file: {SHORT_QUERIES}")
    if not HARD_NEGATIVES.exists():
        raise FileNotFoundError(f"Missing file: {HARD_NEGATIVES}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(BASE_TRAIN)
    short = pd.read_csv(SHORT_QUERIES)
    hard = pd.read_csv(HARD_NEGATIVES)

    base = ensure_cols(base)
    short = ensure_cols(short)
    hard = ensure_cols(hard)

    # Генерация sample_id для новых строк
    short["sample_id"] = [f"patch_short_{i:04d}" for i in range(1, len(short) + 1)]
    hard["sample_id"] = [f"patch_hneg_{i:04d}" for i in range(1, len(hard) + 1)]

    # Объединяем
    patched = pd.concat([base, short, hard], ignore_index=True)

    # Дедупликация:
    # Оставляем только одну строку для уникальной пары (text_normalized, label)
    # Сортируем так, чтобы ручные патчи остались в приоритете (по id 'patch_')
    patched["is_patch"] = patched["sample_id"].str.startswith("patch_")
    patched = patched.sort_values("is_patch", ascending=False)
    patched = patched.drop_duplicates(subset=["text_normalized", "label"], keep="first")
    patched = patched.drop(columns=["is_patch"]).reset_index(drop=True)

    # Сохраняем
    patched.to_csv(OUTPUT_TRAIN, index=False, encoding="utf-8")

    summary = {
        "base_train_rows": int(len(base)),
        "short_queries_added": int(len(short)),
        "hard_negatives_added": int(len(hard)),
        "patched_train_rows": int(len(patched)),
        "distribution_after_patch": patched["label"].value_counts().sort_index().to_dict(),
        "source_distribution": patched["source"].value_counts().sort_values(ascending=False).to_dict()
    }

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Base train rows: {len(base)}")
    print(f"[INFO] Short queries added: {len(short)}")
    print(f"[INFO] Hard negatives added: {len(hard)}")
    print(f"[INFO] Patched train rows (after dedup): {len(patched)}")

    print("\n[INFO] Label distribution in patched train:")
    print(patched["label"].value_counts().sort_index())

    print(f"\n[OK] Saved: {OUTPUT_TRAIN}")
    print(f"[OK] Saved: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()