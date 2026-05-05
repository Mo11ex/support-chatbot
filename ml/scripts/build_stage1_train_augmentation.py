from pathlib import Path
import json
import random
import re
from collections import Counter

import pandas as pd


SEED = 42
TARGET_PER_CLASS = 520
MAX_ATTEMPTS_MULTIPLIER = 50

INPUT_TRAIN_PATH = Path("ml/data/splits/stage1/train_with_hard_negatives.csv")
OUTPUT_AUG_ONLY_PATH = Path("ml/data/splits/stage1/train_augmented_only.csv")
OUTPUT_TRAIN_FINAL_PATH = Path("ml/data/splits/stage1/train_augmented.csv")
OUTPUT_SUMMARY_PATH = Path("ml/logs/reports/stage1_train_augmentation_summary.json")


def normalize_text(text: str) -> str:
    if pd.isna(text):
        return None
    text = str(text)
    text = text.replace("\xa0", " ").replace("\t", " ")
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text if text else None


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "parent_sample_id" not in df.columns:
        df["parent_sample_id"] = ""

    if "augmentation_method" not in df.columns:
        df["augmentation_method"] = ""

    if "hard_negative_pair" not in df.columns:
        df["hard_negative_pair"] = ""

    if "hard_negative_pair_id" not in df.columns:
        df["hard_negative_pair_id"] = ""

    if "hard_negative_pair_similarity" not in df.columns:
        df["hard_negative_pair_similarity"] = ""

    return df


def find_word_spans(text: str):
    return [m.span() for m in re.finditer(r"[а-яА-Яa-zA-ZёЁ]{4,}", text)]


def typo_delete_char(text: str, rng: random.Random) -> str:
    spans = find_word_spans(text)
    if not spans:
        return text
    start, end = rng.choice(spans)
    word = text[start:end]
    if len(word) < 5:
        return text
    idx = rng.randint(1, len(word) - 2)
    new_word = word[:idx] + word[idx + 1:]
    return text[:start] + new_word + text[end:]


def typo_swap_adjacent(text: str, rng: random.Random) -> str:
    spans = find_word_spans(text)
    if not spans:
        return text
    start, end = rng.choice(spans)
    word = text[start:end]
    if len(word) < 5:
        return text
    idx = rng.randint(1, len(word) - 2)
    chars = list(word)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    new_word = "".join(chars)
    return text[:start] + new_word + text[end:]


def typo_duplicate_char(text: str, rng: random.Random) -> str:
    spans = find_word_spans(text)
    if not spans:
        return text
    start, end = rng.choice(spans)
    word = text[start:end]
    idx = rng.randint(1, len(word) - 2)
    new_word = word[:idx] + word[idx] + word[idx:]
    return text[:start] + new_word + text[end:]


def punctuation_variant(text: str, rng: random.Random) -> str:
    variants = []

    if text.endswith("?"):
        variants.append(text[:-1] + "??")
        variants.append(text[:-1] + "?!")
    elif text.endswith("."):
        variants.append(text[:-1] + "?")
        variants.append(text[:-1] + "!")
    else:
        variants.append(text + "?")
        variants.append(text + "!")

    variants.append(text.replace(",", ""))
    variants.append(re.sub(r"[!?]+$", "", text).strip())

    variants = [v for v in variants if v and v != text]
    return rng.choice(variants) if variants else text


def phrase_compression(text: str, rng: random.Random) -> str:
    variants = []

    replacements = [
        (r"\bя хотел бы\b", "хочу"),
        (r"\bя бы хотел\b", "хочу"),
        (r"\bмне нужна помощь\b", "помогите"),
        (r"\bмне нужно\b", "нужно"),
        (r"\bне могли бы вы\b", "можете"),
        (r"\bмогу ли я\b", "можно ли"),
        (r"\bкак я могу\b", "как"),
        (r"\bпожалуйста\b", ""),
        (r"\bне могли бы мне помочь\b", "помогите"),
        (r"\bмогу ли я получить помощь\b", "помогите"),
    ]

    for pattern, repl in replacements:
        candidate = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if candidate and candidate != text:
            variants.append(candidate)

    return rng.choice(variants) if variants else text


AUGMENTERS = [
    ("typo_delete_char", typo_delete_char),
    ("typo_swap_adjacent", typo_swap_adjacent),
    ("typo_duplicate_char", typo_duplicate_char),
    ("phrase_compression", phrase_compression),
    ("punctuation_variant", punctuation_variant),
]


def main():
    rng = random.Random(SEED)

    if not INPUT_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Input train file not found: {INPUT_TRAIN_PATH}")

    OUTPUT_AUG_ONLY_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(INPUT_TRAIN_PATH)
    train = ensure_columns(train)

    # Для аугментации берём только не-аугментированные строки.
    # Hard-negatives не аугментируем, чтобы не переуплотнять train пограничными кейсами.
    augmentation_pool = train[
        (train["is_augmented"] == False) & (train["is_hard_negative"] == False)
    ].copy()

    existing_keys = set(
        zip(
            train["text_normalized"].fillna("").astype(str),
            train["label"].astype(str)
        )
    )

    augmented_rows = []
    method_counter = Counter()
    class_added_counter = Counter()

    labels = sorted(train["label"].unique().tolist())

    for label in labels:
        current_count = int((train["label"] == label).sum())
        need = max(0, TARGET_PER_CLASS - current_count)

        class_pool = augmentation_pool[augmentation_pool["label"] == label].copy()
        if class_pool.empty:
            raise ValueError(f"No augmentation pool rows for label={label}")

        class_pool = class_pool.sample(frac=1, random_state=SEED).reset_index(drop=True)

        print(f"\n[INFO] Label: {label}")
        print(f"[INFO] Current count: {current_count}")
        print(f"[INFO] Target count: {TARGET_PER_CLASS}")
        print(f"[INFO] Need to add: {need}")

        created = 0
        attempts = 0
        pool_index = 0

        max_attempts = max(need * MAX_ATTEMPTS_MULTIPLIER, 1000)

        while created < need and attempts < max_attempts:
            row = class_pool.iloc[pool_index % len(class_pool)]
            pool_index += 1
            attempts += 1

            method_name, method_fn = AUGMENTERS[attempts % len(AUGMENTERS)]

            source_text = str(row["text_original"])
            new_text = method_fn(source_text, rng)
            new_norm = normalize_text(new_text)

            if not new_norm:
                continue

            if new_norm == str(row["text_normalized"]):
                continue

            key = (new_norm, label)
            if key in existing_keys:
                continue

            new_row = row.to_dict()
            new_row["sample_id"] = f"aug_{label}_{created + 1:04d}_{row['sample_id']}"
            new_row["parent_sample_id"] = row["sample_id"]
            new_row["text_original"] = new_text
            new_row["text_normalized"] = new_norm
            new_row["is_augmented"] = True
            new_row["is_hard_negative"] = False
            new_row["augmentation_method"] = method_name
            new_row["source"] = f"{row['source']}|aug_{method_name}"
            new_row["hard_negative_pair"] = ""
            new_row["hard_negative_pair_id"] = ""
            new_row["hard_negative_pair_similarity"] = ""

            augmented_rows.append(new_row)
            existing_keys.add(key)
            method_counter[method_name] += 1
            class_added_counter[label] += 1
            created += 1

        if created < need:
            print(f"[WARN] Could not reach target for {label}. Added only {created}/{need}")
        else:
            print(f"[OK] Added {created} rows for {label} in {attempts} attempts")

    aug_only = pd.DataFrame(augmented_rows)

    if not aug_only.empty:
        aug_only = ensure_columns(aug_only)

    train_final = pd.concat([train, aug_only], ignore_index=True)

    aug_only.to_csv(OUTPUT_AUG_ONLY_PATH, index=False, encoding="utf-8")
    train_final.to_csv(OUTPUT_TRAIN_FINAL_PATH, index=False, encoding="utf-8")

    summary = {
        "seed": SEED,
        "target_per_class": TARGET_PER_CLASS,
        "train_rows_before": int(len(train)),
        "augmented_rows_added": int(len(aug_only)),
        "train_rows_after": int(len(train_final)),
        "class_added_counter": {k: int(v) for k, v in sorted(class_added_counter.items())},
        "augmentation_method_counter": {k: int(v) for k, v in sorted(method_counter.items())},
        "final_distribution": train_final["label"].value_counts().sort_index().to_dict(),
        "output_aug_only_path": str(OUTPUT_AUG_ONLY_PATH),
        "output_train_final_path": str(OUTPUT_TRAIN_FINAL_PATH),
    }

    with open(OUTPUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[INFO] Augmentation method counts:")
    print(pd.Series(method_counter).sort_index())

    print("\n[INFO] Final train distribution:")
    print(train_final["label"].value_counts().sort_index())

    print(f"\n[OK] Saved augmented-only rows: {OUTPUT_AUG_ONLY_PATH}")
    print(f"[OK] Saved final augmented train: {OUTPUT_TRAIN_FINAL_PATH}")
    print(f"[OK] Saved summary JSON: {OUTPUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()