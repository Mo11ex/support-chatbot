from pathlib import Path
import pandas as pd
import yaml


CONFIG_PATH = Path("ml/configs/intent_mapping.yaml")
OUTPUT_PATH = Path("ml/data/processed/full_dataset.csv")


def normalize_text(text):
    if pd.isna(text):
        return None
    text = str(text).strip()
    text = " ".join(text.split())
    return text if text else None


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def process_dataset(dataset_name: str, dataset_cfg: dict, mapping: dict) -> pd.DataFrame:
    path = Path(dataset_cfg["path"])
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    text_col = dataset_cfg["text_column"]
    raw_label_col = dataset_cfg["raw_label_column"]

    out = pd.DataFrame()
    out["text"] = df[text_col].map(normalize_text)
    out["raw_label"] = df[raw_label_col].astype(str).str.strip()
    out["label"] = out["raw_label"].map(mapping)

    if "source_column" in dataset_cfg:
        out["source"] = df[dataset_cfg["source_column"]].astype(str).str.strip()
    else:
        out["source"] = dataset_cfg.get("source", dataset_name)

    out["is_synthetic"] = dataset_cfg.get("is_synthetic", False)

    return out


def main():
    cfg = load_yaml(CONFIG_PATH)

    canonical_labels = set(cfg["canonical_labels"].keys())
    datasets_cfg = cfg["datasets"]
    mappings_cfg = cfg["source_mappings"]

    frames = []

    for dataset_name, dataset_cfg in datasets_cfg.items():
        print(f"[INFO] Processing dataset: {dataset_name}")
        mapping = mappings_cfg[dataset_name]
        df_part = process_dataset(dataset_name, dataset_cfg, mapping)
        frames.append(df_part)

    df = pd.concat(frames, ignore_index=True)

    # remove empty
    df = df.dropna(subset=["text", "label"])

    # remove labels outside canonical set
    df = df[df["label"].isin(canonical_labels)]

    # remove duplicates
    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("\n[OK] Saved full dataset to:", OUTPUT_PATH)
    print("[INFO] Shape:", df.shape)
    print("\n[INFO] Label distribution:")
    print(df["label"].value_counts())

    print("\n[INFO] Source distribution:")
    print(df["source"].value_counts())

    print("\n[INFO] Synthetic distribution:")
    print(df["is_synthetic"].value_counts())


if __name__ == "__main__":
    main()