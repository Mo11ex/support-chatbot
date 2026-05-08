from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


INPUT_PATH = Path("ml/data/processed/full_dataset_stage1_base.csv")
OUTPUT_CSV = Path("ml/data/interim/review/cross_class_suspicious_pairs.csv")
OUTPUT_JSON = Path("ml/logs/reports/stage1_cross_class_summary.json")

LOCAL_MODEL_PATH = Path("backend/app/ml/rag/e5-small")
FALLBACK_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SIMILARITY_THRESHOLD = 0.90
BATCH_SIZE = 64


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if LOCAL_MODEL_PATH.exists():
        model_name = str(LOCAL_MODEL_PATH)
    else:
        model_name = FALLBACK_MODEL_NAME

    print(f"[INFO] Loading embedding model: {model_name}")
    print(f"[INFO] Device: {device}")

    model = SentenceTransformer(model_name, device=device)
    return model, model_name, device


def compute_embeddings(model, texts):
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    required_columns = {"sample_id", "text_original", "text_normalized", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    texts = df["text_normalized"].fillna("").astype(str).tolist()

    model, model_name, device = load_model()
    embeddings = compute_embeddings(model, texts)

    print(f"[INFO] Embeddings shape: {embeddings.shape}")
    print(f"[INFO] Searching nearest cross-class neighbors with threshold > {SIMILARITY_THRESHOLD}")

    sim_matrix = np.matmul(embeddings, embeddings.T)

    rows = []
    labels = df["label"].tolist()

    for i in range(len(df)):
        current_label = labels[i]

        # исключаем самого себя
        sims = sim_matrix[i].copy()
        sims[i] = -1.0

        # оставляем только другие классы
        mask_other_class = np.array([lbl != current_label for lbl in labels])
        sims[~mask_other_class] = -1.0

        best_j = int(np.argmax(sims))
        best_score = float(sims[best_j])

        if best_score > SIMILARITY_THRESHOLD:
            rows.append({
                "sample_id_1": df.iloc[i]["sample_id"],
                "sample_id_2": df.iloc[best_j]["sample_id"],
                "text_1": df.iloc[i]["text_original"],
                "text_2": df.iloc[best_j]["text_original"],
                "label_1": df.iloc[i]["label"],
                "label_2": df.iloc[best_j]["label"],
                "cosine_similarity": best_score,
                "review_status": "",
                "review_comment": ""
            })

        if (i + 1) % 500 == 0 or (i + 1) == len(df):
            print(f"[INFO] Processed rows: {i + 1}/{len(df)}")

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            by=["cosine_similarity", "label_1", "label_2"],
            ascending=[False, True, True]
        ).reset_index(drop=True)

    result.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    pair_distribution_raw = (
    result.groupby(["label_1", "label_2"]).size().sort_values(ascending=False)
    if not result.empty else pd.Series(dtype=int)
    )

    pair_distribution_top20 = {
        f"{label_1}__{label_2}": int(count)
        for (label_1, label_2), count in pair_distribution_raw.head(20).items()
    }

    summary = {
        "input_path": str(INPUT_PATH),
        "model_name": model_name,
        "device": device,
        "total_rows": int(len(df)),
        "embedding_shape": list(embeddings.shape),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "suspicious_pair_rows": int(len(result)),
        "pair_distribution_top20": pair_distribution_top20,
        "output_csv": str(OUTPUT_CSV)
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Total suspicious cross-class rows: {len(result)}")
    print(f"[OK] Saved CSV: {OUTPUT_CSV}")
    print(f"[OK] Saved JSON: {OUTPUT_JSON}")

    if not result.empty:
        print("\n[INFO] Top 20 class-pair counts:")
        top_counts = result.groupby(["label_1", "label_2"]).size().sort_values(ascending=False).head(20)
        print(top_counts)

        print("\n[INFO] Top 15 suspicious pairs:")
        print(
            result[
                ["sample_id_1", "sample_id_2", "label_1", "label_2", "cosine_similarity"]
            ].head(15).to_string(index=False)
        )
    else:
        print("\n[INFO] No suspicious cross-class pairs found above threshold.")


if __name__ == "__main__":
    main()