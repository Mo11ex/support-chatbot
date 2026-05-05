from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


INPUT_PATH = Path("ml/data/processed/full_dataset_stage1_base.csv")
OUTPUT_CSV = Path("ml/data/interim/review/near_duplicates.csv")
OUTPUT_JSON = Path("ml/logs/reports/stage1_near_duplicates_summary.json")

# Если локальная модель есть — используем её, иначе можно заменить на HF-модель
LOCAL_MODEL_PATH = Path("backend/app/ml/rag/e5-small")
FALLBACK_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SIMILARITY_THRESHOLD = 0.95
BATCH_SIZE = 64
CHUNK_SIZE = 512


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


def find_near_duplicates(df, embeddings, threshold=0.95, chunk_size=512):
    n = len(df)
    rows = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = embeddings[start:end]  # shape: [chunk, dim]

        # cosine sim для нормализованных эмбеддингов = dot product
        sim_matrix = np.matmul(chunk, embeddings.T)

        for i_local in range(end - start):
            i = start + i_local

            # смотрим только j > i, чтобы не дублировать пары
            sims = sim_matrix[i_local]
            candidate_js = np.where(sims > threshold)[0]

            for j in candidate_js:
                if j <= i:
                    continue

                rows.append({
                    "sample_id_1": df.iloc[i]["sample_id"],
                    "sample_id_2": df.iloc[j]["sample_id"],
                    "text_1": df.iloc[i]["text_original"],
                    "text_2": df.iloc[j]["text_original"],
                    "label_1": df.iloc[i]["label"],
                    "label_2": df.iloc[j]["label"],
                    "same_label": df.iloc[i]["label"] == df.iloc[j]["label"],
                    "cosine_similarity": float(sims[j]),
                    "review_status": "",
                    "review_comment": ""
                })

        print(f"[INFO] Processed rows: {end}/{n}")

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            by=["cosine_similarity", "same_label"],
            ascending=[False, False]
        ).reset_index(drop=True)

    return result


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
    print(f"[INFO] Searching near-duplicates with threshold > {SIMILARITY_THRESHOLD}")

    near_df = find_near_duplicates(
        df=df,
        embeddings=embeddings,
        threshold=SIMILARITY_THRESHOLD,
        chunk_size=CHUNK_SIZE
    )

    near_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    same_label_count = int(near_df["same_label"].sum()) if not near_df.empty else 0
    cross_label_count = int((~near_df["same_label"]).sum()) if not near_df.empty else 0

    summary = {
        "input_path": str(INPUT_PATH),
        "model_name": model_name,
        "device": device,
        "total_rows": int(len(df)),
        "embedding_shape": list(embeddings.shape),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "near_duplicate_pairs_total": int(len(near_df)),
        "near_duplicate_pairs_same_label": same_label_count,
        "near_duplicate_pairs_cross_label": cross_label_count,
        "output_csv": str(OUTPUT_CSV)
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Total near-duplicate pairs: {len(near_df)}")
    print(f"[INFO] Same-label pairs: {same_label_count}")
    print(f"[INFO] Cross-label pairs: {cross_label_count}")

    print(f"\n[OK] Saved near-duplicate pairs to: {OUTPUT_CSV}")
    print(f"[OK] Saved summary JSON to: {OUTPUT_JSON}")

    if not near_df.empty:
        print("\n[INFO] Top 10 near-duplicate pairs:")
        print(
            near_df[
                ["sample_id_1", "sample_id_2", "label_1", "label_2", "same_label", "cosine_similarity"]
            ].head(10).to_string(index=False)
        )
    else:
        print("\n[INFO] No near-duplicate pairs found above threshold.")


if __name__ == "__main__":
    main()