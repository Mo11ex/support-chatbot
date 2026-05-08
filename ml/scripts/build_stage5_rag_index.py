from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


CONFIG_PATH = Path("ml/configs/stage5_rag_index.yaml")


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).replace("\xa0", " ").replace("\t", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize_for_bm25(text: str, lowercase: bool = True) -> list[str]:
    text = normalize_text(text)
    if lowercase:
        text = text.lower()
    return re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text)


def main():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config: {CONFIG_PATH}")

    cfg = load_yaml(CONFIG_PATH)

    chunks_path = Path(cfg["chunks_path"])
    output_dir = Path(cfg["output_dir"])

    model_name = cfg["dense"]["model_name"]
    batch_size = int(cfg["dense"]["batch_size"])
    passage_prefix = cfg["dense"]["passage_prefix"]
    normalize_embeddings = bool(cfg["dense"]["normalize_embeddings"])

    bm25_lowercase = bool(cfg["bm25"]["lowercase"])

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(chunks_path)

    required_cols = {
        "chunk_id",
        "category",
        "source_group",
        "source_file",
        "source_path",
        "header_h1",
        "header_path_start",
        "header_path_end",
        "text",
        "token_count",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["text"] = df["text"].map(normalize_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df["faiss_row_id"] = np.arange(len(df))

    print(f"[INFO] Stage 5 chunks shape after cleaning: {df.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading dense model: {model_name}")
    print(f"[INFO] Device: {device}")

    model = SentenceTransformer(model_name, device=device)

    passage_texts = [f"{passage_prefix} {t}" for t in df["text"].tolist()]
    embeddings = model.encode(
        passage_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    ).astype("float32")

    print(f"[INFO] Dense embeddings shape: {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss_path = output_dir / "faiss.index"
    metadata_path = output_dir / "metadata.csv"
    bm25_path = output_dir / "bm25.pkl"
    bm25_tokens_path = output_dir / "bm25_tokens.json"
    summary_path = output_dir / "build_summary.json"

    faiss.write_index(index, str(faiss_path))

    df.to_csv(metadata_path, index=False, encoding="utf-8")

    tokenized_corpus = [
        tokenize_for_bm25(text, lowercase=bm25_lowercase)
        for text in df["text"].tolist()
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    with open(bm25_tokens_path, "w", encoding="utf-8") as f:
        json.dump(tokenized_corpus, f, ensure_ascii=False, indent=2)

    summary = {
        "chunks_path": str(chunks_path),
        "output_dir": str(output_dir),
        "dense_model_name": model_name,
        "device": device,
        "batch_size": batch_size,
        "passage_prefix": passage_prefix,
        "normalize_embeddings": normalize_embeddings,
        "num_chunks": int(len(df)),
        "num_categories": int(df["category"].nunique()),
        "categories": df["category"].value_counts().sort_index().to_dict(),
        "embedding_dim": int(dim),
        "faiss_index_type": "IndexFlatIP",
        "files": {
            "faiss_index": str(faiss_path),
            "metadata": str(metadata_path),
            "bm25": str(bm25_path),
            "bm25_tokens": str(bm25_tokens_path),
        }
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[INFO] Build summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\n[OK] Saved FAISS index: {faiss_path}")
    print(f"[OK] Saved metadata: {metadata_path}")
    print(f"[OK] Saved BM25 model: {bm25_path}")
    print(f"[OK] Saved BM25 tokens: {bm25_tokens_path}")
    print(f"[OK] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()