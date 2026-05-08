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


CONFIG_PATH = Path("ml/configs/stage4_hybrid_index.yaml")


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
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    cfg = load_yaml(CONFIG_PATH)

    corpus_path = Path(cfg["corpus_path"])
    output_dir = Path(cfg["output_dir"])

    model_name = cfg["dense"]["model_name"]
    batch_size = int(cfg["dense"]["batch_size"])
    passage_prefix = cfg["dense"]["passage_prefix"]
    normalize_embeddings = bool(cfg["dense"]["normalize_embeddings"])

    bm25_lowercase = bool(cfg["bm25"]["lowercase"])

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_df = pd.read_csv(corpus_path)

    required_cols = {"doc_id", "chunk_id", "title", "text"}
    missing = required_cols - set(corpus_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in corpus: {missing}")

    corpus_df = corpus_df.copy()
    corpus_df["text"] = corpus_df["text"].map(normalize_text)
    corpus_df = corpus_df[corpus_df["text"].str.len() > 0].reset_index(drop=True)
    corpus_df["faiss_row_id"] = np.arange(len(corpus_df))

    print(f"[INFO] Corpus shape after cleaning: {corpus_df.shape}")

    # Dense embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading dense model: {model_name}")
    print(f"[INFO] Device: {device}")

    model = SentenceTransformer(model_name, device=device)

    passage_texts = [f"{passage_prefix} {t}" for t in corpus_df["text"].tolist()]
    embeddings = model.encode(
        passage_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    ).astype("float32")

    print(f"[INFO] Dense embeddings shape: {embeddings.shape}")

    # FAISS IndexFlatIP
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss_path = output_dir / "faiss.index"
    faiss.write_index(index, str(faiss_path))

    # Metadata
    metadata_path = output_dir / "metadata.csv"
    corpus_df[["faiss_row_id", "doc_id", "chunk_id", "title", "text"]].to_csv(
        metadata_path, index=False, encoding="utf-8"
    )

    # BM25
    tokenized_corpus = [
        tokenize_for_bm25(text, lowercase=bm25_lowercase)
        for text in corpus_df["text"].tolist()
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    bm25_path = output_dir / "bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    tokens_path = output_dir / "bm25_tokens.json"
    with open(tokens_path, "w", encoding="utf-8") as f:
        json.dump(tokenized_corpus, f, ensure_ascii=False, indent=2)

    summary = {
        "corpus_path": str(corpus_path),
        "output_dir": str(output_dir),
        "dense_model_name": model_name,
        "device": device,
        "batch_size": batch_size,
        "passage_prefix": passage_prefix,
        "normalize_embeddings": normalize_embeddings,
        "corpus_shape": list(corpus_df.shape),
        "unique_doc_ids": int(corpus_df["doc_id"].nunique()),
        "unique_chunk_ids": int(corpus_df["chunk_id"].nunique()),
        "embedding_dim": int(dim),
        "faiss_index_type": "IndexFlatIP",
        "files": {
            "faiss_index": str(faiss_path),
            "metadata": str(metadata_path),
            "bm25": str(bm25_path),
            "bm25_tokens": str(tokens_path),
        }
    }

    summary_path = output_dir / "build_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[INFO] Build summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\n[OK] Saved FAISS index: {faiss_path}")
    print(f"[OK] Saved metadata: {metadata_path}")
    print(f"[OK] Saved BM25 model: {bm25_path}")
    print(f"[OK] Saved BM25 tokens: {tokens_path}")
    print(f"[OK] Saved build summary: {summary_path}")


if __name__ == "__main__":
    main()