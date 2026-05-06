import faiss
import numpy as np
import pandas as pd
import json
from pathlib import Path
from app.config import settings


class RagService:
    def __init__(self, embedder_service):
        index_dir = Path(settings.rag_index_dir)

        print(f"[RagService] Loading index from: {index_dir}")
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        self.metadata = pd.read_csv(index_dir / "metadata.csv")
        self.embedder = embedder_service

        print(f"[RagService] Loaded {self.index.ntotal} vectors, {len(self.metadata)} chunks")
        print(f"[RagService] Categories: {self.metadata['category'].unique().tolist()}")

    def search(
        self,
        query: str,
        category: str | None = None,
        top_k: int = 5,
        top_n_search: int = 82,
    ) -> dict:
        q_emb = self.embedder.embed_query(query).astype("float32")
        scores, indices = self.index.search(q_emb.reshape(1, -1), top_n_search)

        scores = scores[0]
        indices = indices[0]

        valid = indices >= 0
        scores = scores[valid]
        indices = indices[valid]

        if len(indices) == 0:
            return {
                "top1_score": 0.0,
                "top1_chunk_id": None,
                "top1_text": None,
                "top1_category": None,
                "top1_source_file": None,
                "results": [],
            }

        results_df = self.metadata.iloc[indices].copy().reset_index(drop=True)
        results_df["dense_score"] = scores

        if category is not None:
            filtered = results_df[results_df["category"] == category].copy()
            if not filtered.empty:
                results_df = filtered.reset_index(drop=True)

        results_df = results_df.sort_values("dense_score", ascending=False).reset_index(drop=True)

        top = results_df.iloc[0]
        results = []
        for _, row in results_df.head(top_k).iterrows():
            results.append({
                "chunk_id": str(row["chunk_id"]),
                "category": str(row["category"]),
                "source_file": str(row["source_file"]),
                "score": float(row["dense_score"]),
                "header_path_start": str(row.get("header_path_start", "")),
                "text": str(row.get("text", "")),
            })

        return {
            "top1_score": float(top["dense_score"]),
            "top1_chunk_id": str(top["chunk_id"]),
            "top1_text": str(top.get("text", "")),
            "top1_category": str(top["category"]),
            "top1_source_file": str(top["source_file"]),
            "results": results,
        }