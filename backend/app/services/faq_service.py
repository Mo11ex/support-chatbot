import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from app.config import settings


class FaqService:
    def __init__(self, embedder_service):
        index_dir = Path(settings.faq_index_dir)
        corpus_path = Path(settings.faq_corpus_path)

        print(f"[FaqService] Loading index from: {index_dir}")
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        self.metadata = pd.read_csv(index_dir / "metadata.csv")
        self.corpus = pd.read_csv(corpus_path)
        self.embedder = embedder_service

        print(f"[FaqService] Loaded {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 5) -> dict:
        q_emb = self.embedder.embed_query(query).astype("float32")
        scores, indices = self.index.search(q_emb.reshape(1, -1), top_k)

        scores = scores[0]
        indices = indices[0]

        valid = indices >= 0
        scores = scores[valid]
        indices = indices[valid]

        if len(indices) == 0:
            return {"top1_score": 0.0, "top1_doc_id": None, "top1_text": None, "results": []}

        meta = self.metadata.iloc[indices].copy().reset_index(drop=True)
        meta["score"] = scores

        # Агрегируем до doc_id
        aggregated = (
            meta.sort_values("score", ascending=False)
            .groupby("doc_id", as_index=False)
            .first()
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

        top1 = aggregated.iloc[0]

        # Получаем текст лучшего чанка
        top1_chunk_text = None
        if "text" in meta.columns:
            top1_chunk_text = meta.iloc[0]["text"]

        results = []
        for _, row in aggregated.head(top_k).iterrows():
            results.append({
                "doc_id": row["doc_id"],
                "score": float(row["score"]),
                "title": row.get("title", ""),
            })

        return {
            "top1_score": float(top1["score"]),
            "top1_doc_id": str(top1["doc_id"]),
            "top1_text": top1_chunk_text,
            "results": results,
        }