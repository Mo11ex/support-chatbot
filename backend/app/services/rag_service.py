import faiss
import pandas as pd
import re
from pathlib import Path
from app.config import settings

def clean_rag_text(text: str) -> str:
    # Просто вырезаем служебные заголовки чанкинга
    text = re.sub(r"## Covered sections.*?(\n|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"## Covered subsections.*?(\n|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*#+\s*", "", text) # убираем первый #
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

class RagService:
    def __init__(self, embedder_service):
        index_dir = Path(settings.rag_index_dir)
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        self.metadata = pd.read_csv(index_dir / "metadata.csv")
        self.embedder = embedder_service

    def search(self, query: str, category: str | None = None, top_k: int = 5, top_n_search: int = 82) -> dict:
        q_emb = self.embedder.embed_query(query).astype("float32")
        scores, indices = self.index.search(q_emb.reshape(1, -1), top_n_search)

        scores = scores[0]
        indices = indices[0]
        valid = indices >= 0

        if not valid.any():
            return {"top1_score": 0.0, "top1_text": None, "results": []}

        results_df = self.metadata.iloc[indices[valid]].copy().reset_index(drop=True)
        results_df["dense_score"] = scores[valid]

        if category:
            results_df = results_df[results_df["category"] == category].reset_index(drop=True)

        if results_df.empty:
            return {"top1_score": 0.0, "top1_text": None, "results": []}

        top1 = results_df.iloc[0]
        clean_text = clean_rag_text(str(top1["text"]))

        return {
            "top1_score": float(top1["dense_score"]),
            "top1_text": clean_text,
            "top1_source_file": str(top1["source_file"]),
            "results": []
        }