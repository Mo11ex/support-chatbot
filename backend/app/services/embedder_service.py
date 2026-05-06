import torch
from sentence_transformers import SentenceTransformer
from app.config import settings


class EmbedderService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = settings.embedder_model_name

        print(f"[EmbedderService] Loading model: {model_name}")
        print(f"[EmbedderService] Device: {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_query(self, text: str):
        emb = self.model.encode(
            [f"query: {text}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb[0]

    def embed_passage(self, text: str):
        emb = self.model.encode(
            [f"passage: {text}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb[0]