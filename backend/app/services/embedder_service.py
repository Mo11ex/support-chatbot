from sentence_transformers import SentenceTransformer

class EmbedderService:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        self.model = SentenceTransformer("app/ml/rag/e5-small")

    def embed_query(self, text: str) -> list[float]:
        # Важно для E5: префикс query:
        vec = self.model.encode([f"query: {text}"], normalize_embeddings=True)[0]
        return vec.tolist()

    def embed_passage(self, text: str) -> list[float]:
        # Важно для E5: префикс passage:
        vec = self.model.encode([f"passage: {text}"], normalize_embeddings=True)[0]
        return vec.tolist()