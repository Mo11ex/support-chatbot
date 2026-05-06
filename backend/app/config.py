from pathlib import Path
from pydantic_settings import BaseSettings


# Корень проекта
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://bot_user:bot_pass@localhost:5432/chatbot_db"
    api_key: str = "dev-secret-key-123"

    # Classifier
    classifier_model_dir: str = str(PROJECT_ROOT / "ml/models/classifier/stage3-rubert-tiny2-patched")
    classifier_confidence_high: float = 0.80
    classifier_confidence_mid: float = 0.60
    classifier_max_length: int = 128

    # FAQ retrieval
    faq_index_dir: str = str(PROJECT_ROOT / "ml/models/retriever/stage4_hybrid_index")
    faq_corpus_path: str = str(PROJECT_ROOT / "ml/data/processed/faq_corpus.csv")
    faq_upper_threshold: float = 0.82
    faq_lower_threshold: float = 0.70

    # RAG retrieval
    rag_index_dir: str = str(PROJECT_ROOT / "ml/models/retriever/stage5_rag_index")
    rag_chunks_path: str = str(PROJECT_ROOT / "ml/data/kb_stage5/chunks/stage5_rag_chunks_v2.csv")
    rag_upper_threshold: float = 0.75
    rag_lower_threshold: float = 0.60

    # Embedder
    embedder_model_name: str = "intfloat/multilingual-e5-base"

    # Logging
    request_log_path: str = str(PROJECT_ROOT / "ml/logs/request_log.jsonl")

    # Routing
    classifier_to_rag_category: dict = {
        "order_status": "order_support",
        "account": "account",
        "payment_refund": "payment",
        "return_exchange": None,
        "delivery": "delivery",
        "product_info": "catalog_product",
        "technical_issue": "technical_support",
        "promo_loyalty": None,
        "general_info": None,
        "other": None,
    }

    faq_primary_intents: list = [
        "delivery",
        "payment_refund",
        "return_exchange",
        "promo_loyalty",
    ]

    rag_primary_intents: list = [
        "account",
        "product_info",
        "technical_issue",
        "order_status",
    ]

    fallback_intents: list = [
        "general_info",
        "other",
    ]

    class Config:
        env_file = ".env"


settings = Settings()