from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://bot_user:bot_pass@localhost:5432/chatbot_db"
    api_key: str = "dev-secret-key-123"
    classifier_confidence_high: float = 0.65
    classifier_confidence_mid: float = 0.40
    rag_similarity_threshold: float = 0.70
    faq_similarity_threshold: float = 0.80

    class Config:
        env_file = ".env"


settings = Settings()