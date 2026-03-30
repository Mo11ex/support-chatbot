"""
Главный файл FastAPI приложения
"""
from fastapi import FastAPI
from app.api.v1 import classify, health

app = FastAPI(
    title="Support Chatbot API",
    description="API для чат-бота службы поддержки интернет-магазина",
    version="1.0.0",
)

# Подключение роутеров
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(classify.router, prefix="/api/v1", tags=["Classification"])


@app.on_event("startup")
async def startup():
    """Загрузка ML-моделей при старте сервера"""
    from app.services.classifier_service import ClassifierService
    app.state.classifier = ClassifierService()
    print("Classifier loaded")


@app.get("/")
async def root():
    return {"message": "Support Chatbot API is running"}