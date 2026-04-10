from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.v1 import classify, health, orders, faq, feedback, rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.services.classifier_service import ClassifierService
    from app.services.embedder_service import EmbedderService

    app.state.classifier = ClassifierService()
    print("Classifier loaded")

    app.state.embedder = EmbedderService("intfloat/multilingual-e5-small")
    print("Embedder loaded")

    yield
    print("Shutting down")


app = FastAPI(
    title="Support Chatbot API",
    description="API для чат-бота службы поддержки интернет-магазина",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router,   prefix="/api/v1", tags=["Health"])
app.include_router(classify.router, prefix="/api/v1", tags=["Classification"])
app.include_router(orders.router,   prefix="/api/v1", tags=["Orders"])
app.include_router(faq.router,      prefix="/api/v1", tags=["FAQ"])
app.include_router(feedback.router, prefix="/api/v1", tags=["Feedback"])
app.include_router(rag.router, prefix="/api/v1", tags=["RAG"])


@app.get("/")
async def root():
    return {"message": "Support Chatbot API is running"}