from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.v1 import classify, health, orders, faq, feedback, rag, answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.services.classifier_service import ClassifierService
    from app.services.embedder_service import EmbedderService
    from app.services.faq_service import FaqService
    from app.services.rag_service import RagService
    from app.services.router_service import RouterService
    from app.core.logging_service import RequestLogger

    print("[Startup] Loading Embedder...")
    app.state.embedder = EmbedderService()

    print("[Startup] Loading Classifier...")
    app.state.classifier = ClassifierService()

    print("[Startup] Loading FAQ service...")
    app.state.faq = FaqService(app.state.embedder)

    print("[Startup] Loading RAG service...")
    app.state.rag = RagService(app.state.embedder)

    print("[Startup] Loading Request Logger...")
    app.state.logger = RequestLogger()

    print("[Startup] Building Router...")
    app.state.router = RouterService(
        classifier=app.state.classifier,
        faq_service=app.state.faq,
        rag_service=app.state.rag,
        logger=app.state.logger,
    )

    print("[Startup] All services loaded.")
    yield
    print("[Shutdown] Shutting down...")


app = FastAPI(
    title="Support Chatbot API",
    description="API для чат-бота службы поддержки интернет-магазина",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(health.router,   prefix="/api/v1", tags=["Health"])
app.include_router(classify.router, prefix="/api/v1", tags=["Classification"])
app.include_router(orders.router,   prefix="/api/v1", tags=["Orders"])
app.include_router(faq.router,      prefix="/api/v1", tags=["FAQ Legacy"])
app.include_router(rag.router,      prefix="/api/v1", tags=["RAG Legacy"])
app.include_router(feedback.router, prefix="/api/v1", tags=["Feedback"])
app.include_router(answer.router,   prefix="/api/v1", tags=["Answer"])


@app.get("/")
async def root():
    return {"message": "Support Chatbot API v2.0.0 is running"}