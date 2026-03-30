"""
Classification endpoint
"""
from fastapi import APIRouter, Request
from pydantic import BaseModel
import time


class ClassifyRequest(BaseModel):
    text: str
    session_id: str = "default"


class CategoryScore(BaseModel):
    category: str
    confidence: float


class ClassifyResponse(BaseModel):
    category: str
    category_label: str
    confidence: float
    top_3: list[CategoryScore]
    processing_time_ms: float


router = APIRouter()


@router.post("/classify", response_model=ClassifyResponse)
async def classify_text(request: ClassifyRequest, req: Request):
    start = time.perf_counter()

    classifier = req.app.state.classifier
    result = classifier.predict(request.text)

    processing_time = (time.perf_counter() - start) * 1000

    return ClassifyResponse(
        category=result["category"],
        category_label=result["category_label"],
        confidence=result["confidence"],
        top_3=result["top_3"],
        processing_time_ms=round(processing_time, 1),
    )