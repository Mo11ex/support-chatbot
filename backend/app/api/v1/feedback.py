from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from pydantic import BaseModel
from app.db.database import get_db

router = APIRouter()


class FeedbackRequest(BaseModel):
    session_id: str
    query: str
    answer: str
    category: str | None = None
    rating: int
    reason: str | None = None


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: int


@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(request: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    query = text("""
        INSERT INTO dialog_logs (session_id, direction, message_text, detected_category)
        VALUES (:session_id, 'in', :query, :category)
        RETURNING id
    """)

    result = await db.execute(query, {
        "session_id": request.session_id,
        "query": request.query,
        "category": request.category,
    })
    await db.commit()
    log_id = result.scalar()

    return FeedbackResponse(status="saved", feedback_id=log_id)