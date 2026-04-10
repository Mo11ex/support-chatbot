import os
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.db.database import get_db

router = APIRouter()

FAQ_THRESHOLD = float(os.getenv("FAQ_SIMILARITY_THRESHOLD", "0.80"))

class FaqMatchRequest(BaseModel):
    text: str
    session_id: str | None = None

class FaqMatchResponse(BaseModel):
    matched: bool
    faq_id: int | None = None
    answer: str | None = None
    category: str | None = None
    similarity: float = 0.0

FAQ_SQL = text("""
SELECT
  f.id AS faq_id,
  f.answer_text AS answer,
  c.code AS category,
  similarity(lower(:q), lower(tp.phrase)) AS sim
FROM faq_entries f
CROSS JOIN LATERAL jsonb_array_elements_text(f.trigger_phrases) AS tp(phrase)
LEFT JOIN categories c ON c.id = f.category_id
WHERE f.is_active = TRUE
ORDER BY sim DESC, f.priority DESC
LIMIT 1;
""")

@router.post("/faq/match", response_model=FaqMatchResponse)
async def faq_match(payload: FaqMatchRequest, db: AsyncSession = Depends(get_db)):
    q = payload.text.strip()
    if not q:
        return FaqMatchResponse(matched=False, similarity=0.0)

    result = await db.execute(FAQ_SQL, {"q": q})
    row = result.mappings().first()
    if not row:
        return FaqMatchResponse(matched=False, similarity=0.0)

    sim = float(row["sim"] or 0.0)
    if sim < FAQ_THRESHOLD:
        return FaqMatchResponse(matched=False, similarity=sim)

    return FaqMatchResponse(
        matched=True,
        faq_id=int(row["faq_id"]),
        answer=row["answer"],
        category=row["category"],
        similarity=sim
    )