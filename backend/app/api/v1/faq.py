from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from pydantic import BaseModel
from app.db.database import get_db

router = APIRouter()


class FAQRequest(BaseModel):
    text: str
    session_id: str = "default"


class FAQResponse(BaseModel):
    matched: bool
    faq_id: int | None
    answer: str | None
    category: str | None
    similarity: float


@router.post("/faq/match", response_model=FAQResponse)
async def match_faq(request: FAQRequest, db: AsyncSession = Depends(get_db)):
    user_text = request.text.lower().strip()

    query = text("""
        SELECT f.id, f.answer_text, c.code AS category, f.trigger_phrases
        FROM faq_entries f
        JOIN categories c ON c.id = f.category_id
        WHERE f.is_active = TRUE
        ORDER BY f.priority DESC
    """)

    result = await db.execute(query)
    rows = result.fetchall()

    best_match = None
    best_similarity = 0.0

    for row in rows:
        faq_id, answer, category, triggers_jsonb = row
        triggers = triggers_jsonb if isinstance(triggers_jsonb, list) else []

        for trigger in triggers:
            trigger_lower = str(trigger).lower()

            if trigger_lower in user_text or user_text in trigger_lower:
                similarity = 1.0
            else:
                user_words = set(user_text.split())
                trigger_words = set(trigger_lower.split())
                if not trigger_words:
                    continue
                common = user_words & trigger_words
                similarity = len(common) / max(len(trigger_words), len(user_words))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = {"faq_id": faq_id, "answer": answer, "category": category}

    if best_match and best_similarity >= 0.6:
        await db.execute(
            text("UPDATE faq_entries SET hit_count = hit_count + 1 WHERE id = :id"),
            {"id": best_match["faq_id"]},
        )
        await db.commit()

        return FAQResponse(
            matched=True,
            faq_id=best_match["faq_id"],
            answer=best_match["answer"],
            category=best_match["category"],
            similarity=round(best_similarity, 3),
        )

    return FAQResponse(matched=False, faq_id=None, answer=None, category=None, similarity=round(best_similarity, 3))