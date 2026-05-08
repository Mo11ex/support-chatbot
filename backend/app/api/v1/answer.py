from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.api.v1 import orders


router = APIRouter()


class AnswerRequest(BaseModel):
    text: str
    order_id: str | None = None
    session_id: str | None = None


class AnswerResponse(BaseModel):
    answer: str | None
    branch: str
    intent: str | None = None
    intent_confidence: float | None = None
    source_type: str | None = None
    source_id: str | None = None
    faq_score: float | None = None
    rag_score: float | None = None
    fallback_recommended: bool = False
    latency_ms: float


def format_order_answer(order) -> str:
    lines = []
    lines.append(f"Заказ №{order.order_number}")
    lines.append(f"Статус: {order.status_label}")
    lines.append(f"Дата оформления: {order.created_at}")

    if order.estimated_delivery:
        lines.append(f"Ожидаемая доставка: {order.estimated_delivery}")

    if order.tracking_number:
        lines.append(f"Трек-номер: {order.tracking_number}")

    lines.append(f"Количество товаров: {order.items_count}")
    lines.append(f"Сумма заказа: {order.total_amount:.2f} ₽")

    if order.items:
        lines.append("Состав заказа:")
        for item in order.items[:5]:
            lines.append(f"- {item.product_name} × {item.quantity} ({item.unit_price:.2f} ₽)")
        if len(order.items) > 5:
            lines.append(f"- и ещё {len(order.items) - 5} поз.")

    return "\n".join(lines)


@router.post("/answer", response_model=AnswerResponse)
async def answer(
    request: AnswerRequest,
    req: Request,
    db: AsyncSession = Depends(get_db),
):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty")

    router_service = req.app.state.router
    logger = req.app.state.logger

    result = await router_service.handle(
        text=text,
        order_id=request.order_id,
    )

    # Реальный вызов order API branch
    if result["branch"] == "orders_api":
        order_number = result["source_id"] or request.order_id

        try:
            order_obj = await orders.get_order(order_number, db)
            result["answer"] = format_order_answer(order_obj)

        except HTTPException as exc:
            if exc.status_code == 404:
                result["branch"] = "orders_not_found"
                result["answer"] = (
                    f"Не удалось найти заказ №{order_number}. "
                    f"Проверьте номер заказа и попробуйте снова."
                )
            else:
                result["branch"] = "orders_api_error"
                result["answer"] = (
                    "Не удалось получить данные по заказу из сервиса заказов. "
                    "Попробуйте позже или обратитесь к оператору."
                )

            result["source_type"] = "system"
            result["fallback_recommended"] = True

        except Exception as exc:
            # Любая неожиданная ошибка: БД недоступна, схема не поднята и т.д.
            result["branch"] = "orders_api_unavailable"
            result["answer"] = (
                "Сервис проверки заказов временно недоступен. "
                "Попробуйте позже или обратитесь к оператору."
            )
            result["source_type"] = "system"
            result["fallback_recommended"] = True

        logger.log(
            text=text,
            branch=result["branch"],
            intent=result.get("intent"),
            intent_confidence=result.get("intent_confidence"),
            faq_score=result.get("faq_score"),
            rag_score=result.get("rag_score"),
            source_type=result.get("source_type"),
            source_id=result.get("source_id"),
            answer=result.get("answer"),
            latency_ms=result.get("latency_ms", 0.0),
            fallback_recommended=result.get("fallback_recommended", False),
        )

    return AnswerResponse(**result)