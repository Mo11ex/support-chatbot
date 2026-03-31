from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from pydantic import BaseModel
from app.db.database import get_db

router = APIRouter()


class OrderItemResponse(BaseModel):
    product_name: str
    quantity: int
    unit_price: float


class OrderResponse(BaseModel):
    order_number: str
    status: str
    status_label: str
    created_at: str
    estimated_delivery: str | None
    tracking_number: str | None
    items_count: int
    total_amount: float
    items: list[OrderItemResponse]


@router.get("/orders/{order_number}", response_model=OrderResponse)
async def get_order(order_number: str, db: AsyncSession = Depends(get_db)):
    order_query = text("""
        SELECT
            o.order_number,
            os.code AS status,
            os.label_ru AS status_label,
            o.created_at::text,
            o.estimated_delivery::text,
            o.tracking_number,
            o.items_count,
            o.total_amount::float
        FROM orders o
        JOIN order_statuses os ON os.id = o.status_id
        WHERE o.order_number = :order_number
    """)

    result = await db.execute(order_query, {"order_number": order_number})
    row = result.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Заказ {order_number} не найден")

    items_query = text("""
        SELECT oi.product_name, oi.quantity, oi.unit_price::float
        FROM order_items oi
        JOIN orders o ON o.id = oi.order_id
        WHERE o.order_number = :order_number
    """)

    items_result = await db.execute(items_query, {"order_number": order_number})
    items_rows = items_result.fetchall()

    return OrderResponse(
        order_number=row[0],
        status=row[1],
        status_label=row[2],
        created_at=row[3],
        estimated_delivery=row[4],
        tracking_number=row[5],
        items_count=row[6],
        total_amount=row[7],
        items=[
            OrderItemResponse(product_name=r[0], quantity=r[1], unit_price=r[2])
            for r in items_rows
        ],
    )