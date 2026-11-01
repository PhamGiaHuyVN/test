import logging

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.repository import save_recognition

logger = logging.getLogger("plate-api")


async def record_recognition(factory: async_sessionmaker[AsyncSession], **kwargs) -> None:
    """Ghi lịch sử, KHÔNG BAO GIỜ ném lỗi: DB hỏng không được làm hỏng việc nhận diện."""
    try:
        async with factory() as session:
            await save_recognition(session, **kwargs)
    except Exception:
        logger.exception("Không lưu được lịch sử (request_id=%s)", kwargs.get("request_id"))