import hashlib
import uuid
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import PlateRead, RecognitionRequest
from app.schemas import PlateResult

# Độ dài tối đa của các cột chuỗi (khớp với models.py). Postgres từ chối chuỗi dài hơn.
MAX_REQUEST_ID = 64
MAX_TEXT = 32
MAX_RAW_TEXT = 128


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def save_recognition(
    session: AsyncSession,
    *,
    request_id: str,
    image_sha256: str,
    width: int,
    height: int,
    detector: str,
    elapsed_ms: float,
    plates: list[PlateResult],
    pk: uuid.UUID | None = None,
) -> uuid.UUID:
    """Ghi 1 request và các biển của nó trong CÙNG một transaction. Trả về id của request."""
    row = RecognitionRequest(
        id=pk or uuid.uuid4(),
        request_id=request_id[:MAX_REQUEST_ID],
        image_sha256=image_sha256,
        image_width=width,
        image_height=height,
        detector=detector,
        elapsed_ms=elapsed_ms,
        plate_count=len(plates),
        plates=[
            PlateRead(
                text=p.text[:MAX_TEXT],
                raw_text=p.raw_text[:MAX_RAW_TEXT],
                valid_format=p.valid_format,
                ocr_confidence=p.ocr_confidence,
                det_confidence=p.det_confidence,
                bbox_x=p.bbox.x,
                bbox_y=p.bbox.y,
                bbox_w=p.bbox.w,
                bbox_h=p.bbox.h,
            )
            for p in plates
        ],
    )
    session.add(row)
    await session.commit()
    return row.id  # id do Python sinh nên đọc được ngay sau commit


async def get_request(session: AsyncSession, pk: uuid.UUID) -> RecognitionRequest | None:
    return await session.get(RecognitionRequest, pk)


async def list_plate_reads(
    session: AsyncSession,
    *,
    plate: str | None = None,
    valid_only: bool = False,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[PlateRead], int]:
    """Trả về (các dòng của trang hiện tại, tổng số dòng khớp bộ lọc). Mới nhất trước."""
    stmt = select(PlateRead)
    if plate:
        stmt = stmt.where(PlateRead.text == plate)
    if valid_only:
        stmt = stmt.where(PlateRead.valid_format.is_(True))
    if since:
        stmt = stmt.where(PlateRead.created_at >= since)
    if until:
        stmt = stmt.where(PlateRead.created_at < until)

    total = await session.scalar(select(func.count()).select_from(stmt.subquery()))
    page = stmt.order_by(PlateRead.created_at.desc(), PlateRead.id.desc()).limit(limit).offset(offset)
    rows = (await session.scalars(page)).all()
    return list(rows), int(total or 0)


async def last_seen(session: AsyncSession, plate: str) -> datetime | None:
    """Lần cuối biển này xuất hiện (dùng chỉ mục text + created_at)."""
    return await session.scalar(
        select(func.max(PlateRead.created_at)).where(PlateRead.text == plate)
    )