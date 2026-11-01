import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import PlateRead
from app.db.repository import get_request, last_seen, list_plate_reads
from app.deps import get_session
from app.schemas import (
    BBox,
    ErrorResponse,
    LastSeenResponse,
    PlateHistoryItem,
    PlateHistoryPage,
    RequestHistory,
)
from app.services.plate_format import normalize_plate

router = APIRouter(prefix="/v1/history", tags=["history"])


def _utc(dt: datetime | None) -> datetime | None:
    """Thời gian không kèm múi giờ thì coi là UTC (tránh lỗi so sánh naive/aware)."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _canon(plate: str | None) -> str | None:
    """Chuẩn hóa biển người dùng gõ: '51f 12345' -> '51F-123.45'."""
    if not plate or not plate.strip():
        return None
    text, valid = normalize_plate(plate)
    return text if valid else plate.strip().upper()


def _item(r: PlateRead) -> PlateHistoryItem:
    return PlateHistoryItem(
        id=r.id, history_id=r.request_pk, created_at=r.created_at, text=r.text,
        raw_text=r.raw_text, valid_format=r.valid_format,
        ocr_confidence=r.ocr_confidence, det_confidence=r.det_confidence,
        bbox=BBox(x=r.bbox_x, y=r.bbox_y, w=r.bbox_w, h=r.bbox_h),
    )


@router.get("/plates", response_model=PlateHistoryPage)
async def list_plates(
    plate: str | None = Query(None, description="Lọc theo biển, vd 51F-123.45"),
    valid_only: bool = Query(False, description="Chỉ lấy biển đúng định dạng"),
    since: datetime | None = Query(None, description="Từ thời điểm (gồm). Không có múi giờ = UTC"),
    until: datetime | None = Query(None, description="Đến thời điểm (không gồm)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    rows, total = await list_plate_reads(
        session, plate=_canon(plate), valid_only=valid_only,
        since=_utc(since), until=_utc(until), limit=limit, offset=offset,
    )
    return PlateHistoryPage(total=total, limit=limit, offset=offset, items=[_item(r) for r in rows])


@router.get("/last-seen", response_model=LastSeenResponse, responses={404: {"model": ErrorResponse}})
async def get_last_seen(
    plate: str = Query(..., min_length=1),
    session: AsyncSession = Depends(get_session),
):
    canon = _canon(plate)
    seen = await last_seen(session, canon)
    if seen is None:
        raise HTTPException(404, f"Chưa thấy biển {canon}")
    return LastSeenResponse(plate=canon, last_seen=seen)


@router.get("/requests/{request_pk}", response_model=RequestHistory,
            responses={404: {"model": ErrorResponse}})
async def get_request_detail(request_pk: uuid.UUID, session: AsyncSession = Depends(get_session)):
    row = await get_request(session, request_pk)
    if row is None:
        raise HTTPException(404, "Không tìm thấy bản ghi")
    return RequestHistory(
        id=row.id, request_id=row.request_id, created_at=row.created_at,
        image_sha256=row.image_sha256, image_width=row.image_width,
        image_height=row.image_height, detector=row.detector, elapsed_ms=row.elapsed_ms,
        plate_count=row.plate_count,
        plates=[_item(p) for p in sorted(row.plates, key=lambda p: p.id)],
    )