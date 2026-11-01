import uuid
from datetime import datetime

from pydantic import BaseModel, Field

class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int

class PlateResult(BaseModel):
    text: str = Field(description="Biển số đã chuẩn hóa")
    raw_text: str= Field(description="Raw ocr")
    valid_format: bool = Field(description="Valid format")
    ocr_confidence: float = Field(ge=0, le=1)
    det_confidence: float = Field(ge=0, le=1)
    bbox: BBox

class RecognizeResponse(BaseModel):
    request_id: str
    history_id: uuid.UUID | None = Field(
        default=None, description="Id bản ghi lịch sử (None nếu chưa bật DB). Có thể trễ vài ms mới tra được."
    )
    plates: list[PlateResult]
    elapsed_ms: float

class ErrorResponse(BaseModel):
    detail: str

# lich su
class PlateHistoryItem(BaseModel):
    id: int
    history_id: uuid.UUID
    created_at: datetime
    text: str
    raw_text: str
    valid_format: bool
    ocr_confidence: float
    det_confidence: float
    bbox: BBox

class PlateHistoryPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[PlateHistoryItem]

class LastSeenResponse(BaseModel):
    plate: str
    last_seen: datetime

class RequestHistory(BaseModel):
    id: uuid.UUID
    request_id: str
    created_at: datetime
    image_sha256: str
    image_width: int
    image_height: int
    detector: str
    elapsed_ms: float
    plate_count: int
    plates: list[PlateHistoryItem]
