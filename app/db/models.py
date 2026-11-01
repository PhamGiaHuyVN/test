import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class RecognitionRequest(Base):
    """Một lần gọi API nhận diện (1 ảnh), kể cả khi không tìm thấy biển nào."""

    __tablename__ = "recognition_requests"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # X-Request-ID có thể do client tự gửi nên KHÔNG đảm bảo duy nhất -> chỉ đánh chỉ mục
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    image_sha256: Mapped[str] = mapped_column(String(64), index=True)
    image_width: Mapped[int] = mapped_column(Integer)
    image_height: Mapped[int] = mapped_column(Integer)
    detector: Mapped[str] = mapped_column(String(32))       # classical | yolo
    elapsed_ms: Mapped[float] = mapped_column(Float)
    plate_count: Mapped[int] = mapped_column(Integer)

    plates: Mapped[list["PlateRead"]] = relationship(
        back_populates="request",
        cascade="all, delete-orphan",   # xóa request thì xóa luôn các biển con
        lazy="selectin",                # nạp luôn danh sách biển, tránh lazy-load trong async
    )


class PlateRead(Base):
    """Một biển số đọc được trong một request."""

    __tablename__ = "plate_reads"
    __table_args__ = (
        # Phục vụ truy vấn "biển X xuất hiện lúc nào / lần cuối khi nào"
        Index("ix_plate_reads_text_created_at", "text", "created_at"),
    )

    # BigInteger trên Postgres, Integer trên SQLite (SQLite chỉ tự tăng với INTEGER PRIMARY KEY)
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    request_pk: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("recognition_requests.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    text: Mapped[str] = mapped_column(String(32))           # biển đã chuẩn hóa
    raw_text: Mapped[str] = mapped_column(String(128))      # chuỗi OCR thô
    valid_format: Mapped[bool] = mapped_column(Boolean)
    ocr_confidence: Mapped[float] = mapped_column(Float)
    det_confidence: Mapped[float] = mapped_column(Float)
    bbox_x: Mapped[int] = mapped_column(Integer)
    bbox_y: Mapped[int] = mapped_column(Integer)
    bbox_w: Mapped[int] = mapped_column(Integer)
    bbox_h: Mapped[int] = mapped_column(Integer)

    request: Mapped[RecognitionRequest] = relationship(back_populates="plates")