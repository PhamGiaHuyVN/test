import asyncio

from sqlalchemy import select
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from app.db import models  # noqa: F401  (import để các bảng đăng ký vào Base.metadata)
from app.db.base import Base
from app.db.models import PlateRead, RecognitionRequest
from app.db.session import make_engine, make_sessionmaker

# 1) Xem SQL mà Postgres sẽ nhận (không cần kết nối DB)
for table in Base.metadata.sorted_tables:
    print(str(CreateTable(table).compile(dialect=postgresql.dialect())).strip())
    for ix in sorted(table.indexes, key=lambda i: i.name):
        print(str(CreateIndex(ix).compile(dialect=postgresql.dialect())).strip())
    print("-" * 60)


# 2) Thử ghi/đọc bằng SQLite trong bộ nhớ
async def main():
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    Session = make_sessionmaker(engine)
    async with Session() as s:
        req = RecognitionRequest(
            request_id="abc123", image_sha256="0" * 64, image_width=640, image_height=480,
            detector="classical", elapsed_ms=122.0, plate_count=1,
            plates=[PlateRead(text="80A-338.39", raw_text="80A.33839", valid_format=True,
                              ocr_confidence=0.579, det_confidence=0.5,
                              bbox_x=189, bbox_y=342, bbox_w=139, bbox_h=31)],
        )
        s.add(req)
        await s.commit()

    async with Session() as s:
        r = (await s.execute(select(RecognitionRequest))).scalars().one()
        print("request:", r.request_id, "| tạo lúc:", r.created_at, "| số biển:", len(r.plates))
        print("biển đầu:", r.plates[0].text, r.plates[0].ocr_confidence)
    await engine.dispose()


asyncio.run(main())