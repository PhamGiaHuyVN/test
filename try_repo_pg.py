import asyncio
from datetime import datetime, timedelta, timezone

from app.config import get_settings
from app.db.repository import last_seen, list_plate_reads, save_recognition, sha256_hex
from app.db.session import make_engine, make_sessionmaker
from app.schemas import BBox, PlateResult


def plate(text, valid=True, raw=None):
    return PlateResult(text=text, raw_text=raw or text, valid_format=valid, ocr_confidence=0.9,
                       det_confidence=0.8, bbox=BBox(x=1, y=2, w=3, h=4))


async def save(s, rid, plates):
    return await save_recognition(s, request_id=rid, image_sha256=sha256_hex(b"x"), width=640,
                                  height=480, detector="classical", elapsed_ms=5.0, plates=plates)


async def main():
    engine = make_engine(get_settings().database_url)
    S = make_sessionmaker(engine)
    async with S() as s:
        await save(s, "a", [plate("51F-123.45")])
        await save(s, "b", [plate("51F-123.45"), plate("29A-1234")])
        await save(s, "c", [plate("A" * 100, valid=False, raw="B" * 300)])   # chuỗi dài: phải được cắt
        await save(s, "d", [])
    async with S() as s:
        rows, total = await list_plate_reads(s, plate="51F-123.45")
        print("lọc theo biển:", len(rows), "/", total)
        rows, total = await list_plate_reads(s, valid_only=True, limit=2)
        print("hợp lệ, trang 1:", [r.text for r in rows], "tổng", total)
        now = datetime.now(timezone.utc)
        _, n_future = await list_plate_reads(s, since=now + timedelta(hours=1))
        _, n_past = await list_plate_reads(s, since=now - timedelta(hours=1))
        print("since tương lai / quá khứ:", n_future, n_past)
        ls = await last_seen(s, "51F-123.45")
        print("last_seen:", ls, "| có múi giờ:", ls.tzinfo is not None)
    await engine.dispose()


asyncio.run(main())