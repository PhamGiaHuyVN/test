import asyncio

from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.repository import (
    get_request,
    last_seen,
    list_plate_reads,
    save_recognition,
    sha256_hex,
)
from app.db.session import make_engine, make_sessionmaker
from app.schemas import BBox, PlateResult


def plate(text="51F-123.45", valid=True, raw=None):
    return PlateResult(
        text=text, raw_text=raw or text, valid_format=valid,
        ocr_confidence=0.9, det_confidence=0.8, bbox=BBox(x=1, y=2, w=3, h=4),
    )


def run_db(scenario):
    """Chạy 1 kịch bản async với SQLite trong bộ nhớ, trả về kết quả của nó."""
    async def main():
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        Session = make_sessionmaker(engine)
        try:
            async with Session() as s:
                return await scenario(s)
        finally:
            await engine.dispose()

    return asyncio.run(main())


async def save(s, request_id="r1", plates=None, data=b"img"):
    return await save_recognition(
        s, request_id=request_id, image_sha256=sha256_hex(data), width=640, height=480,
        detector="classical", elapsed_ms=12.5,
        plates=[plate()] if plates is None else plates,
    )


def test_save_and_get():
    async def scenario(s):
        pk = await save(s)
        row = await get_request(s, pk)
        return row.plate_count, [p.text for p in row.plates], row.image_sha256

    count, texts, sha = run_db(scenario)
    assert count == 1 and texts == ["51F-123.45"]
    assert sha == sha256_hex(b"img")


def test_request_without_plates_is_still_recorded():
    async def scenario(s):
        pk = await save(s, plates=[])
        row = await get_request(s, pk)
        return row.plate_count, row.plates

    count, plates = run_db(scenario)
    assert count == 0 and plates == []


def test_filter_and_paginate():
    async def scenario(s):
        await save(s, "a", [plate("51F-123.45")])
        await save(s, "b", [plate("51F-123.45")])
        await save(s, "c", [plate("29A-1234")])
        await save(s, "d", [plate("XYZ", valid=False)])
        by_plate, total_plate = await list_plate_reads(s, plate="51F-123.45")
        valid, total_valid = await list_plate_reads(s, valid_only=True)
        page, total_all = await list_plate_reads(s, limit=1, offset=1)
        return (len(by_plate), total_plate), (len(valid), total_valid), (len(page), total_all)

    plate_res, valid_res, page_res = run_db(scenario)
    assert plate_res == (2, 2)
    assert valid_res == (3, 3)          # XYZ bị loại
    assert page_res == (1, 4)           # trang chỉ có 1 dòng nhưng tổng vẫn là 4


def test_last_seen():
    async def scenario(s):
        await save(s, plates=[plate("51F-123.45")])
        return await last_seen(s, "51F-123.45"), await last_seen(s, "99Z-0000")

    seen, unseen = run_db(scenario)
    assert seen is not None and unseen is None


def test_long_strings_are_truncated():
    async def scenario(s):
        pk = await save(s, request_id="r" * 200, plates=[plate("A" * 100, valid=False, raw="B" * 300)])
        row = await get_request(s, pk)
        return len(row.request_id), len(row.plates[0].text), len(row.plates[0].raw_text)

    assert run_db(scenario) == (64, 32, 128)