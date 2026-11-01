import time
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.db.repository import sha256_hex
from app.deps import get_pipeline
from app.schemas import ErrorResponse, RecognizeResponse
from app.services.history import record_recognition
from app.services.pipeline import Pipeline

health = APIRouter(prefix="/health", tags=["health"])
v1 = APIRouter(prefix="/v1", tags=["recognition"])


@health.get("/live")
def live():
    return {"status": "ok"}


@health.get("/ready")
def ready(request: Request):
    if getattr(request.app.state, "pipeline", None) is None:
        raise HTTPException(503, "Model chưa sẵn sàng")
    return {"status": "ready"}


@v1.post(
    "/recognize",
    response_model=RecognizeResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
               415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def recognize(
    request: Request,
    background: BackgroundTasks,
    file: UploadFile = File(..., description="Ảnh JPEG/PNG/WebP"),
    pipeline: Pipeline = Depends(get_pipeline),
):
    s = request.app.state.settings

    if file.content_type not in s.allowed_content_types:
        raise HTTPException(415, f"Định dạng không hỗ trợ: {file.content_type}")

    data = await file.read(s.max_upload_bytes + 1)
    if len(data) > s.max_upload_bytes:
        raise HTTPException(413, f"Ảnh vượt quá {s.max_upload_mb}MB")

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Không giải mã được ảnh")
    if max(img.shape[:2]) > s.max_image_side:
        raise HTTPException(422, f"Cạnh ảnh vượt quá {s.max_image_side}px")

    start = time.perf_counter()
    async with request.app.state.sem:
        plates = await run_in_threadpool(pipeline.run, img)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

    # Ghi lịch sử SAU khi đã trả response cho client (không làm chậm, không làm hỏng request)
    history_id = None
    factory = getattr(request.app.state, "session_factory", None)
    if factory is not None:
        history_id = uuid.uuid4()
        background.add_task(
            record_recognition, factory,
            pk=history_id,
            request_id=request.state.request_id,
            image_sha256=sha256_hex(data),
            width=img.shape[1],
            height=img.shape[0],
            detector=s.detector,
            elapsed_ms=elapsed_ms,
            plates=plates,
        )

    return RecognizeResponse(
        request_id=request.state.request_id,
        history_id=history_id,
        plates=plates,
        elapsed_ms=elapsed_ms,
    )