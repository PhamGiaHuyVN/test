import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.history import router as history
from app.api.routes import health, v1
from app.config import Settings, get_settings
from app.db.session import make_engine, make_sessionmaker
from app.services.factory import build_pipeline
from app.services.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("plate-api")


async def _ping_db(factory: async_sessionmaker[AsyncSession]) -> None:
    """Chỉ ghi log: DB chưa sẵn sàng thì vẫn cho nhận diện chạy, chỉ không lưu lịch sử."""
    try:
        async with factory() as session:
            await session.execute(text("select 1"))
        logger.info("Lưu lịch sử: BẬT (kết nối DB ổn)")
    except Exception as e:
        logger.warning("Lưu lịch sử: BẬT nhưng chưa kết nối được DB: %s", e)


def create_app(
    settings: Settings | None = None,
    pipeline: Pipeline | None = None,
    load_model: bool = True,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.pipeline is None and load_model:
            logger.info("Loading pipeline (detector=%s)...", settings.detector)
            app.state.pipeline = build_pipeline(settings)
            logger.info("Pipeline ready")

        engine = None
        if app.state.session_factory is None and settings.database_url:
            engine = make_engine(settings.database_url, settings.db_echo)
            app.state.session_factory = make_sessionmaker(engine)
            await _ping_db(app.state.session_factory)
        elif app.state.session_factory is None:
            logger.info("Lưu lịch sử: TẮT (PLATE_DATABASE_URL trống)")

        yield

        app.state.pipeline = None
        if engine is not None:          # chỉ đóng engine do chính mình tạo
            await engine.dispose()
            app.state.session_factory = None

    app = FastAPI(title="License Plate API", version="0.2.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.session_factory = session_factory
    app.state.sem = asyncio.Semaphore(settings.max_concurrency)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        t = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        if not request.url.path.startswith("/health"):
            logger.info("%s %s -> %s %.0fms rid=%s", request.method, request.url.path,
                        response.status_code, (time.perf_counter() - t) * 1000, rid)
        return response

    app.include_router(health)
    app.include_router(v1)
    app.include_router(history)
    return app


app = create_app()