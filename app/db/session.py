from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

def make_engine(url: str, echo: bool = False) -> AsyncEngine:
    """engine quan ly connection pool, tao 1 lan cho ung dung"""
    kwargs: dict = {"echo": echo, "pool_pre_ping": True} # pre_ping: bo ket noi chet truoc khi dung
    if url.startswith("postgres"):
        kwargs.update(pool_size=5, max_overflow=5)

    return create_async_engine(url, **kwargs)

def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """tao session, expire on commit = false de doc obj sau commit khong loi"""
    return async_sessionmaker(engine, expire_on_commit=False)