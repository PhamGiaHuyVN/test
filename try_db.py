import asyncio

from sqlalchemy import text

from app.config import get_settings
from app.db.session import make_engine, make_sessionmaker


async def main():
    s = get_settings()
    if not s.database_url:
        raise SystemExit("Chưa đặt PLATE_DATABASE_URL trong .env")

    engine = make_engine(s.database_url, s.db_echo)

    async with engine.connect() as conn:
        print("select 1 ->", (await conn.execute(text("select 1"))).scalar_one())
        row = (await conn.execute(
            text("select current_database(), current_user"))).one()
        print("database / user ->", tuple(row))

    Session = make_sessionmaker(engine)
    async with Session() as session:
        print("qua session ->", (await session.execute(text("select 2"))).scalar_one())

    await engine.dispose()


asyncio.run(main())