import pytest


@pytest.fixture(autouse=True)
def _never_touch_real_database(monkeypatch):
    """Biến môi trường có ưu tiên cao hơn .env, nên Settings() trong test luôn ra database_url rỗng.
    Nhờ vậy test không bao giờ ghi vào DB thật của bạn."""
    monkeypatch.setenv("PLATE_DATABASE_URL", "")