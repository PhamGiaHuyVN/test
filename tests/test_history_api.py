import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app.config import Settings
from app.db import models  # noqa: F401
from app.db.base import Base
from app.main import create_app
from app.services.detector import Detection
from app.services.ocr import OCRResult
from app.services.pipeline import Pipeline


class FakeDetector:
    def detect(self, img):
        return [Detection((10, 10, 120, 40), 0.9)]


class FakeOCR:
    def read(self, plate):
        return OCRResult("5lF-I23.4S", 0.88)   # chuẩn hóa thành 51F-123.45


def jpeg_bytes():
    return cv2.imencode(".jpg", np.full((100, 200, 3), 255, np.uint8))[1].tobytes()


def post(client):
    return client.post("/v1/recognize", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")})


@pytest.fixture
def client(tmp_path):
    """App thật + SQLite dạng file. Bảng tạo bằng engine đồng bộ để tránh trộn event loop."""
    db_file = tmp_path / "t.db"
    sync_engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(sync_engine)
    sync_engine.dispose()
    settings = Settings(database_url=f"sqlite+aiosqlite:///{db_file}")
    with TestClient(create_app(settings, Pipeline(FakeDetector(), FakeOCR()))) as c:
        yield c


def test_recognize_is_recorded_and_retrievable(client):
    body = post(client).json()
    assert body["history_id"]
    r = client.get(f"/v1/history/requests/{body['history_id']}")
    assert r.status_code == 200
    detail = r.json()
    assert detail["plate_count"] == 1
    assert detail["detector"] == "classical"
    assert detail["plates"][0]["text"] == "51F-123.45"
    assert detail["image_width"] == 200 and detail["image_height"] == 100


def test_search_accepts_unnormalized_plate(client):
    post(client)
    post(client)
    page = client.get("/v1/history/plates", params={"plate": "51f 12345"}).json()
    assert page["total"] == 2 and len(page["items"]) == 2
    assert client.get("/v1/history/plates", params={"plate": "99Z-0000"}).json()["total"] == 0


def test_last_seen(client):
    post(client)
    ok = client.get("/v1/history/last-seen", params={"plate": "51F12345"})
    assert ok.status_code == 200 and ok.json()["plate"] == "51F-123.45"
    assert client.get("/v1/history/last-seen", params={"plate": "99Z-0000"}).status_code == 404


def test_pagination_and_validation(client):
    for _ in range(3):
        post(client)
    page = client.get("/v1/history/plates", params={"limit": 2, "offset": 2}).json()
    assert page["total"] == 3 and len(page["items"]) == 1
    assert client.get("/v1/history/plates", params={"limit": 0}).status_code == 422
    assert client.get("/v1/history/plates", params={"limit": 500}).status_code == 422


def test_since_without_timezone_is_treated_as_utc(client):
    post(client)
    assert client.get("/v1/history/plates", params={"since": "2000-01-01T00:00:00"}).json()["total"] == 1
    assert client.get("/v1/history/plates", params={"since": "2999-01-01T00:00:00"}).json()["total"] == 0


def test_unknown_history_id_is_404(client):
    r = client.get("/v1/history/requests/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


def test_database_disabled():
    app = create_app(Settings(database_url=""), Pipeline(FakeDetector(), FakeOCR()))
    with TestClient(app) as c:
        body = post(c).json()
        assert body["history_id"] is None and len(body["plates"]) == 1
        assert c.get("/v1/history/plates").status_code == 503


def test_db_failure_does_not_break_recognition():
    def broken_factory():
        raise RuntimeError("db down")

    app = create_app(Settings(database_url=""), Pipeline(FakeDetector(), FakeOCR()),
                     session_factory=broken_factory)
    with TestClient(app) as c:
        r = post(c)
        assert r.status_code == 200
        assert r.json()["plates"][0]["text"] == "51F-123.45"