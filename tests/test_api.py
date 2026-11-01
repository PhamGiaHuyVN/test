import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services.detector import Detection
from app.services.ocr import OCRResult
from app.services.pipeline import Pipeline


class FakeDetector:
    def __init__(self, n=1):
        self.n = n

    def detect(self, img):
        return [Detection((10, 10, 120, 40), 0.9)] * self.n


class FakeOCR:
    def read(self, plate):
        return OCRResult("5lF-I23.4S", 0.88)  # cố ý nhầm ký tự để thử sửa lỗi


def jpeg_bytes(w=200, h=100):
    img = np.full((h, w, 3), 255, np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def post(client, data, ctype="image/jpeg"):
    return client.post("/v1/recognize", files={"file": ("a.jpg", data, ctype)})


@pytest.fixture
def client():
    app = create_app(Settings(), Pipeline(FakeDetector(), FakeOCR()))
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 200


def test_recognize_ok(client):
    r = post(client, jpeg_bytes())
    assert r.status_code == 200
    plate = r.json()["plates"][0]
    assert plate["text"] == "51F-123.45"
    assert plate["valid_format"] is True
    assert "X-Request-ID" in r.headers


def test_no_plate():
    app = create_app(Settings(), Pipeline(FakeDetector(0), FakeOCR()))
    with TestClient(app) as c:
        assert post(c, jpeg_bytes()).json()["plates"] == []


def test_wrong_content_type(client):
    assert post(client, b"abc", "text/plain").status_code == 415


def test_corrupt_image(client):
    assert post(client, b"not an image").status_code == 400


def test_too_large():
    app = create_app(Settings(max_upload_mb=0), Pipeline(FakeDetector(), FakeOCR()))
    with TestClient(app) as c:
        assert post(c, jpeg_bytes()).status_code == 413


def test_image_side_too_big():
    app = create_app(Settings(max_image_side=50), Pipeline(FakeDetector(), FakeOCR()))
    with TestClient(app) as c:
        assert post(c, jpeg_bytes(200, 100)).status_code == 422


def test_not_ready():
    # load_model=False: không load EasyOCR thật trong test
    app = create_app(Settings(), None, load_model=False)
    with TestClient(app) as c:
        assert c.get("/health/ready").status_code == 503
        assert post(c, jpeg_bytes()).status_code == 503