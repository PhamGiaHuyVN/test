import numpy as np

from app.services.detector import Detection
from app.services.ocr import OCRResult
from app.services.pipeline import Pipeline


class FakeDetector:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, img):
        return self.dets


class FakeOCR:
    def read(self, plate):
        return OCRResult("5lF-I23.4S", 0.88)  # cố ý đọc nhầm ký tự


IMG = np.full((100, 200, 3), 255, np.uint8)


def test_bbox_path_cleans_text():
    p = Pipeline(FakeDetector([Detection((10, 10, 120, 40), 0.9)]), FakeOCR())
    [r] = p.run(IMG)
    assert r.text == "51F-123.45"          # đã được sửa nhầm ký tự
    assert r.raw_text == "5lF-I23.4S"      # chuỗi thô vẫn được giữ lại
    assert r.valid_format is True
    assert r.bbox.w == 120


def test_quad_path_warps():
    quad = np.array([[10, 10], [130, 12], [128, 50], [8, 48]])
    p = Pipeline(FakeDetector([Detection((8, 10, 122, 40), 0.5, quad)]), FakeOCR())
    assert len(p.run(IMG)) == 1


def test_no_detection_returns_empty():
    assert Pipeline(FakeDetector([]), FakeOCR()).run(IMG) == []


def test_box_outside_image_is_skipped():
    p = Pipeline(FakeDetector([Detection((500, 500, 50, 20), 0.9)]), FakeOCR())
    assert p.run(IMG) == []