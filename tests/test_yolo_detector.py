import sys
import types

import numpy as np

from app.services.detector import YoloDetector


class _Vec(list):
    def tolist(self):
        return list(self)


class _Box:
    def __init__(self, xyxy, conf):
        self.xyxy = [_Vec(xyxy)]
        self.conf = [conf]


def install_fake_ultralytics(monkeypatch, boxes):
    calls = {}

    class FakeYOLO:
        def __init__(self, weights):
            calls["weights"] = weights

        def predict(self, img, **kwargs):
            calls["kwargs"] = kwargs
            return [types.SimpleNamespace(boxes=boxes)]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    return calls


def test_boxes_converted_sorted_and_limited(monkeypatch):
    boxes = [_Box([10, 20, 110, 60], 0.55), _Box([200, 50, 320, 100], 0.91), _Box([5, 5, 25, 15], 0.40)]
    calls = install_fake_ultralytics(monkeypatch, boxes)
    det = YoloDetector("w.pt", conf=0.3, imgsz=416, max_plates=2, classes=[0])
    out = det.detect(np.zeros((480, 640, 3), np.uint8))

    assert [d.conf for d in out] == [0.91, 0.55]            # conf cao nhất trước, cắt còn 2
    assert out[0].bbox == (200, 50, 120, 50)                # xyxy -> (x, y, w, h)
    assert out[0].quad is None                              # YOLO chỉ có hộp, không có 4 góc
    assert calls["weights"] == "w.pt"
    assert calls["kwargs"]["conf"] == 0.3 and calls["kwargs"]["imgsz"] == 416
    assert calls["kwargs"]["classes"] == [0]


def test_no_detection_returns_empty(monkeypatch):
    install_fake_ultralytics(monkeypatch, [])
    assert YoloDetector("w.pt").detect(np.zeros((10, 10, 3), np.uint8)) == []