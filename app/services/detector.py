from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    conf: float
    quad: np.ndarray | None = None

class Detector(Protocol):
    def detect(self, img: np.ndarray) -> list[Detection]: ...

class ClassicalDetector:
    """Canny + contour. Chỉ dùng tạm cho tới khi có model YOLO."""

    def detect(self, img: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 11, 17, 17)

        # Ngưỡng Canny tự thích nghi theo độ sáng trung vị của ảnh
        v = float(np.median(blur))
        edged = cv2.Canny(blur, int(max(0, 0.66 * v)), int(min(255, 1.33 * v)))
        edged = cv2.dilate(edged, None, iterations=1)  # nối các nét đứt

        cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

        out: list[Detection] = []
        for c in cnts:
            # epsilon theo chu vi thay vì cố định 10
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) != 4:
                continue
            x, y, w, h = cv2.boundingRect(approx)
            # lọc theo tỉ lệ khung hình và diện tích tối thiểu
            if 1.2 < w / h < 6 and w * h > 0.002 * gray.size:
                out.append(Detection((x, y, w, h), 0.5, approx.reshape(4, 2)))
        return out[:1]


class YoloDetector:
    """Detector dùng weights YOLO tự train (ultralytics)."""

    def __init__(self, weights: str, conf: float = 0.4, imgsz: int = 640, max_plates: int = 5,
                 classes: list[int] | None = None):
        from ultralytics import YOLO  # import trễ: dùng classical thì không cần cài ultralytics

        self.model = YOLO(weights)
        self.conf, self.imgsz, self.max_plates = conf, imgsz, max_plates
        self.classes = classes or None   # None = lấy mọi lớp

    def detect(self, img: np.ndarray) -> list[Detection]:
        res = self.model.predict(img, conf=self.conf, imgsz=self.imgsz,
                                 classes=self.classes, verbose=False)[0]
        dets = []
        for b in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            dets.append(Detection((x1, y1, x2 - x1, y2 - y1), float(b.conf[0])))
        dets.sort(key=lambda d: d.conf, reverse=True)
        return dets[: self.max_plates]