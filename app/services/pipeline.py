import cv2
import numpy as np

from app.schemas import BBox, PlateResult
from app.services.detector import Detector,Detection
from app.services.ocr import OCREngine
from app.services.plate_format import normalize_plate

def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s, d = pts.sum(axis=1), np.diff(pts, axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]

    return rect

def _warp(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Nắn thẳng biển bị nghiêng thành hình chữ nhật (perspective transform)."""
    rect = _order_points(quad.astype("float32"))
    tl, tr, br, bl = rect
    w = max(int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))), 1)
    h = max(int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))), 1)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    m = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, m, (w, h))

def _trim(crop: np.ndarray, mx: float = 0.04, my: float = 0.08) -> np.ndarray:
    h, w = crop.shape[:2]
    dx, dy = int(w * mx), int(h * my)
    if h - 2 * dy < 10 or w - 2 * dx < 10:
        return crop

    return crop[dy: h - dy, dx: w - dx]

def _crop(img: np.ndarray, det: Detection, pad: float = 0.05) -> np.ndarray:
    if det.quad is not None:
        return _trim(_warp(img, det.quad))     # <-- thêm _trim
    x, y, w, h = det.bbox
    px, py = int(w * pad), int(h * pad)
    H, W = img.shape[:2]
    return img[max(0, y - py): min(H, y + h + py), max(0, x - px): min(W, x + w + px)]

class Pipeline:
    """detect -> crop/warp -> OCR -> chuẩn hóa. Không biết gì về HTTP."""

    def __init__(self, detector: Detector, ocr: OCREngine):
        self.detector = detector
        self.ocr = ocr

    def run(self, img: np.ndarray) -> list[PlateResult]:
        results = []
        for det in self.detector.detect(img):
            crop = _crop(img, det)
            if crop.size == 0:          # hộp nằm ngoài ảnh hoặc rỗng
                continue
            ocr = self.ocr.read(crop)
            text, valid = normalize_plate(ocr.text)
            x, y, w, h = det.bbox
            results.append(PlateResult(
                text=text,
                raw_text=ocr.text,
                valid_format=valid,
                ocr_confidence=round(ocr.confidence, 3),
                det_confidence=round(det.conf, 3),
                bbox=BBox(x=x, y=y, w=w, h=h),
            ))
        return results

