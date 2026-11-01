from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-."

@dataclass
class OCRResult:
    text: str
    confidence: float

class OCREngine(Protocol):
    def read(self, plate_bgr: np.ndarray) -> OCRResult: ...

def group_lines(results: list) -> str:
    """Gom các đoạn OCR thành dòng (trên→dưới), trong dòng sắp trái→phải.
    Cần cho biển 2 dòng của xe máy."""
    items = []
    for box, text, _ in results:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        # (tâm y, chiều cao, x nhỏ nhất, nội dung)
        items.append((sum(ys) / len(ys), max(ys) - min(ys), min(xs), text))
    items.sort(key=lambda t: t[0])

    # Hai đoạn thuộc cùng dòng nếu tâm y lệch nhau chưa tới 0.6 chiều cao chữ
    thresh = 0.6 * float(np.median([i[1] for i in items]))
    lines: list[list] = [[items[0]]]
    for it in items[1:]:
        if it[0] - lines[-1][0][0] > thresh:
            lines.append([it])
        else:
            lines[-1].append(it)

    return " ".join(
        "".join(t[3] for t in sorted(line, key=lambda t: t[2])) for line in lines
    )


class EasyOCREngine:
    def __init__(self, langs: list[str], gpu: bool = False, min_conf: float = 0.3):
        import easyocr

        self.reader = easyocr.Reader(langs, gpu=gpu)  # nặng: chỉ tạo 1 lần
        self.min_conf = min_conf
        self.clahe = cv2.createCLAHE(2.0, (8, 8))

    def read(self, plate_bgr: np.ndarray) -> OCRResult:
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = self.clahe.apply(gray)

        results = [r for r in self.reader.readtext(gray, allowlist=ALLOWLIST)
                   if r[2] >= self.min_conf]
        if not results:
            return OCRResult("", 0.0)

        conf = float(np.mean([r[2] for r in results]))
        return OCRResult(group_lines(results), conf)