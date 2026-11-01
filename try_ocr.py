import cv2

from app.services.ocr import EasyOCREngine
from app.services.plate_format import normalize_plate

IMG_PATH = "detector_out.jpg"   # chỉnh lại đường dẫn

img = cv2.imread(IMG_PATH)
assert img is not None, f"Không đọc được ảnh: {IMG_PATH}"

print("Đang load model (lần đầu sẽ tải về, hơi lâu)...")
ocr = EasyOCREngine(["en"], gpu=False)

res = ocr.read(img)
print("Chuỗi thô     :", repr(res.text))
print("Độ tin cậy    :", round(res.confidence, 3))
print("Sau chuẩn hóa :", normalize_plate(res.text))