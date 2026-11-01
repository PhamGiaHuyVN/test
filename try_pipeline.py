import cv2

from app.services.detector import ClassicalDetector
from app.services.ocr import EasyOCREngine
from app.services.pipeline import Pipeline

IMG_PATH = "../images.jpg"   # ảnh có cả chiếc xe, chỉnh lại đường dẫn

img = cv2.imread(IMG_PATH)
assert img is not None, f"Không đọc được ảnh: {IMG_PATH}"

pipeline = Pipeline(ClassicalDetector(), EasyOCREngine(["en"], gpu=False))
plates = pipeline.run(img)

print(f"Tìm thấy {len(plates)} biển")
out = img.copy()
for p in plates:
    print(p.model_dump_json(indent=2))
    b = p.bbox
    cv2.rectangle(out, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 0, 255), 3)
    cv2.putText(out, p.text, (b.x, max(b.y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imwrite("pipeline_out.jpg", out)
print("Đã lưu pipeline_out.jpg")