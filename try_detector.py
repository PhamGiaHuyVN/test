import cv2

from app.services.detector import ClassicalDetector

IMG_PATH = "../images.jpg"   # chỉnh lại đường dẫn tới ảnh của bạn

img = cv2.imread(IMG_PATH)
assert img is not None, f"Không đọc được ảnh: {IMG_PATH}"

dets = ClassicalDetector().detect(img)
print(f"Tìm thấy {len(dets)} biển")

out = img.copy()
for d in dets:
    x, y, w, h = d.bbox
    print(d.bbox, "conf =", d.conf, "| tỉ lệ w/h =", round(w / h, 2))
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 3)

cv2.imwrite("detector_out.jpg", out)
print("Đã lưu detector_out.jpg, mở ra xem")