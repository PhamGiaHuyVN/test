import cv2
from ultralytics import YOLO

WEIGHTS = r"..\runs\detect\runs\test1\weights\best.pt"
IMG_PATH = "../images.jpg"

model = YOLO(WEIGHTS)
img = cv2.imread(IMG_PATH)
assert img is not None, f"Không đọc được ảnh: {IMG_PATH}"

res = model.predict(img, conf=0.25, imgsz=640, verbose=False)[0]
print(f"Tìm thấy {len(res.boxes)} đối tượng")
for b in res.boxes:
    print(model.names[int(b.cls[0])], round(float(b.conf[0]), 3), [int(v) for v in b.xyxy[0].tolist()])

cv2.imwrite("yolo_out.jpg", res.plot())
print("Đã lưu yolo_out.jpg")