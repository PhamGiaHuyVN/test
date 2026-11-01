from ultralytics import YOLO

WEIGHTS = r"..\runs\detect\runs\test1\weights\best.pt"   # chỉnh lại cho đúng
DATA = r"..\data.yaml"

model = YOLO(WEIGHTS)
print("Các lớp:", model.names)

m = model.val(data=DATA, split="val")
print(f"mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}  precision={m.box.mp:.3f}  recall={m.box.mr:.3f}")