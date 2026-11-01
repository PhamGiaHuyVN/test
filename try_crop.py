import cv2

from app.services.detector import ClassicalDetector
from app.services.pipeline import _crop

img = cv2.imread("../images.jpg")
for i, d in enumerate(ClassicalDetector().detect(img)):
    crop = _crop(img, d)
    print(i, d.bbox, crop.shape)
    cv2.imwrite(f"crop_debug_{i}.jpg", crop)