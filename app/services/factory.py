from app.config import Settings
from app.services.detector import ClassicalDetector, YoloDetector
from app.services.ocr import EasyOCREngine
from app.services.pipeline import Pipeline

def build_pipeline(s: Settings) -> Pipeline:
    if s.detector == "yolo":
        detector = YoloDetector(s.yolo_weights, s.detector_conf, imgsz=s.yolo_imgsz,
                                max_plates=s.max_plates, classes=s.yolo_class_list)
    elif s.detector == "classical":
        detector = ClassicalDetector()
    else:
        raise ValueError(f"PLATE_DETECTOR không hợp lệ:  {s.detector}")

    ocr = EasyOCREngine(s.ocr_lang_list, gpu=s.ocr_gpu, min_conf=s.ocr_min_conf)

    return Pipeline(detector, ocr)

