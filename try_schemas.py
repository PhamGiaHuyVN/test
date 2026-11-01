from pydantic import ValidationError

from app.schemas import BBox, PlateResult, RecognizeResponse

plate = PlateResult(
    text="51F-123.45",
    raw_text="51F-123.45",
    valid_format=True,
    ocr_confidence=0.91,
    det_confidence=0.88,
    bbox=BBox(x=10, y=20, w=120, h=40),
)
resp = RecognizeResponse(request_id="abc123", plates=[plate], elapsed_ms=35.2)
print(resp.model_dump_json(indent=2))

# Thử dữ liệu sai: confidence phải nằm trong [0, 1]
try:
    PlateResult(
        text="x", raw_text="x", valid_format=True,
        ocr_confidence=1.5, det_confidence=0.5,
        bbox=BBox(x=0, y=0, w=1, h=1),
    )
except ValidationError as e:
    print("Bắt được lỗi:", e.errors()[0]["msg"])