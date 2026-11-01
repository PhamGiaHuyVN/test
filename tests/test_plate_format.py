import pytest

from app.services.plate_format import normalize_plate


@pytest.mark.parametrize("raw,expected,valid", [
    ("51F-123.45", "51F-123.45", True),     # đã đúng chuẩn
    ("5lF 1234S", "51F-123.45", True),      # l→1, S→5
    ("59X1-123.45", "59X1-123.45", True),   # xe máy, có ký tự giữa
    ("29A-1234", "29A-1234", True),         # biển cũ 4 số
    ("HELLO", "HELLO", False),              # quá ngắn
    ("", "", False),                        # rỗng
])
def test_normalize(raw, expected, valid):
    assert normalize_plate(raw) == (expected, valid)