from app.services.ocr import group_lines


def box(x, y, w=60, h=30):
    """Hộp 4 góc giống định dạng EasyOCR trả về."""
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def test_single_line_left_to_right():
    # cố ý đưa vào lộn thứ tự
    results = [(box(100, 10), "123", 0.9), (box(10, 12), "51F", 0.9)]
    assert group_lines(results) == "51F123"


def test_two_lines_motorbike():
    results = [
        (box(10, 80), "12345", 0.9),   # dòng dưới
        (box(10, 10), "59", 0.9),      # dòng trên, nửa trái
        (box(80, 12), "X1", 0.9),      # dòng trên, nửa phải
    ]
    assert group_lines(results) == "59X1 12345"