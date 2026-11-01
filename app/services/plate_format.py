import re

# Ở ô bắt buộc là SỐ: chữ nào hay bị nhầm thành số thì đổi lại
TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2",
            "S": "5", "G": "6", "B": "8"}
# Ở ô bắt buộc là CHỮ: số nào hay bị nhầm thành chữ thì đổi lại
TO_ALPHA = {"0": "D", "8": "B", "5": "S", "2": "Z", "6": "G", "4": "A"}

_PATTERN = re.compile(r"^\d{2}[A-Z][A-Z0-9]?\d{4,5}$")

def _fix(chars: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(c, c) for c in chars)

def normalize_plate(raw: str) -> tuple[str, bool]:
    """Trả về (biển đã chuẩn hóa, có hợp lệ không)."""
    # 1. Viết hoa, bỏ mọi thứ không phải chữ/số (khoảng trắng, '-', '.')
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if not 7 <= len(s) <= 10:
        return s, False

    # 2. Cắt theo vị trí rồi sửa từng đoạn
    tail_len = 5 if len(s) >= 8 else 4
    head = _fix(s[:2], TO_DIGIT)          # 2 số đầu: mã tỉnh
    series = _fix(s[2], TO_ALPHA)         # ký tự thứ 3: chữ cái
    middle = s[3:-tail_len]               # 0-1 ký tự (số hoặc chữ) của seri
    tail = _fix(s[-tail_len:], TO_DIGIT)  # 4-5 số cuối

    # 3. Ghép lại và kiểm tra bằng regex
    fixed = f"{head}{series}{middle}{tail}"
    if not _PATTERN.match(fixed):
        return fixed, False

    # 4. Định dạng đẹp: 51F-123.45
    tail_fmt = f"{tail[:3]}.{tail[3:]}" if tail_len == 5 else tail
    return f"{head}{series}{middle}-{tail_fmt}", True