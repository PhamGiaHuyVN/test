from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # thêm extra đễ sửa lỗi extra input
    model_config = SettingsConfigDict(env_file=".env", env_prefix="PLATE_", extra="ignore")

    detector: str = "classical"
    yolo_weights: str = "model/plate_yolo.pt"
    detector_conf: float = 0.4
    yolo_imgsz: int = 640  # nên bằng imgsz lúc train
    yolo_classes: str = ""  # vd "0" hoặc "0,2"; rỗng = mọi lớp
    max_plates: int = 5

    ocr_langs: str = "en"
    ocr_gpu: bool = False
    ocr_min_conf: float = 0.3

    max_upload_mb: int = 8
    max_image_side: int = 4096
    allowed_content_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")

    max_concurrency: int = 2

    # db rong = tat tinh nang luu lich su
    database_url: str = ""
    # in moi sql ra log
    db_echo: bool = False

    @property
    def yolo_class_list(self) -> list[int]:
        return [int(x) for x in self.yolo_classes.split(",") if x.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def ocr_lang_list(self) -> list[str]:
        return [x.strip() for x in self.ocr_langs.split(",") if x.strip()]

@lru_cache
def get_settings() -> Settings:
    return Settings()

