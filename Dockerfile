FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Cài torch bản CPU trước để image nhẹ hơn nhiều (bỏ dòng này nếu dùng GPU)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && pip uninstall -y opencv-python opencv-contrib-python \
    && pip install --force-reinstall --no-deps opencv-python-headless

RUN useradd -m appuser
COPY --chown=appuser app ./app
COPY --chown=appuser alembic.ini ./alembic.ini
COPY --chown=appuser migrations ./migrations
RUN mkdir -p models /home/appuser/.EasyOCR && chown -R appuser /app /home/appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health/live')" || exit 1

# 1 worker: mỗi worker load 1 bản model. Muốn scale thì tăng số container.
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
# Có DB thì migrate trước; "exec" để uvicorn thành tiến trình chính và nhận tín hiệu tắt đúng cách
CMD ["sh", "-c", "if [ -n \"$PLATE_DATABASE_URL\" ]; then alembic upgrade head; fi && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]