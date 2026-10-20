FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    U2NET_HOME=/models/u2net \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg fontconfig fonts-dejavu-core fonts-liberation2 \
        curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

# Subtitle font (best effort: libass falls back to DejaVu via fontconfig if this fails)
RUN mkdir -p /usr/share/fonts/truetype/montserrat \
    && (curl -fsSL -o /usr/share/fonts/truetype/montserrat/Montserrat-ExtraBold.ttf \
        https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/Montserrat-ExtraBold.ttf || true) \
    && fc-cache -f

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && playwright install --with-deps chromium

COPY . .

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["celery", "-A", "src.tasks", "worker", "-Q", "render", "-c", "1", "--loglevel=INFO"]
