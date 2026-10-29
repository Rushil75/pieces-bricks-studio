# Python API for the mosaic engine (the page itself is hosted on Netlify).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PREPROCESS_CACHE=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY engine engine
COPY config config
COPY models models
COPY input input
COPY server.py run_demo.py ./

# Render sets $PORT; default 8000 for docker run.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
