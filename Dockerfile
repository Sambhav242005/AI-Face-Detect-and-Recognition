# Runtime image for AI Face Detect and Recognition.
# Pi-friendly: NO torch/ultralytics here. Browser does WebGPU inference;
# backend only serves API + FAISS. Models must be baked in frontend/models/
# (*.onnx) or mounted at runtime. See README "Running via Docker".
FROM python:3.10-slim

WORKDIR /app

# Keep Python logs visible and avoid .pyc churn inside the image.
# SKIP_MODEL_EXPORT=1: never run torch export inside container (no torch here).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SKIP_MODEL_EXPORT=1

# faiss-cpu needs OpenMP runtime on slim images (incl. ARM/Pi).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache.
COPY requirements.txt ./

# Install runtime dependencies ONLY (fastapi/uvicorn/numpy/faiss-cpu).
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project (frontend/models/*.onnx baked in when present;
# *.pt, venv, sessions excluded via .dockerignore).
COPY . .

RUN mkdir -p backend/sessions

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

# Run from /app so backend/main.py resolves frontend/ and sessions/ correctly.
CMD ["python", "backend/main.py"]
