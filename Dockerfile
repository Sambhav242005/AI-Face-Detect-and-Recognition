# Use the official Python 3.10 slim image
FROM python:3.10-slim

# Set working directory to /app
WORKDIR /app

# Keep Python logs visible and avoid .pyc churn inside the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install native libraries needed by OpenCV/Ultralytics model export.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache.
COPY requirements.txt requirements-models.txt ./

# Install runtime and model-generation dependencies.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-models.txt

# Copy the entire project 
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Specify how to run the application (from the /app directory)
CMD ["python", "backend/main.py"]
