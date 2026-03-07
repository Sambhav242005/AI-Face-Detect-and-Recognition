# Use the official Python 3.10 slim image
FROM python:3.10-slim

# Set working directory to /app
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies (no-cache-dir to keep image light)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project 
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Specify how to run the application (from the /app directory)
CMD ["python", "backend/main.py"]
