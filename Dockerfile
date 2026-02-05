# Base image
FROM python:3.11-slim

# Install system dependencies (needed for OpenCV, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY backend/requirements.txt backend/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r backend/requirements.txt

# Copy the rest of the application
COPY California-Fire-Model/ California-Fire-Model/
COPY reasoning/ reasoning/
COPY backend/ backend/

# Set working directory to backend where server.py resides
WORKDIR /app/backend

# Expose the port
EXPOSE 8080

# Command to run the application (no 'cd' command needed since WORKDIR handles it)
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}"]