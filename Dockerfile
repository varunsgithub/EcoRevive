# EcoRevive Backend Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for rasterio/GDAL
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY backend/requirements.txt ./backend/
COPY reasoning/requirements.txt ./reasoning/

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt
RUN pip install --no-cache-dir -r reasoning/requirements.txt

# Install PyTorch CPU version (smaller image)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY backend/ ./backend/
COPY reasoning/ ./reasoning/
COPY California-Fire-Model/ ./California-Fire-Model/
COPY .env ./.env

# Set Python path
ENV PYTHONPATH=/app:/app/California-Fire-Model

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "backend/server.py"]
