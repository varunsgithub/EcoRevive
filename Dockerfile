FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY backend/requirements.txt backend/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r backend/requirements.txt

# Copy all source code
COPY California-Fire-Model/ California-Fire-Model/
COPY reasoning/ reasoning/
COPY backend/ backend/

EXPOSE 8080

# ✅ CRITICAL FIX: Add BOTH /app and /app/backend to the path
# This allows server.py to import 'reasoning' (from root) AND 'ee_download' (from backend)
ENV PYTHONPATH=/app:/app/backend

# Run the server
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8080"]