# Slim Python base image for lightweight container footprint
FROM python:3.11-slim

# Avoid buffering and bytecode generation
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and trained adapters
COPY src/ /app/src/
COPY adapters/ /app/adapters/

# Expose inference port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI service via Uvicorn
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
