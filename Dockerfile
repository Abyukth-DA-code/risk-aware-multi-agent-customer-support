# ==============================================================================
# Autonomous Multi-Agent Customer Support Resolution Platform
# Production Multi-Stage Dockerfile for AWS / Cloud Deployment
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (build tools, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, configs, and static assets
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY database/ /app/database/
COPY data/ /app/data/
COPY sample_documents/ /app/sample_documents/
COPY hemanth_store_app.py /app/

# Create non-root user for security best practices
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose FastAPI backend port
EXPOSE 8000

# Health check to ensure service readiness on AWS ALB / ECS
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/docs || exit 1

# Default command: Launch production Uvicorn server
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
