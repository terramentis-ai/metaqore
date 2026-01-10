# MetaQore API container image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY metaqore/ ./metaqore/
COPY config/ ./config/

# Create non-root user
RUN useradd --create-home --shell /bin/bash metaqore && \
    chown -R metaqore:metaqore /app
USER metaqore

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/api/v1/health || exit 1

EXPOSE 8001
CMD ["uvicorn", "metaqore.api.app:app", "--host", "0.0.0.0", "--port", "8001"]
