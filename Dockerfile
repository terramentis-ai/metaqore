# Simplified MetaQore Orchestrator container image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy and install simplified requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY metaqore/ ./metaqore/
COPY main.py ./

# Create non-root user
RUN useradd --create-home --shell /bin/bash metaqore && \
    chown -R metaqore:metaqore /app
USER metaqore

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')" || exit 1

EXPOSE 8001
CMD ["python", "main.py"]
