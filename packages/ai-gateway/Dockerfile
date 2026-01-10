# Use Python 3.11 slim image for production
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create non-root user
RUN useradd --create-home --shell /bin/bash metaqore
RUN chown -R metaqore:metaqore /app
USER metaqore

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/health')"

# Expose port
EXPOSE 8000

# Run the service
CMD ["python", "-m", "uvicorn", "src.metaqore_ai_gateway.service:app", "--host", "0.0.0.0", "--port", "8000"]