# Multi-stage build for Profanity Filter API
FROM python:3.11-slim as builder

WORKDIR /app

# Install uv for fast package management
RUN pip install uv

# Copy package files
COPY pyproject.toml ./
COPY src/ ./src/
COPY api/ ./api/

# Install dependencies
RUN uv pip install --system .
RUN uv pip install --system fastapi uvicorn pydantic

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY --from=builder /app/src /app/src
COPY --from=builder /app/api /app/api

# Copy models (these should be mounted as volumes in production)
COPY modernbert_multiclass/ /app/modernbert_multiclass/
COPY modernbert_finetuned/ /app/modernbert_finetuned/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
