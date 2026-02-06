FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
COPY api/ api/

# Install the package with API dependencies
RUN pip install --no-cache-dir -e ".[api]"

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# Expose the API port
EXPOSE 8000

# Configuration via environment variables (see api/config.py for all options)
ENV PP_HOST=0.0.0.0
ENV PP_PORT=8000
ENV PP_MAX_UPLOAD_MB=50
ENV PP_RATE_LIMIT=60

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1

# Run the API server (single worker; use reverse proxy for scaling)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
