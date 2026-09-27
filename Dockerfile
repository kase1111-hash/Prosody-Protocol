FROM python:3.12-slim

WORKDIR /app

# ffmpeg lets AudioToIML decode formats Praat cannot read natively
# (OGG/Opus, WebM, M4A) by transcoding them to WAV first.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Install the package with the REST API extra (includes audio analysis).
# Add ",whisper" to get built-in speech recognition (pulls in PyTorch).
RUN pip install --no-cache-dir ".[api]"

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# Configuration via environment variables
# (see src/prosody_protocol/server/config.py for all options)
ENV PP_HOST=0.0.0.0
ENV PP_PORT=8000
ENV PP_MAX_UPLOAD_MB=50
ENV PP_RATE_LIMIT=60

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/v1/health' % os.environ.get('PP_PORT', '8000'))" || exit 1

# Run the API server (single worker; use a reverse proxy for scaling).
# Host and port come from PP_HOST / PP_PORT.
CMD ["python", "-m", "prosody_protocol.server"]
