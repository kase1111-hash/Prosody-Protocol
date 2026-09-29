FROM python:3.12-slim

WORKDIR /app

# ffmpeg lets AudioToIML decode formats Praat cannot read natively
# (OGG/Opus, WebM, M4A) by transcoding them to WAV first; espeak-ng lets
# /v1/synthesize speak IML instead of rendering a tone preview.
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

# Configuration via environment variables; the full list with defaults is in
# src/prosody_protocol/server/config.py. Also available: PP_CORS_ORIGINS,
# PP_TRUSTED_PROXIES (set this behind a reverse proxy so rate limiting sees
# client addresses), PP_MAX_TEXT_CHARS, PP_MAX_WORDS_CHARS,
# PP_MAX_SYNTH_SECONDS, PP_MAX_AUDIO_SECONDS, PP_MAX_CONCURRENT_JOBS and
# PP_MAX_QUEUED_JOBS, PP_JOB_TIMEOUT_S, PP_MAX_JSON_BYTES and PP_STT_MODEL. Allow roughly
# 450 MB of memory per concurrent job (measured on 10 minutes of audio;
# every recording is analyzed at 16 kHz mono, so the input rate does not
# matter), plus the Whisper model if the whisper extra is installed.
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
