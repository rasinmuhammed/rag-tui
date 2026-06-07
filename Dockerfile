FROM python:3.11-slim

LABEL org.opencontainers.image.title="RAG-TUI"
LABEL org.opencontainers.image.description="Interactive chunking debugger for RAG pipelines"
LABEL org.opencontainers.image.source="https://github.com/rasinmuhammed/rag-tui"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TERM=xterm-256color \
    COLORTERM=truecolor

WORKDIR /app

# Install build deps, then clean up — keeps image slim
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        gcc \
        && \
    rm -rf /var/lib/apt/lists/*

# Install Python package first (layer cache friendly)
COPY pyproject.toml .
COPY rag_tui/ ./rag_tui/
RUN pip install --no-cache-dir -e .

# Persistent cache & preset directories survive container restarts via volume mount
VOLUME ["/root/.rag-tui"]

ENTRYPOINT ["rag-tui"]
CMD []
