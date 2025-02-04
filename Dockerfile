# ---------------------------------------------------
# 1. Builder Stage - Optimized with cache cleaning
# ---------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install system dependencies and build environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential && \
    # Create virtual environment
    python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    # Install Python dependencies
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Clean build dependencies
    apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* && \
    # Remove Python cache files
    find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + && \
    find /opt/venv -type f -name '*.pyc' -delete

# ---------------------------------------------------
# 2. Final Runtime Stage - Minimized
# ---------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Copy optimized virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY research_helper.py .

# Security: Create and switch to non-root user
RUN useradd -m appuser && \
    chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "research_helper:app", "--host", "0.0.0.0", "--port", "8000"]