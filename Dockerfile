# ---------------------------------------------------
# 1. Builder Stage
#    Installs build deps and pinned Python packages
# ---------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /app

# Copy ONLY the requirements file first to leverage Docker layer caching
COPY requirements.txt ./

# Install build-essential (if you have any C-extensions or if you might need it),
# then install pinned dependencies in one shot.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # remove build-essential so the final builder image is slimmer
    apt-get remove -y build-essential && apt-get autoremove -y

# ---------------------------------------------------
# 2. Final Runtime Stage
#    Copies only the venv + your app code
# ---------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Copy over the virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy your Python source code into the final image
# e.g. your Gradio app script "my_app.py":
COPY research_helper.py .

# (Optional) Create a non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["python", "research_helper.py"]
