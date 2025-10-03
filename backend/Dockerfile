FROM python:3.12.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/opt/poetry_cache

# Install Poetry
RUN pip install poetry==1.8.3

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create true \
    && poetry config virtualenvs.in-project true \
    && poetry install --only=main --no-root

# Copy application code
COPY . .

# Install the application
RUN poetry install --only=main

RUN adduser --disabled-password --gecos '' appuser
RUN mkdir -p /app/vector_store && chown -R appuser:appuser /app

FROM builder AS build-index

USER appuser

CMD ["poetry", "run", "python", "scripts/build_index.py"]

FROM python:3.12.11-slim AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/app/.venv
ENV PATH="$VENV_PATH/bin:$PATH"

RUN adduser --disabled-password --gecos '' appuser
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder
COPY --from=builder $VENV_PATH $VENV_PATH

WORKDIR /app
COPY --from=builder /app /app

USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]