# syntax=docker/dockerfile:1

# ---- Build stage: create a venv with the project installed ----
FROM python:3.12-slim AS builder

WORKDIR /app

# psycopg[binary] ships its own libpq, so no build tools are needed.
COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir .

# ---- Runtime stage: only the venv, the code and the data ----
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MIGRATIONS_DIR=/app/migrations \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN groupadd --system eval && useradd --system --gid eval --no-create-home eval

COPY --from=builder /opt/venv /opt/venv
COPY src ./src
COPY migrations ./migrations
COPY scripts ./scripts
COPY data ./data

# The app listens on 8000 and owns no state of its own; Postgres
# holds everything. Files are readable by the non-root user.
RUN chown -R eval:eval /app

USER eval

EXPOSE 8000

# The health endpoint also checks the database, so a passing
# healthcheck means the API and its Postgres are both up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

CMD ["uvicorn", "docker_eval_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
