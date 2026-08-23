FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies and curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy application source code
COPY app/ app/
COPY frontend/ frontend/
COPY migrations/ migrations/
COPY alembic.ini .
COPY .env.example .env

# Expose backend (8000) and frontend (8501) ports
EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
