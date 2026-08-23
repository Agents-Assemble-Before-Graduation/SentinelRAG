#!/usr/bin/env bash
set -e

# Change to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Activate virtualenv if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Export environment variables from .env if present
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
elif [ -f ".env.example" ]; then
    echo "Warning: .env not found, using defaults from .env.example"
    set -a
    source .env.example
    set +a
fi

HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"

echo "Starting SentinelRAG FastAPI Backend on http://$HOST:$PORT ..."
exec uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
