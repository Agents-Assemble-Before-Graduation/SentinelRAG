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
    set -a
    source .env.example
    set +a
fi

PORT="${FRONTEND_PORT:-8501}"

echo "Starting SentinelRAG Streamlit Frontend on port $PORT ..."
exec streamlit run frontend/main.py --server.port "$PORT" --server.address "0.0.0.0"
