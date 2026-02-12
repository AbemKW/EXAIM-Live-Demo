#!/bin/bash
set -e

echo "==================================="
echo "Starting FastAPI backend..."
echo "==================================="

exec uvicorn demos.backend.server:app --host 0.0.0.0 --port 8001 --workers 1
