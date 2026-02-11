#!/bin/bash
set -e

echo "==================================="
echo "Waiting for vLLM server to be ready..."
echo "==================================="

VLLM_URL="http://localhost:8000/v1/models"
MAX_ATTEMPTS=30
SLEEP_INTERVAL=5
TIMEOUT=$((MAX_ATTEMPTS * SLEEP_INTERVAL))

echo "Polling vLLM instance (timeout: ${TIMEOUT}s)"
echo "  - vLLM: ${VLLM_URL}"

for i in $(seq 1 $MAX_ATTEMPTS); do
    echo "[Attempt $i/$MAX_ATTEMPTS] Checking vLLM health..."

    VLLM_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$VLLM_URL" 2>/dev/null || echo "000")

    if [ "$VLLM_CODE" -eq 200 ]; then
        echo "✓ vLLM server is ready! Starting FastAPI backend..."
        exec uvicorn demos.backend.server:app --host 0.0.0.0 --port 8001 --workers 1
    fi

    echo "  vLLM (8000): HTTP $VLLM_CODE"
    echo "  Waiting ${SLEEP_INTERVAL}s..."
    sleep $SLEEP_INTERVAL
done

echo "✗ ERROR: vLLM server failed to start within ${TIMEOUT} seconds"
echo "  Check vLLM logs for details (OOM, CUDA errors, model download failures)"
exit 1
