#!/bin/bash
set -e

echo "==================================="
echo "Waiting for vLLM server to be ready..."
echo "==================================="

VLLM_URL="http://localhost:8001/v1/models"
MAX_ATTEMPTS=24
SLEEP_INTERVAL=5
TIMEOUT=$((MAX_ATTEMPTS * SLEEP_INTERVAL))

echo "Polling ${VLLM_URL} (timeout: ${TIMEOUT}s)"

for i in $(seq 1 $MAX_ATTEMPTS); do
    echo "[Attempt $i/$MAX_ATTEMPTS] Checking vLLM health..."
    
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$VLLM_URL" 2>/dev/null || echo "000")
    
    if [ "$HTTP_CODE" -eq 200 ]; then
        echo "✓ vLLM server is ready! Starting FastAPI backend..."
        exec uvicorn demos.backend.server:app --host 0.0.0.0 --port 8000 --workers 1
    fi
    
    echo "  Response: HTTP $HTTP_CODE (waiting ${SLEEP_INTERVAL}s...)"
    sleep $SLEEP_INTERVAL
done

echo "✗ ERROR: vLLM server failed to start within ${TIMEOUT} seconds"
echo "  Check vLLM logs for details (OOM, CUDA errors, model download failures)"
exit 1
