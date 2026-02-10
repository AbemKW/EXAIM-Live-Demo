#!/bin/bash
set -e

echo "==================================="
echo "Waiting for vLLM servers to be ready..."
echo "==================================="

SUMMARIZER_URL="http://localhost:8001/v1/models"
BUFFER_URL="http://localhost:8002/v1/models"
MAX_ATTEMPTS=30
SLEEP_INTERVAL=5
TIMEOUT=$((MAX_ATTEMPTS * SLEEP_INTERVAL))

echo "Polling both vLLM instances (timeout: ${TIMEOUT}s)"
echo "  - Summarizer: ${SUMMARIZER_URL}"
echo "  - Buffer: ${BUFFER_URL}"

for i in $(seq 1 $MAX_ATTEMPTS); do
    echo "[Attempt $i/$MAX_ATTEMPTS] Checking vLLM health..."

    SUMMARIZER_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SUMMARIZER_URL" 2>/dev/null || echo "000")
    BUFFER_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BUFFER_URL" 2>/dev/null || echo "000")

    if [ "$SUMMARIZER_CODE" -eq 200 ] && [ "$BUFFER_CODE" -eq 200 ]; then
        echo "✓ Both vLLM servers are ready! Starting FastAPI backend..."
        exec uvicorn demos.backend.server:app --host 0.0.0.0 --port 8000 --workers 1
    fi

    echo "  Summarizer (8001): HTTP $SUMMARIZER_CODE"
    echo "  Buffer (8002): HTTP $BUFFER_CODE"
    echo "  Waiting ${SLEEP_INTERVAL}s..."
    sleep $SLEEP_INTERVAL
done

echo "✗ ERROR: vLLM servers failed to start within ${TIMEOUT} seconds"
echo "  Check vLLM logs for details (OOM, CUDA errors, model download failures)"
exit 1
