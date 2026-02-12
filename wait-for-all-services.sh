#!/bin/bash
# wait-for-all-services.sh
# Purpose: Wait for all upstream services (vLLM, Backend, Next.js) to be ready before starting nginx
# This ensures users cannot access the frontend before the full system is operational

set -e

echo "====================================="
echo "Waiting for all services to be ready..."
echo "====================================="

MAX_ATTEMPTS=30
SLEEP_INTERVAL=5
TIMEOUT=$((MAX_ATTEMPTS * SLEEP_INTERVAL))

# Function to check if a service is ready
check_service() {
    local service_name=$1
    local url=$2
    
    echo ""
    echo "Checking $service_name at $url (timeout: ${TIMEOUT}s)"
    
    for attempt in $(seq 1 $MAX_ATTEMPTS); do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
        
        if [ "$HTTP_CODE" -eq 200 ]; then
            echo "✓ $service_name is ready! (HTTP $HTTP_CODE)"
            return 0
        fi
        
        echo "  [Attempt $attempt/$MAX_ATTEMPTS] $service_name not ready (HTTP $HTTP_CODE), waiting ${SLEEP_INTERVAL}s..."
        sleep $SLEEP_INTERVAL
    done
    
    echo "✗ ERROR: $service_name failed to start within ${TIMEOUT}s"
    return 1
}



# Check Backend API (with vLLM connectivity verification)
if ! check_service "Backend" "http://localhost:8000/health"; then
    echo ""
    echo "FATAL: Backend server is not available. Cannot start nginx."
    echo "Check backend logs for errors."
    exit 1
fi

# Check Next.js frontend
if ! check_service "Next.js" "http://localhost:3000"; then
    echo ""
    echo "FATAL: Next.js frontend is not available. Cannot start nginx."
    echo "Check Next.js logs for errors."
    exit 1
fi

echo ""
echo "====================================="
echo "✓ All services are ready!"
echo "====================================="
echo "Starting nginx..."
echo ""
