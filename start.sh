#!/bin/bash
set -e

echo "==================================="
echo "Starting EXAIM on Hugging Face Spaces"
echo "==================================="

# Create log directory for supervisor
mkdir -p /var/log/supervisor

# Set Google Cloud credentials if file exists
for cred_file in /app/gen-lang-client-*.json; do
    if [ -f "$cred_file" ]; then
        export GOOGLE_APPLICATION_CREDENTIALS="$cred_file"
        echo "✓ Google Cloud credentials found: $cred_file"
        break
    fi
done

# Start supervisor which will manage backend, Next.js, and nginx
echo "✓ Starting services with supervisor..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
