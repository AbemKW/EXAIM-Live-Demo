#!/bin/bash
set -e

echo "==================================="
echo "Starting EXAIM on Hugging Face Spaces"
echo "==================================="

# Create log directory for supervisor
mkdir -p /var/log/supervisor

# GCP credentials are managed via Hugging Face Spaces Secrets
# No need to set GOOGLE_APPLICATION_CREDENTIALS manually

# Start supervisor which will manage backend, Next.js, and nginx
echo "✓ Starting services with supervisor..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
