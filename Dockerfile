# Multi-stage Dockerfile for EXAIM with Next.js frontend and FastAPI backend
# Optimized for Hugging Face Spaces deployment

# Stage 1: Build Next.js frontend
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY demos/frontend/package*.json ./

# Install all dependencies (including devDependencies needed for build)
RUN npm ci

# Copy frontend source
COPY demos/frontend ./

# Build Next.js application for production
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# Stage 2: Python backend with built frontend
FROM python:3.11-slim

WORKDIR /app

# Install Node.js 20.x (needed to run Next.js production server)
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install system dependencies for Python packages, nginx, and supervisor
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    nginx \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir uvicorn[standard] fastapi websockets

# Copy application code (EXAIM core logic and models)
COPY exaim_core ./exaim_core
COPY exaid_core ./exaid_core
COPY infra ./infra
COPY demos/cdss_example ./demos/cdss_example
COPY demos/backend ./demos/backend
COPY demos/__init__.py ./demos/__init__.py

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/frontend/.next ./demos/frontend/.next
COPY --from=frontend-builder /app/frontend/public ./demos/frontend/public
COPY --from=frontend-builder /app/frontend/package*.json ./demos/frontend/
COPY --from=frontend-builder /app/frontend/node_modules ./demos/frontend/node_modules

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create writable directories for non-root user
RUN mkdir -p /tmp/nginx /var/lib/nginx /var/log/nginx \
    && chmod -R 777 /tmp /var/lib/nginx /var/log/nginx

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=7860
ENV HOME=/tmp
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers
ENV TORCH_HOME=/tmp/torch
ENV USER=appuser
ENV LOGNAME=appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start supervisor to manage all services (backend, Next.js, nginx)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
