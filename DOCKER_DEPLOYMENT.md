# Docker Deployment for Hugging Face Spaces

This directory contains the Docker-based deployment configuration for EXAIM with Next.js frontend.

## Architecture

The Docker container runs three services via supervisor:
1. **FastAPI Backend** (port 8000) - Python backend with EXAIM logic
2. **Next.js Frontend** (port 3000) - React-based UI
3. **Nginx** (port 7860) - Reverse proxy routing traffic

## Files

- `Dockerfile` - Multi-stage build (Node.js for frontend, Python for backend)
- `nginx.conf` - Routes `/api/` and `/ws` to backend, everything else to frontend
- `supervisord.conf` - Manages all three services
- `start.sh` - Container entrypoint script
- `.dockerignore` - Excludes unnecessary files from build

## Environment Variables

The frontend uses `NEXT_PUBLIC_WS_URL` to connect to the WebSocket backend. In production on Hugging Face Spaces, this should use the Space's URL with `wss://` protocol.

## Building Locally

```bash
docker build -t exaim-app .
```

## Running Locally

```bash
docker run -p 7860:7860 exaim-app
```

Access the app at http://localhost:7860

## Deploying to Hugging Face Spaces

1. Create a new Docker Space on Hugging Face
2. Add a remote to your git repository:
   ```bash
   git remote add docker-space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   ```
3. Push to the Space:
   ```bash
   git push docker-space space-deploy:main
   ```

## Port Configuration

Hugging Face Spaces expects the container to listen on port **7860**. This is configured in:
- Dockerfile: `EXPOSE 7860` and `ENV PORT=7860`
- nginx.conf: `listen 7860;`

## Health Check

The container includes a health check at `/health` that nginx responds to directly.

## Model Access

The backend requires Google Cloud credentials to access Vertex AI models. The credentials file (`gen-lang-client-*.json`) is automatically detected and set as `GOOGLE_APPLICATION_CREDENTIALS` in the startup script.

## Differences from Gradio Deployment

- **UI Framework**: Next.js/React instead of Gradio
- **Communication**: WebSocket for real-time streaming instead of Gradio events
- **Deployment**: Docker container instead of Gradio Python app
- **Routing**: Nginx reverse proxy for clean separation of frontend/backend

The EXAIM core logic and model registration remains unchanged from the Gradio deployment.
