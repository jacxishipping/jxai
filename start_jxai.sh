#!/bin/bash
# JXAI Platform Startup Script
# Starts the complete JXAI stack: Ollama → JXAI API Gateway → Open WebUI → Cloudflare Tunnel
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              JXAI Platform - Full Stack Startup              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 1. Start Ollama
echo "[1/4] Starting Ollama..."
if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    ollama serve &>/tmp/ollama.log &
    sleep 3
    echo "  ✓ Ollama started on http://127.0.0.1:11434"
else
    echo "  ✓ Ollama already running on http://127.0.0.1:11434"
fi

# 2. Start JXAI API Gateway
echo "[2/4] Starting JXAI API Gateway..."
if ! curl -s http://127.0.0.1:8000/health >/dev/null 2>&1; then
    python3 "$SCRIPT_DIR/jxai_gateway.py" &>/tmp/jxai_gateway.log &
    sleep 2
    echo "  ✓ JXAI API Gateway started on http://127.0.0.1:8000"
else
    echo "  ✓ JXAI API Gateway already running on http://127.0.0.1:8000"
fi

# 3. Start Open WebUI
echo "[3/4] Starting Open WebUI..."
if ! curl -s http://127.0.0.1:3000/ >/dev/null 2>&1; then
    docker run -d \
        --name open-webui \
        --restart always \
        -p 3000:8080 \
        -v open-webui-data:/app/backend/data \
        -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
        -e OPENAI_API_BASE_URL=http://host.docker.internal:8000 \
        -e WEBUI_NAME="JXAI" \
        -e WEBUI_SECRET_KEY=jxai-local-dev-key-$(date +%s) \
        ghcr.io/open-webui/open-webui:main >/dev/null 2>&1
    sleep 10
    echo "  ✓ Open WebUI started on http://127.0.0.1:3000"
else
    echo "  ✓ Open WebUI already running on http://127.0.0.1:3000"
fi

# 4. Start Cloudflare Tunnel
echo "[4/4] Starting Cloudflare Tunnel..."
echo "  Starting tunnel (check output for the public URL)..."
cloudflared tunnel --url http://127.0.0.1:3000 2>&1 &
sleep 5

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  JXAI Platform is running!                                  ║"
echo "║                                                              ║"
echo "║  Open WebUI:    http://127.0.0.1:3000                        ║"
echo "║  JXAI API:      http://127.0.0.1:8000                        ║"
echo "║  Ollama API:    http://127.0.0.1:11434                        ║"
echo "║                                                              ║"
echo "║  Models:                                                     ║"
echo "║  - qwen3.6:35b-a3b                                           ║"
echo "║  - qwen3-coder:30b                                            ║"
echo "║                                                              ║"
echo "║  Public URL: Check cloudflared output above                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "⚠️  WARNING: The Cloudflare Quick Tunnel URL provides public,"
echo "   unauthenticated access to the AI service. Anyone with the URL"
echo "   can use the models. This is suitable for temporary testing only."
echo "   Do not expose credentials or internal API keys."
echo ""
