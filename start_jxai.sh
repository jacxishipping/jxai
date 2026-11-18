#!/bin/bash
# JXAI Platform Startup Script
# Starts the JXAI stack: Ollama -> JXAI API Gateway (port 8001) -> Open WebUI (port 8080)
# The native C++ engine (moe_engine serve) uses port 8000; the gateway uses 8001
# so the two no longer collide.
#
# SECURITY (2026-11-17):
#  - Everything binds localhost by default. The gateway requires MOE_API_KEY
#    for its POST endpoints when the variable is set.
#  - Open WebUI runs with auth ENABLED (WEBUI_AUTH=True). No default admin
#    credential is printed; create the admin account in the UI on first run.
#  - The public Cloudflare tunnel is OFF by default. Pass --public to enable
#    it, and only after typing "yes" at the confirmation prompt.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GATEWAY_PORT="${GATEWAY_PORT:-8001}"
ENGINE_PORT=8000
WEBUI_PORT=8080
PUBLIC_MODE=false
if [[ "${1:-}" == "--public" ]]; then
    PUBLIC_MODE=true
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              JXAI Platform - Full Stack Startup              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Ports: engine ${ENGINE_PORT} | gateway ${GATEWAY_PORT} | Open WebUI ${WEBUI_PORT} | Ollama 11434"
echo ""

# 1. Start Ollama
echo "[1/3] Starting Ollama..."
if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    ollama serve &>/tmp/ollama.log &
    sleep 3
    echo "  ✓ Ollama started on http://127.0.0.1:11434"
else
    echo "  ✓ Ollama already running on http://127.0.0.1:11434"
fi

# 2. Start JXAI API Gateway (port 8001; the C++ engine owns 8000)
echo "[2/3] Starting JXAI API Gateway..."
if ! curl -s "http://127.0.0.1:${GATEWAY_PORT}/health" >/dev/null 2>&1; then
    MOE_API_KEY="${MOE_API_KEY:-}" python3 "$SCRIPT_DIR/jxai_gateway.py" --host 127.0.0.1 --port "${GATEWAY_PORT}" &>/tmp/jxai_gateway.log &
    sleep 2
    echo "  ✓ JXAI API Gateway started on http://127.0.0.1:${GATEWAY_PORT}"
    if [[ -n "${MOE_API_KEY:-}" ]]; then
        echo "  ✓ Gateway API-key auth enabled (MOE_API_KEY set)"
    else
        echo "  ! Gateway auth disabled: no MOE_API_KEY set (loopback only)"
    fi
else
    echo "  ✓ JXAI API Gateway already running on http://127.0.0.1:${GATEWAY_PORT}"
fi

# 3. Start Open WebUI (auth ON; secret key from env or generated once)
echo "[3/3] Starting Open WebUI..."
if ! curl -s "http://127.0.0.1:${WEBUI_PORT}/api/v1/auths/signin" >/dev/null 2>&1; then
    if [[ -n "${WEBUI_SECRET_KEY:-}" ]]; then
        SECRET_KEY="${WEBUI_SECRET_KEY}"
        echo "  ✓ Using WEBUI_SECRET_KEY from environment"
    else
        if command -v openssl >/dev/null 2>&1; then
            SECRET_KEY="$(openssl rand -hex 32)"
        else
            SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
        fi
        echo "  ! WEBUI_SECRET_KEY was not set — generated one for this run:"
        echo "    WEBUI_SECRET_KEY=${SECRET_KEY}"
        echo "    (Save it: restarting without the same key invalidates WebUI sessions.)"
    fi
    docker rm -f open-webui 2>/dev/null || true
    docker run -d \
        --name open-webui \
        --restart always \
        --network host \
        -v open-webui:/app/backend/data \
        -e WEBUI_NAME="JXAI" \
        -e WEBUI_AUTH=True \
        -e WEBUI_SECRET_KEY="${SECRET_KEY}" \
        -e OPENAI_API_BASE_URL="http://localhost:${GATEWAY_PORT}/v1" \
        -e OPENAI_API_KEY="${MOE_API_KEY:-}" \
        ghcr.io/open-webui/open-webui:main >/dev/null 2>&1
    echo "  ✓ Open WebUI starting on http://127.0.0.1:${WEBUI_PORT} (auth enabled)"
    echo "  ✓ Create the admin account in the WebUI on first run — no default login exists."
else
    echo "  ✓ Open WebUI already running on http://127.0.0.1:${WEBUI_PORT}"
fi

# 4. Optional public Cloudflare tunnel — OFF by default, explicit opt-in only.
if [[ "${PUBLIC_MODE}" == "true" ]]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  WARNING: --public requested                                  ║"
    echo "║  A Cloudflare Quick Tunnel will expose Open WebUI to the      ║"
    echo "║  PUBLIC INTERNET. Anyone with the tunnel URL can reach the    ║"
    echo "║  login page (auth is on, but the service is still public).    ║"
    echo "║  Only continue if you understand and accept this.             ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    read -r -p "Type \"yes\" to start the public tunnel: " CONFIRM
    if [[ "${CONFIRM}" != "yes" ]]; then
        echo "  Public tunnel NOT started (confirmation was not \"yes\")."
    else
        echo "[4/4] Starting Cloudflare Tunnel (public)..."
        echo "  Starting tunnel (check output for the public URL)..."
        cloudflared tunnel --url "http://127.0.0.1:${WEBUI_PORT}" 2>&1 &
        sleep 5
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  JXAI Platform is running!                                  ║"
echo "║                                                              ║"
echo "║  Open WebUI:    http://127.0.0.1:${WEBUI_PORT}                        ║"
echo "║  JXAI API:      http://127.0.0.1:${GATEWAY_PORT}                        ║"
echo "║  Engine API:    http://127.0.0.1:${ENGINE_PORT}  (moe_engine serve)  ║"
echo "║  Ollama API:    http://127.0.0.1:11434                        ║"
echo "║                                                              ║"
echo "║  Auth: Open WebUI login required. Gateway API key:           ║"
if [[ -n "${MOE_API_KEY:-}" ]]; then
echo "║    MOE_API_KEY is set (gateway POST endpoints require it).   ║"
else
echo "║    MOE_API_KEY not set (gateway is loopback-only, no key).   ║"
fi
echo "║                                                              ║"
echo "║  Models:                                                     ║"
echo "║  - qwen3.6:35b-a3b                                           ║"
echo "║  - qwen3-coder:30b                                            ║"
echo "║  - qwen2.5:1.5b                                              ║"
echo "║  - qwen2.5-coder:1.5b                                        ║"
echo "║                                                              ║"
if [[ "${PUBLIC_MODE}" == "true" ]]; then
echo "║  Public URL: Check cloudflared output above                  ║"
else
echo "║  Public tunnel: OFF (pass --public to enable, with warning)  ║"
fi
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
