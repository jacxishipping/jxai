#!/bin/bash
# JXAI default launcher — native C++ engine only.
#
# Starts the native `moe_engine serve` backend on 127.0.0.1:8001 and serves
# the project's own web UI from the same origin (engine --web-dir web/dist),
# so the UI talks to the engine directly over relative /v1/* and /health URLs.
#
# Backend:  native C++ engine (moe_engine serve) — http://127.0.0.1:8001
# Web UI:   served by the engine itself —      http://127.0.0.1:8001/
# Model:    models/real/qwen2.5-0.5b-instruct-q4_0.gguf (override with
#           --model <path>, MOE_MODEL, or moe_config.json "model_path")
#
# This script NEVER starts Ollama and NEVER falls back to it. If the native
# engine fails to start, it prints the failure and how to fix it, then exits.
# The Ollama proxy (jxai_gateway.py) is a separate, optional integration for
# users who explicitly want Ollama; it is not part of this launcher.
#
# SECURITY:
#  - Binds 127.0.0.1 by default. A non-loopback --host with no MOE_API_KEY
#    is refused by the engine itself unless --allow-insecure-remote is given.
#  - When MOE_API_KEY is set in the environment it is passed through to the
#    engine, which then requires it on mutating/inference endpoints. When it
#    is not set the engine is loopback-only with auth off; the launcher says
#    so plainly instead of pretending otherwise.
#  - Logs go to build/jxai-engine.log (build/ is gitignored). Ctrl-C stops
#    the engine cleanly; no orphaned server is left behind.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ENGINE_PORT="${ENGINE_PORT:-8001}"
ENGINE_HOST="${ENGINE_HOST:-127.0.0.1}"
ENGINE_BIN="build/moe_engine"
ENGINE_LOG="build/jxai-engine.log"
WEB_DIR="web/dist"
DEFAULT_MODEL="models/real/qwen2.5-0.5b-instruct-q4_0.gguf"
NO_BUILD=false
MODEL_ARG=""

usage() {
    cat <<'EOF'
Usage: ./start_jxai.sh [--model <path.gguf>] [--port <n>] [--host <addr>] [--no-build]

Starts the native C++ engine (moe_engine serve) on 127.0.0.1:8001 and serves
the JXAI web UI from the same origin. No Ollama is started or required.

  --model <path>   GGUF model to load (default: MOE_MODEL, then
                   moe_config.json "model_path", then
                   models/real/qwen2.5-0.5b-instruct-q4_0.gguf)
  --port <n>       Engine/UI port (default: 8001; env ENGINE_PORT)
  --host <addr>    Bind host (default: 127.0.0.1; env ENGINE_HOST)
  --no-build       Do not build the engine if the binary is missing; fail
                   instead (useful in CI or on a prebuilt checkout)
  --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_ARG="${2:-}"; shift 2 ;;
        --port) ENGINE_PORT="${2:-}"; shift 2 ;;
        --host) ENGINE_HOST="${2:-}"; shift 2 ;;
        --no-build) NO_BUILD=true; shift ;;
        --help|-h) usage; exit 0 ;;
        --public)
            echo "[ERROR] --public was removed: this launcher no longer starts Open WebUI" >&2
            echo "or a public tunnel. The built-in UI is served by the engine on 127.0.0.1:${ENGINE_PORT}." >&2
            exit 1 ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage >&2
            exit 1 ;;
    esac
done

# --- Model resolution: --model > MOE_MODEL > moe_config.json > default. ---
MODEL="${MODEL_ARG:-${MOE_MODEL:-}}"
if [[ -z "$MODEL" && -f moe_config.json ]]; then
    MODEL="$(python3 -c 'import json,sys; print(json.load(open("moe_config.json")).get("model_path",""))' 2>/dev/null || true)"
fi
MODEL="${MODEL:-$DEFAULT_MODEL}"

if [[ ! -s "$MODEL" ]]; then
    echo "[ERROR] Model file not found or empty: $MODEL" >&2
    echo "" >&2
    echo "The native engine needs a real GGUF model. Either:" >&2
    echo "  1. Download one:  ./build/moe_engine pull Qwen/Qwen2.5-0.5B-Instruct-GGUF" >&2
    echo "     (then pass it with --model, or place it under models/real/)" >&2
    echo "  2. Pass an existing model:  ./start_jxai.sh --model <path/to/model.gguf>" >&2
    echo "" >&2
    echo "Refusing to start without a model: the engine would otherwise" >&2
    echo "auto-forge a synthetic toy model, which is not a real backend." >&2
    exit 1
fi

# --- Locate or build the native engine binary. ---
if [[ ! -x "$ENGINE_BIN" ]]; then
    if [[ "$NO_BUILD" == "true" ]]; then
        echo "[ERROR] Engine binary not found: $ENGINE_BIN (and --no-build was given)." >&2
        echo "Build it with:  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j" >&2
        exit 1
    fi
    if ! command -v cmake >/dev/null 2>&1; then
        echo "[ERROR] Engine binary not found ($ENGINE_BIN) and cmake is not installed." >&2
        echo "Install CMake and a C++ compiler, then run:" >&2
        echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j" >&2
        exit 1
    fi
    echo "[1/3] Engine binary not found — building it (this takes a few minutes the first time)..."
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release || { echo "[ERROR] cmake configure failed. See output above." >&2; exit 1; }
    cmake --build build -j || { echo "[ERROR] Engine build failed. See compiler output above." >&2; exit 1; }
    echo "  ✓ Built $ENGINE_BIN"
else
    echo "[1/3] Native engine binary found: $ENGINE_BIN"
fi

# --- Web UI bundle must exist (served by the engine via --web-dir). ---
if [[ ! -f "$WEB_DIR/index.html" ]]; then
    if command -v npm >/dev/null 2>&1 && [[ -f web/package.json ]]; then
        echo "[2/3] Web UI bundle missing — building it with npm..."
        (cd web && npm ci --no-audit --no-fund && npm run build) || {
            echo "[ERROR] Web UI build failed. Fix the error above, or run 'npm run build' in web/ manually." >&2
            exit 1
        }
    else
        echo "[ERROR] Web UI bundle not found: $WEB_DIR/index.html" >&2
        echo "Build it with:  cd web && npm ci && npm run build" >&2
        exit 1
    fi
else
    echo "[2/3] Web UI bundle found: $WEB_DIR"
fi

HEALTH_URL="http://${ENGINE_HOST}:${ENGINE_PORT}/health"

# Native engine /health includes "architecture" + "hparams". The optional
# Ollama proxy also returns "status":"ok" but adds "ollama_connected" and
# has no architecture — do not treat that as the native engine.
health_body() {
    curl -s --max-time 2 "$HEALTH_URL" 2>/dev/null || true
}

native_engine_healthy() {
    local body
    body="$(health_body)"
    [[ -n "$body" ]] || return 1
    echo "$body" | grep -q '"status"[[:space:]]*:[[:space:]]*"ok"' || return 1
    echo "$body" | grep -q '"architecture"' || return 1
    echo "$body" | grep -q '"hparams"' || return 1
    # Explicit reject of the optional Ollama proxy fingerprint.
    echo "$body" | grep -q '"ollama_connected"' && return 1
    return 0
}

# --- If a healthy native engine is already on the port, reuse it (do not own it). ---
if native_engine_healthy; then
    echo "[3/3] Native engine already healthy on ${ENGINE_HOST}:${ENGINE_PORT} — reusing it (not started by this script)."
    echo ""
    echo "  Backend:  native C++ engine (moe_engine serve)"
    echo "  Engine:   http://${ENGINE_HOST}:${ENGINE_PORT}/health"
    echo "  Web UI:   http://${ENGINE_HOST}:${ENGINE_PORT}/"
    echo "  Model:    $(health_body | grep -o '"model_name":"[^"]*"' | head -1)"
    echo ""
    exit 0
fi

# --- Port conflict: something else is listening but it is not our engine. ---
if curl -s --max-time 2 "http://${ENGINE_HOST}:${ENGINE_PORT}/" >/dev/null 2>&1 || \
   curl -s --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
    echo "[ERROR] Port ${ENGINE_PORT} is already in use by something that is not a healthy native JXAI engine." >&2
    if health_body | grep -q '"ollama_connected"'; then
        echo "Detected the optional Ollama proxy on this port (ollama_connected)." >&2
        echo "Stop it, or pick another port:  ./start_jxai.sh --port <n>" >&2
    else
        echo "Stop that process, or pick another port:  ./start_jxai.sh --port <n>" >&2
    fi
    exit 1
fi

mkdir -p "$(dirname "$ENGINE_LOG")"

# Use all cores for OpenMP matmuls unless the user already pinned a count.
# Single-thread decode is ~4× slower on a 4-core VM.
if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
    export OMP_NUM_THREADS="$(nproc 2>/dev/null || echo 4)"
fi

echo "[3/3] Starting native C++ engine (moe_engine serve)..."
echo "  Model: $MODEL"
echo "  Log:   $ENGINE_LOG"
echo "  Threads: OMP_NUM_THREADS=$OMP_NUM_THREADS"
if [[ -n "${MOE_API_KEY:-}" ]]; then
    echo "  Auth:  API key required (MOE_API_KEY is set)"
else
    echo "  Auth:  OFF — loopback only. Set MOE_API_KEY to require a key on API calls."
fi

# Line-buffer the engine's stdout/stderr (stdbuf) so the log file actually
# contains the startup lines when the engine dies — without it, block
# buffering leaves the log empty in exactly the crash case where the
# launcher prints "last 30 lines of the log".
ENGINE_CMD=("$ENGINE_BIN" serve --model "$MODEL" --host "$ENGINE_HOST" --port "$ENGINE_PORT" --web-dir "$WEB_DIR")
if command -v stdbuf >/dev/null 2>&1; then
    ENGINE_CMD=(stdbuf -oL -eL "${ENGINE_CMD[@]}")
fi

"${ENGINE_CMD[@]}" >"$ENGINE_LOG" 2>&1 &
ENGINE_PID=$!

cleanup() {
    if kill -0 "$ENGINE_PID" 2>/dev/null; then
        echo ""
        echo "[shutdown] Stopping native engine (PID $ENGINE_PID)..."
        kill "$ENGINE_PID" 2>/dev/null
        for _ in $(seq 1 10); do
            kill -0 "$ENGINE_PID" 2>/dev/null || break
            sleep 0.5
        done
        if kill -0 "$ENGINE_PID" 2>/dev/null; then
            echo "[shutdown] Engine did not stop in 5s — forcing."
            kill -9 "$ENGINE_PID" 2>/dev/null
        fi
        wait "$ENGINE_PID" 2>/dev/null
        echo "[shutdown] Engine stopped. Port ${ENGINE_PORT} released."
    fi
    exit 130
}
trap cleanup INT TERM

# --- Wait for a successful native /health; fail loudly on early exit/timeout. ---
WAITED=0
while [[ $WAITED -lt 60 ]]; do
    if native_engine_healthy; then
        break
    fi
    if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
        wait "$ENGINE_PID" 2>/dev/null
        ENGINE_STATUS=$?
        echo "" >&2
        echo "[ERROR] The native engine exited during startup (exit code ${ENGINE_STATUS})." >&2
        echo "Last 30 lines of $ENGINE_LOG:" >&2
        tail -n 30 "$ENGINE_LOG" >&2
        echo "" >&2
        echo "Common causes: missing/corrupt model file, port already in use," >&2
        echo "or an unsupported model. Check the full log and try again." >&2
        echo "No fallback to Ollama is attempted — fix the engine failure above." >&2
        trap - INT TERM
        exit 1
    fi
    sleep 1
    WAITED=$((WAITED + 1))
done

if ! native_engine_healthy; then
    echo "" >&2
    echo "[ERROR] Engine did not become healthy within 60s on ${ENGINE_HOST}:${ENGINE_PORT}." >&2
    echo "Last 30 lines of $ENGINE_LOG:" >&2
    tail -n 30 "$ENGINE_LOG" >&2
    kill "$ENGINE_PID" 2>/dev/null
    wait "$ENGINE_PID" 2>/dev/null
    trap - INT TERM
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  JXAI is running — native C++ engine backend                 ║"
echo "║                                                              ║"
echo "║  Backend:  native C++ engine (moe_engine serve)             ║"
echo "║  Web UI:   http://${ENGINE_HOST}:${ENGINE_PORT}/"
echo "║  Engine:   http://${ENGINE_HOST}:${ENGINE_PORT}/health"
echo "║  Chat API: http://${ENGINE_HOST}:${ENGINE_PORT}/v1/chat/completions"
echo "║  Model:    $MODEL"
if [[ -n "${MOE_API_KEY:-}" ]]; then
echo "║  Auth:     API key required (MOE_API_KEY set)               ║"
else
echo "║  Auth:     OFF — loopback only, set MOE_API_KEY to lock down ║"
fi
echo "║                                                              ║"
echo "║  Press Ctrl-C to stop the engine cleanly.                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Stay in the foreground so Ctrl-C reaches this script's trap; report a crash.
wait "$ENGINE_PID"
ENGINE_STATUS=$?
trap - INT TERM
if [[ $ENGINE_STATUS -ne 0 ]]; then
    echo "" >&2
    echo "[ERROR] The native engine exited unexpectedly (exit code ${ENGINE_STATUS})." >&2
    echo "Last 30 lines of $ENGINE_LOG:" >&2
    tail -n 30 "$ENGINE_LOG" >&2
    exit "$ENGINE_STATUS"
fi
echo "[shutdown] Engine stopped cleanly."
