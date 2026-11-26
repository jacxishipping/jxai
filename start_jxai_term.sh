#!/bin/bash
# Launch the JXAI full-screen Linux terminal client.
#
# Ensures the native engine is healthy on 127.0.0.1:8001 (reuses an existing
# serve if present; otherwise starts one via start_jxai.sh in the background),
# builds jxai_term if needed, then replaces this process with the TUI.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ENGINE_HOST="${ENGINE_HOST:-127.0.0.1}"
ENGINE_PORT="${ENGINE_PORT:-8001}"
TERM_BIN="build/jxai_term"
ENGINE_LOG="build/jxai-engine.log"
STARTED_ENGINE=0

usage() {
    cat <<'EOF'
Usage: ./start_jxai_term.sh [--model <path.gguf>] [--port <n>] [--host <addr>] [jxai_term args...]

Starts (or reuses) moe_engine serve, then opens the full-screen terminal UI.

Extra arguments after the launcher flags are forwarded to jxai_term
(e.g. --temp 0.2 --tokens 256 --system "You are concise.").
EOF
}

MODEL_ARG=""
FORWARD=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --model) MODEL_ARG="${2:-}"; shift 2 ;;
        --port) ENGINE_PORT="${2:-}"; shift 2 ;;
        --host) ENGINE_HOST="${2:-}"; shift 2 ;;
        --) shift; FORWARD+=("$@"); break ;;
        *) FORWARD+=("$1"); shift ;;
    esac
done

engine_healthy() {
    local body
    body="$(curl -sf --max-time 2 "http://${ENGINE_HOST}:${ENGINE_PORT}/health" 2>/dev/null || true)"
    [[ -n "$body" ]] && [[ "$body" == *'"status":"ok"'* || "$body" == *'"status": "ok"'* ]]
}

# --- Build jxai_term if missing. ---
if [[ ! -x "$TERM_BIN" ]]; then
    echo "[jxai_term] building terminal client..."
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++ >/dev/null
    cmake --build build -j --target jxai_term || {
        echo "[ERROR] failed to build jxai_term" >&2
        exit 1
    }
fi

# --- Ensure engine is up. ---
if engine_healthy; then
    echo "[jxai_term] reusing engine at ${ENGINE_HOST}:${ENGINE_PORT}"
else
    echo "[jxai_term] starting native engine..."
    START_ARGS=(--host "$ENGINE_HOST" --port "$ENGINE_PORT")
    if [[ -n "$MODEL_ARG" ]]; then
        START_ARGS+=(--model "$MODEL_ARG")
    fi
    # start_jxai.sh blocks in foreground; run it detached and wait for health.
    # We only want the engine, so invoke moe_engine serve directly if the binary
    # exists and a model is resolvable — otherwise fall back to start_jxai.sh.
    if [[ -x build/moe_engine ]]; then
        MODEL="${MODEL_ARG:-${MOE_MODEL:-}}"
        if [[ -z "$MODEL" && -f moe_config.json ]]; then
            MODEL="$(python3 -c 'import json; print(json.load(open("moe_config.json")).get("model_path",""))' 2>/dev/null || true)"
        fi
        MODEL="${MODEL:-models/real/qwen2.5-0.5b-instruct-q4_0.gguf}"
        if [[ ! -s "$MODEL" ]]; then
            echo "[ERROR] model not found: $MODEL" >&2
            echo "Pass --model <path.gguf> or run ./start_jxai.sh once first." >&2
            exit 1
        fi
        WEB_DIR="web/dist"
        WEB_ARGS=()
        [[ -f "$WEB_DIR/index.html" ]] && WEB_ARGS=(--web-dir "$WEB_DIR")
        mkdir -p build
        nohup ./build/moe_engine serve --model "$MODEL" --host "$ENGINE_HOST" --port "$ENGINE_PORT" \
            "${WEB_ARGS[@]}" >"$ENGINE_LOG" 2>&1 &
        echo $! > build/jxai-engine.pid
        STARTED_ENGINE=1
    else
        echo "[ERROR] build/moe_engine missing. Build with:" >&2
        echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j" >&2
        exit 1
    fi

    ok=0
    for _ in $(seq 1 90); do
        if engine_healthy; then ok=1; break; fi
        if [[ -f build/jxai-engine.pid ]] && ! kill -0 "$(cat build/jxai-engine.pid)" 2>/dev/null; then
            echo "[ERROR] engine exited while starting. Last log lines:" >&2
            tail -n 30 "$ENGINE_LOG" 2>/dev/null || true
            exit 1
        fi
        sleep 1
    done
    if [[ "$ok" != "1" ]]; then
        echo "[ERROR] engine did not become healthy within 90s" >&2
        tail -n 30 "$ENGINE_LOG" 2>/dev/null || true
        exit 1
    fi
    echo "[jxai_term] engine ready"
fi

cleanup() {
    # Do not kill a pre-existing engine we reused; only one we spawned.
    if [[ "$STARTED_ENGINE" == "1" && -f build/jxai-engine.pid ]]; then
        pid="$(cat build/jxai-engine.pid 2>/dev/null || true)"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f build/jxai-engine.pid
    fi
}
trap cleanup EXIT

exec "$TERM_BIN" --host "$ENGINE_HOST" --port "$ENGINE_PORT" "${FORWARD[@]}"
