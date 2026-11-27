#!/bin/bash
# Launch JX Calc — mini calculator UI backed by moe_engine + Gemma (or any GGUF).
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ENGINE_BIN="build/moe_engine"
ENGINE_PORT="${ENGINE_PORT:-8001}"
ENGINE_HOST="${ENGINE_HOST:-127.0.0.1}"
CALC_DIR="web/calculator"
DEFAULT_MODEL="models/real/gemma-3-1b-it-Q4_K_M.gguf"
MODEL="${1:-${MOE_MODEL:-$DEFAULT_MODEL}}"

if [[ ! -x "$ENGINE_BIN" ]]; then
  echo "[jxai-calc] building moe_engine..."
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="${CXX:-/usr/bin/g++}" >/dev/null
  cmake --build build -j --target moe_engine || exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "[ERROR] Model not found: $MODEL" >&2
  echo "Download Gemma first, e.g.:" >&2
  echo "  ./build/moe_engine pull ggml-org/gemma-3-1b-it-GGUF --out models/real" >&2
  exit 1
fi

if [[ ! -f "$CALC_DIR/index.html" ]]; then
  echo "[ERROR] Missing $CALC_DIR/index.html" >&2
  exit 1
fi

echo "[jxai-calc] model=$MODEL"
echo "[jxai-calc] UI    http://${ENGINE_HOST}:${ENGINE_PORT}/"
exec "$ENGINE_BIN" serve --model "$MODEL" --host "$ENGINE_HOST" --port "$ENGINE_PORT" --web-dir "$CALC_DIR"
