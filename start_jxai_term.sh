#!/bin/bash
# Launch the JXAI full-screen Linux terminal client.
#
# On first run, jxai_term opens a setup wizard (HF token → model select →
# download) and then starts moe_engine serve itself. Later runs reuse a
# healthy engine on :8001 when one is already up.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ENGINE_HOST="${ENGINE_HOST:-127.0.0.1}"
ENGINE_PORT="${ENGINE_PORT:-8001}"
TERM_BIN="build/jxai_term"

usage() {
    cat <<'EOF'
Usage: ./start_jxai_term.sh [jxai_term args...]

Builds jxai_term if needed, then opens the terminal UI.
First run shows the setup wizard (HF token, model download, config).

Examples:
  ./start_jxai_term.sh
  ./start_jxai_term.sh --wizard          # force the wizard again
  ./start_jxai_term.sh --skip-wizard
  ./start_jxai_term.sh --temp 0.2 --tokens 256
EOF
}

FORWARD=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        *) FORWARD+=("$1"); shift ;;
    esac
done

# Export saved HF token for child processes (engine pulls, curl).
if [[ -f "${XDG_CONFIG_HOME:-$HOME/.config}/jxai/hf_token" ]]; then
    # shellcheck disable=SC2155
    export HF_TOKEN="$(tr -d '\n\r' < "${XDG_CONFIG_HOME:-$HOME/.config}/jxai/hf_token")"
fi

if [[ ! -x "$TERM_BIN" ]]; then
    echo "[jxai_term] building terminal client..."
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="${CXX:-/usr/bin/g++}" >/dev/null
    cmake --build build -j --target jxai_term || {
        echo "[ERROR] failed to build jxai_term" >&2
        exit 1
    }
fi

# Also ensure moe_engine exists so the TUI can spawn it after the wizard.
if [[ ! -x build/moe_engine ]]; then
    echo "[jxai_term] building moe_engine..."
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="${CXX:-/usr/bin/g++}" >/dev/null
    cmake --build build -j --target moe_engine || {
        echo "[ERROR] failed to build moe_engine" >&2
        exit 1
    }
fi

exec "$TERM_BIN" --host "$ENGINE_HOST" --port "$ENGINE_PORT" "${FORWARD[@]}"
