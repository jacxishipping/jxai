#!/usr/bin/env bash
# Fetch a prebuilt llama.cpp `llama-server` into tools/llama/ for the hybrid
# GPU backend. PCs without a GPU do not need this — moe_engine CPU is default.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/tools/llama"
mkdir -p "$OUT"

TAG="${LLAMA_CPP_TAG:-b5758}"
ARCH="$(uname -m)"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"

pick_asset() {
  case "$OS-$ARCH" in
    linux-x86_64|linux-amd64)
      echo "llama-${TAG}-bin-ubuntu-x64.zip"
      ;;
    darwin-arm64|darwin-aarch64)
      echo "llama-${TAG}-bin-macos-arm64.zip"
      ;;
    darwin-x86_64)
      echo "llama-${TAG}-bin-macos-x64.zip"
      ;;
    *)
      echo "Unsupported platform: $OS $ARCH" >&2
      echo "Build llama.cpp yourself and place llama-server in $OUT" >&2
      exit 1
      ;;
  esac
}

ASSET="$(pick_asset)"
URL="https://github.com/ggerganov/llama.cpp/releases/download/${TAG}/${ASSET}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading $URL"
if command -v curl >/dev/null 2>&1; then
  curl -fsSL -o "$TMP/$ASSET" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -q -O "$TMP/$ASSET" "$URL"
else
  echo "Need curl or wget" >&2
  exit 1
fi

echo "Extracting to $OUT"
unzip -qo "$TMP/$ASSET" -d "$TMP/extract"
# Release zips nest binaries under build/bin or similar — find llama-server.
FOUND="$(find "$TMP/extract" -type f -name 'llama-server' | head -n 1 || true)"
if [[ -z "$FOUND" ]]; then
  echo "llama-server not found in archive; listing:" >&2
  find "$TMP/extract" -type f | head -n 40 >&2
  exit 1
fi
cp -f "$FOUND" "$OUT/llama-server"
chmod +x "$OUT/llama-server"
# Copy sibling shared libs if present next to the binary.
BIN_DIR="$(dirname "$FOUND")"
shopt -s nullglob
for f in "$BIN_DIR"/*.so "$BIN_DIR"/*.dylib "$BIN_DIR"/ggml*; do
  [[ -f "$f" ]] || continue
  cp -f "$f" "$OUT/" 2>/dev/null || true
done

echo "Installed: $OUT/llama-server"
echo "Use: moe_engine serve --model <gguf> --backend auto"
echo "Or set LLAMA_SERVER=$OUT/llama-server"
