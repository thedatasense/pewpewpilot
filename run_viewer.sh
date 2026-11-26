#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"

PORT=${1:-8000}
BIN="./target/release/durlar-server"

if [ ! -x "$BIN" ] || [ Cargo.toml -nt "$BIN" ] || [ src/main.rs -nt "$BIN" ] || [ src/loader.rs -nt "$BIN" ] || [ src/perception.rs -nt "$BIN" ]; then
  echo "Building DurLAR Rust backend (--release)..."
  cargo build --release
fi

echo "DurLAR viewer: http://127.0.0.1:${PORT}"
exec "$BIN" --port "$PORT"
