#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
PORT=${1:-8000}
echo "DurLAR viewer: http://127.0.0.1:${PORT}"
exec python3 app.py --port "$PORT"
