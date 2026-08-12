#!/bin/sh
# Replace the pytorch/nvidia entrypoint. Hub may append dockerStartCmd as
# extra args — ignore them and always start the worker.
echo "[start.sh] alive $(date -u +%Y-%m-%dT%H:%M:%SZ) args=$*" >&2
exec python -u /handler.py
