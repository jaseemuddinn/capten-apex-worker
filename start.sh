#!/bin/sh
# Prove the container actually started before Python/CUDA init.
# Hub often shows only "Waiting for container startup" with zero worker logs.
echo "[start.sh] alive $(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
echo "[start.sh] exec python handler.py" >&2
exec python -u handler.py
