#!/usr/bin/env bash
set -euo pipefail

PUID="${PUID:-99}"
PGID="${PGID:-100}"
UMASK="${UMASK:-022}"

if ! [[ "${PUID}" =~ ^[0-9]+$ && "${PGID}" =~ ^[0-9]+$ ]]; then
    echo "PUID and PGID must be numeric" >&2
    exit 2
fi

if ! getent group appgroup >/dev/null 2>&1; then
    groupadd --non-unique --gid "${PGID}" appgroup
fi
if ! id appuser >/dev/null 2>&1; then
    useradd \
        --non-unique \
        --uid "${PUID}" \
        --gid appgroup \
        --home-dir /config \
        --shell /bin/bash \
        appuser
fi

mkdir -p /config/huggingface /config/gradio /config/torch
chown -R "${PUID}:${PGID}" /config
umask "${UMASK}"

export HOME=/config
export HF_HOME="${HF_HOME:-/config/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/config/huggingface}"
export GRADIO_TEMP_DIR="${GRADIO_TEMP_DIR:-/config/gradio}"
export TORCH_HOME="${TORCH_HOME:-/config/torch}"

exec gosu "${PUID}:${PGID}" "$@"
