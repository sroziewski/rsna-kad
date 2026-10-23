#!/usr/bin/env bash
# Launch a long job in a detached screen session with the project environment.
#   scripts/run_screen.sh <name> <command...>
# Log: logs/<name>_<timestamp>.log. GPUs come from CUDA_VISIBLE_DEVICES in the caller's
# environment (default: none - CPU only); when GPUs are requested nvidia-smi is checked first.
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <session-name> <command...>" >&2
  exit 2
fi
name=$1
shift

proj=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$proj/logs" "$proj/.tmp"
log="$proj/logs/${name}_$(date +%Y%m%d-%H%M%S).log"

export HF_HOME=${HF_HOME:-/raid/sroziewski/cache/huggingface}
export PIP_NO_CACHE_DIR=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TMPDIR="$proj/.tmp"
# Libraries that ignore the explicit cache variables (vLLM telemetry, flashinfer/tilelang JIT
# caches) still resolve "~": point it inside the project so nothing lands in the real home dir.
mkdir -p "$proj/.home"
export HOME="$proj/.home"
export XDG_CACHE_HOME="$proj/.home/.cache" XDG_CONFIG_HOME="$proj/.home/.config"
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_CACHE_ROOT="$proj/.vllm_cache"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}
export PATH="/raid/sroziewski/sandbox/bin:$PATH"
# Provenance for bundle.json `git_sha` (train.git_sha reads this variable); "-dirty" marks
# uncommitted changes to tracked files at launch time.
if [ -z "${RSNA_KAD_GIT_SHA:-}" ] && sha=$(git -C "$proj" rev-parse HEAD 2>/dev/null); then
  if [ -n "$(git -C "$proj" status --porcelain --untracked-files=no 2>/dev/null)" ]; then sha="$sha-dirty"; fi
  export RSNA_KAD_GIT_SHA=$sha
fi

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i "$CUDA_VISIBLE_DEVICES" \
    || { echo "nvidia-smi failed for GPUs $CUDA_VISIBLE_DEVICES" >&2; exit 1; }
fi

if screen -ls | grep -q "[.]${name}[[:space:]]"; then
  echo "screen session '$name' is already running" >&2
  exit 1
fi

cmd=$(printf '%q ' "$@")
printf 'cmd: %s\nstarted: %s  cwd: %s  CUDA_VISIBLE_DEVICES=%s  git %s\n' "$cmd" "$(date -Is)" "$proj" \
  "$CUDA_VISIBLE_DEVICES" "${RSNA_KAD_GIT_SHA:-?}" > "$log"
screen -dmS "$name" bash -c "cd '$proj' && $cmd >> '$log' 2>&1; echo \"exit: \$? at \$(date -Is)\" >> '$log'"
echo "started screen '$name'; log: $log"
