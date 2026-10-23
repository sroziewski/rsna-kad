#!/usr/bin/env bash
# kaggle CLI with the project credentials (architecture §2.8); never writes under $HOME.
#   scripts/kaggle_cli.sh whoami
#   scripts/kaggle_cli.sh dataset-create export/<name>             # first upload of the weights dataset
#   scripts/kaggle_cli.sh dataset-version export/<name> "<message>"
#   scripts/kaggle_cli.sh dataset-status                           # processing state + file list
#   scripts/kaggle_cli.sh kernel-push                              # notebooks/kernel/kernel-metadata.json
#   scripts/kaggle_cli.sh kernel-status
#   scripts/kaggle_cli.sh kernel-output <dir>                      # download the log + submission.csv
#   scripts/kaggle_cli.sh submit <kernel version> "<message>"      # code-competition submission of that version
#   scripts/kaggle_cli.sh submissions
#   scripts/kaggle_cli.sh raw <any kaggle args...>
set -euo pipefail

proj=$(cd "$(dirname "$0")/.." && pwd)
export KAGGLE_CONFIG_DIR=${KAGGLE_CONFIG_DIR:-$proj/.kaggle}
# kagglesdk reads the token only from KAGGLE_API_TOKEN (a value or a file path) or from
# ~/.kaggle/access_token - never from KAGGLE_CONFIG_DIR - so with HOME redirected (run_screen.sh)
# the path form below is the one that works, and the token value never enters the environment.
if [ -z "${KAGGLE_API_TOKEN:-}" ]; then
  for f in "$KAGGLE_CONFIG_DIR/access_token" "$proj/.kaggle/access_token"; do
    if [ -s "$f" ]; then export KAGGLE_API_TOKEN=$f; break; fi
  done
fi
if [ -z "${KAGGLE_API_TOKEN:-}" ]; then
  echo "no Kaggle token: put it in $KAGGLE_CONFIG_DIR/access_token (chmod 600) or export KAGGLE_API_TOKEN" >&2
  exit 2
fi
if [ -f "$KAGGLE_API_TOKEN" ] && grep -q $'\r' "$KAGGLE_API_TOKEN"; then
  # a token pasted from Windows carries a CR that breaks the Authorization header
  t=$(tr -d '\r\n' < "$KAGGLE_API_TOKEN") && (umask 077; printf '%s' "$t" > "$KAGGLE_API_TOKEN.tmp") \
    && mv "$KAGGLE_API_TOKEN.tmp" "$KAGGLE_API_TOKEN"; unset t
fi
export PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
export PATH="/raid/sroziewski/sandbox/bin:$PATH"
export TMPDIR=${TMPDIR:-$proj/.tmp}
mkdir -p "$TMPDIR"

competition=rsna-knee-abnormality-detection
kernel_meta=$proj/notebooks/kernel/kernel-metadata.json

kernel_id() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['id'])" "$kernel_meta"; }
dataset_id() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['dataset_sources'][0])" "$kernel_meta"; }

cmd=${1:-}
shift || true
case "$cmd" in
  whoami)
    python3 - <<'EOF'
import kaggle
kaggle.api.authenticate()
print(kaggle.api.config_values.get("username"))
EOF
    ;;
  dataset-create)   kaggle datasets create -p "$1" --dir-mode zip ;;
  dataset-version)  kaggle datasets version -p "$1" --dir-mode zip -m "${2:-update}" ;;
  dataset-status)   kaggle datasets status "$(dataset_id)"; kaggle datasets files "$(dataset_id)" ;;
  kernel-push)      kaggle kernels push -p "$proj/notebooks/kernel" ;;
  kernel-status)    kaggle kernels status "$(kernel_id)" ;;
  kernel-output)    mkdir -p "${1:?dir}"; kaggle kernels output "$(kernel_id)" -p "$1" ;;
  submit)           kaggle competitions submit "$competition" -k "$(kernel_id)" -v "${1:?kernel version}" \
                      -f submission.csv -m "${2:-$(kernel_id) v$1}" ;;
  submissions)      kaggle competitions submissions "$competition" ;;
  raw)              kaggle "$@" ;;
  *)
    sed -n '2,13p' "$0" >&2
    exit 2
    ;;
esac
