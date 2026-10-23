#!/usr/bin/env bash
# nda-tools with every config/credential path inside the sandbox (no home-dir writes on this server).
#   bash qa/oai/nda.sh login <nda-username>     # once: prompts for the NDA password (not login.gov), stores it 0600
#   bash qa/oai/nda.sh downloadcmd -dp <package-id> -u <nda-username> --verify -d <dir>
# nda-tools reads the password only from a keyring or an interactive prompt; the file keyring keeps later runs
# (screen sessions, file-list downloads) non-interactive.
set -euo pipefail
P=/raid/sroziewski/sandbox/rsna_kad
H=$P/.secrets/nda_home
mkdir -p $H && chmod 700 $P/.secrets $H
export HOME=$H XDG_DATA_HOME=$H/.local/share XDG_CONFIG_HOME=$H/.config
export PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring
umask 077
cmd=$1; shift
if [ "$cmd" = login ]; then
  exec $P/.venv-nda/bin/python -c "
import getpass, keyring, sys
keyring.set_password('nda-tools', sys.argv[1], getpass.getpass('NDA password for %s: ' % sys.argv[1]))
print('stored under', sys.argv[1])" "$1"
fi
exec $P/.venv-nda/bin/$cmd "$@"
