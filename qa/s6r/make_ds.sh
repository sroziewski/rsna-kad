#!/usr/bin/env bash
# s6r weights dataset sroziews/rsna-kad-s6r (private): our OURS_s42_fall.pt + its run_config / DONE marker, and the
# code files leg_s6r.py reads from the weights dataset (nartaa's code_manifest.json + s6_extras.bin, Apache-2.0).
set -euo pipefail
P=/raid/sroziewski/sandbox/rsna_kad
S=$P/.tmp/s6r
M=$S/runs/full
D=$S/ds
rm -rf $D; mkdir -p $D
cp $M/models/OURS_s42_fall.pt $D/
cp $M/run_config.json $D/OURS_s42_fall.run_config.json
cp $M/DONE_OURS_s42_fall.json $D/OURS_s42_fall.DONE.json
cp $S/dl/code_manifest.json $S/dl/s6_extras.bin $D/
/raid/sroziewski/sandbox/bin/python -I - $D <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
d = Path(sys.argv[1])
files = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(d.iterdir())}
(d / "checkpoint-manifest.json").write_text(json.dumps({
    "model": "OURS_s42_fall.pt", "arm": "OURS", "trainer": "nartaa/rsna-knee-s6fall-weights training_code/s6 (Apache-2.0), "
    "targets = rsna_kad v3_pseudo_full", "files": files}, indent=1))
(d / "dataset-metadata.json").write_text(json.dumps({"title": "rsna-kad-s6r", "id": "sroziews/rsna-kad-s6r",
                                                     "licenses": [{"name": "Apache 2.0"}]}, indent=1))
print(json.dumps(files, indent=1))
PYEOF
[ "$(sha256sum $D/OURS_s42_fall.pt | cut -c1-64)" = 8512fadb36be9ed92dc7f71baf68f7aadc20346c1eec31acea85c2bd9d949f21 ]
cd $P
bash scripts/kaggle_cli.sh raw datasets create -p $D
echo DS_CREATED
