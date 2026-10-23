#!/usr/bin/env bash
# main_v4 blend (A_p + E_p + C_p, equal weights) with vs without z-TTA, paired, from qa/ztta/<run>/{tta,notta}_oof.parquet.
# gold58 is the decision arbiter (drop z-TTA if it gains < 0.002); weak/strict are contaminated for pseudo runs.
set -eu
P=/raid/sroziewski/sandbox/bin/python
cd "$(dirname "$0")/../.."
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/.tmp" CUDA_VISIBLE_DEVICES=
if screen -ls | grep -q "[.]ztta[[:space:]]"; then echo "screen ztta still running" >&2; exit 1; fi
E=qa/ztta/eval; rm -rf $E; mkdir -p $E
$P - <<'PY'
import pandas as pd, numpy as np
from pathlib import Path
runs = ["main_convnext_s_384_e8_p", "eff_pseudo_256", "main_armc_convnextv2_t_384_p"]
short = {"main_convnext_s_384_e8_p": "Ap", "eff_pseudo_256": "Ep", "main_armc_convnextv2_t_384_p": "Cp"}
for tag in ("notta", "tta"):
    dfs = [pd.read_parquet(f"qa/ztta/{r}/{tag}_oof.parquet").sort_values("StudyInstanceUID").reset_index(drop=True) for r in runs]
    for r, d in zip(runs, dfs):
        Path(f"qa/ztta/eval/{short[r]}_{tag}").mkdir(parents=True, exist_ok=True)
        d.to_parquet(f"qa/ztta/eval/{short[r]}_{tag}/oof.parquet", index=False)
    assert all((d["StudyInstanceUID"] == dfs[0]["StudyInstanceUID"]).all() for d in dfs)
    cols = [c for c in dfs[0].columns if c.endswith("_pred")]
    b = dfs[0][["StudyInstanceUID", "fold"]].copy()
    b[cols] = np.mean([d[cols].to_numpy(np.float64) for d in dfs], axis=0).astype(np.float32)
    Path(f"qa/ztta/eval/v4_{tag}").mkdir(parents=True, exist_ok=True)
    b.to_parquet(f"qa/ztta/eval/v4_{tag}/oof.parquet", index=False)
PY
$P scripts/evaluate_runs.py --runs $E/v4_notta $E/v4_tta $E/Ap_notta $E/Ap_tta $E/Ep_notta $E/Ep_tta $E/Cp_notta $E/Cp_tta \
  --arbiters gold58 weak strict --n-boot 2000 --labels data/labels/llm_v1.parquet --targets-dir data/targets/v1 \
  --targets-config configs/targets_v1.yaml --out qa/ztta/arbiters.md --json-out qa/ztta/arbiters.json
