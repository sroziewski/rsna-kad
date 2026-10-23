"""Gold-58 check of the OAI probe kernel's e2e run (qa/v2stack/e2e.sh with OURS_W=export/main_v5_C_oai TAG=oai_):
submission.csv must score what qa/oai/judge.py predicted for blend_new (0.9343), and leg 1 alone what it gave oai_C."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
WK = ROOT / ".tmp/v2stack/oai_e2e_full_work"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "qa/full")]
import compare_pw as c  # noqa: E402
from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402

gold = c.gold_frame(Path(DATA_DIR) / "train.csv")
gold[ID_COL] = gold[ID_COL].astype(str)
ids = sorted(gold[ID_COL])
y = gold.set_index(ID_COL).loc[ids, list(LABEL_KEYS)].to_numpy(np.float64) >= 0.5
cols = list(pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", nrows=1).columns[1:])
for name in ("_ours.csv", "_base.csv", "submission.csv"):
    df = pd.read_csv(WK / name, dtype={ID_COL: str}).set_index(ID_COL)
    assert len(df) == 58 and df.loc[ids, cols].notna().all().all(), name
    print(f"{name:15s} gold-58 macro AUC {np.nanmean(c.per_label(y, df.loc[ids, cols].to_numpy(np.float64))):.4f}")
