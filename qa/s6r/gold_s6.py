"""Gold-58 macro AUC of the leg_s6 submission (.tmp/s6r/gold_s6fall.csv)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "qa/full")]
import compare_pw as c  # noqa: E402
from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402

p = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".tmp/s6r/gold_s6fall.csv"
gold = c.gold_frame(Path(DATA_DIR) / "train.csv")
gold[ID_COL] = gold[ID_COL].astype(str)
ids = sorted(gold[ID_COL])
y = gold.set_index(ID_COL).loc[ids, list(LABEL_KEYS)].to_numpy(np.float64) >= 0.5
cols = list(pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", nrows=1).columns[1:])
df = pd.read_csv(p, dtype={ID_COL: str}).set_index(ID_COL)
assert list(df.columns) == cols and sorted(df.index) == ids and df.notna().all().all()
pl = c.per_label(y, df.loc[ids, cols].to_numpy(np.float64))
print("gold-58 macro AUC %.4f" % np.nanmean(pl))
print("  " + ", ".join("%s %.3f" % (k, v) for k, v in zip(cols, pl)))
