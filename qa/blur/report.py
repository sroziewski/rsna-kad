"""Score qa/blur/gold_<variant>.csv on gold-58 vs no blur: macro AUC, paired bootstrap delta, per finding.
Sanity: 'none' vs the six runs' own gold predictions (runs/<run>/full/oof.parquet, mean of the seeds)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "qa/full")]
import compare_pw as c  # noqa: E402
from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402
from rsna_kad.train import PRED_COLS  # noqa: E402

from blur_gold import RUN, SEEDS, VARIANTS  # noqa: E402

gold = c.gold_frame(Path(DATA_DIR) / "train.csv")
ids = sorted(gold[ID_COL].astype(str))
y = gold.assign(**{ID_COL: gold[ID_COL].astype(str)}).set_index(ID_COL).loc[ids, list(LABEL_KEYS)].to_numpy(float) >= 0.5
P = {v: pd.read_csv(ROOT / f"qa/blur/gold_{v}.csv", dtype={ID_COL: str}).set_index(ID_COL).loc[ids, PRED_COLS].to_numpy()
     for v in VARIANTS if (ROOT / f"qa/blur/gold_{v}.csv").exists()}
own = np.mean([pd.read_parquet(ROOT / "runs" / RUN.format(s) / "full/oof.parquet").assign(
    **{ID_COL: lambda d: d[ID_COL].astype(str)}).set_index(ID_COL).loc[ids, PRED_COLS].to_numpy() for s in SEEDS], axis=0)
m = lambda p, ix=slice(None): np.nanmean(c.per_label(y[ix], p[ix]))
print(f"sanity: none vs runs' own gold preds max |d| {np.abs(P['none'] - own).max():.2e}; own macro {m(own):.4f}")
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]
base = P["none"]
print(f"none: macro {m(base):.4f}")
for v, p in P.items():
    if v == "none":
        continue
    d = np.array([m(p, ix) - m(base, ix) for ix in boot])
    pl = c.per_label(y, p) - c.per_label(y, base)
    print(f"{v:8s} macro {m(p):.4f} delta {m(p) - m(base):+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] "
          f"P(>0)={np.mean(d > 0):.2f}")
    print("         " + ", ".join(f"{k} {x:+.3f}" for k, x in zip(LABEL_KEYS, pl)))
