"""Gold-58 AUC of the label tables themselves (2026-10-15 pre-check for the Opus-label retrain): Qile's Opus Blend
soft labels vs our calibrated v1.1 report targets (v3_pseudo_full's label part, before the teacher mix)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "qa/full")]
import compare_pw as c  # noqa: E402
from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402
from rsna_kad.labels.targets import Calibration, TargetsConfig, make_targets  # noqa: E402

gold = c.gold_frame(Path(DATA_DIR) / "train.csv")
gold[ID_COL] = gold[ID_COL].astype(str)
ids = sorted(gold[ID_COL])
y = gold.set_index(ID_COL).loc[ids, list(LABEL_KEYS)].to_numpy(np.float64) >= 0.5
cols = list(pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", nrows=1).columns[1:])
print("key<->column:", list(zip(LABEL_KEYS, cols)))

op = pd.read_csv(ROOT / ".tmp/qile/in/labels_opus_blend.csv", dtype={ID_COL: str}).set_index(ID_COL)
opus = op.loc[ids, cols].to_numpy(np.float64)
conf = op.loc[ids, [f"{k}_confidence" for k in cols]].to_numpy(np.float64)

V3 = ROOT / "data/targets/v3_pseudo_full"
man = json.loads((V3 / "manifest.json").read_text())
labels = pd.read_parquet(ROOT / man["labels"]); labels[ID_COL] = labels[ID_COL].astype(str)
lt = make_targets(labels, Calibration.from_json(V3 / "calibration_full.json"), TargetsConfig.load(ROOT / man["config"]),
                  exclude_ids=set())
lt[ID_COL] = lt[ID_COL].astype(str)
ours = lt.set_index(ID_COL).loc[ids, [f"{k}_target" for k in LABEL_KEYS]].to_numpy(np.float64)

# all-studies agreement between the two label sources (non-gold), as a diversity read
allid = sorted(set(lt[ID_COL]) & set(op.index) - set(ids))
o_all = op.loc[allid, cols].to_numpy(np.float64)
u_all = lt.set_index(ID_COL).loc[allid, [f"{k}_target" for k in LABEL_KEYS]].to_numpy(np.float64)

po, pu, pm = c.per_label(y, opus), c.per_label(y, ours), c.per_label(y, 0.5 * opus + 0.5 * ours)
print(f"manifest labels={man['labels']} config={man['config']}")
print(f"gold-58 macro AUC: opus {np.nanmean(po):.4f} | ours v1.1 targets {np.nanmean(pu):.4f} | mean {np.nanmean(pm):.4f}")
print(f"{'finding':16s} opus   ours   mean   spearman(non-gold {len(allid)})  opus_conf_gold")
for j, k in enumerate(cols):
    r = pd.Series(o_all[:, j]).corr(pd.Series(u_all[:, j]), method="spearman")
    print(f"  {k:16s} {po[j]:.3f}  {pu[j]:.3f}  {pm[j]:.3f}  {r:.3f}  {conf[:, j].mean():.2f}")
rng = np.random.default_rng(0); d = []
for _ in range(2000):
    i = rng.integers(0, len(ids), len(ids))
    d.append(np.nanmean(c.per_label(y[i], opus[i])) - np.nanmean(c.per_label(y[i], ours[i])))
d = np.array(d)
print(f"opus - ours: {np.nanmean(po) - np.nanmean(pu):+.4f} [{np.nanpercentile(d, 2.5):+.4f}, {np.nanpercentile(d, 97.5):+.4f}] P(<=0)={np.mean(d <= 0):.2f}")
