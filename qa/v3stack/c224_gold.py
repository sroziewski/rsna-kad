"""Gold-58 check of nartaa's c224 Raptor checkpoint inside v3stack (2026-10-13, user-requested).

Arms, both rebuilt from the v3 e2e run (.tmp/v2stack/v3_e2e_full_work) so the only difference is c224:
  v3      = 0.7 * rank(0.5 * rank(Raptor 384 raw) + 0.5 * rank(compact R7)) + 0.3 * rank(main_v5_C)   (57150980, LB 0.957)
  v3_c224 = the same with the Raptor slot = 0.65 * rank(Raptor 384) + 0.35 * rank(c224), varshithakolupuri's A224 = 0.35.
c224 gold predictions: qa/oaistack/gold_c224.csv (nartaa's notebook, oaistack gold run).
Pre-declared: v3_c224 > v3 on gold-58 -> one LB probe.  nartaa selected both checkpoints on gold-58, so this check is
biased in c224's favour; a small gold gain is weak evidence.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
WK = ROOT / ".tmp/v2stack/v3_e2e_full_work"
A224, W = 0.35, 0.30
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "qa/full")]
import compare_pw as c  # noqa: E402
from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402

gold = c.gold_frame(Path(DATA_DIR) / "train.csv")
gold[ID_COL] = gold[ID_COL].astype(str)
ids = sorted(gold[ID_COL])
y = gold.set_index(ID_COL).loc[ids, list(LABEL_KEYS)].to_numpy(np.float64) >= 0.5
cols = list(pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", nrows=1).columns[1:])


def rk(x):
    return pd.DataFrame(x).rank(pct=True).to_numpy()


def leg(path):
    df = pd.read_csv(path, dtype={ID_COL: str}).set_index(ID_COL)
    assert df.loc[ids, cols].notna().all().all(), path
    return rk(df.loc[ids, cols].to_numpy(np.float64))


r384, r7, ours = leg(WK / "raptor_raw.csv"), leg(WK / "ours_sub.csv"), leg(WK / "_ours.csv")
r224, g384 = leg(ROOT / "qa/oaistack/gold_c224.csv"), leg(ROOT / "qa/oaistack/gold_c384.csv")
print("rho(Raptor 384 in v3 run, oaistack c384) per label min %.3f" %
      min(np.corrcoef(r384[:, j], g384[:, j])[0, 1] for j in range(len(cols))))
print("rho(Raptor 384, c224) per label mean %.3f" % np.mean([np.corrcoef(r384[:, j], r224[:, j])[0, 1] for j in range(len(cols))]))


def v3(raptor):
    return (1 - W) * rk(0.5 * raptor + 0.5 * r7) + W * ours


cand = {"c384": r384, "c224": r224, "r7": r7, "v3": v3(r384), "v3_c224": v3((1 - A224) * r384 + A224 * r224),
        "v3_sub_check": leg(WK / "submission.csv")}
pl = {n: c.per_label(y, p) for n, p in cand.items()}
for n in cand:
    print(f"  {n:13s} {np.nanmean(pl[n]):.4f}")
rng = np.random.default_rng(2026)
d = np.empty(2000)
for i in range(2000):
    ix = rng.integers(0, len(ids), len(ids))
    d[i] = np.nanmean(c.per_label(y[ix], cand["v3_c224"][ix])) - np.nanmean(c.per_label(y[ix], cand["v3"][ix]))
delta = np.nanmean(pl["v3_c224"]) - np.nanmean(pl["v3"])
print(f"v3_c224 - v3: {delta:+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] P={np.mean(d > 0):.2f}")
print("label             c384  c224  v3    v3_c224")
for j, k in enumerate(cols):
    print(f"  {k:16s} {pl['c384'][j]:.3f} {pl['c224'][j]:.3f} {pl['v3'][j]:.3f} {pl['v3_c224'][j]:.3f}")
print("VERDICT:", "LB PROBE (c224 ahead on gold)" if delta > 0 else "STOP (c224 not ahead on gold)")
