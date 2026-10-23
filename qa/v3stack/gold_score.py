"""Gold-58 verdict for v3stack (2026-10-13): heliosli's R7 notebook (Raptor 0.5 + compact R7 0.5, public ~0.957) as the
base, main_v5_C rank-blended at 0.3 - the weight 57084313 used, fixed before any R7 gold number was seen.

Pre-declared: blend > R7 base on gold-58 -> one LB probe of the blend; the plain R7 notebook is submitted regardless as
the slot-1 safety baseline.  heliosli trained R7 with the gold studies in its pool and Raptor was selected on them, so
the base's gold score is optimistic and the comparison is biased against our leg.  The v2 base and 57084313's blend
(.tmp/v2stack/e2e_full_work) are printed for reference.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
WK = ROOT / ".tmp/v2stack/v3_e2e_full_work"
V2 = ROOT / ".tmp/v2stack/e2e_full_work"
W = 0.30
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "qa/full")]
import compare_pw as c  # noqa: E402
from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402

gold = c.gold_frame(Path(DATA_DIR) / "train.csv")
gold[ID_COL] = gold[ID_COL].astype(str)
ids = sorted(gold[ID_COL])
y = gold.set_index(ID_COL).loc[ids, list(LABEL_KEYS)].to_numpy(np.float64) >= 0.5
cols = list(pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", nrows=1).columns[1:])


def leg(path):
    df = pd.read_csv(path, dtype={ID_COL: str}).set_index(ID_COL)
    assert len(df) == 58 and df.loc[ids, cols].notna().all().all(), path
    return df.loc[ids, cols].rank(pct=True).to_numpy()


cand = {"r7_base": leg(WK / "_base.csv"), "main_v5_C": leg(WK / "_ours.csv"), "v3_blend": leg(WK / "submission.csv"),
        "v2_base": leg(V2 / "_base.csv"), "v2_blend_57084313": leg(V2 / "submission.csv")}
expect = (1 - W) * cand["r7_base"] + W * cand["main_v5_C"]
print("kernel blend == 0.7 r7 + 0.3 ours (rank):", np.allclose(pd.DataFrame(expect).rank(pct=True), cand["v3_blend"])
      or float(np.corrcoef(expect.ravel(), cand["v3_blend"].ravel())[0, 1]))
pl = {n: c.per_label(y, p) for n, p in cand.items()}
for n in cand:
    print(f"  {n:18s} {np.nanmean(pl[n]):.4f}")
rng = np.random.default_rng(2026)
for a, b in (("v3_blend", "r7_base"), ("r7_base", "v2_base"), ("v3_blend", "v2_blend_57084313")):
    d = np.empty(2000)
    for i in range(2000):
        ix = rng.integers(0, len(ids), len(ids))
        d[i] = np.nanmean(c.per_label(y[ix], cand[a][ix])) - np.nanmean(c.per_label(y[ix], cand[b][ix]))
    print(f"{a} - {b}: {np.nanmean(pl[a]) - np.nanmean(pl[b]):+.4f} "
          f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] P={np.mean(d > 0):.2f}")
print("label            r7    ours  v3bl  v2bas v2bl  rho(r7,ours) rho(v2,ours)")
for j, k in enumerate(cols):
    print(f"  {k:16s} {pl['r7_base'][j]:.3f} {pl['main_v5_C'][j]:.3f} {pl['v3_blend'][j]:.3f} {pl['v2_base'][j]:.3f}"
          f" {pl['v2_blend_57084313'][j]:.3f}  {np.corrcoef(cand['r7_base'][:, j], cand['main_v5_C'][:, j])[0, 1]:+.2f}"
          f" {np.corrcoef(cand['v2_base'][:, j], cand['main_v5_C'][:, j])[0, 1]:+.2f}")
d = np.nanmean(pl["v3_blend"]) - np.nanmean(pl["r7_base"])
print("VERDICT:", "LB PROBE (blend ahead of R7 base on gold)" if d > 0 else "NO BLEND PROBE (R7 base alone not beaten)")
