#!/usr/bin/env python3
"""Gold-58 verdict for main_v5_C + OAI students (qa/oai, 2026-10-12).

Pre-declared before training: the deciding comparison is blend(v2 base 0.7, OAI C 0.3) vs blend(v2 base 0.7,
main_v5_C 0.3), i.e. what would replace slot 1 (57084313, LB 0.955).  If the delta is > 0, one LB probe is
submitted; adoption is decided on that probe.  OAI C alone vs main_v5_C, the three OAI-mapped findings and the
rank correlation with the base are reported for information (the stk lesson: a student that moves toward the blend
partner adds nothing).  The v2 base's gold score is optimistic (Raptor was selected on gold-58), equally in both arms.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "qa/full")]
import compare_pw as c  # noqa: E402
from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402

W = 0.30
SEEDS = [int(s) for s in sys.argv[1:]] or list(range(2026, 2600, 100))
gold = c.gold_frame(Path(DATA_DIR) / "train.csv")
gold[ID_COL] = gold[ID_COL].astype(str)
ids = sorted(gold[ID_COL])
y = gold.set_index(ID_COL).loc[ids, list(LABEL_KEYS)].to_numpy(np.float64) >= 0.5
cols = list(pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", nrows=1).columns[1:])


def rk(a):
    return pd.DataFrame(a).rank(pct=True).to_numpy()


base = (pd.read_csv(ROOT / ".tmp/v2stack/e2e_full_work/_base.csv", dtype={ID_COL: str}).set_index(ID_COL)
        .loc[ids, cols].rank(pct=True).to_numpy())
old = np.mean([c.load(f"main_armc_convnextv2_t_384_p_full_s{s}", ids) for s in SEEDS], axis=0)
new = np.mean([c.load(f"main_armc_convnextv2_t_384_p_full_oai_s{s}", ids) for s in SEEDS], axis=0)
cand = {"v2base": base, "main_v5_C": old, "oai_C": new,
        "blend_old": (1 - W) * base + W * rk(old), "blend_new": (1 - W) * base + W * rk(new)}
pl = {n: c.per_label(y, p) for n, p in cand.items()}
print("seeds", SEEDS)
print("gold-58 macro AUC")
for n in cand:
    print(f"  {n:10s} {np.nanmean(pl[n]):.4f}")
rng = np.random.default_rng(2026)
for a, b in (("blend_new", "blend_old"), ("oai_C", "main_v5_C")):
    d = np.empty(2000)
    for i in range(2000):
        ix = rng.integers(0, len(ids), len(ids))
        d[i] = np.nanmean(c.per_label(y[ix], cand[a][ix])) - np.nanmean(c.per_label(y[ix], cand[b][ix]))
    print(f"{a} - {b}: {np.nanmean(pl[a]) - np.nanmean(pl[b]):+.4f} "
          f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] P={np.mean(d > 0):.2f}")
seed_scores = [np.nanmean(c.per_label(y, c.load(f"main_armc_convnextv2_t_384_p_full_oai_s{s}", ids))) for s in SEEDS]
print("oai_C single seeds:", " ".join(f"{v:.4f}" for v in seed_scores))
print("label            base   old_C  oai_C  b_old  b_new  rho(base,old) rho(base,new)")
for j, k in enumerate(cols):
    ro = np.corrcoef(base[:, j], rk(old)[:, j])[0, 1]
    rn = np.corrcoef(base[:, j], rk(new)[:, j])[0, 1]
    print(f"  {k:16s} {pl['v2base'][j]:.3f}  {pl['main_v5_C'][j]:.3f}  {pl['oai_C'][j]:.3f}  {pl['blend_old'][j]:.3f}"
          f"  {pl['blend_new'][j]:.3f}  {ro:+.2f} {rn:+.2f}")
d = np.nanmean(pl["blend_new"]) - np.nanmean(pl["blend_old"])
print("VERDICT:", "LB PROBE (blend_new ahead on gold)" if d > 0 else "STOP (blend_new not ahead on gold)")
