"""Joint-blend judge for the Opus-label C leg (pre-declared 2026-10-15). The joint kernel = 0.5 v3stack + 0.5 Qile,
v3stack = 0.7 R7 base + 0.3 our C leg (ranks). Swap our leg: 3-seed mean of the Opus-label seeds vs the 3-seed mean
of the current Cfull seeds (same seeds 2026/2126/2226, both never trained on gold), everything else fixed. Paired
bootstrap over the 58 studies (2000). Rebuild the joint kernel only at delta >= +0.004 with P(delta > 0) >= 0.95.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "qa/full")]
import compare_pw as c  # noqa: E402
from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402

SEEDS = (2026, 2126, 2226)
WK = ROOT / ".tmp/v2stack/v3_e2e_full_work"
Q = ROOT / ".tmp/qile/in/predictions/blend_mtr01_mth04_olp05_LB0957.csv"
gold = c.gold_frame(Path(DATA_DIR) / "train.csv")
gold[ID_COL] = gold[ID_COL].astype(str)
ids = sorted(gold[ID_COL])
y = gold.set_index(ID_COL).loc[ids, list(LABEL_KEYS)].to_numpy(np.float64) >= 0.5
cols = list(pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", nrows=1).columns[1:])


def pct(a):
    return pd.DataFrame(a).rank(pct=True).to_numpy()


def seeds(fmt):
    return pct(np.mean([c.load(fmt.format(s), ids) for s in SEEDS], 0))


def rd(p, gold_only=False):
    df = pd.read_csv(p, dtype={ID_COL: str})
    if gold_only:
        df = df[df["set"] == "gold58"]
    return df.set_index(ID_COL).loc[ids, cols].rank(pct=True).to_numpy()


base, qb = rd(WK / "_base.csv"), rd(Q, True)
old, new = seeds("main_armc_convnextv2_t_384_p_full_s{}"), seeds("main_armc_convnextv2_t_384_p_full_opus_s{}")
auc = lambda p: c.per_label(y, p)
cand = {"C old 3 seeds": old, "C opus 3 seeds": new, "Qile": qb,
        "v3 (old C)": 0.7 * base + 0.3 * old, "v3 (opus C)": 0.7 * base + 0.3 * new}
cand["JOINT (old C)"] = 0.5 * cand["v3 (old C)"] + 0.5 * qb
cand["JOINT (opus C)"] = 0.5 * cand["v3 (opus C)"] + 0.5 * qb
for n, p in cand.items():
    print(f"  {n:18s} {np.nanmean(auc(p)):.4f}")
sp = lambda a, b: np.mean([pd.Series(a[:, j]).corr(pd.Series(b[:, j]), method="spearman") for j in range(len(cols))])
print(f"spearman old C ~ Qile {sp(old, qb):.3f} | opus C ~ Qile {sp(new, qb):.3f} | old ~ opus C {sp(old, new):.3f}")
A, B = cand["JOINT (opus C)"], cand["JOINT (old C)"]
pa, pb = auc(A), auc(B)
print("per finding (old -> opus joint): " + ", ".join(f"{k} {pa[j] - pb[j]:+.3f}" for j, k in enumerate(cols)))
rng = np.random.default_rng(0)
d = []
for _ in range(2000):
    i = rng.integers(0, len(ids), len(ids))
    d.append(np.nanmean(c.per_label(y[i], A[i])) - np.nanmean(c.per_label(y[i], B[i])))
d = np.array(d)
delta, p = np.nanmean(pa) - np.nanmean(pb), float(np.mean(d > 0))
print(f"JOINT opus - old: {delta:+.4f} [{np.nanpercentile(d, 2.5):+.4f}, {np.nanpercentile(d, 97.5):+.4f}] P(>0)={p:.3f}")
print("VERDICT:", "REBUILD joint with opus C" if delta >= 0.004 and p >= 0.95 else "TIE/REJECT - keep current joint")
