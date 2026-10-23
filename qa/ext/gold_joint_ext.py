"""Gold-58 judge for the external-knee C leg (pre-declared 2026-10-16, docs: "C leg + Andrey's external knees").
Same construction as qa/opus/gold_joint_opus.py: joint v2 = 0.5 v3stack + 0.5 Qile, v3stack = 0.7 R7 base + 0.3 C
(ranks); swap the C leg's 3-seed mean (2026/2126/2226, both never trained on gold), everything else fixed. Gold is
information only (it under-rates extra-knee models) with a floor: the joint with the ext C must not lose > 0.004.
Passing the floor earns one LB probe, adopted only at LB >= 0.960. The joint3 form (.4 V / .4 P / .2 A, A = Andrey's
hx48 leg from the e2e ok case) is printed for information.
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
FLOOR = -0.004
WK = ROOT / ".tmp/v2stack/v3_e2e_full_work"
Q = ROOT / ".tmp/qile/in/predictions/blend_mtr01_mth04_olp05_LB0957.csv"
A_LEG = ROOT / ".tmp/andrey/e2e/ok_safe/A.csv"
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


def boot(a, b):
    pa, pb = c.per_label(y, a), c.per_label(y, b)
    rng = np.random.default_rng(0)
    d = []
    for _ in range(2000):
        i = rng.integers(0, len(ids), len(ids))
        d.append(np.nanmean(c.per_label(y[i], a[i])) - np.nanmean(c.per_label(y[i], b[i])))
    d = np.array(d)
    return np.nanmean(pa) - np.nanmean(pb), np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5), float(np.mean(d > 0))


base, qb, al = rd(WK / "_base.csv"), rd(Q, True), rd(A_LEG)
old, new = seeds("main_armc_convnextv2_t_384_p_full_s{}"), seeds("main_armc_convnextv2_t_384_p_full_ext_s{}")
auc = lambda p: c.per_label(y, p)
cand = {"C old 3 seeds": old, "C ext 3 seeds": new, "Qile": qb, "Andrey A": al,
        "v3 (old C)": 0.7 * base + 0.3 * old, "v3 (ext C)": 0.7 * base + 0.3 * new}
cand["JOINT (old C)"] = 0.5 * cand["v3 (old C)"] + 0.5 * qb
cand["JOINT (ext C)"] = 0.5 * cand["v3 (ext C)"] + 0.5 * qb
cand["joint3 (old C)"] = 0.4 * cand["v3 (old C)"] + 0.4 * qb + 0.2 * al
cand["joint3 (ext C)"] = 0.4 * cand["v3 (ext C)"] + 0.4 * qb + 0.2 * al
for n, p in cand.items():
    print(f"  {n:18s} {np.nanmean(auc(p)):.4f}")
sp = lambda a, b: np.mean([pd.Series(a[:, j]).corr(pd.Series(b[:, j]), method="spearman") for j in range(len(cols))])
print(f"spearman old C ~ Qile {sp(old, qb):.3f} | ext C ~ Qile {sp(new, qb):.3f} | old C ~ A {sp(old, al):.3f} | "
      f"ext C ~ A {sp(new, al):.3f} | old ~ ext C {sp(old, new):.3f}")
d, lo, hi, p = boot(new, old)
print(f"C leg ext - old: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] P(>0)={p:.3f}")
pa, pb = auc(cand["JOINT (ext C)"]), auc(cand["JOINT (old C)"])
print("per finding (old -> ext joint): " + ", ".join(f"{k} {pa[j] - pb[j]:+.3f}" for j, k in enumerate(cols)))
d3, lo3, hi3, p3 = boot(cand["joint3 (ext C)"], cand["joint3 (old C)"])
print(f"joint3 ext - old (info): {d3:+.4f} [{lo3:+.4f}, {hi3:+.4f}] P(>0)={p3:.3f}")
d, lo, hi, p = boot(cand["JOINT (ext C)"], cand["JOINT (old C)"])
print(f"JOINT ext - old: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] P(>0)={p:.3f}  (floor {FLOOR:+.3f})")
print("VERDICT:", "FLOOR PASSED - one LB probe of the joint with ext C, adopt only at LB >= 0.960" if d >= FLOOR
      else "FLOOR FAILED - reject, no LB probe")
