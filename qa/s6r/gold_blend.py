"""s6r gold-58 judge: joint3 (.4 V / .4 P / .2 A, the 57327062 form) with an S6 leg added at the weight fixed before any
result, 0.15 (joint3 legs scaled by 0.85), per-finding rank blend. Information only (the C labels and the P/A weights saw
gold; hx48 is a lucky seed on gold, Andrey 10-18). Gate: joint3 + S6 must not lose > 0.001 vs joint3.
`python qa/s6r/gold_blend.py <s6 gold submission.csv> [label]`
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "qa/full")]
import compare_pw as c  # noqa: E402
from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402

W_S6 = 0.15
gold = c.gold_frame(Path(DATA_DIR) / "train.csv")
gold[ID_COL] = gold[ID_COL].astype(str)
ids = sorted(gold[ID_COL])
y = gold.set_index(ID_COL).loc[ids, list(LABEL_KEYS)].to_numpy(np.float64) >= 0.5
cols = list(pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", nrows=1).columns[1:])


def rd(p, gold_only=False):
    df = pd.read_csv(p, dtype={ID_COL: str})
    if gold_only:
        df = df[df["set"] == "gold58"]
    df = df.set_index(ID_COL)
    assert set(ids) <= set(df.index), p
    return df.loc[ids, cols].rank(pct=True).to_numpy()


V = rd(ROOT / ".tmp/v2stack/v3_e2e_full_work/submission.csv")
P = rd(ROOT / ".tmp/qile/in/predictions/blend_mtr01_mth04_olp05_LB0957.csv", True)
A = rd(ROOT / ".tmp/andrey/e2e/ok_safe/A.csv")
S = rd(sys.argv[1])
label = sys.argv[2] if len(sys.argv) > 2 else Path(sys.argv[1]).stem
j3 = 0.4 * V + 0.4 * P + 0.2 * A
new = (1 - W_S6) * j3 + W_S6 * S
m = lambda p: np.nanmean(c.per_label(y, p))
rng = np.random.default_rng(0)
d = np.empty(2000)
for i in range(2000):
    ix = rng.integers(0, len(ids), len(ids))
    d[i] = np.nanmean(c.per_label(y[ix], new[ix])) - np.nanmean(c.per_label(y[ix], j3[ix]))
sp = np.mean([pd.Series(S[:, j]).corr(pd.Series(j3[:, j]), method="spearman") for j in range(len(cols))])
pa, pb = c.per_label(y, new), c.per_label(y, j3)
print(f"[{label}] S6 alone {m(S):.4f} | joint3 {m(j3):.4f} | joint3 + S6 at {W_S6}: {m(new):.4f} | "
      f"delta {m(new) - m(j3):+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] P(>0)={np.mean(d > 0):.2f} | "
      f"spearman S6 ~ joint3 {sp:.3f}")
print("  per finding: " + ", ".join(f"{k} {pa[j] - pb[j]:+.3f}" for j, k in enumerate(cols)))
print("  GATE (>= -0.001):", "PASS" if m(new) - m(j3) >= -0.001 else "FAIL")
