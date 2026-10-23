"""Gold-58 of leg combinations around joint v2 (2026-10-17, user question "maybe use Andrey's model and Qile"):
V old (v3stack) / V new (r6stack) / P Qile / A Andrey hx48 (joint3 e2e ok case), per-finding rank blends.
Information only: P's blend weights were tuned on gold, A's per-finding mix was picked on gold, Raptor was selected
on gold."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "qa/full")]
import compare_pw as c  # noqa: E402
from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402

gold = c.gold_frame(Path(DATA_DIR) / "train.csv")
gold[ID_COL] = gold[ID_COL].astype(str)
ids = sorted(gold[ID_COL])
y = gold.set_index(ID_COL).loc[ids, list(LABEL_KEYS)].to_numpy(np.float64) >= 0.5
cols = list(pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", nrows=1).columns[1:])


def rd(p, gold_only=False):
    df = pd.read_csv(p, dtype={ID_COL: str})
    if gold_only:
        df = df[df["set"] == "gold58"]
    return df.set_index(ID_COL).loc[ids, cols].rank(pct=True).to_numpy()


L = {"Vold": rd(ROOT / ".tmp/v2stack/v3_e2e_full_work/submission.csv"),
     "Vnew": rd(ROOT / ".tmp/v2stack/r6_e2e_full_work/submission.csv"),
     "P": rd(ROOT / ".tmp/qile/in/predictions/blend_mtr01_mth04_olp05_LB0957.csv", True),
     "A": rd(ROOT / ".tmp/andrey/e2e/ok_safe/A.csv")}
mac = lambda w: float(np.nanmean(c.per_label(y, sum(v * L[k] for k, v in w.items()))))
ref = {"Vold": .5, "P": .5}
rng = np.random.default_rng(0)
B = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]


def boot(w):
    p1, p0 = sum(v * L[k] for k, v in w.items()), sum(v * L[k] for k, v in ref.items())
    d = np.array([np.nanmean(c.per_label(y[b], p1[b])) - np.nanmean(c.per_label(y[b], p0[b])) for b in B])
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


combos = {
    "joint v2 = .5 Vold + .5 P (ref)": ref,
    ".5 Vnew + .5 P": {"Vnew": .5, "P": .5},
    "P + A .5/.5": {"P": .5, "A": .5},
    "P + A .6/.4": {"P": .6, "A": .4},
    "joint3 .4 Vold/.4 P/.2 A": {"Vold": .4, "P": .4, "A": .2},
    "joint3 .4 Vnew/.4 P/.2 A": {"Vnew": .4, "P": .4, "A": .2},
    "1/3 Vold P A": {"Vold": 1 / 3, "P": 1 / 3, "A": 1 / 3},
    "1/3 Vnew P A": {"Vnew": 1 / 3, "P": 1 / 3, "A": 1 / 3},
}
for k in L:
    print(f"  {k:30s} alone {mac({k: 1}):.4f}")
for n, w in combos.items():
    d, lo, hi, p = boot(w)
    print(f"  {n:30s} {mac(w):.4f}  vs joint v2 {mac(w) - mac(ref):+.4f} [{lo:+.4f}, {hi:+.4f}] P(>0)={p:.2f}")
