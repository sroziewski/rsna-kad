"""Score qa/blur/run_r7.sh on gold-58: parity (sigma 0 vs the unpatched V run), R7 alone / V / joint3 with the blurred R7.
joint3 = .4 V + .4 P + .2 A (per-finding ranks), P and A from qa/blur/e2e_blur.sh (hx48 v6): unblurred (orig) or
blurred (s0.65, the pending 57405419 form). Information only (58 studies).   python qa/blur/score_r7.py [sigma_mm]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "qa/full")]
import compare_pw as c  # noqa: E402
from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402

SIG = sys.argv[1] if len(sys.argv) > 1 else "0.288"
S = ROOT / ".tmp/v2stack"
gold = c.gold_frame(Path(DATA_DIR) / "train.csv")
gold[ID_COL] = gold[ID_COL].astype(str)
ids = sorted(gold[ID_COL])
y = gold.set_index(ID_COL).loc[ids, list(LABEL_KEYS)].to_numpy(np.float64) >= 0.5
cols = list(pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", nrows=1).columns[1:])
raw = lambda p: pd.read_csv(p, dtype={ID_COL: str}).set_index(ID_COL).loc[ids, cols]
rk = lambda p: raw(p).rank(pct=True).to_numpy()
m = lambda p, ix=slice(None): np.nanmean(c.per_label(y[ix], p[ix]))
W = {"orig": S / "v3_e2e_full_work", "s0": S / "v3blur0_e2e_full_work", "blur": S / f"v3blur{SIG}_e2e_full_work"}
# a blur that raises inside an R7 worker drops every slice -> mean-score fallbacks, which would read as "blur hurts"
for k in ("s0", "blur"):
    r = json.loads((W[k] / "_prediction/worker0/receipt.json").read_text())
    print(f"R7 {k}: scored {r['scored']}/{r['studies']}, fallback {r['fallback']}, side flips {r['side_check']['flipped']}")
    assert r["scored"] == r["studies"] == len(ids) and not r["fallback"], f"R7 {k} did not score every study"
for part, f in (("R7", "ours_sub.csv"), ("base", "_base.csv"), ("V", "submission.csv")):
    d = (raw(W["orig"] / f) - raw(W["s0"] / f)).abs().to_numpy().max()
    print(f"parity {part}: sigma 0 vs unpatched max |d| {d:.2e}")
for part, f in (("Raptor", "raptor_raw.csv"), ("C", "_ours.csv")):   # only R7 may change with the blur
    d = (raw(W["s0"] / f) - raw(W["blur"] / f)).abs().to_numpy().max()
    print(f"unchanged {part}: sigma 0 vs blur max |d| {d:.2e}")
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]


def cmp(name, a, b):
    d = np.array([m(b, ix) - m(a, ix) for ix in boot])
    pl = c.per_label(y, b) - c.per_label(y, a)
    print(f"{name}: {m(a):.4f} -> {m(b):.4f} delta {m(b) - m(a):+.4f} [{np.percentile(d, 5):+.4f}, {np.percentile(d, 95):+.4f}] "
          f"(90 %) P(>0)={np.mean(d > 0):.2f}")
    print("   " + ", ".join(f"{k} {x:+.3f}" for k, x in zip(cols, pl)))


base = "s0"   # same code path as the blurred run, blur off
for part, f in (("R7 alone", "ours_sub.csv"), ("base (Raptor + R7)", "_base.csv"), ("V", "submission.csv")):
    cmp(f"{part} sigma {SIG} mm", rk(W[base] / f), rk(W["blur"] / f))
B = ROOT / ".tmp/blur"
PA = {"P/A unblurred": (B / "e2e_orig/safe/P.csv", B / "e2e_orig/safe/A.csv"),
      "P/A blurred 0.65 px": (B / "e2e_s0.65/safe/P.csv", B / "e2e_s0.65/safe/A.csv")}
j = lambda v, pa: 0.4 * rk(v) + 0.4 * rk(pa[0]) + 0.2 * rk(pa[1])
v0, vb = W[base] / "submission.csv", W["blur"] / "submission.csv"
cmp("joint3: + R7 blur (P/A unblurred)", j(v0, PA["P/A unblurred"]), j(vb, PA["P/A unblurred"]))
cmp("joint3: + R7 blur + P/A blur", j(v0, PA["P/A unblurred"]), j(vb, PA["P/A blurred 0.65 px"]))
cmp("joint3 P/A blur (pending 57405419) -> + R7 blur", j(v0, PA["P/A blurred 0.65 px"]), j(vb, PA["P/A blurred 0.65 px"]))
