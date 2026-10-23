"""Score qa/blur/e2e_blur.sh: parity (s0 == orig per leg), gold-58 macro AUC per leg, and joint3 (.4 V / .4 P / .2 A,
per-finding rank) with the blurred P and A vs the unpatched ones; V = the v3stack gold run. Information only (58 studies).
  python qa/blur/score_legs.py [sigma]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "qa/full")]
import compare_pw as c  # noqa: E402
from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402

SIG = sys.argv[1] if len(sys.argv) > 1 else "0.65"
B = ROOT / ".tmp/blur"
gold = c.gold_frame(Path(DATA_DIR) / "train.csv")
gold[ID_COL] = gold[ID_COL].astype(str)
ids = sorted(gold[ID_COL])
y = gold.set_index(ID_COL).loc[ids, list(LABEL_KEYS)].to_numpy(np.float64) >= 0.5
cols = list(pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", nrows=1).columns[1:])


def raw(p):
    return pd.read_csv(p, dtype={ID_COL: str}).set_index(ID_COL).loc[ids, cols]


def rk(p):
    return raw(p).rank(pct=True).to_numpy()


m = lambda p, ix=slice(None): np.nanmean(c.per_label(y[ix], p[ix]))
legs = {case: {k: B / f"e2e_{case}/safe/{k}.csv" for k in "PA"} for case in ("orig", "s0", f"s{SIG}")}
ok = True
o2 = B / "e2e_orig2/safe"
for k in "PA":
    d = (raw(legs["orig"][k]) - raw(legs["s0"][k])).abs().to_numpy().max()
    # parity = s0 no further from orig than a repeat of the unpatched leg (orig2; hx48 is not bit-deterministic on GPU)
    floor = (raw(legs["orig"][k]) - raw(o2 / f"{k}.csv")).abs().to_numpy().max() if (o2 / f"{k}.csv").exists() else 0.0
    rho = np.mean([raw(legs["orig"][k])[x].corr(raw(legs["s0"][k])[x], method="spearman") for x in cols])
    print(f"parity {k}: s0 vs orig max |d| {d:.2e} (orig vs orig2 {floor:.2e}), spearman {rho:.5f}",
          "PASS" if d <= floor + 1e-9 else "FAIL")
    ok &= d <= floor + 1e-9
print(f"A orig vs Andrey's gold58_hx48_v6.csv max |d| {(raw(legs['orig']['A']) - raw(ROOT / '.tmp/andrey/gold58_hx48_v6.csv')).abs().to_numpy().max():.2e}")
rng = np.random.default_rng(0)
boot = [rng.integers(0, len(ids), len(ids)) for _ in range(2000)]


def cmp(name, a, b):
    d = np.array([m(b, ix) - m(a, ix) for ix in boot])
    pl = c.per_label(y, b) - c.per_label(y, a)
    print(f"{name}: {m(a):.4f} -> {m(b):.4f} delta {m(b) - m(a):+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}] "
          f"P(>0)={np.mean(d > 0):.2f}")
    print("   " + ", ".join(f"{k} {x:+.3f}" for k, x in zip(cols, pl)))


for k in "PA":
    cmp(f"leg {k} sigma {SIG}", rk(legs["orig"][k]), rk(legs[f"s{SIG}"][k]))
V = rk(ROOT / ".tmp/v2stack/v3_e2e_full_work/submission.csv")
j = lambda case: 0.4 * V + 0.4 * rk(legs[case]["P"]) + 0.2 * rk(legs[case]["A"])
cmp(f"joint3 sigma {SIG}", j("orig"), j(f"s{SIG}"))
print("PARITY", "PASS" if ok else "FAIL")
