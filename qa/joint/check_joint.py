"""Checks of the joint kernel's gold-58 e2e (qa/joint/e2e_joint.sh): argv = joint dir, reference v3stack e2e work dir.

ok / pfirst: submission == (1 - w) rank(V) + w rank(P) and leg V reproduces the reference v3stack run;
fail / partial / sleep: submission == leg V's file and the report names the dropped partner.  Prints gold-58 macro
AUC of V, the stand-in P and the blend (information only: the stand-in is not the partner's model).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "qa/full")]
import compare_pw as c  # noqa: E402
from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402

J, REF = Path(sys.argv[1]), Path(sys.argv[2])
gold = c.gold_frame(Path(DATA_DIR) / "train.csv")
gold[ID_COL] = gold[ID_COL].astype(str)
ids = sorted(gold[ID_COL])
y = gold.set_index(ID_COL).loc[ids, list(LABEL_KEYS)].to_numpy(np.float64) >= 0.5
cols = list(pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", nrows=1).columns[1:])


def rd(p):
    return pd.read_csv(p, dtype={ID_COL: str}).set_index(ID_COL).loc[ids, cols]


ok = True
ref = rd(REF / "submission.csv")
# the R7/Raptor base is not bit-reproducible on GPU (raw diffs ~1e-2 seen); on 58 studies that swaps neighbouring
# ranks, so leg V is held to the reference within a few rank steps, not exactly
V_TOL = 3 / 58
for case in ("ok", "pfirst", "fail", "partial", "sleep"):
    sub, v = rd(J / f"{case}_work/submission.csv"), rd(J / f"{case}_safe/V.csv")
    rep = json.load(open(J / f"{case}_work/joint_report.json"))
    v_ref = float((v - ref).abs().max().max())
    if case in ("ok", "pfirst"):
        p = rd(J / f"{case}_safe/P.csv")
        w = rep["w_partner"]
        want = (1 - w) * v.rank(pct=True) + w * p.rank(pct=True)
        good = rep["submission"] == "blend" and np.allclose(sub.to_numpy(), want.to_numpy()) and v_ref < V_TOL
        g = {n: np.nanmean(c.per_label(y, d.to_numpy())) for n, d in (("V", v), ("P", p), ("blend", sub))}
        print(f"{case:8s} {'PASS' if good else 'FAIL'}  blend == {1 - w}V + {w}P; |V - ref| {v_ref:.1e}; gold "
              + " ".join(f"{n} {s:.4f}" for n, s in g.items()) + f"; spearman {rep.get('spearman_v_p')}")
    else:
        good = (rep["submission"] == "V" and "error" in rep["P"] and np.allclose(sub.to_numpy(), v.to_numpy())
                and v_ref < V_TOL)
        print(f"{case:8s} {'PASS' if good else 'FAIL'}  submission == V; partner dropped: {rep['P'].get('error', '')[:90]}")
    ok &= good
print("ALL PASS" if ok else "SOME FAILED")
