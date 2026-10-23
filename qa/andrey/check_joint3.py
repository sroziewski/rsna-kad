"""Checks of the three-leg joint e2e (qa/andrey/e2e_joint3.sh): argv = e2e dir.

ok: submission == (0.4 V + 0.4 P + 0.2 A) on per-leg ranks; leg A reproduces Andrey's own gold file (latest_gold.csv)
and leg P Qile's offline gold58. afail: leg A dropped, submission == 0.5 V + 0.5 P (the current joint). Gold-58 macro
AUC of every leg and of the submission.
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

J = Path(sys.argv[1])
gold = c.gold_frame(Path(DATA_DIR) / "train.csv")
gold[ID_COL] = gold[ID_COL].astype(str)
ids = sorted(gold[ID_COL])
y = gold.set_index(ID_COL).loc[ids, list(LABEL_KEYS)].to_numpy(np.float64) >= 0.5
cols = list(pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", nrows=1).columns[1:])


def rd(p, gold_only=False):
    df = pd.read_csv(p, dtype={ID_COL: str})
    if gold_only:
        df = df[df["set"] == "gold58"]
    return df.set_index(ID_COL).loc[ids, cols]


auc = lambda d: float(np.nanmean(c.per_label(y, d.to_numpy())))
sp = lambda a, b: float(np.mean([a[k].corr(b[k], method="spearman") for k in cols]))
ok = True
for case in ("ok", "afail"):
    sub = rd(J / f"{case}_work/submission.csv")
    rep = json.load(open(J / f"{case}_work/joint_report.json"))
    legs = {k: rd(J / f"{case}_safe/{k}.csv").rank(pct=True) for k in ("V", "P", "A") if (J / f"{case}_safe/{k}.csv").exists()}
    if case == "ok":
        want = 0.4 * legs["V"] + 0.4 * legs["P"] + 0.2 * legs["A"]
        a_ref = rd(ROOT / ".tmp/andrey/in/latest_gold.csv")
        q_ref = rd(ROOT / ".tmp/qile/in/predictions/blend_mtr01_mth04_olp05_LB0957.csv", True)
        s_a, s_p = sp(legs["A"], a_ref), sp(legs["P"], q_ref)
        good = (rep["submission"] == "blend" and set(rep["weights_used"]) == {"V", "P", "A"}
                and np.allclose(sub.to_numpy(), want.to_numpy(), atol=1e-6) and s_a > 0.99 and s_p > 0.999)
        print(f"ok     {'PASS' if good else 'FAIL'}  sub == .4V+.4P+.2A; A ~ Andrey's file spearman {s_a:.4f} (his gold "
              f"{auc(a_ref):.4f}); P ~ Qile offline {s_p:.4f}")
    else:
        want = 0.5 * legs["V"] + 0.5 * legs["P"]
        good = ("error" in rep["A"] and rep["submission"] == "blend" and "A" not in legs
                and np.allclose(sub.to_numpy(), want.to_numpy(), atol=1e-6))
        print(f"afail  {'PASS' if good else 'FAIL'}  A dropped ({rep['A'].get('error', '')[:60]}); sub == .5V+.5P")
    print("        gold " + " ".join(f"{k} {auc(v):.4f}" for k, v in legs.items()) + f" | submission {auc(sub):.4f}"
          + f" | legs {json.dumps({k: rep[k] for k in ('V', 'P', 'A') if k in rep})}")
    ok &= good
print("ALL PASS" if ok else "SOME FAILED")
