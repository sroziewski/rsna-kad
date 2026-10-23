"""Gold-58 + runtime check of the r6base V leg (2026-10-17, user-requested): goodpjw2008's 0.958 notebook as the base
of our V leg in place of heliosli's R7 notebook, main_v5_C at 0.3 unchanged; joint v2 form = 0.5 V + 0.5 Qile.

Gold is information only: nartaa selected the Raptor models on gold-58, and both bases carry that bias. Hidden runtime
is projected from the e2e's A6000 seconds with the ratio of the old V leg's hidden time to its e2e time.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "qa/full")]
import compare_pw as c  # noqa: E402
from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402

OLD = ROOT / ".tmp/v2stack/v3_e2e_full_work"
NEW = ROOT / ".tmp/v2stack/r6_e2e_full_work"
LOG = ROOT / ".tmp/v2stack/r6_e2e_full.log"
Q = ROOT / ".tmp/qile/in/predictions/blend_mtr01_mth04_olp05_LB0957.csv"
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
    assert set(ids) <= set(df.index) and df.loc[ids, cols].notna().all().all(), p
    return df.loc[ids, cols].rank(pct=True).to_numpy()


def boot(a, b):
    rng = np.random.default_rng(0)
    d = np.empty(2000)
    for i in range(2000):
        ix = rng.integers(0, len(ids), len(ids))
        d[i] = np.nanmean(c.per_label(y[ix], a[ix])) - np.nanmean(c.per_label(y[ix], b[ix]))
    return (np.nanmean(c.per_label(y, a)) - np.nanmean(c.per_label(y, b)), np.percentile(d, 2.5),
            np.percentile(d, 97.5), float(np.mean(d > 0)))


qb = rd(Q, True)
cand = {"R7 base (old)": rd(OLD / "_base.csv"), "R6 0.958 base (new)": rd(NEW / "_base.csv"),
        "main_v5_C": rd(NEW / "_ours.csv"), "V old (v3stack)": rd(OLD / "submission.csv"),
        "V new (r6stack)": rd(NEW / "submission.csv"), "Qile": qb}
cand["JOINT v2 old"] = 0.5 * cand["V old (v3stack)"] + 0.5 * qb
cand["JOINT v2 new"] = 0.5 * cand["V new (r6stack)"] + 0.5 * qb
pl = {n: c.per_label(y, p) for n, p in cand.items()}
for n in cand:
    print(f"  {n:22s} {np.nanmean(pl[n]):.4f}")
print("kernel V new == 0.7 base + 0.3 C (rank):", np.allclose(
    pd.DataFrame(0.7 * cand["R6 0.958 base (new)"] + 0.3 * cand["main_v5_C"]).rank(pct=True), cand["V new (r6stack)"]))
sp = lambda a, b: np.mean([pd.Series(a[:, j]).corr(pd.Series(b[:, j]), method="spearman") for j in range(len(cols))])
print(f"spearman R7 ~ R6-0.958 base {sp(cand['R7 base (old)'], cand['R6 0.958 base (new)']):.3f} | "
      f"new base ~ Qile {sp(cand['R6 0.958 base (new)'], qb):.3f} | old base ~ Qile {sp(cand['R7 base (old)'], qb):.3f}")
for a, b in (("R6 0.958 base (new)", "R7 base (old)"), ("V new (r6stack)", "V old (v3stack)"),
             ("JOINT v2 new", "JOINT v2 old")):
    d, lo, hi, p = boot(cand[a], cand[b])
    print(f"{a} - {b}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] P(>0)={p:.2f}")
pa, pb = pl["JOINT v2 new"], pl["JOINT v2 old"]
print("per finding (joint old -> new): " + ", ".join(f"{k} {pa[j] - pb[j]:+.3f}" for j, k in enumerate(cols)))

rep_new = json.loads((NEW / "v2stack_report.json").read_text())
rep_old = json.loads((OLD / "v2stack_report.json").read_text())
print(f"V leg e2e seconds (58 studies, A6000): old {rep_old['total_s']} (C {rep_old['ours_s']}, base "
      f"{rep_old['base_s']}) | new {rep_new['total_s']} (C {rep_new['ours_s']}, base {rep_new['base_s']}) | "
      f"ratio {rep_new['total_s'] / rep_old['total_s']:.2f}")
log = LOG.read_text(errors="replace")
for pat in (r"RAPTOR_STAGE_DONE seconds \d+", r"RAPTOR224_STAGE_DONE seconds \d+", r'"seconds": [\d.]+',
            r"own leg: blended .*? in \d+s", r"BLEND_FALLBACK.*", r"Part B.*skipp.*"):
    for m in re.findall(pat, log):
        print("  log:", m[:160])
print("receipt:", rep_new.get("base_receipt"))
