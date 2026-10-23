"""Q4 - ensemble anatomy: arm correlations, leave-one-out, every equal-weight member set, OOF->LB map.
Writes s4_ensemble.md + s4_subsets.csv."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from common import (LABEL_KEYS, LB, OUT, _rank_auc, aligned, blend, bootstrap_subset_auc, fmt, load_all,
                    paired_delta, subset_auc)

ctx = load_all()
subs, preds = ctx["subs"], ctx["preds"]
ARMS = ["A", "A8", "E", "C", "MVP"]
W = {"weak": 0.58, "consensus": 0.38, "gold": 0.04}  # evaluate_runs.py composite weights
L = ["# S4 ensemble anatomy", ""]

# ---- correlations ------------------------------------------------------------------------------
subW = subs["weak"]
PW = {a: aligned(ctx, subW, preds[a]) for a in ARMS}
L += ["## Per-label Pearson correlation of OOF predictions (4,349 weak rows)", "",
      "| pair | " + " | ".join(LABEL_KEYS) + " | mean |", "|---|" + "---|" * 13]
for a, b in combinations(["E", "A", "A8", "C"], 2):
    r = [np.corrcoef(PW[a][:, j], PW[b][:, j])[0, 1] for j in range(12)]
    L.append(f"| {a}-{b} | " + " | ".join(f"{v:.2f}" for v in r) + f" | {np.mean(r):.3f} |")

# ---- every member set --------------------------------------------------------------------------
sets = [c for n in range(1, 6) for c in combinations(ARMS, n)]
rows = []
for s in sets:
    p = blend(preds, s)
    rec = {"set": "+".join(s), "n": len(s)}
    for arb, sub in subs.items():
        rec[arb] = subset_auc(sub, aligned(ctx, sub, p))[0]
    rec["composite"] = sum(W[k] * rec[k] for k in W) / sum(W.values())
    rows.append(rec)
S = pd.DataFrame(rows).sort_values("weak", ascending=False)
S.to_csv(OUT / "s4_subsets.csv", index=False)

# paired bootstraps vs the shipped A+E+C
key = ["A+E+C", "A+A8+E+C", "A+A8+C", "A+C", "A8+E+C", "A+E", "A+A8+E+C+MVP", "A+E+C+MVP", "A8+E+C", "A+A8+E",
       "A8+C", "C"]
key = [k for k in key if k in set(S["set"])] + [s for s in S["set"].head(5) if s not in key]
boots = {}
for arb, sub in subs.items():
    arr = {k: aligned(ctx, sub, blend(preds, k.split("+"))) for k in key}
    boots[arb] = bootstrap_subset_auc(sub, arr, n_boot=2000 if arb != "gold" else 2000, seed=2026)
L += ["", "## Equal-weight member sets (all 31 scored; top 10 by weak + key sets)", "",
      "Paired delta vs the shipped A+E+C, 2,000 study-level bootstrap reps, CI95 and P(>0). Selecting the max "
      "of 31 sets on the same OOF adds a small optimism (~ one half-width).", "",
      "| set | weak | strict | consensus | gold | composite | d weak [CI] P | d consensus [CI] P | d gold [CI] |",
      "|---|---|---|---|---|---|---|---|---|"]
show = list(dict.fromkeys(list(S["set"].head(10)) + key))
Si = S.set_index("set")
for k in show:
    r = Si.loc[k]
    cells = []
    for arb in ["weak", "consensus", "gold"]:
        if k in boots[arb] and k != "A+E+C":
            d = paired_delta(boots[arb][k], boots[arb]["A+E+C"])
            cells.append(f"{d['mean']:+.4f} [{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}]" + (f" {d['p_gt']:.2f}" if arb != "gold" else ""))
        else:
            cells.append("-")
    L.append(f"| {k} | {r.weak:.4f} | {r.strict:.4f} | {r.consensus:.4f} | {r.gold:.4f} | {r.composite:.4f} | "
             + " | ".join(cells) + " |")

# ---- leave-one-out, per label ------------------------------------------------------------------
L += ["", "## Leave-one-out per label (weak AUC of base minus base-without-member; + = member helps)", ""]
for base in (["A", "E", "C"], ["A", "A8", "E", "C"]):
    pb = blend(preds, base)
    _, perb = subset_auc(subW, aligned(ctx, subW, pb))
    L += [f"Base {'+'.join(base)}:", "", "| drop | macro | " + " | ".join(LABEL_KEYS) + " |", "|---|---|" + "---|" * 12]
    for m in base:
        rest = [x for x in base if x != m]
        mac, per = subset_auc(subW, aligned(ctx, subW, blend(preds, rest)))
        L.append(f"| {m} | {np.mean(list(perb.values())) - mac:+.4f} | "
                 + " | ".join(f"{perb[k] - per[k]:+.3f}" for k in LABEL_KEYS) + " |")
    L.append("")

# ---- OOF -> LB ---------------------------------------------------------------------------------
ent = {"MVP": "MVP", "E": "E", "A": "A", "A+E": "A+E", "A+E+C": "A+E+C"}
L += ["## OOF -> public LB (5 scored entries)", "",
      "| entry | weak | strict | consensus | gold | composite | LB | LB - weak |", "|---|---|---|---|---|---|---|---|"]
xs, ys = [], []
for e, sname in ent.items():
    r = Si.loc[sname]
    L.append(f"| {e} | {r.weak:.4f} | {r.strict:.4f} | {r.consensus:.4f} | {r.gold:.4f} | {r.composite:.4f} | "
             f"{LB[e]:.3f} | {LB[e] - r.weak:+.4f} |")
    xs.append(r.weak)
    ys.append(LB[e])
xs, ys = np.array(xs), np.array(ys)
b, a = np.polyfit(xs, ys, 1)
res = ys - (a + b * xs)
b4, a4 = np.polyfit(xs[1:], ys[1:], 1)
L += ["", f"OLS on all 5: LB = {a:.3f} + {b:.3f} x weak, residual SD {res.std(ddof=2):.4f}, Pearson "
      f"{np.corrcoef(xs, ys)[0, 1]:.3f}. Without the MVP (4 v1-trained entries): slope {b4:.3f}, Pearson "
      f"{np.corrcoef(xs[1:], ys[1:])[0, 1]:.3f}.", ""]

# LB noise: subsample 390 weak studies (public ~30 % of ~1,300) and look at the spread of levels and paired deltas.
rng = np.random.default_rng(7)
n = len(subW.ids)
cand = {k: aligned(ctx, subW, blend(preds, k.split("+"))) for k in ["A", "A+E", "A+E+C"]}
lv, d1, d2 = [], [], []
for _ in range(400):
    r = rng.choice(n, 390, replace=False)
    sc = {}
    for k, p in cand.items():
        vals = []
        for j in range(12):
            kk = r[subW.keep[r, j]]
            vals.append(_rank_auc(subW.y[kk, j], p[kk, j]))
        sc[k] = np.nanmean(vals)
    lv.append(sc["A+E+C"])
    d1.append(sc["A+E+C"] - sc["A+E"])
    d2.append(sc["A+E"] - sc["A"])
lv, d1, d2 = map(np.array, (lv, d1, d2))
L += [f"390-study subsamples of the weak rows (400 draws): SD of the A+E+C level {lv.std():.4f}; SD of the paired "
      f"delta (A+E+C)-(A+E) {d1.std():.4f} (mean {d1.mean():+.4f}, P(<0) {np.mean(d1 < 0):.2f}); (A+E)-A "
      f"{d2.std():.4f} (mean {d2.mean():+.4f}, P(<0) {np.mean(d2 < 0):.2f}). Weak rows have fewer mcl/fracture "
      "positives than gold-like test labels, so these SDs are if anything conservative for level, and the "
      "label-definition shift (weak vs radiologist) adds on top."]
(OUT / "s4_ensemble.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
