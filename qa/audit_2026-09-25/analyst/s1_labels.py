"""Q1 - where is AUC lost: per-label AUC for every arm and blend on weak / strict / consensus / gold,
per-label CIs for the shipped blend A+E+C, the label-quality ceiling (soft targets vs gold) and a
headroom ranking. Writes s1_labels.md + s1_labels.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (LABEL_KEYS, OUT, PROJ, _rank_auc, aligned, ci95, fmt, gold_targets_frame, load_all,
                    per_label_boot, subset_auc)

ctx = load_all()
subs, preds = ctx["subs"], ctx["preds"]
cands = ["MVP", "E", "A", "A8", "C", "A+E", "A+E+C"]
V3 = "A+E+C"
rows = []
for arb, sub in subs.items():
    for c in cands:
        m, per = subset_auc(sub, aligned(ctx, sub, preds[c]))
        rows.append({"arbiter": arb, "cand": c, "macro": m, **per})
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "s1_labels.csv", index=False)

# Per-label bootstrap CIs of the shipped blend, and of (blend - best single arm) paired.
ci = {}
dl = {}
for arb, sub in subs.items():
    nb = 1000
    rows_b = np.random.default_rng(2026).integers(0, len(sub.ids), (nb, len(sub.ids)))
    bv = per_label_boot(sub, aligned(ctx, sub, preds[V3]), rows=rows_b)
    ci[arb] = [ci95(bv[:, j]) for j in range(12)]
    singles = {c: per_label_boot(sub, aligned(ctx, sub, preds[c]), rows=rows_b) for c in ["E", "A", "A8", "C"]}
    t = tab[tab.arbiter == arb].set_index("cand")
    dl[arb] = {}
    for j, k in enumerate(LABEL_KEYS):
        best = max(["E", "A", "A8", "C"], key=lambda c: t.loc[c, k])
        d = bv[:, j] - singles[best][:, j]
        d = d[np.isfinite(d)]
        dl[arb][k] = (best, t.loc[V3, k] - t.loc[best, k], *ci95(d), float((d > 0).mean()))

# Label-quality ceiling: AUC of the calibrated soft weak target (fold-honest) against gold, per label.
gt = gold_targets_frame(ctx)
g = ctx["gold"]
ceil_v1 = {k: _rank_auc(g[k].to_numpy() >= 0.5, gt[f"{k}_target"].to_numpy(np.float64)) for k in LABEL_KEYS}
ceil_g3 = {}
try:
    gt3 = gold_targets_frame(ctx, "data/targets/v1_1_gemma3", "configs/targets_v1_1.yaml",
                             "data/labels/merged_v1_1_gemma3.parquet")
    ceil_g3 = {k: _rank_auc(g[k].to_numpy() >= 0.5, gt3[f"{k}_target"].to_numpy(np.float64)) for k in LABEL_KEYS}
except Exception as e:  # noqa: BLE001
    print("gemma3 ceiling unavailable:", e)

L = ["# S1 per-label AUC (OOF, 4,349 weak rows / 58 gold)", ""]
for arb in subs:
    t = tab[tab.arbiter == arb].set_index("cand")
    L += [f"## {arb}", "", "| label | " + " | ".join(cands) + f" | {V3} CI95 |", "|---|" + "---|" * (len(cands) + 1)]
    for k in ["macro"] + list(LABEL_KEYS):
        cells = [fmt(t.loc[c, k]) for c in cands]
        cic = "" if k == "macro" else "[%s, %s]" % tuple(fmt(v, 3) for v in ci[arb][LABEL_KEYS.index(k)])
        L.append(f"| {k} | " + " | ".join(cells) + f" | {cic} |")
    L.append("")

L += ["## Headroom per label", "",
      "`loss` = (1 - AUC)/12 = the label's share of the macro shortfall of A+E+C. `blend-best` = A+E+C minus the "
      "best single arm on that label (paired bootstrap CI, 1000 reps). `ceiling` = AUC of the fold-honest "
      "calibrated soft target vs gold (label quality); `g3` = same on targets v1_1_gemma3.", "",
      "| label | weak A+E+C | loss weak | gold A+E+C | loss gold | best arm (weak) | blend-best weak [CI] P | "
      "blend-best gold | ceiling v1 | ceiling g3 | gold - ceiling |", "|---|---|---|---|---|---|---|---|---|---|---|"]
tw = tab[tab.arbiter == "weak"].set_index("cand")
tg = tab[tab.arbiter == "gold"].set_index("cand")
hr = []
for k in LABEL_KEYS:
    best, d, lo, hi, p = dl["weak"][k]
    bg, dg, *_ = dl["gold"][k]
    hr.append((k, tw.loc[V3, k], (1 - tw.loc[V3, k]) / 12, tg.loc[V3, k], (1 - tg.loc[V3, k]) / 12))
    L.append(f"| {k} | {fmt(tw.loc[V3, k])} | {fmt((1 - tw.loc[V3, k]) / 12)} | {fmt(tg.loc[V3, k])} | "
             f"{fmt((1 - tg.loc[V3, k]) / 12)} | {best} | {d:+.4f} [{lo:+.4f},{hi:+.4f}] {p:.2f} | "
             f"{dg:+.4f} ({bg}) | {fmt(ceil_v1[k])} | {fmt(ceil_g3.get(k, np.nan))} | "
             f"{tg.loc[V3, k] - ceil_v1[k]:+.4f} |")
L += ["", f"macro ceiling v1 {np.mean(list(ceil_v1.values())):.4f}; g3 "
      f"{np.mean(list(ceil_g3.values())) if ceil_g3 else float('nan'):.4f}; A+E+C gold {tg.loc[V3, 'macro']:.4f}.", ""]
L.append("Ranking by weak loss: " + ", ".join(f"{k} {l:.4f}" for k, _, l, _, _ in sorted(hr, key=lambda r: -r[2])))
L.append("Ranking by gold loss: " + ", ".join(f"{k} {l:.4f}" for k, _, _, _, l in sorted(hr, key=lambda r: -r[4])))
(OUT / "s1_labels.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
