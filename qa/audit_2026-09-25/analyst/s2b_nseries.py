"""Q2 follow-up: is the n_series >= 8 deficit a series-selection effect or a site effect?
Compares the 8+ studies against same-site studies with fewer series (site-matched), and splits 8+
by how many same-type siblings the main slots had. Writes s2b_nseries.md."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import LABEL_KEYS, OUT, PROJ, _rank_auc, aligned, ci95, load_all

ctx = load_all()
subW = ctx["subs"]["weak"]
P = aligned(ctx, subW, ctx["preds"]["A+E+C"])
att = pd.read_csv(OUT / "s2_attributes.csv", index_col=0).loc[subW.ids]
ci = pd.read_csv(PROJ / "analysis/out/cache_index_v1.csv")
ci = ci[ci.split == "train"]
sib = ci[ci.slot.isin(["sag_fs", "cor_fs", "ax_fs", "sag_t1"])].groupby("StudyInstanceUID").slot.apply(
    lambda s: int(s.value_counts().max()))
att["max_siblings"] = sib.reindex(att.index).fillna(1).astype(int).values
big = (att.n_series == "8+").to_numpy()


def mac(mask, labs=None):
    per = {}
    for j, k in enumerate(LABEL_KEYS):
        kk = subW.keep[:, j] & mask
        y = subW.y[kk, j]
        if (labs is None and (y.sum() < 10 or (~y).sum() < 10)) or (labs is not None and k not in labs):
            continue
        per[k] = _rank_auc(y, P[kk, j])
    return per


def boot(mask, labs, nb=500, seed=1):
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(mask)
    out = []
    for _ in range(nb):
        r = rng.choice(idx, idx.size)
        v = [_rank_auc(subW.y[r[subW.keep[r, j]], j], P[r[subW.keep[r, j]], j]) for j, k in enumerate(LABEL_KEYS) if k in labs]
        out.append(np.nanmean(v))
    return ci95(np.array(out))


L = ["# S2b - the n_series 8+ slice", ""]
sites_big = att[big].site_key.value_counts()
L.append(f"8+ studies: {big.sum()}, spread over {len(sites_big)} site_keys; top: " +
         "; ".join(f"{s.split('|')[0]} {s.split('|')[1]} {s.split('|')[-1]} {n}" for s, n in sites_big.head(5).items()))
# site-matched comparison: sites that have both 8+ and <8 studies, weight each site equally via pooled cells
sites = [s for s in sites_big.index if ((att.site_key == s) & ~big).sum() >= 20 and sites_big[s] >= 10]
mb = big & att.site_key.isin(sites).to_numpy()
ms = ~big & att.site_key.isin(sites).to_numpy()
pb = mac(mb)
labs = [k for k in pb if k in mac(ms)]
a_b, a_s = mac(mb, labs), mac(ms, labs)
L += ["", f"Site-matched ({len(sites)} sites with >= 10 8+ and >= 20 other studies): 8+ n={mb.sum()} "
      f"{np.mean(list(a_b.values())):.4f} {tuple(round(x, 3) for x in boot(mb, labs))} vs same sites <8 series "
      f"n={ms.sum()} {np.mean(list(a_s.values())):.4f} {tuple(round(x, 3) for x in boot(ms, labs))} on {len(labs)} labels.",
      "", "Per label (8+ minus <8, same sites): " + ", ".join(f"{k} {a_b[k] - a_s[k]:+.3f}" for k in labs)]
L += ["", "| max same-type siblings in a main slot | n | macro | CI95 | labels |", "|---|---|---|---|---|"]
for v in sorted(att.max_siblings.unique()):
    m = (att.max_siblings == v).to_numpy()
    per = mac(m)
    if len(per) < 3:
        L.append(f"| {v} | {m.sum()} | too few | | |")
        continue
    lo, hi = boot(m, list(per))
    L.append(f"| {v} | {m.sum()} | {np.mean(list(per.values())):.4f} | [{lo:.3f}, {hi:.3f}] | {len(per)} |")
(OUT / "s2b_nseries.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))

# Composition of the "side unknown" slice (is it a site proxy?)
unk = att.side_source == "unknown"
comp = att[unk].manufacturer.value_counts(normalize=True)
extra = ("", "side_source=unknown composition: " + ", ".join(f"{k} {v:.2f}" for k, v in comp.items())
         + f"; GE share in the other studies {(att[~unk].manufacturer == 'GE').mean():.2f}")
with open(OUT / "s2b_nseries.md", "a", encoding="utf-8") as fh:
    fh.write("\n".join(extra) + "\n")
print("\n".join(extra))
