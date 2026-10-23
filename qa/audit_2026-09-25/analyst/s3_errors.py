"""Q3 - error analysis of the shipped blend A+E+C.

(a) Confident errors vs weak labels: a negative scored above the MEDIAN POSITIVE of its label (cFP),
    a positive scored below the MEDIAN NEGATIVE (cFN); rates split by LLM state, voter agreement,
    severity and language, to separate "model wrong" from "label silent/wrong".
(b) Gold-58: per cell, misrank share = fraction of opposite-class gold cells of the same label ranked
    on the wrong side; a cell FAILS for an arm if share > 0.5. Cells failed by every arm are listed.
(c) Medial/lateral: in weak studies where exactly one of (med, lat) is positive, AUC of
    (rank_med - rank_lat) for telling med+lat- from lat+med-, overall and by knee side / side source.
Writes s3_errors.md."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import ID_COL, LABEL_KEYS, OUT, _rank_auc, aligned, ci95, load_all, voter_codes

ctx = load_all()
V3 = "A+E+C"
subW = ctx["subs"]["weak"]
lab = ctx["lab_weak"]
pw = aligned(ctx, subW, ctx["preds"][V3])
codes = voter_codes(lab)
L = ["# S3 error analysis (A+E+C)", ""]

# ---- (a) confident errors vs weak labels -------------------------------------------------------
recs = []
for j, k in enumerate(LABEL_KEYS):
    kj = subW.keep[:, j]
    y = subW.y[:, j]
    p = pw[:, j]
    med_pos = np.median(p[kj & y])
    med_neg = np.median(p[kj & ~y])
    cfp = kj & ~y & (p > med_pos)
    cfn = kj & y & (p < med_neg)
    st = lab[f"{k}_state"].astype(str).to_numpy()
    sev = lab[f"{k}_severity"].astype(str).to_numpy()
    for i in np.flatnonzero(kj):
        recs.append({"label": k, "y": bool(y[i]), "cfp": bool(cfp[i]), "cfn": bool(cfn[i]), "state": st[i],
                     "sev": sev[i], "agree": {3: "all3", 2: "llm2", 1: "split"}[int(codes[i, j])],
                     "lang": lab["lang"].iat[i]})
E = pd.DataFrame(recs)
neg, pos = E[~E.y], E[E.y]
L += ["## (a) Confident errors vs weak labels", "",
      f"Negatives {len(neg):,}, of which cFP {neg.cfp.sum():,} ({neg.cfp.mean():.3%}); positives {len(pos):,}, "
      f"cFN {pos.cfn.sum():,} ({pos.cfn.mean():.3%}).", "",
      "| label | neg | cFP rate | cFP among negated | cFP among absent | pos | cFN rate | cFN mild/hedged | cFN other |",
      "|---|---|---|---|---|---|---|---|---|"]
for k in LABEL_KEYS:
    n_, p_ = neg[neg.label == k], pos[pos.label == k]
    ng, ab = n_[n_.state == "negated"], n_[n_.state == "absent"]
    mild = p_[(p_.sev == "mild") | (p_.state == "hedged")]
    oth = p_[~((p_.sev == "mild") | (p_.state == "hedged"))]
    L.append(f"| {k} | {len(n_)} | {n_.cfp.mean():.3f} | {ng.cfp.mean():.3f} (n={len(ng)}) | "
             f"{ab.cfp.mean() if len(ab) else float('nan'):.3f} (n={len(ab)}) | {len(p_)} | {p_.cfn.mean():.3f} | "
             f"{mild.cfn.mean() if len(mild) else float('nan'):.3f} (n={len(mild)}) | {oth.cfn.mean():.3f} (n={len(oth)}) |")
L += ["", "By voter agreement (all labels pooled):", "", "| agree | neg | cFP rate | pos | cFN rate |", "|---|---|---|---|---|"]
for a in ["all3", "llm2", "split"]:
    n_, p_ = neg[neg.agree == a], pos[pos.agree == a]
    L.append(f"| {a} | {len(n_):,} | {n_.cfp.mean():.3f} | {len(p_):,} | {p_.cfn.mean():.3f} |")
L += ["", "By report language (pooled):", "", "| lang | neg | cFP | pos | cFN |", "|---|---|---|---|---|"]
for lg, g in E.groupby("lang"):
    n_, p_ = g[~g.y], g[g.y]
    L.append(f"| {lg} | {len(n_):,} | {n_.cfp.mean():.3f} | {len(p_):,} | {p_.cfn.mean():.3f} |")

# ---- (b) gold rows -----------------------------------------------------------------------------
gold = ctx["gold"]
subG = ctx["subs"]["gold"]
arms = ["MVP", "E", "A", "A8", "C", V3]
G = {a: aligned(ctx, subG, ctx["preds"][a]) for a in arms}
from common import gold_targets_frame  # noqa: E402
gt = gold_targets_frame(ctx)
labs_all = ctx["labels"].set_index(ID_COL)
fails = []
share_tab = {a: [] for a in arms}
for j, k in enumerate(LABEL_KEYS):
    y = subG.y[:, j]
    for a in arms:
        p = G[a][:, j]
        sh = np.where(y, [(p[~y] >= v).mean() for v in p], [(p[y] <= v).mean() for v in p])
        share_tab[a].append(sh)
    for i in range(len(y)):
        if all(share_tab[a][j][i] > 0.5 for a in ["E", "A", "A8", "C", V3]):
            sid = subG.ids[i]
            fails.append({"study": sid[-10:], "label": k, "gold": int(y[i]), "weak_target": gt[f"{k}_target"].iat[i],
                          "llm_state": f"{labs_all.at[sid, k + '_state']}/{labs_all.at[sid, k + '_severity']}",
                          "lang": labs_all.at[sid, "lang"], "share_V3": share_tab[V3][j][i]})
F = pd.DataFrame(fails)
F["label_agrees"] = np.where(F.gold == 1, F.weak_target >= 0.5, F.weak_target < 0.5)
n_cells = subG.keep.sum()
L += ["", "## (b) Gold-58 cells failed by every arm (E, A, A8, C and the blend)", "",
      f"{len(F)} of {n_cells} gold cells ({len(F) / n_cells:.1%}); {int((F.gold == 1).sum())} missed positives, "
      f"{int((F.gold == 0).sum())} false alarms. The weak target points the same (wrong) way in "
      f"{int((~F.label_agrees).sum())} of them (label-driven), the opposite (right) way in {int(F.label_agrees.sum())} "
      "(image-driven: the label was right and every model still failed).", "",
      "Per label: " + ", ".join(f"{k} {int((F.label == k).sum())}" for k in LABEL_KEYS), "",
      "Failure rate per arm (share > 0.5), all 696 cells: " +
      ", ".join(f"{a} {np.mean(np.concatenate(share_tab[a]) > 0.5):.3f}" for a in arms), "",
      "| study (last 10) | lang | label | gold | weak target | LLM state/sev | V3 misrank share | label agrees w/ gold |",
      "|---|---|---|---|---|---|---|---|"]
for _, r in F.sort_values(["study", "label"]).iterrows():
    L.append(f"| {r.study} | {r.lang} | {r.label} | {r.gold} | {r.weak_target:.2f} | {r.llm_state} | "
             f"{r.share_V3:.2f} | {'yes' if r.label_agrees else 'NO'} |")
st_counts = F.groupby("study").size().sort_values(ascending=False)
L += ["", "Studies with >= 3 all-arm failures: " + ", ".join(f"{s} ({n})" for s, n in st_counts[st_counts >= 3].items())]
F.to_csv(OUT / "s3_gold_failures.csv", index=False)

# ---- (c) medial vs lateral ---------------------------------------------------------------------
att = pd.read_csv(OUT / "s2_attributes.csv", index_col=0) if (OUT / "s2_attributes.csv").exists() else None
rank = np.empty_like(pw)
for j in range(12):
    rank[:, j] = pd.Series(pw[:, j]).rank(pct=True).to_numpy()
L += ["", "## (c) Medial vs lateral", "",
      "Discordant weak studies (exactly one side positive, the other negative, both cells kept). "
      "`AUC` = how well rank(med) - rank(lat) separates med+/lat- (1) from lat+/med- (0); 1.0 = sides never "
      "confused. `swap` = share of discordant studies where the WRONG side ranks higher. Bootstrap 1000.", "",
      "| pair | group | n med+lat- | n lat+med- | AUC [CI95] | swap rate |", "|---|---|---|---|---|---|"]
rng = np.random.default_rng(2026)
for med, lat in (("med_men", "lat_men"), ("med_oa", "lat_oa")):
    jm, jl = LABEL_KEYS.index(med), LABEL_KEYS.index(lat)
    both = subW.keep[:, jm] & subW.keep[:, jl]
    disc = both & (subW.y[:, jm] != subW.y[:, jl])
    ym = subW.y[:, jm]
    diff = rank[:, jm] - rank[:, jl]
    groups = {"all": np.ones(len(ym), bool)}
    if att is not None:
        a = att.loc[subW.ids]
        for col in ("side_est", "side_source"):
            for v in sorted(a[col].astype(str).unique()):
                groups[f"{col}={v}"] = (a[col].astype(str) == v).to_numpy()
    for gname, gm in groups.items():
        m = disc & gm
        if m.sum() < 20 or ym[m].sum() < 5 or (~ym[m]).sum() < 5:
            continue
        idx = np.flatnonzero(m)
        auc = _rank_auc(ym[idx], diff[idx])
        bs = [_rank_auc(ym[r], diff[r]) for r in (rng.choice(idx, idx.size) for _ in range(1000))]
        lo, hi = ci95(np.array(bs))
        swap = np.mean(np.where(ym[idx], diff[idx] < 0, diff[idx] > 0))
        L.append(f"| {med}/{lat} | {gname} | {int(ym[idx].sum())} | {int((~ym[idx]).sum())} | {auc:.3f} [{lo:.3f}, {hi:.3f}] | {swap:.3f} |")
# Control: the same statistic for two unrelated labels with similar prevalence tells what "no side information" looks like.
for a_, b_ in (("med_men", "acl"),):
    ja, jb = LABEL_KEYS.index(a_), LABEL_KEYS.index(b_)
    m = subW.keep[:, ja] & subW.keep[:, jb] & (subW.y[:, ja] != subW.y[:, jb])
    idx = np.flatnonzero(m)
    L.append(f"| control {a_}/{b_} | all | {int(subW.y[idx, ja].sum())} | {int((~subW.y[idx, ja]).sum())} | "
             f"{_rank_auc(subW.y[idx, ja], rank[idx, ja] - rank[idx, jb]):.3f} | "
             f"{np.mean(np.where(subW.y[idx, ja], rank[idx, ja] < rank[idx, jb], rank[idx, ja] > rank[idx, jb])):.3f} |")
# Gold: discordant gold studies.
for med, lat in (("med_men", "lat_men"), ("med_oa", "lat_oa")):
    jm, jl = LABEL_KEYS.index(med), LABEL_KEYS.index(lat)
    y = subG.y
    pg = G[V3]
    rg = np.column_stack([pd.Series(pg[:, j]).rank(pct=True) for j in range(12)])
    m = y[:, jm] != y[:, jl]
    idx = np.flatnonzero(m)
    diff = rg[idx, jm] - rg[idx, jl]
    swap = np.where(y[idx, jm], diff < 0, diff > 0)
    L.append(f"| gold {med}/{lat} | gold-58 | {int(y[idx, jm].sum())} | {int((~y[idx, jm]).sum())} | "
             f"{_rank_auc(y[idx, jm], diff):.3f} | {swap.mean():.3f} ({int(swap.sum())}/{len(idx)}) |")
(OUT / "s3_errors.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
