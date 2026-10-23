"""Q2 - OOF AUC of the shipped blend A+E+C (and arm A) by data slice.

Macro AUC inside a slice is taken over the labels that have >= MIN_CLS positives and negatives in
that slice; `ref` is the whole-population macro over the SAME labels, so `delta` is not an artefact
of which labels a small slice can score. CI = 500 study-level bootstrap replicates inside the slice.
Consensus (frozen voter states) is shown as a label-definition check for language/site slices.
Writes s2_slices.md + s2_slices.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (ID_COL, LABEL_KEYS, OUT, PROJ, _rank_auc, aligned, ci95, fmt, load_all)

MIN_CLS = 10
NB = 500
ctx = load_all()
ids_w = ctx["weak_ids"]
subW, subC = ctx["subs"]["weak"], ctx["subs"]["consensus"]
P = {c: aligned(ctx, subW, ctx["preds"][c]) for c in ["A+E+C", "A"]}
PC = aligned(ctx, subC, ctx["preds"]["A+E+C"])

# ---- per-study slice attributes -------------------------------------------------------------
ss = pd.read_csv(PROJ / "analysis/out/study_summary.csv", dtype={"study_dir": str})
ss = ss.set_index("study_dir")
ci = pd.read_csv(PROJ / "analysis/out/cache_index_v1.csv")
ci = ci[(ci["split"] == "train")]
sel = ci[ci["selected"].astype(bool)]
folds = ctx["folds"].set_index(ID_COL)


def manu_group(m: str) -> str:
    m = str(m).lower()
    for k, v in (("siemens", "Siemens"), ("ge", "GE"), ("philips", "Philips"), ("toshiba", "Toshiba/Canon"),
                 ("canon", "Toshiba/Canon"), ("fuji", "Fujifilm/Hitachi"), ("hitachi", "Fujifilm/Hitachi")):
        if m.startswith(k) or k in m:
            return v
    return "other"


att = pd.DataFrame(index=ids_w)
att["lang"] = folds.loc[ids_w, "lang"].values
att["site_key"] = folds.loc[ids_w, "site_key"].values
att["fold"] = folds.loc[ids_w, "fold"].astype(str).values
att["manufacturer"] = [manu_group(m) for m in ss.loc[ids_w, "manu"]]
fld = ss.loc[ids_w, "field"]
att["field"] = np.where(fld.isna(), "missing", np.where(fld >= 2.5, "3T", np.where(fld >= 1.4, "1.5T", "<1.4T")))
ns = ss.loc[ids_w, "n_series"].astype(int)
att["n_series"] = pd.cut(ns, [0, 4, 5, 6, 7, 99], labels=["3-4", "5", "6", "7", "8+"]).astype(str).values
slots = sel.groupby("StudyInstanceUID")["slot"].apply(set)
att["sag_fs"] = ["present" if "sag_fs" in slots.get(s, set()) else "MISSING->sag_t1" for s in ids_w]
att["cor_fs"] = ["present" if "cor_fs" in slots.get(s, set()) else "MISSING->cor_t1" for s in ids_w]
att["sag_t1"] = ["present" if "sag_t1" in slots.get(s, set()) else "MISSING" for s in ids_w]
main = sel[sel["slot"].isin(["sag_fs", "cor_fs", "ax_fs", "sag_t1"])]
any3d = main.groupby("StudyInstanceUID")["is_3d"].any()
att["any_3d_in_main_slots"] = ["3D" if any3d.get(s, False) else "2D only" for s in ids_w]
small_fov = main.groupby("StudyInstanceUID")["fov_mm"].min()
att["min_fov"] = ["<130mm (padded)" if small_fov.get(s, 999) < 130 else ">=130mm" for s in ids_w]
st = ci.groupby("StudyInstanceUID").agg(side_est=("side_est", "first"), side_conf=("side_conf", "max"))
conf = st["side_conf"].reindex(ids_w).fillna(0).to_numpy()
att["side_source"] = np.select([conf == 0, np.isclose(conf, 0.95), np.isclose(conf, 0.6), np.isclose(conf, 0.8)],
                               ["unknown", "tag", "tag-contradictory", "desc"], "geometry")
att["side_est"] = st["side_est"].reindex(ids_w).fillna("unknown").values
att.to_csv(OUT / "s2_attributes.csv")


def score(sub, pred, mask, labels=None):
    per = {}
    for j, k in enumerate(LABEL_KEYS):
        kj = sub.keep[:, j] & mask
        y = sub.y[kj, j]
        if labels is None and (y.sum() < MIN_CLS or (~y).sum() < MIN_CLS):
            continue
        if labels is not None and k not in labels:
            continue
        per[k] = _rank_auc(y, pred[kj, j])
    return per


def boot(sub, pred, mask, labels, rng):
    idx = np.flatnonzero(mask)
    out = np.empty(NB)
    for b in range(NB):
        r = rng.choice(idx, idx.size)
        vals = []
        for j, k in enumerate(LABEL_KEYS):
            if k not in labels:
                continue
            kk = r[sub.keep[r, j]]
            vals.append(_rank_auc(sub.y[kk, j], pred[kk, j]))
        out[b] = np.nanmean(vals)
    return out


full = {c: score(subW, P[c], np.ones(len(ids_w), bool), list(LABEL_KEYS)) for c in P}
fullC = score(subC, PC, np.ones(len(ids_w), bool), list(LABEL_KEYS))
rng = np.random.default_rng(2026)
rows = []
dims = ["manufacturer", "field", "lang", "any_3d_in_main_slots", "sag_fs", "cor_fs", "sag_t1", "min_fov",
        "side_source", "side_est", "n_series", "fold", "site_key"]
for dim in dims:
    vals = att[dim].value_counts()
    if dim == "site_key":
        vals = vals[vals >= 100]
    for v, n in vals.items():
        m = (att[dim] == v).to_numpy()
        per = score(subW, P["A+E+C"], m)
        if len(per) < 3:
            rows.append({"dim": dim, "value": v, "n": int(n), "labels": len(per)})
            continue
        labs = list(per)
        mac = float(np.mean(list(per.values())))
        ref = float(np.mean([full["A+E+C"][k] for k in labs]))
        b = boot(subW, P["A+E+C"], m, labs, rng)
        lo, hi = ci95(b)
        perA = score(subW, P["A"], m, labs)
        macA = float(np.nanmean(list(perA.values())))
        refA = float(np.mean([full["A"][k] for k in labs]))
        perC = score(subC, PC, m)
        labsC = list(perC)
        cm = float(np.mean(list(perC.values()))) if perC else np.nan
        cref = float(np.mean([fullC[k] for k in labsC])) if perC else np.nan
        worst = min(per, key=lambda k: per[k] - full["A+E+C"][k])
        rows.append({"dim": dim, "value": v, "n": int(n), "labels": len(labs), "weak": mac, "ci_lo": lo, "ci_hi": hi,
                     "ref": ref, "delta": mac - ref, "armA_delta": macA - refA,
                     "cons_delta": cm - cref, "worst_label": f"{worst} {per[worst] - full['A+E+C'][worst]:+.3f}"})
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "s2_slices.csv", index=False)
L = ["# S2 slices - OOF-weak macro AUC of A+E+C per slice (weak rows)", "",
     f"Labels scored in a slice need >= {MIN_CLS} pos and neg there; `ref` = whole-population macro on the same "
     f"labels; CI = {NB} bootstrap reps inside the slice; `cons_delta` = same delta on the consensus arbiter; "
     "`armA_delta` = the delta for arm A alone.", "",
     "| dim | value | n | labels | weak [CI95] | ref | delta | armA delta | cons delta | worst label (vs pop) |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for _, r in tab.iterrows():
    if not np.isfinite(r.get("weak", np.nan)):
        L.append(f"| {r['dim']} | {r['value']} | {r['n']} | {r['labels']} | too few | | | | | |")
        continue
    L.append(f"| {r['dim']} | {r['value']} | {r['n']} | {r['labels']} | {r['weak']:.4f} [{r['ci_lo']:.3f}, "
             f"{r['ci_hi']:.3f}] | {r['ref']:.4f} | {r['delta']:+.4f} | {r['armA_delta']:+.4f} | "
             f"{fmt(r['cons_delta'])} | {r['worst_label']} |")
(OUT / "s2_slices.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
