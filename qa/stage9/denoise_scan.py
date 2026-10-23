"""Stage 9 CPU analysis: confident-learning contradiction counts between targets v1 and the
image models' OOF, plus the pseudo-label headroom on the under-reported keys.

Read-only: touches nothing under data/targets/v1 or data/labels.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from rsna_kad.constants import ID_COL, LABEL_KEYS, UNDER_REPORTED_KEYS  # noqa: E402

ROOT = Path(".")
FOLDS = pd.read_csv("data/folds/folds_v1.csv")
FOLDS[ID_COL] = FOLDS[ID_COL].astype(str)
UNDER = tuple(k for k in LABEL_KEYS if k in UNDER_REPORTED_KEYS)

HI, LO = 0.9, 0.1


def targets_eval_table():
    """For study s in fold j, the row from data/targets/v1/fold{j}.parquet (the convention
    assemble_oof/compare_runs use). Gold rows are absent from every v1 table."""
    parts = []
    for k in sorted(FOLDS["fold"].unique()):
        t = pd.read_parquet(f"data/targets/v1/fold{k}.parquet")
        t[ID_COL] = t[ID_COL].astype(str)
        ids = set(FOLDS.loc[FOLDS["fold"] == k, ID_COL])
        parts.append(t[t[ID_COL].isin(ids)])
    return pd.concat(parts, ignore_index=True)


def load_oof(run):
    o = pd.read_parquet(f"runs/{run}/oof.parquet")
    o[ID_COL] = o[ID_COL].astype(str)
    return o[[ID_COL] + [f"{k}_pred" for k in LABEL_KEYS]]


def main():
    tg = targets_eval_table()
    cn = load_oof("main_convnext_s_384").rename(columns={f"{k}_pred": f"{k}_cn" for k in LABEL_KEYS})
    ef = load_oof("eff_effv2s_256").rename(columns={f"{k}_pred": f"{k}_ef" for k in LABEL_KEYS})
    lab = pd.read_parquet("data/labels/llm_v1.parquet")
    lab[ID_COL] = lab[ID_COL].astype(str)
    df = tg.merge(cn, on=ID_COL).merge(ef, on=ID_COL).merge(FOLDS[[ID_COL, "fold", "lang"]], on=ID_COL)
    print(f"rows: targets {len(tg)}  joined {len(df)}  (gold excluded from v1 targets)")

    rows = []
    hand = []
    for k in LABEL_KEYS:
        t = df[f"{k}_target"].to_numpy(dtype=float)
        w = df[f"{k}_weight"].to_numpy(dtype=float)
        pc, pe = df[f"{k}_cn"].to_numpy(dtype=float), df[f"{k}_ef"].to_numpy(dtype=float)
        pm = (pc + pe) / 2.0
        st = lab.set_index(ID_COL).loc[df[ID_COL], f"{k}_state"].astype(str).to_numpy()
        pos_contra = (t >= HI) & (pm <= LO)          # label says yes, both models say no
        neg_contra = (t <= LO) & (pm >= 1 - LO)      # label says no, both models say yes
        both_c = ((t >= HI) & (pc <= LO)) | ((t <= LO) & (pc >= 1 - LO))
        both_e = ((t >= HI) & (pe <= LO)) | ((t <= LO) & (pe >= 1 - LO))
        agree = both_c & both_e
        rows.append({
            "key": k, "under": k in UNDER, "n": len(df),
            "n_hi": int((t >= HI).sum()), "n_lo": int((t <= LO).sum()),
            "pos_contra": int(pos_contra.sum()), "neg_contra": int(neg_contra.sum()),
            "contra": int((pos_contra | neg_contra).sum()),
            "contra_pct": 100.0 * (pos_contra | neg_contra).mean(),
            "both_models_agree": int(agree.sum()),
            "w_mean_contra": float(w[pos_contra | neg_contra].mean()) if (pos_contra | neg_contra).any() else float("nan"),
        })
        idx = np.where(pos_contra | neg_contra)[0]
        for i in idx:
            hand.append({"key": k, ID_COL: df[ID_COL].iloc[i], "fold": int(df["fold"].iloc[i]),
                         "lang": df["lang"].iloc[i], "state": st[i], "target": round(float(t[i]), 4),
                         "weight": round(float(w[i]), 3), "cn": round(float(pc[i]), 4),
                         "ef": round(float(pe[i]), 4), "kind": "pos_contra" if pos_contra[i] else "neg_contra"})
    r = pd.DataFrame(rows)
    print("\n=== confident-learning contradictions (target>=0.9 & OOF<=0.1, or target<=0.1 & OOF>=0.9; OOF = mean of the two arms)")
    print(r.to_string(index=False))
    print(f"\nTOTAL contradicted cells: {r['contra'].sum()} of {len(df) * 12} "
          f"({100.0 * r['contra'].sum() / (len(df) * 12):.3f} %)")
    print(f"  pos_contra {r['pos_contra'].sum()}  neg_contra {r['neg_contra'].sum()}")
    print(f"  under-reported keys: {r.loc[r['under'], 'contra'].sum()}  "
          f"well-reported keys: {r.loc[~r['under'], 'contra'].sum()}")

    h = pd.DataFrame(hand)
    h.to_csv(".tmp/devj/contradictions.csv", index=False)
    print(f"\nwrote .tmp/devj/contradictions.csv ({len(h)} rows)")
    print("\nby state of the LLM label:")
    print(h.groupby(["kind", "state"]).size().to_string())

    # ---- pseudo-label headroom: what the OOF says on the low-weight 'absent' cells of the
    # under-reported keys (the cells the LLM cannot see).
    print("\n=== under-reported 'absent' cells (weight 0.1) - OOF distribution")
    for k in UNDER:
        st = lab.set_index(ID_COL).loc[df[ID_COL], f"{k}_state"].astype(str).to_numpy()
        w = df[f"{k}_weight"].to_numpy(dtype=float)
        m = (st == "absent") & (w <= 0.15)
        pm = (df[f"{k}_cn"].to_numpy(dtype=float) + df[f"{k}_ef"].to_numpy(dtype=float)) / 2.0
        t = df[f"{k}_target"].to_numpy(dtype=float)
        print(f"  {k:10s} n={m.sum():5d} target={t[m][0] if m.any() else float('nan'):.3f} "
              f"OOF mean {pm[m].mean():.3f} p10 {np.percentile(pm[m], 10):.3f} "
              f"p50 {np.percentile(pm[m], 50):.3f} p90 {np.percentile(pm[m], 90):.3f} "
              f"| frac>0.5 {(pm[m] > 0.5).mean():.3f}")

    # ---- correlation of the two arms on those cells (are they independent evidence?)
    print("\n=== spearman(convnext, eff) OOF per key, all weak rows")
    for k in LABEL_KEYS:
        a, b = df[f"{k}_cn"].rank(), df[f"{k}_ef"].rank()
        print(f"  {k:10s} {np.corrcoef(a, b)[0, 1]:.3f}")
    json.dump({"contra_total": int(r["contra"].sum()), "n_cells": int(len(df) * 12)},
              open(".tmp/devj/denoise_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
