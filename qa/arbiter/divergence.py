#!/usr/bin/env python3
"""Evidence for qa/arbiter/report.md §1: where the weak labels and the image-derived gold
labels disagree, and whether blending gains exactly on the untrustworthy cells.

  python qa/arbiter/divergence.py --runs runs/main_convnext_s_384 runs/eff_effv2s_256 \
      runs/mvp_effv2s_256 --out qa/arbiter/divergence.md

Cell types come from `metrics.voter_codes` (LLM V1 / LLM V2 / rule voter). Nothing here is
used by training; it only measures the labels.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", default="data/labels/llm_v1.parquet")
    ap.add_argument("--targets-dir", default="data/targets/v1")
    ap.add_argument("--folds-csv", default="data/folds/folds_v1.csv")
    ap.add_argument("--train-csv", default="/raid/sroziewski/data/rsna_kad/train.csv")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="qa/arbiter/divergence.md")
    ap.add_argument("--json-out", default="qa/arbiter/divergence.json")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

    from rsna_kad.constants import ID_COL, LABEL_KEYS, UNDER_REPORTED_KEYS  # noqa: E402
    from rsna_kad.metrics import (AGREE_ALL3, AGREE_LLM2, AGREE_SPLIT, Subset, evidence_mask,  # noqa: E402
                                  subset_auc, voter_codes, weak_subset)
    from rsna_kad.labels.targets import gold_frame, state_score  # noqa: E402

    labels = pd.read_parquet(args.labels)
    labels[ID_COL] = labels[ID_COL].astype(str)
    labels = labels.sort_values(ID_COL).reset_index(drop=True)
    folds = pd.read_csv(args.folds_csv)
    folds[ID_COL] = folds[ID_COL].astype(str)
    gold = gold_frame(args.train_csv)
    gold[ID_COL] = gold[ID_COL].astype(str)

    preds: dict[str, pd.DataFrame] = {}
    for rd in args.runs:
        df = pd.read_parquet(Path(rd) / "oof.parquet")
        df[ID_COL] = df[ID_COL].astype(str)
        preds[Path(rd).name] = df.sort_values(ID_COL).reset_index(drop=True)
    names = list(preds)

    # ---- the blend that was actually submitted: equal-weight mean of arm A and the efficiency arm
    base = preds[names[0]][[ID_COL]].copy()
    pcols = [f"{k}_pred" for k in LABEL_KEYS]
    blend_members = [n for n in names if n in ("main_convnext_s_384", "eff_effv2s_256")] or names[:2]
    arr = np.mean([preds[n].set_index(ID_COL).loc[base[ID_COL], pcols].to_numpy(np.float64)
                   for n in blend_members], axis=0)
    blend = pd.concat([base, pd.DataFrame(arr, columns=pcols)], axis=1)
    preds["blend(" + "+".join(m.split("_")[0] for m in blend_members) + ")"] = blend
    names = list(preds)

    codes = voter_codes(labels, LABEL_KEYS)
    ev_ok = evidence_mask(labels, LABEL_KEYS)
    states = np.column_stack([labels[f"{k}_state"].astype(str).to_numpy() for k in LABEL_KEYS])
    sev = np.column_stack([labels[f"{k}_severity"].astype(str).to_numpy() for k in LABEL_KEYS])
    weak_claim = np.column_stack([
        np.array([state_score(a, b, k) for a, b in zip(states[:, j], sev[:, j])]) >= 0.5
        for j, k in enumerate(LABEL_KEYS)])
    under = np.array([k in UNDER_REPORTED_KEYS for k in LABEL_KEYS])

    L: list[str] = ["# Label divergence: weak labels vs gold-58, by cell type", ""]
    out: dict = {}

    # ------------------------------------------------------------------ 1. cell census
    n_cells = codes.size
    census = {"all3": int((codes == AGREE_ALL3).sum()), "llm2": int((codes == AGREE_LLM2).sum()),
              "split": int((codes == AGREE_SPLIT).sum()), "n_cells": int(n_cells),
              "absent_under": int(((states == "absent") & under[None, :]).sum()),
              "evidence_ok": int(ev_ok.sum())}
    out["census"] = census
    L += [f"{len(labels)} studies x 12 labels = {n_cells:,} cells.", "",
          "| cell type | cells | share |", "|---|---|---|",
          f"| all three voters agree (`all3`) | {census['all3']:,} | {census['all3'] / n_cells:.3f} |",
          f"| LLM V1==V2, rule voter differs (`llm2`) | {census['llm2']:,} | {census['llm2'] / n_cells:.3f} |",
          f"| LLM variants disagree (`split`) | {census['split']:,} | {census['split'] / n_cells:.3f} |",
          f"| primary state absent on an under-reported key | {census['absent_under']:,} | {census['absent_under'] / n_cells:.3f} |",
          f"| evidence_ok and no mention_conflict | {census['evidence_ok']:,} | {census['evidence_ok'] / n_cells:.3f} |", ""]

    L += ["Per label, share of cells with three-voter agreement and the state mix of those cells:", "",
          "| label | all3 | llm2 | split | all3 affirmed | all3 negated | all3 absent |", "|---|---|---|---|---|---|---|"]
    per_label_census = {}
    for j, k in enumerate(LABEL_KEYS):
        c = codes[:, j]
        a3 = c == AGREE_ALL3
        row = {"all3": float(a3.mean()), "llm2": float((c == AGREE_LLM2).mean()),
               "split": float((c == AGREE_SPLIT).mean()),
               "all3_affirmed": int((a3 & (states[:, j] == "affirmed")).sum()),
               "all3_negated": int((a3 & (states[:, j] == "negated")).sum()),
               "all3_absent": int((a3 & (states[:, j] == "absent")).sum())}
        per_label_census[k] = row
        L.append(f"| {k} | {row['all3']:.3f} | {row['llm2']:.3f} | {row['split']:.3f} | "
                 f"{row['all3_affirmed']:,} | {row['all3_negated']:,} | {row['all3_absent']:,} |")
    out["per_label_census"] = per_label_census
    L.append("")

    # ------------------------------------------------------------------ 2. label quality vs gold
    gi = gold.merge(labels[[ID_COL]], on=ID_COL).reset_index(drop=True)
    pos = labels[ID_COL].searchsorted(gi[ID_COL].to_numpy())
    g_y = gi[list(LABEL_KEYS)].to_numpy(np.float64) >= 0.5
    g_codes, g_claim, g_states, g_ev = codes[pos], weak_claim[pos], states[pos], ev_ok[pos]
    strata = {
        "all3": g_codes == AGREE_ALL3,
        "llm2": g_codes == AGREE_LLM2,
        "split": g_codes == AGREE_SPLIT,
        "absent_under": (g_states == "absent") & under[None, :],
        "evidence_ok": g_ev,
        "all": np.ones_like(g_codes, bool),
    }
    L += ["## 2. How often the weak label is right, per cell type (gold-58, 696 cells)", "",
          "`weak claim` = state->score mapping of the primary voter binarised at 0.5 "
          "(affirmed/hedged positive, negated/absent negative).", "",
          "| cell type | cells | weak claim == gold | claim pos rate | gold pos rate |", "|---|---|---|---|---|"]
    quality = {}
    for tag, m in strata.items():
        n = int(m.sum())
        if n == 0:
            continue
        acc = float((g_claim[m] == g_y[m]).mean())
        quality[tag] = {"n": n, "acc": acc, "claim_pos": float(g_claim[m].mean()),
                        "gold_pos": float(g_y[m].mean())}
        L.append(f"| {tag} | {n:,} | {acc:.3f} | {quality[tag]['claim_pos']:.3f} | {quality[tag]['gold_pos']:.3f} |")
    out["gold_cell_quality"] = quality
    L.append("")

    # ------------------------------------------------------------------ 3. error reproduction
    L += ["## 3. Do the models reproduce the extractor's errors?", "",
          "On the gold cells where the weak claim disagrees with gold, AUC of the model's "
          "prediction **against the weak claim**. 0.5 = the model ignores the extractor's mistake; "
          "above 0.5 = it reproduces it (and OOF-weak pays it for that).", "",
          "| run | gold AUC (all 696 cells) | AUC vs weak claim on the "
          f"{int((g_claim != g_y).sum())} disagreeing cells [CI95] | AUC vs gold on the same cells |",
          "|---|---|---|---|"]
    from rsna_kad.metrics import bootstrap_subset_auc, ci95, paired_delta  # noqa: E402
    repro = {}
    wrong = g_claim != g_y
    claim_sub = Subset("claim_on_wrong", gi[ID_COL].to_numpy(), wrong, g_claim)
    gold_sub = Subset("gold_on_wrong", gi[ID_COL].to_numpy(), wrong, g_y)
    aligned = {n: preds[n].set_index(ID_COL).loc[gi[ID_COL], pcols].to_numpy(np.float64) for n in names}
    boot_claim = bootstrap_subset_auc(claim_sub, aligned, n_boot=args.n_boot, seed=2026)
    for n in names:
        p = aligned[n]
        g_all, _ = subset_auc(Subset("g", gi[ID_COL].to_numpy(), np.ones_like(g_y, bool), g_y), p)
        a_claim, _ = subset_auc(claim_sub, p)
        a_gold, _ = subset_auc(gold_sub, p)
        lo, hi = ci95(boot_claim[n])
        repro[n] = {"gold_all": g_all, "auc_vs_claim": a_claim, "auc_vs_gold": a_gold,
                    "claim_ci": [lo, hi]}
        L.append(f"| {n} | {g_all:.4f} | {a_claim:.4f} [{lo:.3f}, {hi:.3f}] | {a_gold:.4f} |")
    out["error_reproduction"] = repro
    L += ["", f"Cells: {int(wrong.sum())} of {wrong.size} gold cells have weak claim != gold. "
          f"CI = {args.n_boot} bootstrap replicates over the 58 gold studies.", ""]

    # ------------------------------------------------------------------ 4. blend gain by cell type
    parts = []
    for k in sorted(folds["fold"].unique()):
        t = pd.read_parquet(Path(args.targets_dir) / f"fold{k}.parquet")
        t[ID_COL] = t[ID_COL].astype(str)
        parts.append(t[t[ID_COL].isin(set(folds.loc[folds["fold"] == k, ID_COL]))])
    targets = pd.concat(parts, ignore_index=True).sort_values(ID_COL).reset_index(drop=True)
    weak = weak_subset(targets)
    wpos = labels[ID_COL].searchsorted(weak.ids)
    w_codes, w_ev, w_states = codes[wpos], ev_ok[wpos], states[wpos]

    L += ["## 4. Where does the blend's OOF-weak gain come from?", "",
          "OOF-weak AUC restricted to each cell type (macro over the labels that keep both "
          "classes there). `delta` = blend - arm A.", "",
          "| cell type | kept cells | " + " | ".join(names) + " | delta blend-armA [CI95] |",
          "|---|---|" + "---|" * (len(names) + 1)]
    blend_name = [n for n in names if n.startswith("blend")][0]
    arma = "main_convnext_s_384" if "main_convnext_s_384" in names else names[0]
    strata_w = {
        "all": np.ones_like(w_codes, bool),
        "all3": w_codes == AGREE_ALL3,
        "llm2": w_codes == AGREE_LLM2,
        "split": w_codes == AGREE_SPLIT,
        "absent_under": (w_states == "absent") & under[None, :],
        "evidence_ok": w_ev,
        "evidence_bad": ~w_ev,
    }
    gain = {}
    for tag, m in strata_w.items():
        sub = weak.restrict(m, tag)
        if sub.coverage == 0:
            continue
        al = {n: sub.align(preds[n]) for n in names}
        vals = {n: subset_auc(sub, al[n])[0] for n in names}
        bt = bootstrap_subset_auc(sub, {blend_name: al[blend_name], arma: al[arma]},
                                  n_boot=args.n_boot, seed=2026)
        d = paired_delta(bt[blend_name], bt[arma])
        gain[tag] = {"n": sub.coverage, **vals, "delta": vals[blend_name] - vals[arma], "delta_ci": d}
        L.append(f"| {tag} | {sub.coverage:,} | " + " | ".join(f"{vals[n]:.4f}" for n in names) +
                 f" | {vals[blend_name] - vals[arma]:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}] |")
    out["blend_gain_by_cell_type"] = gain
    L.append("")

    text = "\n".join(L) + "\n"
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    Path(args.json_out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
