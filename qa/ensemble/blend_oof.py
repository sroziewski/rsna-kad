#!/usr/bin/env python3
"""OOF blend evaluation for the main track (architecture §2.6: equal weights only, no LB tuning).

  python qa/ensemble/blend_oof.py --runs runs/main_convnext_s_384 runs/eff_effv2s_256 \
      --runs-optional runs/mvp_effv2s_256 --targets v1 --out qa/ensemble/blend_tables.md

Every candidate (each single run, the equal-weight mean of sigmoids, the equal-weight rank
average) is scored on identical rows with `metrics.macro_auc` against targets v1, plus the
strict OOF AUC and the gold-58 AUC with a bootstrap CI. A paired bootstrap over studies gives
the probability that a blend beats the best single run, and the per-label Pearson/Spearman
correlation between the two arms is the diversity check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _avg_ranks(s: np.ndarray) -> np.ndarray:
    """Average ranks (1-based, ties averaged), vectorised. `metrics.auc` computes the same
    ranks with a per-element Python loop, which a 10^5-call bootstrap cannot afford."""
    n = s.shape[0]
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(sorted_s[1:], sorted_s[:-1], out=new[1:])
    group = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], n) - 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = (starts[group] + ends[group]) / 2.0 + 1.0
    return ranks


def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney AUC, identical to `metrics.auc` but vectorised (asserted in main)."""
    n_pos = int(y_true.sum())
    n_neg = y_true.shape[0] - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _avg_ranks(y_score)
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def rank_normalise(a: np.ndarray) -> np.ndarray:
    """Per-column average ranks mapped to (0,1); AUC is rank-invariant, so this only changes
    how the members are combined, never how a single member scores."""
    out = np.empty_like(a, dtype=np.float64)
    for j in range(a.shape[1]):
        out[:, j] = _avg_ranks(a[:, j]) / (a.shape[0] + 1.0)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs that must be in the blend")
    ap.add_argument("--runs-optional", nargs="*", default=[], help="run dirs tried as extra members")
    ap.add_argument("--targets", default="v1")
    ap.add_argument("--targets-dir", default="data/targets")
    ap.add_argument("--folds-csv", default="data/folds/folds_v1.csv")
    ap.add_argument("--labels", default=None, help="label table for the strict AUC")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="qa/ensemble/blend_tables.md")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

    from rsna_kad.constants import ID_COL, LABEL_KEYS  # noqa: E402
    from rsna_kad.cv import load_folds  # noqa: E402
    from rsna_kad.data import load_targets  # noqa: E402
    from rsna_kad.metrics import auc, binarise, macro_auc  # noqa: E402
    from rsna_kad.train import (PRED_COLS, gold_frame_or_none, labels_path, strict_auc)  # noqa: E402
    from rsna_kad.config import load_config  # noqa: E402

    all_dirs = [Path(r) for r in list(args.runs) + list(args.runs_optional)]
    required = [Path(r) for r in args.runs]
    oofs: dict[str, pd.DataFrame] = {}
    cfgs = {}
    for rd in all_dirs:
        cfg = load_config(rd / "config.yaml")
        df = pd.read_parquet(rd / "oof.parquet")
        df[ID_COL] = df[ID_COL].astype(str)
        oofs[cfg.run_id] = df.sort_values(ID_COL).reset_index(drop=True)
        cfgs[cfg.run_id] = cfg
    names = list(oofs)
    req_names = [load_config(rd / "config.yaml").run_id for rd in required]

    common = sorted(set.intersection(*(set(df[ID_COL]) for df in oofs.values())))
    folds_df = load_folds(args.folds_csv)
    folds_df[ID_COL] = folds_df[ID_COL].astype(str)
    gold_ids = set(folds_df.loc[folds_df["is_gold"], ID_COL])

    # Each fold's rows are scored against that fold's target table (fold-local calibration, §2.5).
    parts = []
    for k in sorted(folds_df["fold"].unique()):
        t = load_targets(Path(args.targets_dir) / args.targets / f"fold{k}.parquet")
        t[ID_COL] = t[ID_COL].astype(str)
        parts.append(t[t[ID_COL].isin(set(folds_df.loc[folds_df["fold"] == k, ID_COL]))])
    targets = pd.concat(parts, ignore_index=True)
    targets = targets[targets[ID_COL].isin(common)].sort_values(ID_COL).reset_index(drop=True)
    weak_ids = list(targets[ID_COL])

    P = {n: oofs[n][oofs[n][ID_COL].isin(common)].set_index(ID_COL).loc[common, PRED_COLS].to_numpy(np.float64)
         for n in names}
    idx_weak = np.array([common.index(s) for s in weak_ids])
    y_weak = targets[[f"{k}_target" for k in LABEL_KEYS]].to_numpy(np.float64)
    w_weak = targets[[f"{k}_weight" for k in LABEL_KEYS]].to_numpy(np.float64)

    gold = gold_frame_or_none()
    gold = gold[gold[ID_COL].isin(common)].sort_values(ID_COL).reset_index(drop=True)
    gold_rows = list(gold[ID_COL])
    idx_gold = np.array([common.index(s) for s in gold_rows])
    y_gold = gold[[k for k in LABEL_KEYS]].to_numpy(np.float64)

    lp = Path(args.labels) if args.labels else labels_path(cfgs[req_names[0]])
    labels_tab = pd.read_parquet(lp) if lp and Path(lp).exists() else None

    R = {n: rank_normalise(P[n]) for n in names}

    cands: dict[str, np.ndarray] = {n: P[n] for n in names}
    combos = [tuple(req_names)]
    for extra in args.runs_optional:
        rid = load_config(Path(extra) / "config.yaml").run_id
        combos.append(tuple(req_names) + (rid,))
    for combo in combos:
        tag = " + ".join(combo)
        cands[f"mean({tag})"] = np.mean([P[n] for n in combo], axis=0)
        cands[f"rank({tag})"] = np.mean([R[n] for n in combo], axis=0)

    # --- pre-computed per-label keep masks so the bootstrap only resamples row indices
    keep, ybin = {}, {}
    for j, k in enumerate(LABEL_KEYS):
        keep[k], ybin[k] = binarise(y_weak[:, j], w_weak[:, j])

    rng = np.random.default_rng(args.seed)
    chk = rng.random(400)
    chk[:40] = chk[:40].round(2)  # ties, so the tie-averaging path is exercised too
    chk_y = rng.random(400) > 0.6
    assert abs(fast_auc(chk_y, chk) - auc(chk_y, chk)) < 1e-12, "fast_auc disagrees with metrics.auc"

    boot_rows = rng.integers(0, len(weak_ids), (args.n_boot, len(weak_ids)))
    boot_gold = rng.integers(0, len(gold_rows), (args.n_boot, len(gold_rows)))

    def weak_macro_boot(pred: np.ndarray) -> np.ndarray:
        """Macro AUC of every bootstrap replicate (rows resampled once, shared across candidates)."""
        out = np.full((args.n_boot, len(LABEL_KEYS)), np.nan)
        for j, k in enumerate(LABEL_KEYS):
            # Studies are resampled, then this label's keep mask is applied, so a replicate's
            # kept count varies exactly as the study-level bootstrap says it should.
            kj, yj, pj = keep[k], ybin[k], pred[idx_weak, j]
            for b in range(args.n_boot):
                sel = boot_rows[b][kj[boot_rows[b]]]
                out[b, j] = fast_auc(yj[sel], pj[sel])
        return np.nanmean(out, axis=1)

    def gold_macro_boot(pred: np.ndarray) -> np.ndarray:
        out = np.full((args.n_boot, len(LABEL_KEYS)), np.nan)
        for j in range(len(LABEL_KEYS)):
            yj, pj = y_gold[:, j] >= 0.5, pred[idx_gold, j]
            for b in range(args.n_boot):
                sel = boot_gold[b]
                out[b, j] = fast_auc(yj[sel], pj[sel])
        return np.nanmean(out, axis=1)

    rows = {}
    for name, pred in cands.items():
        weak, per_label = macro_auc(y_weak, pred[idx_weak], w_weak)
        g_macro, g_per = macro_auc(y_gold, pred[idx_gold], None, pos_thr=0.5, neg_thr=0.5, neg_min_weight=0.0)
        gb = gold_macro_boot(pred)
        pr = pd.DataFrame({ID_COL: common, **{c: pred[:, j] for j, c in enumerate(PRED_COLS)}})
        st, st_per = (strict_auc(pr[pr[ID_COL].isin(weak_ids)], labels_tab) if labels_tab is not None
                      else (float("nan"), {}))
        rows[name] = {"weak": weak, "per_label": per_label, "strict": st, "per_label_strict": st_per,
                      "gold": g_macro, "per_label_gold": g_per,
                      "gold_ci": [float(np.nanpercentile(gb, 2.5)), float(np.nanpercentile(gb, 97.5))],
                      "weak_boot": weak_macro_boot(pred)}

    best_single = max(names, key=lambda n: rows[n]["weak"])
    best_boot = rows[best_single]["weak_boot"]
    boots = {n: rows[n]["weak_boot"] for n in rows}
    for name, r in rows.items():
        d = r["weak_boot"] - best_boot
        r["delta_vs_best"] = r["weak"] - rows[best_single]["weak"]
        r["p_beats_best"] = float(np.mean(d > 0))
        r["delta_ci"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        r.pop("weak_boot")

    # --- diversity
    a, b = req_names[0], req_names[1]
    corr = {}
    for j, k in enumerate(LABEL_KEYS):
        x, y = P[a][:, j], P[b][:, j]
        rx, ry = R[a][:, j], R[b][:, j]
        corr[k] = {"pearson": float(np.corrcoef(x, y)[0, 1]), "spearman": float(np.corrcoef(rx, ry)[0, 1])}

    def f(x, nd=4) -> str:
        return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"

    order = names + [n for n in cands if n not in names]
    L = [f"# Ensemble OOF analysis (targets {args.targets})", "",
         f"{len(common)} common studies ({len(weak_ids)} weak, {len(gold_rows)} gold); "
         f"OOF-weak per §2.5 (binarised targets, macro over 12 labels), gold-58 vs `train.csv`. "
         f"Bootstrap: {args.n_boot} replicates, seed {args.seed}, rows resampled once and reused for every "
         f"candidate, so the deltas are paired.", "",
         "| candidate | OOF-weak | delta vs best single | 95% CI of delta | P(blend > best) | strict | gold-58 | gold CI95 |",
         "|---|---|---|---|---|---|---|---|"]
    for n in order:
        r = rows[n]
        L.append(f"| {n} | {f(r['weak'])} | {r['delta_vs_best']:+.4f} | "
                 f"[{r['delta_ci'][0]:+.4f}, {r['delta_ci'][1]:+.4f}] | {r['p_beats_best']:.3f} | "
                 f"{f(r['strict'])} | {f(r['gold'])} | [{f(r['gold_ci'][0], 3)}, {f(r['gold_ci'][1], 3)}] |")
    short = {n: (n if n in names else n.split("(")[0] + "(" +
                 "+".join(p.strip().split("_")[0] for p in n.split("(")[1].rstrip(")").split("+")) + ")")
             for n in order}
    L += ["", f"Best single run on OOF-weak: **{best_single}**.", "",
          "## Paired bootstrap: P(row's OOF-weak > column's)", "",
          "| | " + " | ".join(short[n] for n in order) + " |", "|---|" + "---|" * len(order)]
    for a_ in order:
        cells = ["-" if a_ == b_ else f"{np.mean(boots[a_] - boots[b_] > 0):.3f}" for b_ in order]
        L.append(f"| {short[a_]} | " + " | ".join(cells) + " |")
    L += ["", "## Per-label OOF-weak AUC", "",
          "| label | " + " | ".join(order) + " |", "|---|" + "---|" * len(order)]
    for k in LABEL_KEYS:
        L.append(f"| {k} | " + " | ".join(f(rows[n]["per_label"].get(k, float("nan"))) for n in order) + " |")
    L += ["", "## Per-label gold-58 AUC", "", "| label | " + " | ".join(order) + " |",
          "|---|" + "---|" * len(order)]
    for k in LABEL_KEYS:
        L.append(f"| {k} | " + " | ".join(f(rows[n]["per_label_gold"].get(k, float("nan"))) for n in order) + " |")
    L += ["", f"## Diversity: OOF correlation {a} vs {b}", "", "| label | Pearson | Spearman |", "|---|---|---|"]
    for k in LABEL_KEYS:
        L.append(f"| {k} | {corr[k]['pearson']:.3f} | {corr[k]['spearman']:.3f} |")
    L.append(f"| **mean** | {np.mean([c['pearson'] for c in corr.values()]):.3f} | "
             f"{np.mean([c['spearman'] for c in corr.values()]):.3f} |")
    text = "\n".join(L) + "\n"
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"rows": rows, "corr": corr, "best_single": best_single,
                                                   "n_common": len(common)}, indent=1, default=float),
                                       encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
