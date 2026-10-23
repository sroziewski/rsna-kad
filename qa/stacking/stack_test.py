"""Does stacking our 10 finished models beat the equal-weight main_v4 blend?

Stackers are fitted with nested folds: the weights used to predict fold k are fitted only on
the out-of-fold predictions of the other four folds, against the v1 weak targets binarised the
way the metrics do. Every candidate is then judged on the 58 gold studies, which no model and no
stacker ever trains on, with a paired bootstrap over studies.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS, LABELS

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "A": "runs/main_convnext_s_384", "A8": "runs/main_convnext_s_384_e8", "E": "runs/eff_effv2s_256",
    "C": "runs/main_armc_convnextv2_t_384", "D": "runs/main_armd_convnextv2_t_384",
    "Ap": "runs/main_convnext_s_384_e8_p", "Ep": "runs/eff_pseudo_256", "Cp": "runs/main_armc_convnextv2_t_384_p",
    "Ep2": "qa/main_v4/snap/Ep2_frozen_0313", "Cp2": "runs/main_armc_convnextv2_t_384_p2",
}
MAIN_V4 = ["Ap", "Ep", "Cp"]
POS_THR, NEG_THR, MIN_W = 0.6, 0.2, 0.5


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def main() -> None:
    folds = pd.read_csv(ROOT / "data/folds/folds_v1.csv")
    ids = folds[ID_COL].astype(str).tolist()
    fold_of = dict(zip(folds[ID_COL].astype(str), folds["fold"]))
    names = list(RUNS)
    # preds[name][key] -> array aligned to `ids`
    preds = {}
    for n, r in RUNS.items():
        o = pd.read_parquet(ROOT / r / "oof.parquet").assign(**{ID_COL: lambda d: d[ID_COL].astype(str)})
        o = o.set_index(ID_COL).loc[ids]
        preds[n] = {k: o[f"{k}_pred"].to_numpy(float) for k in LABEL_KEYS}

    # weak targets for each study from its own fold's table (calibrated without that fold's gold)
    tgt = {k: np.full(len(ids), np.nan) for k in LABEL_KEYS}
    wgt = {k: np.zeros(len(ids)) for k in LABEL_KEYS}
    pos = {sid: i for i, sid in enumerate(ids)}
    for f in range(5):
        t = pd.read_parquet(ROOT / f"data/targets/v1/fold{f}.parquet")
        t = t[t[ID_COL].astype(str).map(fold_of) == f]
        idx = t[ID_COL].astype(str).map(pos).to_numpy()
        for k in LABEL_KEYS:
            tgt[k][idx] = t[f"{k}_target"].to_numpy()
            wgt[k][idx] = t[f"{k}_weight"].to_numpy()
    fold_arr = np.array([fold_of[s] for s in ids])

    gold = pd.read_csv(f"{DATA_DIR}/train.csv")
    gold = gold[gold[LABELS[0]].notna()].assign(**{ID_COL: lambda d: d[ID_COL].astype(str)}).set_index(ID_COL)
    gidx = np.array([pos[s] for s in gold.index])
    gy = {k: gold[lab].to_numpy(int) for k, lab in zip(LABEL_KEYS, LABELS)}

    def equal(members):
        return {k: np.mean([preds[m][k] for m in members], axis=0) for k in LABEL_KEYS}

    def stack(members, per_label: bool, C: float):
        out = {k: np.zeros(len(ids)) for k in LABEL_KEYS}
        for f in range(5):
            tr, te = fold_arr != f, fold_arr == f
            if per_label:
                for k in LABEL_KEYS:
                    X = np.stack([logit(preds[m][k]) for m in members], 1)
                    ok = tr & (wgt[k] >= MIN_W) & ((tgt[k] >= POS_THR) | (tgt[k] <= NEG_THR))
                    y = (tgt[k][ok] >= POS_THR).astype(int)
                    lr = LogisticRegression(C=C, max_iter=2000).fit(X[ok], y)
                    out[k][te] = lr.decision_function(X[te])
            else:  # one weight per member shared by all labels, plus a per-label intercept
                Xs, ys, lab_ix = [], [], []
                for j, k in enumerate(LABEL_KEYS):
                    ok = tr & (wgt[k] >= MIN_W) & ((tgt[k] >= POS_THR) | (tgt[k] <= NEG_THR))
                    X = np.stack([logit(preds[m][k]) for m in members], 1)[ok]
                    onehot = np.zeros((len(X), len(LABEL_KEYS))); onehot[:, j] = 1
                    Xs.append(np.hstack([X, onehot])); ys.append((tgt[k][ok] >= POS_THR).astype(int))
                lr = LogisticRegression(C=C, max_iter=4000, fit_intercept=False).fit(np.vstack(Xs), np.concatenate(ys))
                w = lr.coef_[0][: len(members)]
                for k in LABEL_KEYS:
                    X = np.stack([logit(preds[m][k]) for m in members], 1)
                    out[k][te] = X[te] @ w
        return out

    def gold_macro(p, rows):
        vals = []
        for k in LABEL_KEYS:
            y, s = gy[k][rows], p[k][gidx][rows]
            if 0 < y.sum() < len(y):
                vals.append(roc_auc_score(y, s))
        return float(np.mean(vals))

    cands = {
        "main_v4 equal (Ap+Ep+Cp)": equal(MAIN_V4),
        "equal all 10": equal(names),
        "stack all 10, global weights": stack(names, per_label=False, C=1.0),
        "stack all 10, per-label LR": stack(names, per_label=True, C=1.0),
        "stack all 10, per-label LR, strong shrink": stack(names, per_label=True, C=0.01),
        "stack main_v4 3, per-label LR": stack(MAIN_V4, per_label=True, C=1.0),
    }
    base = "main_v4 equal (Ap+Ep+Cp)"
    rng = np.random.default_rng(2026)
    n = len(gidx)
    boots = rng.integers(0, n, (1000, n))
    point = {c: gold_macro(p, np.arange(n)) for c, p in cands.items()}
    rows = []
    for c, p in cands.items():
        d = np.array([gold_macro(p, b) - gold_macro(cands[base], b) for b in boots])
        rows.append((c, point[c], point[c] - point[base], np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()))
        print(f"{c:44s} gold {point[c]:.4f}  delta {point[c]-point[base]:+.4f} "
              f"[{np.percentile(d,2.5):+.4f},{np.percentile(d,97.5):+.4f}] P={(d>0).mean():.3f}")
    (ROOT / "qa/stacking/result.json").write_text(json.dumps(
        [dict(zip(["candidate", "gold", "delta", "lo", "hi", "p"], map(lambda v: v if isinstance(v, str) else round(float(v), 4), r))) for r in rows], indent=1))


if __name__ == "__main__":
    main()
