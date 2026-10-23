#!/usr/bin/env python3
"""How close are the v3 targets to gold, before any student is trained?

Each gold study gets the target it WOULD receive if it were a training row: v1_1 label target
from its own fold's calibration (which never saw it), and the teacher's OOF prediction (from
the fold model that never trained on it). Paired bootstrap over the 58 studies. Clean: gold is
in no training set and no calibration used here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_runs import gold_targets  # noqa: E402
from rsna_kad.constants import ID_COL, LABEL_KEYS  # noqa: E402
from rsna_kad.labels.pseudo import teacher_mean  # noqa: E402
from rsna_kad.labels.targets import TargetsConfig, gold_frame  # noqa: E402
from rsna_kad.metrics import _rank_auc  # noqa: E402

TEACHERS = ("runs/main_convnext_s_384", "runs/eff_effv2s_256", "runs/main_armc_convnextv2_t_384")

gold = gold_frame(ROOT / "data/train.csv") if (ROOT / "data/train.csv").exists() else None
if gold is None:
    from rsna_kad.constants import DATA_DIR
    gold = gold_frame(Path(DATA_DIR) / "train.csv")
gold[ID_COL] = gold[ID_COL].astype(str)
folds = pd.read_csv(ROOT / "data/folds/folds_v1.csv"); folds[ID_COL] = folds[ID_COL].astype(str)
labels = pd.read_parquet(ROOT / "data/labels/merged_v1_1.parquet"); labels[ID_COL] = labels[ID_COL].astype(str)
cfg = TargetsConfig.load(ROOT / "configs/targets_v1_1.yaml")
lab_g = labels[labels[ID_COL].isin(set(gold[ID_COL]))]
lt = gold_targets(lab_g, folds, cfg, ROOT / "data/targets/v1_1").set_index(ID_COL)
te = teacher_mean([pd.read_parquet(ROOT / t / "oof.parquet") for t in TEACHERS]).set_index(ID_COL)
ids = sorted(set(lt.index) & set(te.index) & set(gold[ID_COL]))
g = gold.set_index(ID_COL).loc[ids, list(LABEL_KEYS)].to_numpy(float) >= 0.5
L = lt.loc[ids, [f"{k}_target" for k in LABEL_KEYS]].to_numpy(float)
T = te.loc[ids, [f"{k}_pred" for k in LABEL_KEYS]].to_numpy(float)
cands = {"label v1_1": L, "teacher": T, "v3 = 0.5L+0.5T": 0.5 * L + 0.5 * T}


def macro(y, p):
    v = [_rank_auc(y[:, j], p[:, j]) for j in range(y.shape[1])]
    return float(np.nanmean(v)), v


rng = np.random.default_rng(2026)
B = 2000
boots = {n: np.empty(B) for n in cands}
for b in range(B):
    ix = rng.integers(0, len(ids), len(ids))
    for n, p in cands.items():
        boots[n][b] = macro(g[ix], p[ix])[0]
out = {"n": len(ids), "macro": {}, "per_label": {}, "delta_v3_minus_label": None}
for n, p in cands.items():
    m, v = macro(g, p)
    out["macro"][n] = m
    out["per_label"][n] = dict(zip(LABEL_KEYS, v))
d = boots["v3 = 0.5L+0.5T"] - boots["label v1_1"]
out["delta_v3_minus_label"] = {"mean": float(np.mean(d)), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                               "p_gt0": float((d > 0).mean())}
print(json.dumps(out, indent=1))
(ROOT / "qa/pseudo_v3/target_fidelity.json").write_text(json.dumps(out, indent=1))
