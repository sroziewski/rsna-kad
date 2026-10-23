"""Shared loaders for the 2026-09-25 analyst audit (CPU only, read-only on runs/ and data/).

Run from the project root on the server:
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 \
    /raid/sroziewski/sandbox/bin/python qa/audit_2026-09-25/analyst/<script>.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ / "src"))
sys.path.insert(0, str(PROJ / "scripts"))

from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402
from rsna_kad.labels.targets import TargetsConfig, gold_frame  # noqa: E402
from rsna_kad.metrics import (_rank_auc, bootstrap_subset_auc, ci95, consensus_subset,  # noqa: E402
                              gold_subset, paired_delta, strict_subset, subset_auc, voter_codes,
                              weak_subset)

OUT = Path(__file__).resolve().parent
RUNS = {"MVP": "mvp_effv2s_256", "E": "eff_effv2s_256", "A": "main_convnext_s_384",
        "A8": "main_convnext_s_384_e8", "C": "main_armc_convnextv2_t_384"}
PCOLS = [f"{k}_pred" for k in LABEL_KEYS]
# The five scored Kaggle entries (public LB) and the OOF candidate that each one shipped.
LB = {"MVP": 0.865, "E": 0.913, "A": 0.931, "A+E": 0.929, "A+E+C": 0.933}


def load_preds() -> tuple[list[str], dict[str, np.ndarray]]:
    ids = None
    out = {}
    for tag, run in RUNS.items():
        df = pd.read_parquet(PROJ / "runs" / run / "oof.parquet")
        df[ID_COL] = df[ID_COL].astype(str)
        df = df.set_index(ID_COL).sort_index()
        if ids is None:
            ids = list(df.index)
        out[tag] = df.loc[ids, PCOLS].to_numpy(np.float64)
    return ids, out


def blend(preds: dict[str, np.ndarray], members) -> np.ndarray:
    return np.mean([preds[m] for m in members], axis=0)


def name_of(members) -> str:
    return "+".join(members)


def load_all():
    """Everything the section scripts need, aligned to one sorted id list."""
    ids, preds = load_preds()
    for combo in (("A", "E"), ("A", "E", "C")):
        preds[name_of(combo)] = blend(preds, combo)
    folds = pd.read_csv(PROJ / "data/folds/folds_v1.csv")
    folds[ID_COL] = folds[ID_COL].astype(str)
    folds = folds.set_index(ID_COL).loc[ids].reset_index()
    labels = pd.read_parquet(PROJ / "data/labels/llm_v1.parquet")
    labels[ID_COL] = labels[ID_COL].astype(str)
    labels = labels.set_index(ID_COL).loc[ids].reset_index()
    gold = gold_frame(Path(DATA_DIR) / "train.csv")
    gold[ID_COL] = gold[ID_COL].astype(str)
    gold_ids = set(gold[ID_COL])
    weak_ids = [s for s in ids if s not in gold_ids]
    parts = []
    for k in range(5):
        t = pd.read_parquet(PROJ / f"data/targets/v1/fold{k}.parquet")
        t[ID_COL] = t[ID_COL].astype(str)
        parts.append(t[t[ID_COL].isin(set(folds.loc[folds["fold"] == k, ID_COL]))])
    targets = pd.concat(parts).set_index(ID_COL).loc[weak_ids].reset_index()
    lab_weak = labels.set_index(ID_COL).loc[weak_ids].reset_index()
    gold = gold.set_index(ID_COL).loc[[s for s in ids if s in gold_ids]].reset_index()
    subs = {"weak": weak_subset(targets), "strict": strict_subset(lab_weak),
            "consensus": consensus_subset(lab_weak), "gold": gold_subset(gold)}
    row_of = {s: i for i, s in enumerate(ids)}
    return dict(ids=ids, preds=preds, folds=folds, labels=labels, gold=gold, targets=targets,
                lab_weak=lab_weak, weak_ids=weak_ids, subs=subs, row_of=row_of)


def aligned(ctx, sub, pred: np.ndarray) -> np.ndarray:
    idx = [ctx["row_of"][s] for s in sub.ids]
    return pred[idx]


def gold_targets_frame(ctx, targets_dir="data/targets/v1", cfg="configs/targets_v1.yaml",
                       labels_path="data/labels/llm_v1.parquet"):
    """Soft calibrated targets of the gold rows, each with its own (never-seen-it) fold calibration."""
    from evaluate_runs import gold_targets
    lab = pd.read_parquet(PROJ / labels_path)
    lab[ID_COL] = lab[ID_COL].astype(str)
    lab = lab[lab[ID_COL].isin(set(ctx["gold"][ID_COL]))].sort_values(ID_COL).reset_index(drop=True)
    folds = pd.read_csv(PROJ / "data/folds/folds_v1.csv")
    folds[ID_COL] = folds[ID_COL].astype(str)
    gt = gold_targets(lab, folds, TargetsConfig.load(PROJ / cfg), PROJ / targets_dir)
    return gt.set_index(ID_COL).loc[ctx["gold"][ID_COL]].reset_index()


def per_label_boot(sub, pred_al: np.ndarray, n_boot=1000, seed=2026, rows=None):
    """(n_boot, n_labels) per-label AUC on study-resampled replicates."""
    n = len(sub.ids)
    if rows is None:
        rows = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    out = np.full((rows.shape[0], len(sub.keys)), np.nan)
    for j in range(len(sub.keys)):
        kj, yj, pj = sub.keep[:, j], sub.y[:, j], pred_al[:, j]
        for b in range(rows.shape[0]):
            sel = rows[b][kj[rows[b]]]
            out[b, j] = _rank_auc(yj[sel], pj[sel])
    return out


def fmt(x, nd=4):
    return "-" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) if isinstance(v, (float, np.floating)) else str(v)
                                       for v in r.values) + " |")
    return "\n".join(lines)


__all__ = ["PROJ", "OUT", "RUNS", "PCOLS", "LB", "LABEL_KEYS", "ID_COL", "load_all", "blend", "name_of",
           "aligned", "gold_targets_frame", "per_label_boot", "fmt", "md_table", "subset_auc",
           "bootstrap_subset_auc", "paired_delta", "ci95", "_rank_auc", "voter_codes", "combinations"]
