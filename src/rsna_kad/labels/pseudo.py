"""Stage 9: image-model pseudo-labels for the cells the reports are silent about.

The LLM can only label what a radiologist wrote down. For the seven under-reported keys
(`constants.UNDER_REPORTED_KEYS`) "not mentioned" is not a negative - synovitis is affirmed in
12.5 % of reports but positive in 46.6 % of gold studies - so `targets.make_targets` gives those
`absent` cells the calibrated absent-rate at weight 0.1: enough to keep the loss defined, not
enough to teach anything. This module replaces exactly those cells with 0.5 x prior +
0.5 x image-model prediction (research.md §3, RSNA-2024 lumbar 2nd) at a weight that matches a
well-reported `absent`.

Nothing else is touched: a cell outside `keys`, a cell whose state is not `absent`, a cell whose
weight shows the row was overridden (parse failure / postoperative), and any study without a
prediction all keep their v1 value bit-for-bit.

LEAKAGE. `pred` must come from a model that saw neither the study it labels nor the fold the
next model will be validated on - a fold-k model's OOF does NOT qualify, because it was trained
on every other fold including the one being held out. `build_fold_targets` enforces the first
half (rows of fold k are never pseudo-labelled in fold k's table, and fold k is that model's
validation fold); the second half is the caller's job and is argued in qa/stage9/report.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rsna_kad.constants import ID_COL, LABEL_KEYS, UNDER_REPORTED_KEYS

# 0.5 real + 0.5 pseudo, verbatim from the RSNA-2024 lumbar 2nd-place recipe (research.md §3).
# Fixed a priori and never tuned on the held-out fold.
PSEUDO_MIX = 0.5
# A pseudo-labelled silent cell is now backed by an independent image model, which makes it about
# as trustworthy as an `absent` on a well-reported key - `targets.weights.absent_well_reported`.
PSEUDO_WEIGHT = 0.5
# `absent` on an under-reported key is the only cell v1 gives weight 0.1; this tolerance
# identifies it without re-reading the label table.
SILENT_WEIGHT_MAX = 0.15


def silent_mask(targets: pd.DataFrame, key: str, weight_max: float = SILENT_WEIGHT_MAX) -> np.ndarray:
    """Cells v1 marked as "the report says nothing about this" (weight 0.1)."""
    return targets[f"{key}_weight"].to_numpy(dtype=np.float64) <= weight_max


def apply_pseudo(targets: pd.DataFrame, pred: pd.DataFrame, keys: tuple[str, ...] = (),
                 mix: float = PSEUDO_MIX, weight: float = PSEUDO_WEIGHT,
                 only_ids: set[str] | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    """v1 targets with the silent cells of `keys` replaced by mix-of-prior-and-prediction.

    `pred` has `StudyInstanceUID` + `{key}_pred`. `only_ids` restricts the rewrite to those
    studies (everything else is returned unchanged). Returns the new table and the number of
    cells rewritten per key."""
    keys = keys or tuple(k for k in LABEL_KEYS if k in UNDER_REPORTED_KEYS)
    out = targets.copy()
    p = pred.set_index(pred[ID_COL].astype(str))
    have = out[ID_COL].astype(str).isin(p.index).to_numpy()
    if only_ids is not None:
        have &= out[ID_COL].astype(str).isin(only_ids).to_numpy()
    aligned = p.reindex(out[ID_COL].astype(str))
    counts: dict[str, int] = {}
    for key in keys:
        m = silent_mask(out, key) & have
        if not m.any():
            counts[key] = 0
            continue
        pv = aligned[f"{key}_pred"].to_numpy(dtype=np.float64)
        t = out[f"{key}_target"].to_numpy(dtype=np.float64)
        w = out[f"{key}_weight"].to_numpy(dtype=np.float64)
        t = np.where(m, (1.0 - mix) * t + mix * pv, t)
        w = np.where(m, weight, w)
        out[f"{key}_target"] = np.clip(t, 0.0, 1.0).astype(np.float32)
        out[f"{key}_weight"] = np.clip(w, 0.0, 1.0).astype(np.float32)
        counts[key] = int(m.sum())
    return out, counts


def build_fold_targets(fold: int, v1_targets: pd.DataFrame, folds: pd.DataFrame,
                       pred: pd.DataFrame, out_dir: str | Path, **kw) -> tuple[pd.DataFrame, dict[str, int]]:
    """Write `<out_dir>/fold{fold}.parquet`: pseudo-labels on the rows of the *other* folds only.

    Fold k's own rows stay byte-identical to v1, which keeps two things true at once - they are
    never trained on by the fold-k model anyway, and the validation metric fold k reports is
    still the v1 metric, so the run is paired with a v1 run cell for cell."""
    folds = folds.assign(**{ID_COL: folds[ID_COL].astype(str)})
    train_ids = set(folds.loc[folds["fold"].astype(int) != int(fold), ID_COL])
    out, counts = apply_pseudo(v1_targets, pred, only_ids=train_ids, **kw)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / f"fold{fold}.parquet", index=False)
    return out, counts


# ------------------------------------------------------------------ v3: teacher mix on every cell
#
# Stage 9 (above) touched only the silent cells and fed them from leave-two-out sources, and was
# judged on OOF-weak - a yardstick built from the report labels, which by construction penalises
# a prediction that departs from them. v3 follows what the top-50 teams report instead (audit
# 2026-09-25 §2): EVERY cell becomes 0.5 x label + 0.5 x teacher, with the teacher = the equal-
# weight OOF blend of the entry's own models. Masking the silent cells "did nothing" for them and
# rounding to 0/1 hurt, so neither is offered here.
#
# LEAKAGE (differs from stage 9 on purpose; argued in qa/pseudo_v3/report.md): the teacher OOF
# of a fold-j study comes from a model that trained on fold k's weak rows, so the fold-k
# student's weak OOF is optimistic. The 58 gold studies are in no training set of teacher or
# student, so gold-58 and the leaderboard remain clean yardsticks - weak OOF does not.

V3_MIX = 0.5
# `absent` on an under-reported key: v1's 0.1 said "the report is silent, do not learn from it".
# Half of the target is now an image-model reading, so the cell is about as informative as an
# `absent` on a well-reported key - `weights.absent_well_reported` = 0.5. Fixed before any
# student existed and applied to every under-reported key alike; not tuned.
V3_SILENT_WEIGHT = 0.5


def teacher_mean(oofs: list[pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight mean of the `{key}_pred` probabilities of several OOF tables, on the
    studies every table has. OOF preds are already sigmoids (§3.2), so this is the
    mean-of-sigmoids blend the ensemble entry ships."""
    if not oofs:
        raise ValueError("no teacher OOF given")
    cols = [f"{k}_pred" for k in LABEL_KEYS]
    idx = [o.assign(**{ID_COL: o[ID_COL].astype(str)}).set_index(ID_COL) for o in oofs]
    for o in idx:
        if not o.index.is_unique:
            raise ValueError("teacher OOF has duplicate studies")
        v = o[cols].to_numpy(np.float64)
        if np.isnan(v).any() or (v < 0).any() or (v > 1).any():
            raise ValueError("teacher OOF must hold probabilities in [0, 1]")
    common = sorted(set.intersection(*(set(o.index) for o in idx)))
    arr = np.mean([o.loc[common, cols].to_numpy(np.float64) for o in idx], axis=0)
    out = pd.DataFrame(arr, columns=cols)
    out.insert(0, ID_COL, common)
    return out


def silent_weight_targets(label_table: pd.DataFrame, calib, cfg, exclude_ids: set[str] | None = None,
                          silent_weight: float = V3_SILENT_WEIGHT) -> pd.DataFrame:
    """`targets.make_targets` with only `weights.absent_under_reported` replaced.

    Re-running the §2.1 weight rule (voter disagreement factor, hallucination cap, parse-fail /
    postoperative override) instead of patching weights after the fact keeps every other weight
    exactly as the label targets have it; the target columns are untouched by the change."""
    from dataclasses import replace

    from rsna_kad.labels.targets import make_targets
    cfg2 = replace(cfg, weights={**cfg.weights, "absent_under_reported": float(silent_weight)})
    return make_targets(label_table, calib, cfg2, exclude_ids=exclude_ids)


def mix_with_teacher(label_targets: pd.DataFrame, teacher: pd.DataFrame, weights: pd.DataFrame,
                     train_ids: set[str], mix: float = V3_MIX) -> tuple[pd.DataFrame, dict[str, int]]:
    """target = (1 - mix) x label + mix x teacher and weight from `weights`, on every cell of the
    rows in `train_ids` that have a teacher prediction; every other row keeps `label_targets`
    bit-for-bit. `weights` must be row-aligned with `label_targets` and share its targets
    (it is the same table re-weighted by `silent_weight_targets`). Returns the table and the
    number of rows rewritten (`rows`) and without a teacher (`no_teacher`)."""
    ids = label_targets[ID_COL].astype(str)
    if not np.array_equal(ids.to_numpy(), weights[ID_COL].astype(str).to_numpy()):
        raise ValueError("label targets and weights are not row-aligned")
    for k in LABEL_KEYS:
        if not np.array_equal(label_targets[f"{k}_target"].to_numpy(), weights[f"{k}_target"].to_numpy()):
            raise ValueError(f"{k}: the re-weighted table changed a target - wrong label table or calibration")
    t = teacher.set_index(teacher[ID_COL].astype(str))
    has_t = ids.isin(t.index).to_numpy()
    m = ids.isin(train_ids).to_numpy() & has_t
    aligned = t.reindex(ids)
    out = label_targets.copy()
    for k in LABEL_KEYS:
        lab = label_targets[f"{k}_target"].to_numpy(np.float64)
        pv = aligned[f"{k}_pred"].to_numpy(np.float64)
        tgt = np.where(m, (1.0 - mix) * lab + mix * np.nan_to_num(pv), lab)
        w = np.where(m, weights[f"{k}_weight"].to_numpy(np.float64), label_targets[f"{k}_weight"].to_numpy(np.float64))
        out[f"{k}_target"] = np.where(m, np.clip(tgt, 0.0, 1.0).astype(np.float32), label_targets[f"{k}_target"])
        out[f"{k}_weight"] = np.where(m, np.clip(w, 0.0, 1.0).astype(np.float32), label_targets[f"{k}_weight"])
    return out, {"rows": int(m.sum()), "no_teacher": int((ids.isin(train_ids).to_numpy() & ~has_t).sum())}
