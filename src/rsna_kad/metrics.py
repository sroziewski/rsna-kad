"""Macro AUC on soft targets (binarised with a dropped middle band), the gold-58 report, and
the arbiter subsets of `qa/arbiter/report.md` (consensus / evidence / strict / gold)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from rsna_kad.constants import ID_COL, LABEL_KEYS, UNDER_REPORTED_KEYS

POS_THR = 0.6
NEG_THR = 0.2
NEG_MIN_WEIGHT = 0.5
THR_EPS = 1e-6


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney AUC with tie handling (average ranks); NaN if one class is missing."""
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos, n_neg = int(y_true.sum()), int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binarise(y_soft: np.ndarray, weight: np.ndarray | None, pos_thr: float = POS_THR,
             neg_thr: float = NEG_THR, neg_min_weight: float = NEG_MIN_WEIGHT) -> tuple[np.ndarray, np.ndarray]:
    """(keep_mask, y_bin): positive if target >= pos_thr, negative if target <= neg_thr and
    weight >= neg_min_weight; everything else (the middle band, low-weight negatives) is dropped."""
    y_soft = np.asarray(y_soft, dtype=np.float64)
    weight = np.ones_like(y_soft) if weight is None else np.asarray(weight, dtype=np.float64)
    # Targets are stored as float32 (0.2f32 == 0.2000000030 > 0.2): a target sitting exactly on a
    # threshold must count as inside it, hence the tolerance.
    pos = y_soft >= pos_thr - THR_EPS
    neg = (y_soft <= neg_thr + THR_EPS) & (weight >= neg_min_weight - THR_EPS)
    return pos | neg, pos


def macro_auc(y_soft: np.ndarray | pd.DataFrame, y_pred: np.ndarray | pd.DataFrame,
              weight: np.ndarray | pd.DataFrame | None = None, pos_thr: float = POS_THR,
              neg_thr: float = NEG_THR, neg_min_weight: float = NEG_MIN_WEIGHT,
              keys: tuple[str, ...] = LABEL_KEYS) -> tuple[float, dict[str, float]]:
    """Macro AUC over labels; labels whose kept rows lack a class are skipped (NaN)."""
    y_soft, y_pred = np.asarray(y_soft, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    if weight is not None:
        weight = np.asarray(weight, dtype=np.float64)
    if y_soft.ndim == 1:
        y_soft, y_pred = y_soft[:, None], y_pred[:, None]
        weight = None if weight is None else weight[:, None]
    per_label: dict[str, float] = {}
    for j in range(y_soft.shape[1]):
        keep, y_bin = binarise(y_soft[:, j], None if weight is None else weight[:, j],
                               pos_thr, neg_thr, neg_min_weight)
        name = keys[j] if j < len(keys) else str(j)
        per_label[name] = auc(y_bin[keep], y_pred[keep, j])
    vals = [v for v in per_label.values() if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), per_label


def bootstrap_macro_auc(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000,
                        seed: int = 0) -> tuple[float, float]:
    """95 % percentile CI of the macro AUC (rows resampled with replacement)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m, _ = macro_auc(y_true[idx], y_pred[idx], None, pos_thr=0.5, neg_thr=0.5, neg_min_weight=0.0)
        if not np.isnan(m):
            stats.append(m)
    if not stats:
        return float("nan"), float("nan")
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def gold_report(pred: pd.DataFrame, gold: pd.DataFrame, n_boot: int = 1000, seed: int = 0,
                langs: tuple[str, ...] = ("en", "es", "tr")) -> dict[str, Any]:
    """Macro AUC of `pred` ({key}_pred columns) vs `gold` ({key} 0/1 columns) on the joined
    StudyInstanceUID rows, with per-label values, a bootstrap CI and per-language values."""
    df = gold.merge(pred, on="StudyInstanceUID", how="inner", suffixes=("", "_p"))
    y_true = df[[k for k in LABEL_KEYS]].to_numpy(dtype=np.float64)
    y_pred = df[[f"{k}_pred" for k in LABEL_KEYS]].to_numpy(dtype=np.float64)
    macro, per_label = macro_auc(y_true, y_pred, None, pos_thr=0.5, neg_thr=0.5, neg_min_weight=0.0)
    lo, hi = bootstrap_macro_auc(y_true, y_pred, n_boot=n_boot, seed=seed)
    out: dict[str, Any] = {"n": int(len(df)), "macro_auc": macro, "ci95": [lo, hi], "per_label": per_label}
    if "lang" in df.columns:
        per_lang: dict[str, Any] = {}
        for lang in langs:
            sub = df["lang"].astype(str) == lang
            if sub.sum() < 3:
                continue
            m, pl = macro_auc(y_true[sub.to_numpy()], y_pred[sub.to_numpy()], None,
                              pos_thr=0.5, neg_thr=0.5, neg_min_weight=0.0)
            per_lang[lang] = {"n": int(sub.sum()), "macro_auc": m, "per_label": pl}
        out["per_lang"] = per_lang
    return out


# ====================================================================== arbiter subsets
# Everything below is additive: `oof_weak_auc`, `oof_strict_auc` and `gold_auc` keep using
# the functions above unchanged. The subsets exist because `oof_weak_auc` scores a model
# against LLM-derived targets, so a model that reproduces the extractor's mistakes is
# rewarded for it (qa/arbiter/report.md). A subset restricts the comparison to cells whose
# label is corroborated by an independent voter instead of re-weighting the same cells.

AFFIRMED, HEDGED, NEGATED, ABSENT = "affirmed", "hedged", "negated", "absent"
# Voter-agreement codes of a (study, label) cell.
AGREE_ALL3 = 3   # LLM V1 == LLM V2 == rule voter
AGREE_LLM2 = 2   # the two LLM variants agree, the rule voter differs
AGREE_SPLIT = 1  # the two LLM variants disagree


def _rank_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """`auc` without its per-element Python loop; a 10^5-call bootstrap cannot afford that one.
    A test pins the two against each other, ties included."""
    n = y_score.shape[0]
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == n:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    s = y_score[order]
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(s[1:], s[:-1], out=new[1:])
    group = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    ends = np.append(starts[1:], n) - 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = (starts[group] + ends[group]) / 2.0 + 1.0
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * (n - n_pos)))


def _state(labels: pd.DataFrame, col: str, n: int) -> np.ndarray:
    return labels[col].astype(str).to_numpy() if col in labels.columns else np.full(n, "", dtype=object)


def voter_codes(labels: pd.DataFrame, keys: Sequence[str] = LABEL_KEYS) -> np.ndarray:
    """(n_rows, n_keys) of AGREE_ALL3 / AGREE_LLM2 / AGREE_SPLIT.

    The rule voter emits only affirmed/negated/absent, so an LLM `hedged` cell can never reach
    AGREE_ALL3 - that is intended: a hedged cell is exactly the kind of cell no arbiter should
    treat as ground truth."""
    n = len(labels)
    out = np.empty((n, len(keys)), dtype=np.int8)
    for j, key in enumerate(keys):
        s1 = _state(labels, f"{key}_state", n)
        s2 = _state(labels, f"{key}_state_v2", n)
        ru = _state(labels, f"{key}_rules_state", n)
        same_llm = s1 == s2
        out[:, j] = np.where(same_llm & (s1 == ru), AGREE_ALL3,
                             np.where(same_llm, AGREE_LLM2, AGREE_SPLIT))
    return out


def evidence_mask(labels: pd.DataFrame, keys: Sequence[str] = LABEL_KEYS) -> np.ndarray:
    """(n_rows, n_keys): the quote was found in the report and the rule voter did not see a
    mention the LLM called absent."""
    n = len(labels)
    out = np.ones((n, len(keys)), dtype=bool)
    for j, key in enumerate(keys):
        if f"{key}_evidence_ok" in labels.columns:
            out[:, j] &= labels[f"{key}_evidence_ok"].to_numpy(dtype=bool)
        if f"{key}_mention_conflict" in labels.columns:
            out[:, j] &= ~labels[f"{key}_mention_conflict"].to_numpy(dtype=bool)
    return out


@dataclass(frozen=True)
class Subset:
    """A scoring subset: which (study, label) cells count and what their binary label is.

    `keep` and `y` are (n_rows, n_keys); `y` is only meaningful where `keep`. Rows are studies,
    so a bootstrap that resamples rows resamples whole studies - the unit the LB also samples."""

    name: str
    ids: np.ndarray
    keep: np.ndarray
    y: np.ndarray
    keys: tuple[str, ...] = LABEL_KEYS

    @property
    def coverage(self) -> int:
        return int(self.keep.sum())

    def per_label_coverage(self) -> dict[str, dict[str, int]]:
        out = {}
        for j, key in enumerate(self.keys):
            k = self.keep[:, j]
            out[key] = {"n": int(k.sum()), "n_pos": int((self.y[:, j] & k).sum()),
                        "n_neg": int((~self.y[:, j] & k).sum())}
        return out

    def restrict(self, mask: np.ndarray, name: str | None = None) -> "Subset":
        return Subset(name or self.name, self.ids, self.keep & mask, self.y, self.keys)

    def align(self, pred: pd.DataFrame) -> np.ndarray:
        """Predictions of `pred` ({key}_pred columns) in this subset's row order."""
        p = pred.set_index(pred[ID_COL].astype(str))
        missing = [s for s in self.ids if s not in p.index]
        if missing:
            raise KeyError(f"{len(missing)} subset studies missing from the predictions")
        return p.loc[self.ids, [f"{k}_pred" for k in self.keys]].to_numpy(dtype=np.float64)


def weak_subset(targets: pd.DataFrame, keys: Sequence[str] = LABEL_KEYS, name: str = "weak",
                pos_thr: float = POS_THR, neg_thr: float = NEG_THR,
                neg_min_weight: float = NEG_MIN_WEIGHT) -> Subset:
    """The current primary metric as a Subset: `binarise` of the calibrated soft targets."""
    n = len(targets)
    keep = np.zeros((n, len(keys)), bool)
    y = np.zeros((n, len(keys)), bool)
    for j, key in enumerate(keys):
        keep[:, j], y[:, j] = binarise(targets[f"{key}_target"].to_numpy(),
                                       targets[f"{key}_weight"].to_numpy(),
                                       pos_thr, neg_thr, neg_min_weight)
    return Subset(name, targets[ID_COL].astype(str).to_numpy(), keep, y, tuple(keys))


def strict_subset(labels: pd.DataFrame, keys: Sequence[str] = LABEL_KEYS,
                  name: str = "strict") -> Subset:
    """Affirmed vs negated cells of the primary voter (§2.5 metric 2) as a Subset."""
    n = len(labels)
    keep = np.zeros((n, len(keys)), bool)
    y = np.zeros((n, len(keys)), bool)
    for j, key in enumerate(keys):
        st = _state(labels, f"{key}_state", n)
        keep[:, j] = (st == AFFIRMED) | (st == NEGATED)
        y[:, j] = st == AFFIRMED
    return Subset(name, labels[ID_COL].astype(str).to_numpy(), keep, y, tuple(keys))


def consensus_subset(labels: pd.DataFrame, keys: Sequence[str] = LABEL_KEYS,
                     include_agreed_absent: bool = False,
                     under_reported: Iterable[str] = UNDER_REPORTED_KEYS,
                     name: str | None = None) -> Subset:
    """Cells where LLM V1, LLM V2 and the rule voter all produced the same state, kept as
    positive (affirmed) or negative (negated).

    With `include_agreed_absent` a unanimously *absent* cell of a well-reported label is added
    as a negative - the same convention §2.1 uses when it gives those cells target 0.06. It is
    off by default: unanimous silence is agreement between three voters that read the same
    report, not evidence about the image, and for the under-reported keys it is demonstrably
    wrong (synovitis is gold-positive in 47 % of studies and mentioned in 12 %)."""
    n = len(labels)
    codes = voter_codes(labels, keys)
    under = set(under_reported)
    keep = np.zeros((n, len(keys)), bool)
    y = np.zeros((n, len(keys)), bool)
    for j, key in enumerate(keys):
        st = _state(labels, f"{key}_state", n)
        all3 = codes[:, j] == AGREE_ALL3
        pos, neg = all3 & (st == AFFIRMED), all3 & (st == NEGATED)
        if include_agreed_absent and key not in under:
            neg = neg | (all3 & (st == ABSENT))
        keep[:, j], y[:, j] = pos | neg, pos
    tag = name or ("consensus+absent" if include_agreed_absent else "consensus")
    return Subset(tag, labels[ID_COL].astype(str).to_numpy(), keep, y, tuple(keys))


def gold_subset(gold: pd.DataFrame, keys: Sequence[str] = LABEL_KEYS, name: str = "gold58") -> Subset:
    """The 58 image-derived rows of train.csv; every cell counts."""
    y = gold[list(keys)].to_numpy(dtype=np.float64) >= 0.5
    return Subset(name, gold[ID_COL].astype(str).to_numpy(), np.ones_like(y, bool), y, tuple(keys))


def subset_auc(subset: Subset, y_pred: np.ndarray) -> tuple[float, dict[str, float]]:
    """Macro AUC over the labels of `subset` that have both classes among their kept cells."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    per_label: dict[str, float] = {}
    for j, key in enumerate(subset.keys):
        k = subset.keep[:, j]
        per_label[key] = _rank_auc(subset.y[k, j], y_pred[k, j]) if k.any() else float("nan")
    vals = [v for v in per_label.values() if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), per_label


def _boot_rows(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, (n_boot, n))


def bootstrap_subset_auc(subset: Subset, preds: dict[str, np.ndarray], n_boot: int = 2000,
                         seed: int = 2026) -> dict[str, np.ndarray]:
    """Macro AUC of every candidate on every bootstrap replicate, sharing one set of resampled
    studies so differences between candidates are paired (the same replicate for both sides)."""
    rows = _boot_rows(len(subset.ids), n_boot, seed)
    out: dict[str, np.ndarray] = {}
    for name, pred in preds.items():
        pred = np.asarray(pred, dtype=np.float64)
        vals = np.full((n_boot, len(subset.keys)), np.nan)
        for j in range(len(subset.keys)):
            kj, yj, pj = subset.keep[:, j], subset.y[:, j], pred[:, j]
            for b in range(n_boot):
                sel = rows[b][kj[rows[b]]]
                vals[b, j] = _rank_auc(yj[sel], pj[sel])
        with np.errstate(invalid="ignore"):
            out[name] = np.nanmean(vals, axis=1)
    return out


def ci95(samples: np.ndarray) -> tuple[float, float]:
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def paired_delta(boot_a: np.ndarray, boot_b: np.ndarray) -> dict[str, float]:
    """Distribution of (a - b) over the shared bootstrap replicates: CI, P(a > b) and the
    half-width that decides what an arbiter can and cannot resolve."""
    d = np.asarray(boot_a, dtype=np.float64) - np.asarray(boot_b, dtype=np.float64)
    d = d[np.isfinite(d)]
    lo, hi = ci95(d)
    return {"mean": float(d.mean()) if d.size else float("nan"), "ci_lo": lo, "ci_hi": hi,
            "half_width": (hi - lo) / 2.0 if d.size else float("nan"),
            "p_gt": float((d > 0).mean()) if d.size else float("nan")}


def weighted_arbiter(values: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean of several arbiters' macro AUCs over the names present in both dicts."""
    pairs = [(weights[k], values[k]) for k in weights if k in values and np.isfinite(values[k])]
    if not pairs:
        return float("nan")
    wsum = sum(w for w, _ in pairs)
    return float(sum(w * v for w, v in pairs) / wsum) if wsum > 0 else float("nan")
