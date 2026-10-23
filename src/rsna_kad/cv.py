"""Site-grouped, language-stratified folds. Generated once; the md5 of the CSV is
pinned by tests/test_cv.py so the split can never silently change."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from rsna_kad.constants import ID_COL

FOLD_COLUMNS = [ID_COL, "fold", "site_key", "lang", "is_gold"]


def _study_summary(path: str | os.PathLike[str], group_col: str, strat_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    id_col = ID_COL if ID_COL in df.columns else "study_dir"
    gold_col = "labelled" if "labelled" in df.columns else "is_gold"
    out = pd.DataFrame({
        ID_COL: df[id_col].astype(str),
        "site_key": df[group_col].astype(str),
        "lang": df[strat_col].astype(str),
        "is_gold": df[gold_col].astype(str).str.lower().isin(["true", "1", "yes"]),
    })
    if out[ID_COL].duplicated().any():
        raise ValueError(f"{path}: duplicate {ID_COL}")
    return out.sort_values(ID_COL).reset_index(drop=True)


def make_folds(study_summary_csv: str | os.PathLike[str], n_splits: int = 5, seed: int = 2026,
               group_col: str = "site_key", strat_col: str = "lang", method: str = "balanced",
               n_trials: int = 200) -> pd.DataFrame:
    """Grouped folds. `sklearn` = StratifiedGroupKFold(groups=site, y=lang); `balanced` = the
    same grouping but a seeded greedy search that also balances fold size and gold count,
    because sklearn's variant only optimises the y distribution and, with language ≈ site,
    lands at 14-25 % fold sizes for every seed tried on this data."""
    df = _study_summary(study_summary_csv, group_col, strat_col)
    if method == "sklearn":
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold = np.full(len(df), -1, dtype=int)
        for k, (_, test_idx) in enumerate(splitter.split(df, df["lang"], df["site_key"])):
            fold[test_idx] = k
    elif method == "balanced":
        fold = _balanced_group_folds(df, n_splits, seed, n_trials)
    else:
        raise ValueError(f"unknown fold method {method!r}")
    if (fold < 0).any():
        raise RuntimeError("fold assignment left studies unassigned")
    df["fold"] = fold
    return df[FOLD_COLUMNS]


def _balanced_group_folds(df: pd.DataFrame, k: int, seed: int, n_trials: int,
                          w_size: float = 1.0, w_gold: float = 1.0, w_lang: float = 0.5) -> np.ndarray:
    """Best of n_trials greedy assignments (groups in size order with seeded jitter, each to
    the fold that minimises the squared imbalance of size, gold count and language shares)."""
    langs = sorted(df["lang"].unique())
    groups = df.groupby("site_key")
    names = list(groups.groups)
    n = groups.size().reindex(names).to_numpy(dtype=float)
    gold = groups["is_gold"].sum().reindex(names).to_numpy(dtype=float)
    lang = pd.crosstab(df["site_key"], df["lang"]).reindex(index=names, columns=langs).to_numpy(dtype=float)
    n_tot, g_tot, l_tot = n.sum(), max(gold.sum(), 1.0), np.maximum(lang.sum(axis=0), 1.0)
    target = 1.0 / k
    rng = np.random.default_rng(seed)

    def cost(fn: np.ndarray, fg: np.ndarray, fl: np.ndarray) -> float:
        return (w_size * ((fn / n_tot - target) ** 2).sum()
                + w_gold * ((fg / g_tot - target) ** 2).sum()
                + w_lang * ((fl / l_tot - target) ** 2).sum(axis=0).mean())

    best_cost, best_assign = np.inf, None
    for _ in range(n_trials):
        order = np.argsort(-n * rng.uniform(0.6, 1.4, size=len(n)), kind="stable")
        fn, fg, fl = np.zeros(k), np.zeros(k), np.zeros((k, len(langs)))
        assign = np.empty(len(n), dtype=int)
        for gi in order:
            trial_costs = []
            for f in range(k):
                fn[f] += n[gi]; fg[f] += gold[gi]; fl[f] += lang[gi]
                trial_costs.append(cost(fn, fg, fl))
                fn[f] -= n[gi]; fg[f] -= gold[gi]; fl[f] -= lang[gi]
            f = int(np.argmin(trial_costs))
            assign[gi] = f
            fn[f] += n[gi]; fg[f] += gold[gi]; fl[f] += lang[gi]
        total = cost(fn, fg, fl)
        if total < best_cost:
            best_cost, best_assign = total, assign.copy()
    fold_of_group = dict(zip(names, best_assign))
    return df["site_key"].map(fold_of_group).to_numpy(dtype=int)


def fold_summary(folds: pd.DataFrame) -> pd.DataFrame:
    per_fold = folds.groupby("fold").agg(
        n=(ID_COL, "size"), frac=(ID_COL, lambda s: len(s) / len(folds)),
        n_gold=("is_gold", "sum"), n_sites=("site_key", "nunique"))
    langs = pd.crosstab(folds["fold"], folds["lang"], normalize="index").round(3)
    return per_fold.join(langs)


def check_folds(folds: pd.DataFrame, n_splits: int = 5, frac_range: tuple[float, float] = (0.18, 0.22),
                gold_range: tuple[int, int] = (8, 16)) -> list[str]:
    """Violations of the architecture §4 stage-1 fold acceptance criteria (empty = OK)."""
    problems = []
    if sorted(folds["fold"].unique()) != list(range(n_splits)):
        problems.append(f"folds present: {sorted(folds['fold'].unique())}")
    sites_per_fold = folds.groupby("site_key")["fold"].nunique()
    if (sites_per_fold > 1).any():
        problems.append(f"{int((sites_per_fold > 1).sum())} site_keys span several folds")
    for k, grp in folds.groupby("fold"):
        frac = len(grp) / len(folds)
        if not frac_range[0] <= frac <= frac_range[1]:
            problems.append(f"fold {k}: {frac:.3f} of studies")
        n_gold = int(grp["is_gold"].sum())
        if not gold_range[0] <= n_gold <= gold_range[1]:
            problems.append(f"fold {k}: {n_gold} gold rows")
    return problems


def load_folds(path: str | os.PathLike[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={ID_COL: str, "fold": int, "site_key": str, "lang": str})
    df["is_gold"] = df["is_gold"].astype(str).str.lower().isin(["true", "1"])
    return df[FOLD_COLUMNS]


def write_folds(folds: pd.DataFrame, path: str | os.PathLike[str]) -> str:
    """Writes with a fixed row order and line ending so the md5 is platform-independent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = folds.sort_values(ID_COL).to_csv(index=False, lineterminator="\n")
    path.write_bytes(text.encode("utf-8"))
    return md5_of(path)


def md5_of(path: str | os.PathLike[str]) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()
