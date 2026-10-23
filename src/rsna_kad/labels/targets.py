"""States -> soft targets and weights (architecture §2.1), gold-58 evaluation of a label table."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from rsna_kad.constants import LABEL_KEYS, LABELS, LABEL_TO_KEY, UNDER_REPORTED_KEYS
from rsna_kad.labels.merge import voter_suffixes
from rsna_kad.metrics import gold_report, macro_auc

CELLS: tuple[str, ...] = ("affirmed", "affirmed_mild", "hedged", "absent", "negated")
MONOTONE_CHAIN: tuple[str, ...] = ("affirmed", "hedged", "absent", "negated")
# Labels where the host thresholds make a "mild" report finding gold-negative (§2.1).
MILD_PENALTY_KEYS: frozenset[str] = frozenset({"med_men", "lat_men", "effusion", "mcl", "med_oa", "lat_oa", "pf_oa"})
STATE_SCORE: dict[str, float] = {"affirmed": 1.0, "hedged": 0.66, "absent": 0.33, "negated": 0.0}
MILD_PENALTY = 0.35


# ------------------------------------------------------------------ gold evaluation


def state_score(state: str, severity: str, key: str) -> float:
    s = STATE_SCORE[state]
    if state == "affirmed" and severity == "mild" and key in MILD_PENALTY_KEYS:
        s -= MILD_PENALTY
    return s


def label_scores(table: pd.DataFrame) -> pd.DataFrame:
    """StudyInstanceUID + {key}_pred: state->score mapping, averaged over every voter present
    (`{key}_state` plus each `{key}_state_v*` of a merged table)."""
    out = pd.DataFrame({"StudyInstanceUID": table["StudyInstanceUID"].to_numpy()})
    for key in LABEL_KEYS:
        cols = []
        for suffix in voter_suffixes(table):
            sc, vc = f"{key}_state{suffix}", f"{key}_severity{suffix}"
            st, sv = table[sc].astype(str).to_numpy(), table[vc].astype(str).to_numpy()
            cols.append(np.array([state_score(a, b, key) for a, b in zip(st, sv)]))
        out[f"{key}_pred"] = np.mean(cols, axis=0).astype(np.float32)
    return out


def gold_frame(train_csv: str | Path) -> pd.DataFrame:
    """The 58 gold rows of train.csv with snake-case 0/1 columns."""
    df = pd.read_csv(train_csv)
    gold = df[df[LABELS[0]].notna()].copy()
    gold = gold.rename(columns=LABEL_TO_KEY)
    return gold[["StudyInstanceUID"] + list(LABEL_KEYS)].reset_index(drop=True)


def gold_state_report(table: pd.DataFrame, gold: pd.DataFrame, n_boot: int = 1000) -> dict[str, Any]:
    pred = label_scores(table)
    if "lang" in table.columns:
        pred = pred.merge(table[["StudyInstanceUID", "lang"]], on="StudyInstanceUID")
    return gold_report(pred, gold, n_boot=n_boot)


def gold_state_auc(table: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Per-label AUC of the state->score mapping vs gold, plus a 'macro' row."""
    rep = gold_state_report(table, gold, n_boot=0)
    rows = [{"label": k, "auc": v} for k, v in rep["per_label"].items()]
    rows.append({"label": "macro", "auc": rep["macro_auc"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ config


@dataclass
class TargetsConfig:
    version: str
    prior_strength: float
    calibrate: bool
    base: dict[str, Any]
    weights: dict[str, float]
    under_reported: tuple[str, ...] = tuple(sorted(UNDER_REPORTED_KEYS))

    @classmethod
    def load(cls, path: str | Path) -> "TargetsConfig":
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        known = {"version", "prior_strength", "calibrate", "base", "weights", "under_reported"}
        unknown = set(doc) - known
        if unknown:
            raise ValueError(f"unknown keys in {path}: {sorted(unknown)}")
        if "under_reported" in doc:
            doc["under_reported"] = tuple(doc["under_reported"])
        return cls(**doc)


def state_prior_table(cfg: TargetsConfig) -> pd.DataFrame:
    """Base value per (key, cell) before calibration; columns key, cell, base."""
    rows = []
    mild = cfg.base["affirmed_mild"]
    for key in LABEL_KEYS:
        under = key in cfg.under_reported
        vals = {
            "affirmed": cfg.base["affirmed"],
            "affirmed_mild": mild.get(key, mild["default"]),
            "hedged": cfg.base["hedged"],
            "absent": cfg.base["absent_under_reported"] if under else cfg.base["absent_well_reported"],
            "negated": cfg.base["negated"],
        }
        for cell in CELLS:
            rows.append({"key": key, "cell": cell, "base": float(vals[cell])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ calibration


def cell_of(state: str, severity: str) -> str:
    if state == "affirmed" and severity == "mild":
        return "affirmed_mild"
    return state


def pava_decreasing(values: list[float], weights: list[float]) -> list[float]:
    """Weighted pool-adjacent-violators for a non-increasing sequence."""
    blocks: list[list[float]] = []  # [value, weight, count]
    for v, w in zip(values, weights):
        blocks.append([v, w, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v1, w1, c1 = blocks[-2]
            v2, w2, c2 = blocks[-1]
            w = w1 + w2
            blocks[-2:] = [[(v1 * w1 + v2 * w2) / w if w > 0 else (v1 + v2) / 2, w, c1 + c2]]
    out: list[float] = []
    for v, _, c in blocks:
        out.extend([v] * c)
    return out


@dataclass
class Calibration:
    version: str
    fold: int | None
    n_gold_train: int
    values: dict[str, dict[str, float]] = field(default_factory=dict)   # key -> cell -> value
    counts: dict[str, dict[str, list[int]]] = field(default_factory=dict)  # key -> cell -> [n, n_pos]

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"version": self.version, "fold": self.fold,
                                          "n_gold_train": self.n_gold_train, "values": self.values,
                                          "counts": self.counts}, indent=1), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "Calibration":
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(doc["version"], doc["fold"], doc["n_gold_train"], doc["values"], doc["counts"])

    def is_monotone(self) -> bool:
        for key in LABEL_KEYS:
            v = [self.values[key][c] for c in MONOTONE_CHAIN]
            if any(a < b - 1e-9 for a, b in zip(v, v[1:])):
                return False
            if not (self.values[key]["negated"] - 1e-9 <= self.values[key]["affirmed_mild"] <= self.values[key]["affirmed"] + 1e-9):
                return False
        return True


def calibrate(label_table: pd.DataFrame, gold: pd.DataFrame | None, train_ids: set[str] | None,
              cfg: TargetsConfig, fold: int | None = None) -> Calibration:
    """Empirical-Bayes posterior mean per (key, cell) around the base value, using the gold rows
    whose StudyInstanceUID is in `train_ids`, then PAVA on affirmed >= hedged >= absent >= negated.
    With cfg.calibrate False (or no gold) the base table is returned unchanged."""
    prior = state_prior_table(cfg)
    base = {(r.key, r.cell): r.base for r in prior.itertuples()}
    values: dict[str, dict[str, float]] = {k: {} for k in LABEL_KEYS}
    counts: dict[str, dict[str, list[int]]] = {k: {c: [0, 0] for c in CELLS} for k in LABEL_KEYS}
    n_train = 0
    if cfg.calibrate and gold is not None and train_ids:
        g = gold[gold["StudyInstanceUID"].isin(train_ids)]
        g = g.merge(label_table, on="StudyInstanceUID", how="inner", suffixes=("", "_lt"))
        n_train = len(g)
        for key in LABEL_KEYS:
            cells = [cell_of(s, v) for s, v in zip(g[f"{key}_state"].astype(str), g[f"{key}_severity"].astype(str))]
            y = g[key].to_numpy(dtype=float)
            for c, yy in zip(cells, y):
                counts[key][c][0] += 1
                counts[key][c][1] += int(yy > 0.5)
    s = float(cfg.prior_strength)
    for key in LABEL_KEYS:
        for c in CELLS:
            n, n_pos = counts[key][c]
            values[key][c] = (s * base[(key, c)] + n_pos) / (s + n)
        chain = [values[key][c] for c in MONOTONE_CHAIN]
        w = [s + counts[key][c][0] for c in MONOTONE_CHAIN]
        for c, v in zip(MONOTONE_CHAIN, pava_decreasing(chain, w)):
            values[key][c] = float(v)
        values[key]["affirmed_mild"] = float(min(max(values[key]["affirmed_mild"], values[key]["negated"]), values[key]["affirmed"]))
    return Calibration(cfg.version, fold, n_train, values, counts)


# ------------------------------------------------------------------ targets


def _weight_of(cell: str, key: str, cfg: TargetsConfig) -> float:
    w = cfg.weights
    if cell in ("affirmed", "affirmed_mild"):
        return w["affirmed"]
    if cell == "negated":
        return w["negated"]
    if cell == "hedged":
        return w["hedged"]
    return w["absent_under_reported"] if key in cfg.under_reported else w["absent_well_reported"]


def make_targets(label_table: pd.DataFrame, calib: Calibration, cfg: TargetsConfig,
                 exclude_ids: set[str] | None = None) -> pd.DataFrame:
    """Targets + weights per (study, label). Rows in exclude_ids (gold) are dropped.

    Voter disagreement -> target = mean of the voters' mapped values, weight x a factor that
    falls from 1.0 at unanimity to cfg.weights["disagreement"] at an even split:

        m = d + (1 - d) * (2f - 1),   f = share of voters holding the modal cell, d = disagreement

    With the two voters of targets v1 that is exactly v1's rule (f is 1.0 or 0.5, so m is 1.0 or
    d) and a third voter grades the middle instead of treating 2-of-3 like 1-of-2.

    parse failure / postoperative -> every weight = parse_fail / postoperative; hallucinated
    evidence ({key}_evidence_ok False) -> hallucinated_evidence."""
    df = label_table
    if exclude_ids:
        df = df[~df["StudyInstanceUID"].isin(exclude_ids)]
    df = df.reset_index(drop=True)
    out = pd.DataFrame({"StudyInstanceUID": df["StudyInstanceUID"].to_numpy()})
    parse_ok = df["parse_ok"].to_numpy(dtype=bool) if "parse_ok" in df.columns else np.ones(len(df), bool)
    postop = df["postoperative"].to_numpy(dtype=bool) if "postoperative" in df.columns else np.zeros(len(df), bool)
    row_override = np.where(~parse_ok, cfg.weights["parse_fail"], np.where(postop, cfg.weights["postoperative"], np.nan))
    suffixes = voter_suffixes(df)
    d = float(cfg.weights["disagreement"])
    for key in LABEL_KEYS:
        vals = calib.values[key]
        per_voter = []
        for suffix in suffixes:
            st = df[f"{key}_state{suffix}"].astype(str).to_numpy()
            sv = df[f"{key}_severity{suffix}"].astype(str).to_numpy()
            per_voter.append([cell_of(a, b) for a, b in zip(st, sv)])
        cells = per_voter[0]
        target = np.mean([[vals[c] for c in cl] for cl in per_voter], axis=0)
        weight = np.array([_weight_of(c, key, cfg) for c in cells], dtype=np.float64)
        if len(per_voter) > 1:
            votes = np.array(per_voter, dtype=object).T          # (n_rows, n_voters)
            share = np.array([max(np.unique(r, return_counts=True)[1]) for r in votes]) / len(per_voter)
            weight = weight * (d + (1.0 - d) * (2.0 * share - 1.0))
        if f"{key}_evidence_ok" in df.columns:
            bad = ~df[f"{key}_evidence_ok"].to_numpy(dtype=bool)
            weight = np.where(bad, np.minimum(weight, cfg.weights["hallucinated_evidence"]), weight)
        weight = np.where(np.isnan(row_override), weight, row_override)
        out[f"{key}_target"] = np.clip(target, 0.0, 1.0).astype(np.float32)
        out[f"{key}_weight"] = np.clip(weight, 0.0, 1.0).astype(np.float32)
    return out


def validate_targets(df: pd.DataFrame, n_rows: int | None = None) -> None:
    cols = [f"{k}_{f}" for k in LABEL_KEYS for f in ("target", "weight")]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"targets lack columns: {missing}")
    if n_rows is not None and len(df) != n_rows:
        raise ValueError(f"targets have {len(df)} rows, expected {n_rows}")
    vals = df[cols].to_numpy(dtype=np.float64)
    if np.isnan(vals).any():
        raise ValueError("targets contain NaN")
    if (vals < 0).any() or (vals > 1).any():
        raise ValueError("targets outside [0, 1]")


def build_fold_targets(label_table: pd.DataFrame, folds: pd.DataFrame, gold: pd.DataFrame,
                       cfg: TargetsConfig, out_dir: str | Path) -> dict[int, pd.DataFrame]:
    """One targets table per fold, calibrated on the gold rows of the other folds; gold rows are
    excluded from every table. Writes fold{k}.parquet + calibration_fold{k}.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_ids = set(gold["StudyInstanceUID"])
    fold_of = dict(zip(folds["StudyInstanceUID"], folds["fold"].astype(int)))
    result: dict[int, pd.DataFrame] = {}
    for k in sorted(folds["fold"].astype(int).unique()):
        train_gold = {sid for sid in gold_ids if fold_of.get(sid) != k}
        calib = calibrate(label_table, gold, train_gold, cfg, fold=int(k))
        if not calib.is_monotone():
            raise RuntimeError(f"fold {k}: calibration is not monotone")
        targets = make_targets(label_table, calib, cfg, exclude_ids=gold_ids)
        validate_targets(targets)
        targets.to_parquet(out_dir / f"fold{k}.parquet", index=False)
        calib.to_json(out_dir / f"calibration_fold{k}.json")
        result[int(k)] = targets
    return result


def oof_weak_auc(targets: pd.DataFrame, pred: pd.DataFrame) -> tuple[float, dict[str, float]]:
    df = targets.merge(pred, on="StudyInstanceUID", how="inner")
    y = df[[f"{k}_target" for k in LABEL_KEYS]].to_numpy()
    w = df[[f"{k}_weight" for k in LABEL_KEYS]].to_numpy()
    p = df[[f"{k}_pred" for k in LABEL_KEYS]].to_numpy()
    return macro_auc(y, p, w)
