"""Combine the two LLM prompt variants and the rule voter into one label table with
agreement / mention-conflict / evidence flags. Disagreement is kept (both states stored),
never resolved by majority vote (research.md §2)."""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

from rsna_kad.constants import LABEL_KEYS
from rsna_kad.labels.schema import LABEL_TABLE_COLUMNS, finalize_label_table

EVIDENCE_MIN_RATIO = 0.8

try:  # optional, ~50x faster than difflib for the sliding-window match
    from rapidfuzz import fuzz as _fuzz
except ImportError:  # pragma: no cover
    _fuzz = None


def _squash(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def evidence_ratio(evidence: str, report: str) -> float:
    """Best fuzzy match ratio of the quote against any window of the report (1.0 = substring)."""
    ev, rep = _squash(evidence), _squash(report)
    if not ev or not rep:
        return 0.0
    if ev in rep:
        return 1.0
    if _fuzz is not None:
        return float(_fuzz.partial_ratio(ev, rep)) / 100.0
    n = len(ev)
    step = max(1, n // 4)
    best = 0.0
    for start in range(0, max(1, len(rep) - n + 1), step):
        r = difflib.SequenceMatcher(None, ev, rep[start:start + n + step]).ratio()
        if r > best:
            best = r
            if best >= 0.999:
                break
    return best


def evidence_found(evidence: str, report: str, min_ratio: float = EVIDENCE_MIN_RATIO) -> bool:
    return evidence_ratio(evidence, report) >= min_ratio


def add_evidence_flags(table: pd.DataFrame, reports: dict[str, str]) -> pd.DataFrame:
    """{key}_evidence_ok per cell: True for absent cells, else the quote must be found in the report."""
    table = table.copy()
    ids = table["StudyInstanceUID"].to_numpy()
    for key in LABEL_KEYS:
        states = table[f"{key}_state"].astype(str).to_numpy()
        evs = table[f"{key}_evidence"].fillna("").astype(str).to_numpy()
        ok = np.ones(len(table), dtype=bool)
        for i, (sid, st, ev) in enumerate(zip(ids, states, evs)):
            if st == "absent":
                continue
            ok[i] = evidence_found(ev, reports.get(sid, ""))
        table[f"{key}_evidence_ok"] = ok
    return table


def _indexed(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("StudyInstanceUID", drop=False)


def merge_votes(llm_v1: pd.DataFrame, llm_v2: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    """Merged label table: V1 is the primary voter, V2 is kept in `{key}_state_v2`, the rule
    voter in `{key}_rules_state`; flags `{key}_agree`, `{key}_mention_conflict`, `{key}_evidence_ok`."""
    v1, v2, ru = _indexed(llm_v1), _indexed(llm_v2), _indexed(rules)
    ids = list(v1.index)
    missing = set(ids) - set(v2.index)
    if missing:
        raise ValueError(f"{len(missing)} studies missing from V2")
    missing = set(ids) - set(ru.index)
    if missing:
        raise ValueError(f"{len(missing)} studies missing from the rules table")
    v2, ru = v2.loc[ids], ru.loc[ids]
    ok1, ok2 = v1["parse_ok"].to_numpy(bool), v2["parse_ok"].to_numpy(bool)
    use_v2 = ~ok1 & ok2
    out: dict[str, Any] = {
        "StudyInstanceUID": ids,
        "lang": v1["lang"].astype(str).to_numpy(),
        "extractor": "merged",
        "variant": "v1+v2+rules",
        "parse_ok": ok1 | ok2,
        "knee_side": np.where(v1["knee_side"].astype(str) != "unknown", v1["knee_side"].astype(str),
                              np.where(v2["knee_side"].astype(str) != "unknown", v2["knee_side"].astype(str),
                                       ru["knee_side"].astype(str))),
        "postoperative": v1["postoperative"].to_numpy(bool) | v2["postoperative"].to_numpy(bool),
        "normal_template": v1["normal_template"].to_numpy(bool) & v2["normal_template"].to_numpy(bool),
    }
    for key in LABEL_KEYS:
        s1, s2 = v1[f"{key}_state"].astype(str).to_numpy(), v2[f"{key}_state"].astype(str).to_numpy()
        g1, g2 = v1[f"{key}_severity"].astype(str).to_numpy(), v2[f"{key}_severity"].astype(str).to_numpy()
        e1 = v1[f"{key}_evidence"].fillna("").astype(str).to_numpy()
        e2 = v2[f"{key}_evidence"].fillna("").astype(str).to_numpy()
        out[f"{key}_state"] = np.where(use_v2, s2, s1)
        out[f"{key}_severity"] = np.where(use_v2, g2, g1)
        out[f"{key}_evidence"] = np.where(use_v2, e2, e1)
        out[f"{key}_state_v2"] = np.where(use_v2, s1, s2)
        out[f"{key}_severity_v2"] = np.where(use_v2, g1, g2)
        out[f"{key}_evidence_v2"] = np.where(use_v2, e1, e2)
        out[f"{key}_agree"] = (s1 == s2) & ok1 & ok2
        rs = ru[f"{key}_state"].astype(str).to_numpy()
        mentioned = ru[f"{key}_mentioned"].to_numpy(bool) if f"{key}_mentioned" in ru.columns else (rs != "absent")
        out[f"{key}_rules_state"] = rs
        out[f"{key}_mention_conflict"] = (out[f"{key}_state"] == "absent") & mentioned
        ok_col = f"{key}_evidence_ok"
        o1 = v1[ok_col].to_numpy(bool) if ok_col in v1.columns else np.ones(len(ids), bool)
        o2 = v2[ok_col].to_numpy(bool) if ok_col in v2.columns else np.ones(len(ids), bool)
        out[ok_col] = np.where(use_v2, o2, o1 & (o2 | ~ok2))
    return finalize_label_table(pd.DataFrame(out))


def voter_suffixes(table: pd.DataFrame, keys: tuple[str, ...] = LABEL_KEYS) -> tuple[str, ...]:
    """Column suffixes of the voters present: `("",)`, `("", "_v2")`, `("", "_v2", "_v3")`, ...

    The primary voter has no suffix. Order is the column order of the table, which is the order
    the voters were added, so a caller can rely on `[0]` being the primary one."""
    key = keys[0]
    out = [""] if f"{key}_state" in table.columns else []
    out += [c[len(key) + 6:] for c in table.columns
            if c.startswith(f"{key}_state_v") and c != f"{key}_state"]
    return tuple(out)


def add_voter(merged: pd.DataFrame, table: pd.DataFrame, suffix: str,
              keys: tuple[str, ...] = LABEL_KEYS) -> pd.DataFrame:
    """Append one more voter's states as `{key}_state{suffix}` / `_severity{suffix}`.

    Studies the new voter does not cover keep the primary voter's cell, so a partial table
    abstains instead of voting "absent" on the rows it never saw (`report_labels_v2.csv` is one
    study short of our 4,407). `{key}_evidence_ok` and `{key}_mention_conflict` are NOT touched:
    they are statements about the quote of the primary voter, and a voter that publishes no
    quotes must not be able to trip the hallucination weight."""
    if not suffix.startswith("_v"):
        raise ValueError(f"voter suffix must look like '_v3', got {suffix!r}")
    if f"{keys[0]}_state{suffix}" in merged.columns:
        raise ValueError(f"voter {suffix} is already present")
    out = merged.copy()
    src = table.set_index("StudyInstanceUID")
    ids = out["StudyInstanceUID"].to_numpy()
    covered = np.isin(ids, src.index.to_numpy())
    idx = pd.Index(ids[covered])
    for key in keys:
        for field in ("state", "severity", "evidence"):
            base = out[f"{key}_{field}"].astype(str).to_numpy().copy()
            base[covered] = src.loc[idx, f"{key}_{field}"].astype(str).to_numpy()
            out[f"{key}_{field}{suffix}"] = base
    out.attrs = dict(merged.attrs)
    out.attrs[f"voter{suffix}"] = {"extractor": str(table["extractor"].iloc[0]),
                                   "variant": str(table["variant"].iloc[0]),
                                   "n_covered": int(covered.sum()), "n_rows": int(len(out))}
    return out


def voter_agreement(merged: pd.DataFrame, keys: tuple[str, ...] = LABEL_KEYS) -> pd.DataFrame:
    """Per-label unanimity rate over every voter present, and the pairwise rates against the
    primary voter - the diagnostic that says whether a new voter is diverse or redundant."""
    sufs = voter_suffixes(merged, keys)
    rows = []
    for key in keys:
        states = [merged[f"{key}_state{s}"].astype(str).to_numpy() for s in sufs]
        unan = np.ones(len(merged), bool)
        for s in states[1:]:
            unan &= s == states[0]
        row = {"label": key, "n_voters": len(sufs), "unanimous_rate": float(unan.mean())}
        for s, arr in zip(sufs[1:], states[1:]):
            row[f"agree{s}"] = float((arr == states[0]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def merge_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """Per-label agreement / conflict / evidence rates and affirmed rates."""
    rows = []
    for key in LABEL_KEYS:
        st = merged[f"{key}_state"].astype(str)
        rows.append({
            "label": key,
            "affirmed_rate": float((st == "affirmed").mean()),
            "hedged_rate": float((st == "hedged").mean()),
            "negated_rate": float((st == "negated").mean()),
            "absent_rate": float((st == "absent").mean()),
            "agree_rate": float(merged[f"{key}_agree"].mean()),
            "mention_conflict_rate": float(merged[f"{key}_mention_conflict"].mean()),
            "evidence_ok_rate": float(merged.loc[st != "absent", f"{key}_evidence_ok"].mean()) if (st != "absent").any() else float("nan"),
        })
    return pd.DataFrame(rows)


MERGED_EXTRA_COLUMNS: tuple[str, ...] = tuple(
    [f"{k}_{f}" for k in LABEL_KEYS for f in ("state_v2", "severity_v2", "evidence_v2", "agree", "rules_state", "mention_conflict", "evidence_ok")])
MERGED_COLUMNS: tuple[str, ...] = LABEL_TABLE_COLUMNS + MERGED_EXTRA_COLUMNS
