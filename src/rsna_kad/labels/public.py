"""Adapter for an external, already-published report-label table (PATH 2 of targets v1.1).

`pilkwang/rsna-knee-llm-labels` / `report_labels_v2.csv` publishes three columns per finding -
`<Label>` (score), `<Label>__conf` and `<Label>__verdict` - and nothing else. Measured on the
whole file (`qa/competitor/public_table.md`) the three columns carry exactly five distinct cells:

    verdict NO   score 0.08  conf 0.85
    verdict UNK  score 0.28  conf 0.05
    verdict YES  score 0.68 / 0.82 / 0.94  conf 0.95

so `conf` is a function of `verdict` and adds nothing, and the whole table is a 5-level ordinal
per cell. That is what the mapping below encodes: the three YES levels are the only place where
the table says more than a bare state, and they map onto our severity axis (§2.1). There are no
evidence quotes, so cells from this voter are never subject to the hallucination check - they
carry no quote to check, and scoring them as a failed quote would silently down-weight them.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from rsna_kad.constants import ID_COL, LABELS, LABEL_TO_KEY
from rsna_kad.labels.schema import finalize_label_table

# YES score -> (state, severity). Their levels are ordinal within YES; the boundaries sit between
# the three observed values, so the table is reproduced exactly and a future file with slightly
# different levels still maps monotonically.
YES_LEVELS: tuple[tuple[float, str, str], ...] = (
    (0.90, "affirmed", "severe"),
    (0.75, "affirmed", "unspecified"),
    (0.00, "affirmed", "mild"),
)
# Alternative for the lowest YES level, measured against the gold rows in `pick_low_yes_mapping`:
# 0.68 sits next to our own `hedged` score of 0.66, so "is their weakest YES a hedge?" is an
# empirical question, not a naming one.
LOW_YES_ALTERNATIVES: dict[str, tuple[str, str]] = {
    "mild": ("affirmed", "mild"),
    "hedged": ("hedged", "unspecified"),
}
VERDICT_STATES: dict[str, tuple[str, str]] = {
    "NO": ("negated", "none"),
    "UNK": ("absent", "none"),
}


def _cell(verdict: str, score: float, low_yes: str) -> tuple[str, str]:
    if verdict in VERDICT_STATES:
        return VERDICT_STATES[verdict]
    if verdict != "YES":
        return "absent", "none"
    for thr, state, severity in YES_LEVELS:
        if score >= thr:
            return LOW_YES_ALTERNATIVES[low_yes] if severity == "mild" else (state, severity)
    return "absent", "none"


def load_public_table(path: str | Path, lang_of: dict[str, str] | None = None,
                      extractor: str = "pilkwang_v2", variant: str = "public",
                      low_yes: str = "mild") -> pd.DataFrame:
    """The external CSV as a §3.2 label table (no evidence quotes, parse_ok always True)."""
    if low_yes not in LOW_YES_ALTERNATIVES:
        raise ValueError(f"low_yes must be one of {sorted(LOW_YES_ALTERNATIVES)}")
    df = pd.read_csv(path)
    missing = [c for c in (ID_COL, *LABELS) if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    ids = df[ID_COL].astype(str).to_numpy()
    out: dict[str, object] = {
        ID_COL: ids,
        "lang": [(lang_of or {}).get(i, "en") for i in ids],
        "extractor": extractor, "variant": variant, "parse_ok": True,
        "knee_side": "unknown", "postoperative": False, "normal_template": False,
    }
    for label in LABELS:
        key = LABEL_TO_KEY[label]
        verdicts = df[f"{label}__verdict"].astype(str).str.strip().str.upper().to_numpy()
        scores = df[label].astype(float).to_numpy()
        cells = [_cell(v, s, low_yes) for v, s in zip(verdicts, scores)]
        out[f"{key}_state"] = [c[0] for c in cells]
        out[f"{key}_severity"] = [c[1] for c in cells]
        out[f"{key}_evidence"] = ""
    return finalize_label_table(pd.DataFrame(out))


def pick_low_yes_mapping(path: str | Path, gold: pd.DataFrame,
                         lang_of: dict[str, str] | None = None) -> dict[str, object]:
    """Which reading of their weakest YES level reproduces their own ordering best.

    Their raw `score` column is the reference: it is what their pipeline trains on, and the
    published 0.870 gold AUC belongs to it. Whichever of our mappings scores closer to it on the
    gold rows is the one that loses least in translation."""
    from rsna_kad.labels.targets import gold_state_report
    from rsna_kad.metrics import gold_report

    df = pd.read_csv(path)
    raw = pd.DataFrame({ID_COL: df[ID_COL].astype(str)})
    for label in LABELS:
        raw[f"{LABEL_TO_KEY[label]}_pred"] = df[label].astype(float)
    ref = gold_report(raw, gold, n_boot=0)["macro_auc"]
    cand = {name: gold_state_report(load_public_table(path, lang_of, low_yes=name),
                                    gold, n_boot=0)["macro_auc"]
            for name in LOW_YES_ALTERNATIVES}
    chosen = max(cand, key=lambda n: cand[n])
    return {"raw_score_macro_auc": ref, "candidates": cand, "chosen": chosen,
            "loss_vs_raw": float(ref - cand[chosen])}
