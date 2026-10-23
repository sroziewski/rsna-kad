"""The per-report label record: dataclass, JSON schema for guided decoding, parsing
and the flat row layout of the label tables (architecture §3.2)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from rsna_kad.constants import LABEL_KEYS, SEVERITIES, STATES

KNEE_SIDES: tuple[str, ...] = ("left", "right", "unknown")
EVIDENCE_MAX_CHARS = 120


@dataclass(frozen=True)
class LabelCell:
    state: str = "absent"
    severity: str = "none"
    evidence: str = ""


@dataclass
class ReportLabels:
    knee_side: str = "unknown"
    postoperative: bool = False
    normal_template: bool = False
    labels: dict[str, LabelCell] = field(default_factory=dict)

    def cell(self, key: str) -> LabelCell:
        return self.labels.get(key, LabelCell())

    def to_dict(self) -> dict[str, Any]:
        return {
            "knee_side": self.knee_side,
            "postoperative": self.postoperative,
            "normal_template": self.normal_template,
            "labels": {k: {"state": c.state, "severity": c.severity, "evidence": c.evidence}
                       for k, c in ((k, self.cell(k)) for k in LABEL_KEYS)},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


_CELL_SCHEMA = {
    "type": "object",
    "properties": {
        "state": {"type": "string", "enum": list(STATES)},
        "severity": {"type": "string", "enum": list(SEVERITIES)},
        "evidence": {"type": "string", "maxLength": EVIDENCE_MAX_CHARS},
    },
    "required": ["state", "severity", "evidence"],
    "additionalProperties": False,
}

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "knee_side": {"type": "string", "enum": list(KNEE_SIDES)},
        "postoperative": {"type": "boolean"},
        "normal_template": {"type": "boolean"},
        "labels": {
            "type": "object",
            "properties": {k: _CELL_SCHEMA for k in LABEL_KEYS},
            "required": list(LABEL_KEYS),
            "additionalProperties": False,
        },
    },
    "required": ["knee_side", "postoperative", "normal_template", "labels"],
    "additionalProperties": False,
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _clean_text(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _FENCE_RE.sub("", text.strip()).strip()
    # Tolerate chatter around the object: take the outermost {...}.
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return ""
    return text[start:end + 1]


def parse_output(text: str | None) -> ReportLabels | None:
    """Parse model output into ReportLabels; None when it does not satisfy the schema.

    Strict on the label keys and the enums (guided decoding guarantees them, so a
    violation means truncation or a broken run), lenient on surrounding text."""
    if not text:
        return None
    try:
        doc = json.loads(_clean_text(text))
    except ValueError:
        return None
    if not isinstance(doc, dict) or not isinstance(doc.get("labels"), dict):
        return None
    side = doc.get("knee_side", "unknown")
    if side not in KNEE_SIDES:
        return None
    labels: dict[str, LabelCell] = {}
    for key in LABEL_KEYS:
        cell = doc["labels"].get(key)
        if not isinstance(cell, dict):
            return None
        state, severity = cell.get("state"), cell.get("severity", "unspecified")
        if state not in STATES or severity not in SEVERITIES:
            return None
        evidence = cell.get("evidence", "")
        if not isinstance(evidence, str):
            return None
        if state in ("negated", "absent"):
            severity = "none"
        elif severity == "none":
            severity = "unspecified"
        if state == "absent":
            evidence = ""
        labels[key] = LabelCell(state, severity, evidence.strip()[:EVIDENCE_MAX_CHARS])
    return ReportLabels(
        knee_side=side,
        postoperative=bool(doc.get("postoperative", False)),
        normal_template=bool(doc.get("normal_template", False)),
        labels=labels,
    )


def label_row(study_id: str, lang: str, extractor: str, variant: str,
              parsed: ReportLabels | None) -> dict[str, Any]:
    """One flat row of a label table; a failed parse gives absent states with parse_ok=False."""
    rec = parsed if parsed is not None else ReportLabels()
    row: dict[str, Any] = {
        "StudyInstanceUID": study_id, "lang": lang, "extractor": extractor, "variant": variant,
        "parse_ok": parsed is not None, "knee_side": rec.knee_side,
        "postoperative": rec.postoperative, "normal_template": rec.normal_template,
    }
    for key in LABEL_KEYS:
        cell = rec.cell(key)
        row[f"{key}_state"] = cell.state
        row[f"{key}_severity"] = cell.severity
        row[f"{key}_evidence"] = cell.evidence
    return row


LABEL_TABLE_COLUMNS: tuple[str, ...] = tuple(
    ["StudyInstanceUID", "lang", "extractor", "variant", "parse_ok", "knee_side",
     "postoperative", "normal_template"]
    + [f"{k}_{f}" for k in LABEL_KEYS for f in ("state", "severity", "evidence")])


def finalize_label_table(df):
    """Enforce the §3.2 dtypes and column order on a label table."""
    import pandas as pd

    df = df.copy()
    for col in ("parse_ok", "postoperative", "normal_template"):
        df[col] = df[col].astype(bool)
    df["knee_side"] = pd.Categorical(df["knee_side"], categories=list(KNEE_SIDES))
    for key in LABEL_KEYS:
        df[f"{key}_state"] = pd.Categorical(df[f"{key}_state"], categories=list(STATES))
        df[f"{key}_severity"] = pd.Categorical(df[f"{key}_severity"], categories=list(SEVERITIES))
        df[f"{key}_evidence"] = df[f"{key}_evidence"].fillna("").astype(str)
    extra = [c for c in df.columns if c not in LABEL_TABLE_COLUMNS]
    return df[list(LABEL_TABLE_COLUMNS) + extra]


def validate_label_table(df, n_rows: int | None = 4407) -> None:
    missing = [c for c in LABEL_TABLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"label table lacks columns: {missing}")
    if n_rows is not None and len(df) != n_rows:
        raise ValueError(f"label table has {len(df)} rows, expected {n_rows}")
    if df["StudyInstanceUID"].duplicated().any():
        raise ValueError("label table has duplicate StudyInstanceUID")
    for key in LABEL_KEYS:
        bad = set(df[f"{key}_state"].astype(str).unique()) - set(STATES)
        if bad:
            raise ValueError(f"{key}_state has values outside STATES: {bad}")
        bad = set(df[f"{key}_severity"].astype(str).unique()) - set(SEVERITIES)
        if bad:
            raise ValueError(f"{key}_severity has values outside SEVERITIES: {bad}")


def write_label_table(df, path) -> None:
    """Parquet + csv copy next to it (§3.2)."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = finalize_label_table(df)
    df.to_parquet(path, index=False)
    df.to_csv(path.with_suffix(".csv"), index=False)
