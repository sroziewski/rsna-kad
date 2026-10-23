import numpy as np
import pandas as pd

from rsna_kad.constants import LABEL_KEYS
from rsna_kad.labels.merge import add_evidence_flags, evidence_found, evidence_ratio, merge_summary, merge_votes
from rsna_kad.labels.schema import LabelCell, ReportLabels, label_row

REPORT = "Findings: There is a horizontal undersurface tear of the posterior horn of the medial meniscus. No joint effusion."


def _table(sid_states: dict[str, dict[str, tuple[str, str, str]]], variant: str, parse_ok=None):
    rows = []
    for sid, cells in sid_states.items():
        rec = ReportLabels(labels={k: LabelCell(*v) for k, v in cells.items()})
        row = label_row(sid, "en", "llm", variant, rec)
        if parse_ok is not None and sid in parse_ok:
            row["parse_ok"] = parse_ok[sid]
        rows.append(row)
    return pd.DataFrame(rows)


def test_evidence_found_exact_fuzzy_and_missing():
    assert evidence_found("horizontal undersurface tear", REPORT)
    assert evidence_found("Horizontal  undersurface TEAR of the posterior horn", REPORT)  # case/whitespace
    assert evidence_ratio("horizontal undersurfase tear of the posterior horn", REPORT) >= 0.8  # one typo
    assert not evidence_found("complete ACL rupture with bone bruise", REPORT)
    assert not evidence_found("", REPORT)


def test_add_evidence_flags():
    t = _table({"s": {"med_men": ("affirmed", "unspecified", "horizontal undersurface tear"),
                      "acl": ("negated", "none", "made up quote about the ACL"),
                      "baker": ("absent", "none", "")}}, "v1")
    t = add_evidence_flags(t, {"s": REPORT})
    assert t.loc[0, "med_men_evidence_ok"] and not t.loc[0, "acl_evidence_ok"] and t.loc[0, "baker_evidence_ok"]


def test_merge_votes_flags():
    v1 = _table({"s1": {"acl": ("affirmed", "severe", "acl tear"), "effusion": ("negated", "none", "no effusion")},
                 "s2": {"acl": ("absent", "none", "")}}, "v1", parse_ok={"s2": False})
    v2 = _table({"s1": {"acl": ("hedged", "unspecified", "possible acl tear"), "effusion": ("negated", "none", "no effusion")},
                 "s2": {"acl": ("negated", "none", "acl intact")}}, "v2")
    rules = _table({"s1": {"acl": ("affirmed", "unspecified", "acl tear")}, "s2": {"mcl": ("negated", "none", "mcl normal")}}, "rules")
    rules["mcl_mentioned"] = [False, True]
    rules["acl_mentioned"] = [True, False]
    for k in LABEL_KEYS:
        if f"{k}_mentioned" not in rules.columns:
            rules[f"{k}_mentioned"] = False
    v1["acl_evidence_ok"] = [True, True]
    v2["acl_evidence_ok"] = [False, True]
    m = merge_votes(v1, v2, rules)
    assert list(m["StudyInstanceUID"]) == ["s1", "s2"]
    r1 = m.set_index("StudyInstanceUID").loc["s1"]
    assert r1["acl_state"] == "affirmed" and r1["acl_state_v2"] == "hedged" and not r1["acl_agree"]
    assert r1["effusion_agree"] and r1["effusion_state"] == "negated"
    assert r1["acl_rules_state"] == "affirmed"
    assert not r1["acl_evidence_ok"]  # v2's quote failed
    assert not r1["acl_mention_conflict"]
    r2 = m.set_index("StudyInstanceUID").loc["s2"]
    # v1 failed to parse: v2 becomes primary, mcl absent in the LLM but mentioned by the rules
    assert r2["acl_state"] == "negated" and r2["parse_ok"]
    assert r2["mcl_mention_conflict"] and not r2["acl_mention_conflict"]
    assert m["extractor"].iloc[0] == "merged"
    s = merge_summary(m)
    assert len(s) == len(LABEL_KEYS)
    assert np.isclose(s.set_index("label").loc["acl", "agree_rate"], 0.0)


def test_merge_requires_same_ids():
    v1 = _table({"s1": {}}, "v1")
    v2 = _table({"s2": {}}, "v2")
    rules = _table({"s1": {}}, "rules")
    import pytest

    with pytest.raises(ValueError):
        merge_votes(v1, v2, rules)
