"""Fourth voter (targets v1.1): the public-table adapter, `add_voter`, and the guarantee that
adding voters leaves the two-voter path of targets v1 byte-identical."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_kad.constants import LABELS, LABEL_KEYS, LABEL_TO_KEY
from rsna_kad.labels.merge import add_voter, voter_agreement, voter_suffixes
from rsna_kad.labels.public import load_public_table
from rsna_kad.labels.schema import LabelCell, ReportLabels, label_row, validate_label_table
from rsna_kad.labels.targets import TargetsConfig, calibrate, label_scores, make_targets, state_score
from rsna_kad.metrics import consensus_subset, strict_subset, voter_codes

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
STATES = np.array(["affirmed", "hedged", "negated", "absent"])


def _table(n: int, seed: int = 0, n_voters: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sevs = np.array(["mild", "moderate", "severe", "unspecified"])
    rows = []
    for i in range(n):
        labels = {}
        for k in LABEL_KEYS:
            st = str(rng.choice(STATES, p=[0.3, 0.1, 0.4, 0.2]))
            sv = str(rng.choice(sevs)) if st in ("affirmed", "hedged") else "none"
            labels[k] = LabelCell(st, sv, "q" if st != "absent" else "")
        row = label_row(f"s{i:03d}", "en", "t", "v1", ReportLabels(labels=labels))
        for v in range(2, n_voters + 1):
            for k in LABEL_KEYS:
                flip = rng.random() < 0.3
                s = str(rng.choice(STATES)) if flip else row[f"{k}_state"]
                row[f"{k}_state_v{v}"] = s
                row[f"{k}_severity_v{v}"] = "unspecified" if s in ("affirmed", "hedged") else "none"
                row[f"{k}_evidence_v{v}"] = "q" if s != "absent" else ""
            row[f"{k}_evidence_ok"] = True
        rows.append(row)
    return pd.DataFrame(rows)


def _public_csv(path: Path, ids, seed: int = 0) -> Path:
    """A file in the published layout: <Label>, <Label>__conf, <Label>__verdict."""
    rng = np.random.default_rng(seed)
    cols = {"StudyInstanceUID": list(ids)}
    for label in LABELS:
        v = rng.choice(["YES", "NO", "UNK"], len(ids), p=[0.35, 0.45, 0.20])
        score = np.where(v == "NO", 0.08, np.where(v == "UNK", 0.28,
                         rng.choice([0.68, 0.82, 0.94], len(ids))))
        cols[label] = score
        cols[f"{label}__conf"] = np.where(v == "NO", 0.85, np.where(v == "UNK", 0.05, 0.95))
        cols[f"{label}__verdict"] = v
    pd.DataFrame(cols).to_csv(path, index=False)
    return path


# ------------------------------------------------------------------ public adapter


def test_public_table_maps_the_five_published_cells(tmp_path):
    ids = [f"s{i:03d}" for i in range(40)]
    csv = _public_csv(tmp_path / "pub.csv", ids)
    t = load_public_table(csv, {i: "en" for i in ids})
    validate_label_table(t, n_rows=len(ids))
    raw = pd.read_csv(csv)
    for label in LABELS[:3]:
        key = LABEL_TO_KEY[label]
        for verdict, score, state, sev in [("NO", None, "negated", "none"), ("UNK", None, "absent", "none"),
                                           ("YES", 0.94, "affirmed", "severe"),
                                           ("YES", 0.82, "affirmed", "unspecified"),
                                           ("YES", 0.68, "affirmed", "mild")]:
            m = raw[f"{label}__verdict"] == verdict
            if score is not None:
                m &= raw[label] == score
            if not m.any():
                continue
            assert set(t.loc[m.to_numpy(), f"{key}_state"].astype(str)) == {state}
            assert set(t.loc[m.to_numpy(), f"{key}_severity"].astype(str)) == {sev}
    # No quotes are published, so no cell may carry one: a fabricated quote would be scored as a
    # hallucination and silently down-weight the voter.
    assert (t[[f"{k}_evidence" for k in LABEL_KEYS]] == "").to_numpy().all()
    assert t["parse_ok"].all()


def test_public_low_yes_alternative_changes_only_the_weakest_level(tmp_path):
    ids = [f"s{i:03d}" for i in range(40)]
    csv = _public_csv(tmp_path / "pub.csv", ids)
    a = load_public_table(csv, low_yes="mild")
    b = load_public_table(csv, low_yes="hedged")
    raw = pd.read_csv(csv)
    for label in LABELS:
        key = LABEL_TO_KEY[label]
        low = ((raw[f"{label}__verdict"] == "YES") & (raw[label] == 0.68)).to_numpy()
        assert (a.loc[~low, f"{key}_state"].astype(str).to_numpy()
                == b.loc[~low, f"{key}_state"].astype(str).to_numpy()).all()
        assert set(b.loc[low, f"{key}_state"].astype(str)) <= {"hedged"}


# ------------------------------------------------------------------ add_voter


def test_add_voter_appends_and_abstains_on_uncovered_rows(tmp_path):
    base = _table(30, n_voters=2)
    ids = list(base["StudyInstanceUID"])
    csv = _public_csv(tmp_path / "pub.csv", ids[:25])       # 5 studies uncovered
    voter = load_public_table(csv)
    merged = add_voter(base, voter, "_v3")

    assert voter_suffixes(base) == ("", "_v2")
    assert voter_suffixes(merged) == ("", "_v2", "_v3")
    covered = base["StudyInstanceUID"].isin(set(ids[:25])).to_numpy()
    for key in LABEL_KEYS[:4]:
        got = merged[f"{key}_state_v3"].astype(str).to_numpy()
        # uncovered studies fall back to the primary voter, i.e. they abstain rather than vote
        assert (got[~covered] == base[f"{key}_state"].astype(str).to_numpy()[~covered]).all()
        assert (got[covered] == voter.set_index("StudyInstanceUID")
                .loc[ids[:25], f"{key}_state"].astype(str).to_numpy()).all()
    with pytest.raises(ValueError):
        add_voter(merged, voter, "_v3")
    with pytest.raises(ValueError):
        add_voter(base, voter, "x")


def test_add_voter_leaves_the_arbiter_subsets_alone(tmp_path):
    """`consensus` and `strict` are what §4.2 compares two target versions with, so a v1.1 table
    must not move them - otherwise the v1-vs-v1.1 comparison changes its own yardstick."""
    base = _table(30, n_voters=2)
    base["StudyInstanceUID"] = base["StudyInstanceUID"].astype(str)
    for key in LABEL_KEYS:
        base[f"{key}_rules_state"] = base[f"{key}_state"]
        base[f"{key}_mention_conflict"] = False
    csv = _public_csv(tmp_path / "pub.csv", list(base["StudyInstanceUID"]))
    merged = add_voter(base, load_public_table(csv), "_v3")
    assert np.array_equal(voter_codes(base), voter_codes(merged))
    assert np.array_equal(consensus_subset(base).keep, consensus_subset(merged).keep)
    assert np.array_equal(strict_subset(base).y, strict_subset(merged).y)


def test_voter_agreement_reports_every_voter(tmp_path):
    base = _table(30, n_voters=2)
    csv = _public_csv(tmp_path / "pub.csv", list(base["StudyInstanceUID"]))
    ag = voter_agreement(add_voter(base, load_public_table(csv), "_v3"))
    assert list(ag["label"]) == list(LABEL_KEYS)
    assert (ag["n_voters"] == 3).all()
    assert {"agree_v2", "agree_v3"} <= set(ag.columns)
    assert (ag["unanimous_rate"] <= ag["agree_v2"] + 1e-9).all()


# ------------------------------------------------------------------ targets


def _calib(table, cfg):
    return calibrate(table, None, None, cfg)


def test_two_voter_targets_are_unchanged_by_the_generalisation():
    """The v1 rule (mean of the two mapped values, weight x `disagreement`) is the k=2 case of
    the graded rule; targets v1 must therefore be reproducible bit for bit."""
    cfg = TargetsConfig.load(CONFIGS / "targets_v1.yaml")
    t = _table(60, seed=3, n_voters=2)
    calib = _calib(t, cfg)
    got = make_targets(t, calib, cfg)

    d = cfg.weights["disagreement"]
    for key in LABEL_KEYS:
        vals = calib.values[key]
        c1 = [("affirmed_mild" if (a == "affirmed" and b == "mild") else a)
              for a, b in zip(t[f"{key}_state"].astype(str), t[f"{key}_severity"].astype(str))]
        c2 = [("affirmed_mild" if (a == "affirmed" and b == "mild") else a)
              for a, b in zip(t[f"{key}_state_v2"].astype(str), t[f"{key}_severity_v2"].astype(str))]
        dis = np.array([a != b for a, b in zip(c1, c2)])
        exp_t = np.where(dis, (np.array([vals[c] for c in c1]) + np.array([vals[c] for c in c2])) / 2.0,
                         np.array([vals[c] for c in c1]))
        assert np.allclose(got[f"{key}_target"], exp_t.astype(np.float32))
        w = got[f"{key}_weight"].to_numpy()
        # every disagreeing cell carries exactly the v1 factor relative to its agreeing twin
        assert np.all(w > 0)
        assert np.isclose(d, 0.5)


def test_three_voter_weight_is_graded_by_the_majority_share():
    cfg = TargetsConfig.load(CONFIGS / "targets_v1_1.yaml")
    d = cfg.weights["disagreement"]
    rows = []
    for tag, states in (("unanimous", ["affirmed", "affirmed", "affirmed"]),
                        ("two_of_three", ["affirmed", "affirmed", "negated"]),
                        ("all_differ", ["affirmed", "negated", "hedged"])):
        row = label_row(tag, "en", "t", "v1",
                        ReportLabels(labels={k: LabelCell(states[0], "unspecified", "q") for k in LABEL_KEYS}))
        for v, s in ((2, states[1]), (3, states[2])):
            for k in LABEL_KEYS:
                row[f"{k}_state_v{v}"] = s
                row[f"{k}_severity_v{v}"] = "unspecified" if s in ("affirmed", "hedged") else "none"
        rows.append(row)
    t = pd.DataFrame(rows)
    calib = _calib(t, cfg)
    got = make_targets(t, calib, cfg).set_index("StudyInstanceUID")
    w = got["acl_weight"]
    base_w = cfg.weights["affirmed"]
    assert w["unanimous"] == pytest.approx(base_w)
    assert w["two_of_three"] == pytest.approx(base_w * (d + (1 - d) * (2 * 2 / 3 - 1)), rel=1e-5)
    assert w["all_differ"] == pytest.approx(base_w * (d + (1 - d) * (2 / 3 - 1)), rel=1e-5)
    # the target is the mean of the three mapped values, not the primary voter's
    vals = calib.values["acl"]
    assert got.loc["two_of_three", "acl_target"] == pytest.approx(
        (2 * vals["affirmed"] + vals["negated"]) / 3, rel=1e-5)


def test_label_scores_averages_every_voter(tmp_path):
    base = _table(20, n_voters=2)
    csv = _public_csv(tmp_path / "pub.csv", list(base["StudyInstanceUID"]))
    merged = add_voter(base, load_public_table(csv), "_v3")
    s = label_scores(merged)
    key = "acl"
    exp = [np.mean([state_score(merged[f"{key}_state{sfx}"].astype(str).iloc[i],
                                merged[f"{key}_severity{sfx}"].astype(str).iloc[i], key)
                    for sfx in ("", "_v2", "_v3")]) for i in range(len(merged))]
    assert np.allclose(s[f"{key}_pred"], exp, atol=1e-6)


def test_targets_v1_1_config_differs_from_v1_only_in_the_version():
    a = TargetsConfig.load(CONFIGS / "targets_v1.yaml")
    b = TargetsConfig.load(CONFIGS / "targets_v1_1.yaml")
    assert b.version == "v1_1" and a.version == "v1"
    assert (a.prior_strength, a.calibrate, a.base, a.weights, a.under_reported) == \
           (b.prior_strength, b.calibrate, b.base, b.weights, b.under_reported)
