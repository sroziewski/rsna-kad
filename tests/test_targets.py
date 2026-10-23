import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_kad.constants import LABEL_KEYS, UNDER_REPORTED_KEYS
from rsna_kad.labels.schema import LabelCell, ReportLabels, label_row
from rsna_kad.labels.targets import (CELLS, MONOTONE_CHAIN, Calibration, TargetsConfig, build_fold_targets,
                                     calibrate, cell_of, gold_state_auc, label_scores, make_targets,
                                     pava_decreasing, state_prior_table, state_score, validate_targets)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _cfg(name="targets_v1.yaml") -> TargetsConfig:
    return TargetsConfig.load(CONFIGS / name)


def _table(n: int, seed: int = 0, variant2: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    states = np.array(["affirmed", "hedged", "negated", "absent"])
    sevs = np.array(["mild", "moderate", "severe", "unspecified"])
    rows = []
    for i in range(n):
        labels = {}
        for k in LABEL_KEYS:
            st = str(rng.choice(states, p=[0.3, 0.1, 0.4, 0.2]))
            sv = str(rng.choice(sevs)) if st in ("affirmed", "hedged") else "none"
            labels[k] = LabelCell(st, sv, "q" if st != "absent" else "")
        row = label_row(f"s{i}", "en", "t", "v1", ReportLabels(labels=labels))
        if variant2:
            for k in LABEL_KEYS:
                flip = rng.random() < 0.3
                row[f"{k}_state_v2"] = str(rng.choice(states)) if flip else row[f"{k}_state"]
                row[f"{k}_severity_v2"] = "unspecified" if row[f"{k}_state_v2"] in ("affirmed", "hedged") else "none"
                row[f"{k}_evidence_ok"] = rng.random() > 0.05
        rows.append(row)
    return pd.DataFrame(rows)


def _gold(table: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """Gold rows correlated with the states (affirmed mostly positive, negated mostly negative)."""
    rng = np.random.default_rng(seed)
    g = {"StudyInstanceUID": table["StudyInstanceUID"].to_numpy()}
    for k in LABEL_KEYS:
        p = table[f"{k}_state"].astype(str).map({"affirmed": 0.85, "hedged": 0.5, "absent": 0.3, "negated": 0.1}).to_numpy()
        g[k] = (rng.random(len(table)) < p).astype(int)
    return pd.DataFrame(g)


def test_configs_load_and_unknown_key_rejected(tmp_path):
    for name in ("targets_v0.yaml", "targets_v1.yaml"):
        cfg = _cfg(name)
        assert cfg.prior_strength == 8 and set(cfg.under_reported) == UNDER_REPORTED_KEYS
    assert _cfg("targets_v0.yaml").calibrate is False and _cfg().calibrate is True
    bad = tmp_path / "bad.yaml"
    bad.write_text((CONFIGS / "targets_v1.yaml").read_text() + "\nbogus: 1\n")
    with pytest.raises(ValueError):
        TargetsConfig.load(bad)


def test_state_score_mapping():
    assert state_score("affirmed", "unspecified", "acl") == 1.0
    assert state_score("affirmed", "mild", "med_men") == pytest.approx(0.65)
    assert state_score("affirmed", "mild", "fracture") == 1.0  # no mild penalty for fracture
    assert state_score("hedged", "mild", "effusion") == pytest.approx(0.66)
    assert state_score("absent", "none", "acl") == pytest.approx(0.33)
    assert state_score("negated", "none", "acl") == 0.0


def test_label_scores_averages_variants():
    t = _table(5, variant2=True)
    s = label_scores(t)
    assert set(s.columns) == {"StudyInstanceUID", *[f"{k}_pred" for k in LABEL_KEYS]}
    k = "acl"
    exp = [(state_score(a, b, k) + state_score(c, d, k)) / 2 for a, b, c, d in
           zip(t[f"{k}_state"], t[f"{k}_severity"], t[f"{k}_state_v2"], t[f"{k}_severity_v2"])]
    assert np.allclose(s[f"{k}_pred"], exp)


def test_prior_table_values():
    cfg = _cfg()
    p = state_prior_table(cfg).set_index(["key", "cell"])["base"]
    assert p[("acl", "affirmed")] == 0.92 and p[("mcl", "affirmed_mild")] == 0.20
    assert p[("contusion", "affirmed_mild")] == 0.70 and p[("acl", "affirmed_mild")] == 0.35
    assert p[("acl", "absent")] == 0.06 and p[("synovitis", "absent")] == 0.15
    assert len(p) == 12 * len(CELLS)


def test_pava_decreasing():
    assert pava_decreasing([0.9, 0.5, 0.6, 0.1], [1, 1, 1, 1]) == pytest.approx([0.9, 0.55, 0.55, 0.1])
    assert pava_decreasing([0.9, 0.5, 0.3, 0.1], [1, 1, 1, 1]) == [0.9, 0.5, 0.3, 0.1]
    assert pava_decreasing([0.1, 0.9], [1, 3]) == pytest.approx([0.7, 0.7])


def test_calibrate_moves_towards_gold_and_stays_monotone():
    cfg = _cfg()
    t = _table(300)
    g = _gold(t)
    ids = set(t["StudyInstanceUID"])
    calib = calibrate(t, g, ids, cfg, fold=0)
    assert calib.is_monotone() and calib.n_gold_train == 300
    base = state_prior_table(cfg).set_index(["key", "cell"])["base"]
    for k in LABEL_KEYS:
        v = calib.values[k]
        assert v["absent"] > base[(k, "absent")]  # gold positive rate 0.3 among absent pulls it up
        assert v["negated"] > base[(k, "negated")]
        for a, b in zip(MONOTONE_CHAIN, MONOTONE_CHAIN[1:]):
            assert v[a] >= v[b] - 1e-9
    # no calibration: identical to the base table
    base_calib = calibrate(t, g, ids, _cfg("targets_v0.yaml"))
    for k in LABEL_KEYS:
        for c in CELLS:
            assert base_calib.values[k][c] == pytest.approx(base[(k, c)])


def test_calibrate_prior_dominates_with_few_rows():
    cfg = _cfg()
    t = _table(4)
    g = _gold(t)
    calib = calibrate(t, g, set(t["StudyInstanceUID"]), cfg)
    assert calib.is_monotone()
    assert abs(calib.values["acl"]["affirmed"] - 0.92) < 0.3


def test_calibration_json_roundtrip(tmp_path):
    cfg = _cfg()
    t = _table(50)
    calib = calibrate(t, _gold(t), set(t["StudyInstanceUID"]), cfg, fold=2)
    calib.to_json(tmp_path / "c.json")
    back = Calibration.from_json(tmp_path / "c.json")
    assert back.values == calib.values and back.fold == 2


def test_make_targets_ranges_weights_and_disagreement():
    cfg = _cfg()
    t = _table(200, variant2=True)
    t.loc[0, "parse_ok"] = False
    t.loc[1, "postoperative"] = True
    calib = calibrate(t, None, None, cfg)
    out = make_targets(t, calib, cfg, exclude_ids={"s2", "s3"})
    assert len(out) == 198 and "s2" not in set(out["StudyInstanceUID"])
    validate_targets(out, n_rows=198)
    r0, r1 = out.iloc[0], out.iloc[1]
    assert all(r0[f"{k}_weight"] == pytest.approx(0.2) for k in LABEL_KEYS)
    assert all(r1[f"{k}_weight"] == pytest.approx(0.2) for k in LABEL_KEYS)
    tt = t.set_index("StudyInstanceUID").loc[out["StudyInstanceUID"]]
    for k in LABEL_KEYS:
        c1 = [cell_of(a, b) for a, b in zip(tt[f"{k}_state"].astype(str), tt[f"{k}_severity"].astype(str))]
        c2 = [cell_of(a, b) for a, b in zip(tt[f"{k}_state_v2"].astype(str), tt[f"{k}_severity_v2"].astype(str))]
        dis = np.array([a != b for a, b in zip(c1, c2)])
        v = calib.values[k]
        exp = np.array([(v[a] + v[b]) / 2 if d else v[a] for a, b, d in zip(c1, c2, dis)])
        assert np.allclose(out[f"{k}_target"], exp, atol=1e-6)
        w = out[f"{k}_weight"].to_numpy()
        ok = tt[f"{k}_evidence_ok"].to_numpy(bool)
        clean = ok & ~dis & (np.arange(len(tt)) >= 2)
        st = tt[f"{k}_state"].astype(str).to_numpy()
        expected_w = {"affirmed": 1.0, "negated": 1.0, "hedged": 0.7,
                      "absent": 0.1 if k in cfg.under_reported else 0.5}
        assert all(w[i] == pytest.approx(expected_w[st[i]]) for i in np.where(clean)[0])
        assert all(w[i] <= 0.5 * expected_w[st[i]] + 1e-6 for i in np.where(dis & (np.arange(len(tt)) >= 2))[0])
        assert all(w[i] <= 0.3 + 1e-6 for i in np.where(~ok)[0])


def test_validate_targets_catches_nan_and_range():
    t = _table(5)
    cfg = _cfg("targets_v0.yaml")
    out = make_targets(t, calibrate(t, None, None, cfg), cfg)
    validate_targets(out, n_rows=5)
    bad = out.copy()
    bad.loc[0, "acl_target"] = np.nan
    with pytest.raises(ValueError):
        validate_targets(bad)
    bad = out.copy()
    bad.loc[0, "acl_weight"] = 1.5
    with pytest.raises(ValueError):
        validate_targets(bad)


def test_build_fold_targets_with_synthetic_folds(tmp_path):
    cfg = _cfg()
    t = _table(120)
    ids = t["StudyInstanceUID"].to_numpy()
    rng = np.random.default_rng(5)
    folds = pd.DataFrame({"StudyInstanceUID": ids, "fold": rng.integers(0, 5, len(ids)),
                          "site_key": "k", "lang": "en", "is_gold": False})
    gold = _gold(t.head(20))  # 20 gold rows, like the 58 real ones
    folds.loc[folds["StudyInstanceUID"].isin(gold["StudyInstanceUID"]), "is_gold"] = True
    res = build_fold_targets(t, folds, gold, cfg, tmp_path)
    assert set(res) == {0, 1, 2, 3, 4}
    for k, df in res.items():
        assert len(df) == 100 and not set(df["StudyInstanceUID"]) & set(gold["StudyInstanceUID"])
        assert (tmp_path / f"fold{k}.parquet").exists()
        cal = json.loads((tmp_path / f"calibration_fold{k}.json").read_text())
        n_gold_in_fold = int(((folds["fold"] == k) & folds["is_gold"]).sum())
        assert cal["n_gold_train"] == 20 - n_gold_in_fold
        assert Calibration.from_json(tmp_path / f"calibration_fold{k}.json").is_monotone()
    # calibrations differ between folds (fold-local)
    assert res[0][[f"{k}_target" for k in LABEL_KEYS]].to_numpy().tolist() != res[1][[f"{k}_target" for k in LABEL_KEYS]].to_numpy().tolist()


def test_gold_state_auc_frame():
    t = _table(58)
    g = _gold(t)
    df = gold_state_auc(t, g)
    assert list(df["label"]) == list(LABEL_KEYS) + ["macro"]
    assert df.set_index("label").loc["macro", "auc"] > 0.6
