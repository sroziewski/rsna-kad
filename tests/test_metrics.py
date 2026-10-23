import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from rsna_kad.constants import LABEL_KEYS
from rsna_kad.metrics import (AGREE_ALL3, AGREE_LLM2, AGREE_SPLIT, Subset, _rank_auc, auc,
                              binarise, bootstrap_macro_auc, bootstrap_subset_auc, ci95,
                              consensus_subset, evidence_mask, gold_report, gold_subset,
                              macro_auc, paired_delta, strict_subset, subset_auc, voter_codes,
                              weak_subset, weighted_arbiter)


def test_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 50)
        if y.min() == y.max():
            continue
        s = np.round(rng.random(50), 1)  # many ties
        assert auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_auc_single_class_is_nan():
    assert np.isnan(auc(np.ones(5), np.arange(5)))


def test_binarise_drops_middle_band_and_low_weight_negatives():
    y = np.array([0.9, 0.6, 0.5, 0.3, 0.2, 0.1, 0.0])
    w = np.array([1.0, 1.0, 1.0, 1.0, 0.4, 0.6, 0.1])
    keep, pos = binarise(y, w)
    assert keep.tolist() == [True, True, False, False, False, True, False]
    assert pos[keep].tolist() == [True, True, False]


def test_binarise_float32_targets_exactly_on_thresholds():
    # 0.2 and 0.6 stored as float32 are not exactly 0.2 / 0.6 in float64
    y = np.array([0.2, 0.6, 0.5], dtype=np.float32)
    w = np.array([0.5, 1.0, 1.0], dtype=np.float32)
    keep, pos = binarise(y, w)
    assert keep.tolist() == [True, True, False]
    assert pos.tolist() == [False, True, False]
    keep64, pos64 = binarise(y.astype(np.float64), w.astype(np.float64))
    assert keep64.tolist() == keep.tolist() and pos64.tolist() == pos.tolist()


def test_macro_auc_against_sklearn_on_toy_case():
    rng = np.random.default_rng(1)
    n = 200
    y_soft = rng.random((n, 12))
    weight = rng.random((n, 12))
    pred = y_soft + rng.normal(0, 0.3, (n, 12))
    macro, per_label = macro_auc(y_soft, pred, weight)
    expected = []
    for j, key in enumerate(LABEL_KEYS):
        keep = (y_soft[:, j] >= 0.6) | ((y_soft[:, j] <= 0.2) & (weight[:, j] >= 0.5))
        ref = roc_auc_score((y_soft[keep, j] >= 0.6).astype(int), pred[keep, j])
        assert per_label[key] == pytest.approx(ref, abs=1e-12)
        expected.append(ref)
    assert macro == pytest.approx(np.mean(expected), abs=1e-12)


def test_macro_auc_skips_single_class_labels():
    y = np.zeros((10, 2))
    y[:, 0] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    pred = np.tile(np.arange(10)[:, None], (1, 2)).astype(float)
    macro, per = macro_auc(y, pred, None, keys=("a", "b"))
    assert per["a"] == 1.0 and np.isnan(per["b"]) and macro == 1.0


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, (58, 12)).astype(float)
    p = y + rng.normal(0, 0.8, y.shape)
    m, _ = macro_auc(y, p, None, pos_thr=0.5, neg_thr=0.5, neg_min_weight=0.0)
    lo, hi = bootstrap_macro_auc(y, p, n_boot=200, seed=0)
    assert lo <= m <= hi and hi - lo < 0.3


def test_gold_report_per_label_and_language():
    rng = np.random.default_rng(3)
    ids = [f"s{i}" for i in range(58)]
    gold = pd.DataFrame({"StudyInstanceUID": ids, **{k: rng.integers(0, 2, 58) for k in LABEL_KEYS}})
    pred = pd.DataFrame({"StudyInstanceUID": ids, "lang": ["en"] * 30 + ["es"] * 20 + ["tr"] * 8,
                         **{f"{k}_pred": gold[k] * 0.8 + rng.random(58) * 0.4 for k in LABEL_KEYS}})
    rep = gold_report(pred, gold, n_boot=50)
    assert rep["n"] == 58 and 0.5 < rep["macro_auc"] <= 1.0
    assert set(rep["per_label"]) == set(LABEL_KEYS)
    assert set(rep["per_lang"]) == {"en", "es", "tr"} and rep["per_lang"]["tr"]["n"] == 8
    assert rep["ci95"][0] <= rep["macro_auc"] <= rep["ci95"][1]


# --------------------------------------------------------------------- arbiter subsets


def _label_table(n: int = 40, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    states = np.array(["affirmed", "hedged", "negated", "absent"])
    cols = {"StudyInstanceUID": [f"s{i:03d}" for i in range(n)]}
    for k in LABEL_KEYS:
        cols[f"{k}_state"] = rng.choice(states, n)
        cols[f"{k}_state_v2"] = np.where(rng.random(n) < 0.7, cols[f"{k}_state"], rng.choice(states, n))
        cols[f"{k}_rules_state"] = np.where(rng.random(n) < 0.6, cols[f"{k}_state_v2"],
                                            rng.choice(states[[0, 2, 3]], n))
        cols[f"{k}_severity"] = "unspecified"
        cols[f"{k}_evidence_ok"] = rng.random(n) > 0.1
        cols[f"{k}_mention_conflict"] = rng.random(n) < 0.05
    return pd.DataFrame(cols)


def _preds(ids, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"StudyInstanceUID": list(ids),
                         **{f"{k}_pred": rng.random(len(ids)) for k in LABEL_KEYS}})


def test_rank_auc_matches_auc_including_ties():
    rng = np.random.default_rng(11)
    for _ in range(30):
        y = rng.integers(0, 2, 120).astype(bool)
        s = np.round(rng.random(120), 1)  # heavy ties
        if y.all() or not y.any():
            continue
        assert _rank_auc(y, s) == pytest.approx(auc(y, s), abs=1e-12)
    assert np.isnan(_rank_auc(np.ones(5, bool), np.arange(5.0)))


def test_voter_codes_classify_the_three_voters():
    lab = pd.DataFrame({
        "StudyInstanceUID": ["a", "b", "c", "d"],
        "acl_state": ["affirmed", "affirmed", "negated", "hedged"],
        "acl_state_v2": ["affirmed", "affirmed", "absent", "hedged"],
        "acl_rules_state": ["affirmed", "negated", "negated", "hedged"],
    })
    assert voter_codes(lab, ("acl",))[:, 0].tolist() == [AGREE_ALL3, AGREE_LLM2, AGREE_SPLIT, AGREE_ALL3]


def test_consensus_subset_keeps_only_unanimous_affirmed_or_negated():
    lab = pd.DataFrame({
        "StudyInstanceUID": list("abcde"),
        # unanimous affirmed / unanimous negated / unanimous absent / LLM-only agreement / split
        "acl_state": ["affirmed", "negated", "absent", "affirmed", "affirmed"],
        "acl_state_v2": ["affirmed", "negated", "absent", "affirmed", "negated"],
        "acl_rules_state": ["affirmed", "negated", "absent", "negated", "affirmed"],
    })
    sub = consensus_subset(lab, ("acl",))
    assert sub.keep[:, 0].tolist() == [True, True, False, False, False]
    assert sub.y[sub.keep[:, 0], 0].tolist() == [True, False]
    assert sub.coverage == 2 and sub.per_label_coverage()["acl"] == {"n": 2, "n_pos": 1, "n_neg": 1}


def test_consensus_absent_counts_only_for_well_reported_labels():
    lab = pd.DataFrame({"StudyInstanceUID": ["a"],
                        **{f"{k}_{c}": ["absent"] for k in ("acl", "synovitis")
                           for c in ("state", "state_v2", "rules_state")}})
    sub = consensus_subset(lab, ("acl", "synovitis"), include_agreed_absent=True)
    # acl is well reported (unanimous silence == negative), synovitis is not (§2.1).
    assert sub.keep[0].tolist() == [True, False]
    assert not sub.y[0, 0]
    assert consensus_subset(lab, ("acl", "synovitis")).coverage == 0


def test_strict_subset_is_affirmed_vs_negated_of_the_primary_voter():
    lab = _label_table()
    sub = strict_subset(lab)
    st = lab["acl_state"].to_numpy()
    assert sub.keep[:, 0].tolist() == ((st == "affirmed") | (st == "negated")).tolist()
    assert sub.y[:, 0].tolist() == (st == "affirmed").tolist()


def test_weak_subset_reproduces_binarise_and_macro_auc():
    rng = np.random.default_rng(12)
    n = 300
    t = {"StudyInstanceUID": [f"s{i:03d}" for i in range(n)]}
    for k in LABEL_KEYS:
        t[f"{k}_target"] = rng.random(n).astype(np.float32)
        t[f"{k}_weight"] = rng.random(n).astype(np.float32)
    targets = pd.DataFrame(t)
    pred = _preds(targets["StudyInstanceUID"])
    sub = weak_subset(targets)
    y = targets[[f"{k}_target" for k in LABEL_KEYS]].to_numpy()
    w = targets[[f"{k}_weight" for k in LABEL_KEYS]].to_numpy()
    p = sub.align(pred)
    macro_ref, per_ref = macro_auc(y, p, w)
    macro, per = subset_auc(sub, p)
    assert macro == pytest.approx(macro_ref, abs=1e-12)
    for k in LABEL_KEYS:
        assert per[k] == pytest.approx(per_ref[k], abs=1e-12)


def test_evidence_mask_and_restrict_only_remove_cells():
    lab = _label_table()
    ev = evidence_mask(lab)
    base = strict_subset(lab)
    narrowed = base.restrict(ev, "strict_ev")
    assert narrowed.name == "strict_ev" and narrowed.coverage <= base.coverage
    assert (narrowed.keep <= base.keep).all() and (narrowed.y == base.y).all()
    assert ev.sum() < ev.size  # the fixture does contain flagged cells


def test_subset_align_orders_predictions_by_subset_ids_and_rejects_gaps():
    lab = _label_table(12)
    sub = strict_subset(lab)
    pred = _preds(lab["StudyInstanceUID"][::-1])
    aligned = sub.align(pred)
    ref = pred.set_index("StudyInstanceUID").loc[sub.ids, [f"{k}_pred" for k in LABEL_KEYS]].to_numpy()
    assert np.array_equal(aligned, ref)
    with pytest.raises(KeyError):
        sub.align(pred.iloc[2:])


def test_gold_subset_keeps_every_cell():
    rng = np.random.default_rng(13)
    gold = pd.DataFrame({"StudyInstanceUID": [f"g{i}" for i in range(58)],
                         **{k: rng.integers(0, 2, 58).astype(float) for k in LABEL_KEYS}})
    sub = gold_subset(gold)
    assert sub.keep.all() and sub.coverage == 58 * 12
    macro, _ = subset_auc(sub, sub.align(_preds(gold["StudyInstanceUID"])))
    assert 0.0 <= macro <= 1.0


def test_paired_bootstrap_is_paired_and_brackets_the_point_estimate():
    lab = _label_table(200, seed=14)
    sub = strict_subset(lab)
    rng = np.random.default_rng(15)
    good = np.where(sub.y, 0.7, 0.3) + rng.normal(0, 0.25, sub.y.shape)
    bad = rng.random(sub.y.shape)
    m_good, _ = subset_auc(sub, good)
    m_bad, _ = subset_auc(sub, bad)
    boots = bootstrap_subset_auc(sub, {"good": good, "bad": bad}, n_boot=200, seed=3)
    lo, hi = ci95(boots["good"])
    assert lo <= m_good <= hi
    d = paired_delta(boots["good"], boots["bad"])
    assert d["mean"] == pytest.approx(m_good - m_bad, abs=0.03)
    assert d["p_gt"] > 0.95 and d["ci_lo"] > 0
    assert d["half_width"] == pytest.approx((d["ci_hi"] - d["ci_lo"]) / 2, abs=1e-12)
    # A candidate compared with itself has a degenerate delta - the pairing is real.
    same = paired_delta(boots["good"], boots["good"])
    assert same["mean"] == 0.0 and same["half_width"] == 0.0


def test_bootstrap_subset_auc_shares_rows_across_candidates():
    lab = _label_table(60, seed=16)
    sub = strict_subset(lab)
    rng = np.random.default_rng(17)
    a = rng.random(sub.y.shape)
    both = bootstrap_subset_auc(sub, {"a": a, "b": a.copy()}, n_boot=50, seed=5)
    assert np.allclose(both["a"], both["b"], equal_nan=True)


def test_weighted_arbiter_ignores_missing_and_nan_parts():
    assert weighted_arbiter({"a": 0.9, "b": 0.8}, {"a": 1.0, "b": 1.0}) == pytest.approx(0.85)
    assert weighted_arbiter({"a": 0.9, "b": float("nan")}, {"a": 0.5, "b": 0.5}) == pytest.approx(0.9)
    assert weighted_arbiter({"a": 0.9}, {"a": 0.5, "c": 0.5}) == pytest.approx(0.9)
    assert np.isnan(weighted_arbiter({}, {"a": 1.0}))


def test_subset_auc_skips_labels_without_both_classes():
    ids = ["a", "b", "c"]
    keep = np.ones((3, 2), bool)
    y = np.array([[True, True], [False, True], [True, True]])
    sub = Subset("t", np.array(ids), keep, y, ("x", "z"))
    macro, per = subset_auc(sub, np.array([[0.9, 0.1], [0.1, 0.5], [0.8, 0.2]]))
    assert per["x"] == 1.0 and np.isnan(per["z"]) and macro == 1.0
