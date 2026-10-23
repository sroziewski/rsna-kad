import numpy as np
import pandas as pd
import pytest

from rsna_kad.constants import ID_COL, LABEL_KEYS, UNDER_REPORTED_KEYS
from rsna_kad.labels.pseudo import PSEUDO_MIX, PSEUDO_WEIGHT, apply_pseudo, build_fold_targets

UNDER = tuple(k for k in LABEL_KEYS if k in UNDER_REPORTED_KEYS)
WELL = tuple(k for k in LABEL_KEYS if k not in UNDER_REPORTED_KEYS)


def _targets(n: int, seed: int = 0) -> pd.DataFrame:
    """v1-shaped table: the under-reported keys carry the weight-0.1 'silent' cells."""
    rng = np.random.default_rng(seed)
    out = pd.DataFrame({ID_COL: [f"s{i}" for i in range(n)]})
    for k in LABEL_KEYS:
        out[f"{k}_target"] = rng.uniform(0.02, 0.95, n).astype(np.float32)
        w = np.where(rng.random(n) < 0.5, 1.0, 0.5)
        if k in UNDER:
            w = np.where(rng.random(n) < 0.6, 0.1, w)
        out[f"{k}_weight"] = w.astype(np.float32)
    return out


def _pred(ids, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = pd.DataFrame({ID_COL: list(ids)})
    for k in LABEL_KEYS:
        out[f"{k}_pred"] = rng.uniform(0, 1, len(out)).astype(np.float32)
    return out


def test_only_silent_cells_of_under_reported_keys_move():
    t = _targets(200)
    p = _pred(t[ID_COL])
    out, counts = apply_pseudo(t, p)
    for k in WELL:
        pd.testing.assert_series_equal(out[f"{k}_target"], t[f"{k}_target"])
        pd.testing.assert_series_equal(out[f"{k}_weight"], t[f"{k}_weight"])
    for k in UNDER:
        moved = t[f"{k}_weight"].to_numpy() <= 0.15
        assert counts[k] == int(moved.sum()) > 0
        assert np.allclose(out.loc[~moved, f"{k}_target"], t.loc[~moved, f"{k}_target"])
        assert np.allclose(out.loc[~moved, f"{k}_weight"], t.loc[~moved, f"{k}_weight"])
        assert np.allclose(out.loc[moved, f"{k}_weight"], PSEUDO_WEIGHT)
        want = (1 - PSEUDO_MIX) * t.loc[moved, f"{k}_target"] + PSEUDO_MIX * p.loc[moved, f"{k}_pred"]
        assert np.allclose(out.loc[moved, f"{k}_target"], want.astype(np.float32), atol=1e-6)


def test_study_without_a_prediction_is_untouched():
    t = _targets(50)
    p = _pred(t[ID_COL][:20])
    out, _ = apply_pseudo(t, p)
    tail = slice(20, None)
    for k in LABEL_KEYS:
        assert np.allclose(out[f"{k}_target"][tail], t[f"{k}_target"][tail])
        assert np.allclose(out[f"{k}_weight"][tail], t[f"{k}_weight"][tail])


def test_build_fold_targets_leaves_the_held_out_fold_bit_identical(tmp_path):
    t = _targets(150)
    folds = pd.DataFrame({ID_COL: t[ID_COL], "fold": np.arange(len(t)) % 5})
    out, counts = build_fold_targets(2, t, folds, _pred(t[ID_COL]), tmp_path)
    held = folds["fold"].to_numpy() == 2
    for k in LABEL_KEYS:
        assert (out.loc[held, f"{k}_target"].to_numpy() == t.loc[held, f"{k}_target"].to_numpy()).all()
        assert (out.loc[held, f"{k}_weight"].to_numpy() == t.loc[held, f"{k}_weight"].to_numpy()).all()
    assert sum(counts.values()) > 0
    written = pd.read_parquet(tmp_path / "fold2.parquet")
    assert len(written) == len(t)
    assert written.dtypes[f"{UNDER[0]}_target"] == np.float32


@pytest.mark.parametrize("key", UNDER)
def test_values_stay_in_range(key):
    t = _targets(80, seed=3)
    out, _ = apply_pseudo(t, _pred(t[ID_COL], seed=4))
    v = out[[f"{key}_target", f"{key}_weight"]].to_numpy()
    assert np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()


# ------------------------------------------------------------------ v3: teacher mix on every cell

from rsna_kad.labels.pseudo import (V3_MIX, V3_SILENT_WEIGHT, mix_with_teacher,  # noqa: E402
                                    silent_weight_targets, teacher_mean)
from rsna_kad.labels.targets import calibrate, make_targets  # noqa: E402
from test_targets import _cfg, _gold, _table  # noqa: E402


def _v3_inputs(n=120, seed=0):
    table = _table(n, seed=seed, variant2=True)
    table.loc[3, "parse_ok"] = False
    cfg = _cfg()
    calib = calibrate(table, _gold(table), set(table[ID_COL][: n // 2]), cfg)
    return table, cfg, calib, make_targets(table, calib, cfg)


def test_teacher_mean_is_the_mean_of_probabilities_on_common_studies():
    a, b = _pred([f"s{i}" for i in range(10)], seed=1), _pred([f"s{i}" for i in range(2, 12)], seed=2)
    t = teacher_mean([a, b])
    assert list(t[ID_COL]) == sorted(f"s{i}" for i in range(2, 10))
    k = f"{LABEL_KEYS[0]}_pred"
    want = (a.set_index(ID_COL).loc[t[ID_COL], k].to_numpy(np.float64)
            + b.set_index(ID_COL).loc[t[ID_COL], k].to_numpy(np.float64)) / 2
    assert np.allclose(t[k], want)
    bad = a.copy()
    bad[k] = bad[k] * 5 - 2   # logits, not probabilities
    with pytest.raises(ValueError):
        teacher_mean([bad])


def test_silent_weight_raises_only_under_reported_absent_cells():
    table, cfg, calib, lt = _v3_inputs()
    rw = silent_weight_targets(table, calib, cfg)
    assert cfg.weights["absent_under_reported"] == 0.1          # the caller's config is not mutated
    for k in LABEL_KEYS:
        assert np.array_equal(rw[f"{k}_target"].to_numpy(), lt[f"{k}_target"].to_numpy())
        w0, w1 = lt[f"{k}_weight"].to_numpy(), rw[f"{k}_weight"].to_numpy()
        silent = (table[f"{k}_state"].astype(str) == "absent").to_numpy() & table["parse_ok"].to_numpy(bool)
        if k in UNDER_REPORTED_KEYS:
            # 0.1 -> 0.5 is x5 under the disagreement factor, unless the hallucination cap binds
            ok = table[f"{k}_evidence_ok"].to_numpy(bool)
            want = w0 * V3_SILENT_WEIGHT / 0.1
            want = np.where(ok, want, np.minimum(want, cfg.weights["hallucinated_evidence"]))
            assert np.allclose(w1[silent], want[silent], atol=1e-6)
            assert (w1[silent] > w0[silent]).all()
            assert np.array_equal(w1[~silent], w0[~silent])
        else:
            assert np.array_equal(w0, w1)
    assert all(rw.loc[3, f"{k}_weight"] == pytest.approx(cfg.weights["parse_fail"]) for k in LABEL_KEYS)


def test_mix_with_teacher_mixes_every_train_cell_and_freezes_the_rest():
    table, cfg, calib, lt = _v3_inputs()
    rw = silent_weight_targets(table, calib, cfg)
    ids = lt[ID_COL].astype(str)
    teacher = _pred(ids[:-5], seed=7)                            # last 5 studies have no teacher
    train = set(ids[::2])
    out, cnt = mix_with_teacher(lt, teacher, rw, train)
    has = ids.isin(set(teacher[ID_COL])).to_numpy()
    m = ids.isin(train).to_numpy() & has
    assert cnt == {"rows": int(m.sum()), "no_teacher": int((ids.isin(train).to_numpy() & ~has).sum())}
    tv = teacher.set_index(ID_COL).reindex(ids)
    for k in LABEL_KEYS:
        want = (1 - V3_MIX) * lt[f"{k}_target"].to_numpy(np.float64) + V3_MIX * tv[f"{k}_pred"].to_numpy(np.float64)
        assert np.allclose(out.loc[m, f"{k}_target"], want[m], atol=1e-6)
        assert np.array_equal(out.loc[m, f"{k}_weight"].to_numpy(), rw.loc[m, f"{k}_weight"].to_numpy())
        for f in ("target", "weight"):
            assert np.array_equal(out.loc[~m, f"{k}_{f}"].to_numpy(), lt.loc[~m, f"{k}_{f}"].to_numpy())
            assert out[f"{k}_{f}"].dtype == np.float32


def test_mix_with_teacher_rejects_a_reweighted_table_with_other_targets():
    table, cfg, calib, lt = _v3_inputs()
    rw = silent_weight_targets(table, calib, cfg)
    rw[f"{LABEL_KEYS[0]}_target"] = rw[f"{LABEL_KEYS[0]}_target"] * np.float32(0.9)
    with pytest.raises(ValueError):
        mix_with_teacher(lt, _pred(lt[ID_COL]), rw, set(lt[ID_COL]))
    with pytest.raises(ValueError):
        mix_with_teacher(lt, _pred(lt[ID_COL]), rw.iloc[::-1].reset_index(drop=True), set(lt[ID_COL]))
