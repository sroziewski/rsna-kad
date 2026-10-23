#!/usr/bin/env python3
"""data/targets/v3_pseudo_full/full.parquet: the v3 construction for a full-data run.

  python qa/full/make_targets_full.py

label part = targets v1_1's recipe (merged_v1_1 labels, configs/targets_v1_1.yaml) with the empirical-
Bayes calibration fitted on ALL 58 gold rows (a full-data model has no held-out fold whose gold
must stay out of the calibration); target = 0.5 x label + 0.5 x main_v3 teacher OOF on every cell
of every row; weight = the v1_1 weight rule with absent_under_reported 0.1 -> 0.5. Gold rows are
excluded from the table. data/targets/v3_pseudo is not touched.

Self-check before writing: the same code with calibration_fold0.json of v1_1 must reproduce
data/targets/v1_1/fold0.parquet bit-for-bit, so the only difference from v1_1 is the calibration.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS, UNDER_REPORTED_KEYS  # noqa: E402
from rsna_kad.cv import load_folds  # noqa: E402
from rsna_kad.data import load_targets  # noqa: E402
from rsna_kad.labels.pseudo import (V3_MIX, V3_SILENT_WEIGHT, mix_with_teacher,  # noqa: E402
                                    silent_weight_targets, teacher_mean)
from rsna_kad.labels.targets import (Calibration, TargetsConfig, calibrate, gold_frame,  # noqa: E402
                                     make_targets, validate_targets)

TEACHERS = ("runs/main_convnext_s_384", "runs/eff_effv2s_256", "runs/main_armc_convnextv2_t_384")
BASE = ROOT / "data/targets/v1_1"
OUT = ROOT / "data/targets/v3_pseudo_full"
N_ROWS = 4349


def main() -> int:
    man = json.loads((BASE / "manifest.json").read_text())
    cfg = TargetsConfig.load(ROOT / man["config"])
    labels = pd.read_parquet(ROOT / man["labels"])
    labels[ID_COL] = labels[ID_COL].astype(str)
    folds = load_folds(ROOT / "data/folds/folds_v1.csv")
    folds[ID_COL] = folds[ID_COL].astype(str)
    gold_ids = set(folds.loc[folds["is_gold"], ID_COL])
    gold = gold_frame(Path(DATA_DIR) / "train.csv")
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    assert set(gold["StudyInstanceUID"]) == gold_ids and len(gold_ids) == 58

    ref = load_targets(BASE / "fold0.parquet")
    again = make_targets(labels, Calibration.from_json(BASE / "calibration_fold0.json"), cfg, exclude_ids=gold_ids)
    for c in ref.columns:
        assert np.array_equal(ref[c].to_numpy(), again[c].astype(ref[c].dtype).to_numpy()), f"v1_1 recipe drift: {c}"

    calib = calibrate(labels, gold, gold_ids, cfg, fold=None)
    assert calib.is_monotone() and calib.n_gold_train == 58, calib.n_gold_train
    lt = make_targets(labels, calib, cfg, exclude_ids=gold_ids)
    lt[ID_COL] = lt[ID_COL].astype(str)
    rw = silent_weight_targets(labels, calib, cfg, exclude_ids=gold_ids)
    rw[ID_COL] = rw[ID_COL].astype(str)
    teacher = teacher_mean([pd.read_parquet(ROOT / t / "oof.parquet") for t in TEACHERS])
    teacher = teacher[~teacher[ID_COL].isin(gold_ids)].reset_index(drop=True)
    new, cnt = mix_with_teacher(lt, teacher, rw, set(lt[ID_COL]))
    assert cnt == {"rows": N_ROWS, "no_teacher": 0}, cnt

    validate_targets(new, n_rows=N_ROWS)
    assert not set(new[ID_COL]) & gold_ids
    tv = teacher.set_index(ID_COL).loc[new[ID_COL]]
    stats = []
    for k in LABEL_KEYS:
        want = (1 - V3_MIX) * lt[f"{k}_target"].to_numpy(np.float64) + V3_MIX * tv[f"{k}_pred"].to_numpy(np.float64)
        assert np.allclose(new[f"{k}_target"].to_numpy(np.float64), want, atol=1e-6), k
        w0, w1 = lt[f"{k}_weight"].to_numpy(), new[f"{k}_weight"].to_numpy()
        if k not in UNDER_REPORTED_KEYS:
            assert np.array_equal(w0, w1), k
        else:
            assert (w1 >= w0 - 1e-7).all(), k
        stats.append({"key": k, "label_target": float(lt[f"{k}_target"].mean()),
                      "fold0_label_target": float(ref[f"{k}_target"].mean()),
                      "v3_full_target": float(new[f"{k}_target"].mean()), "weight": float(w1.mean())})

    OUT.mkdir(parents=True, exist_ok=True)
    new.to_parquet(OUT / "full.parquet", index=False)
    calib.to_json(OUT / "calibration_full.json")
    (OUT / "manifest.json").write_text(json.dumps({
        "version": "v3_pseudo_full", "base_recipe": "v1_1", "labels": man["labels"], "config": man["config"],
        "calibration": "all 58 gold", "teachers": list(TEACHERS), "mix": V3_MIX,
        "absent_under_reported_weight": V3_SILENT_WEIGHT, "n_rows": N_ROWS,
        "builder": "qa/full/make_targets_full.py"}, indent=1))
    print(pd.DataFrame(stats).round(3).to_string(index=False))
    print(f"OK: {OUT / 'full.parquet'} {len(new)} rows, all mixed, no gold; calibration on {calib.n_gold_train} gold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
