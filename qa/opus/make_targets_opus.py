#!/usr/bin/env python3
"""data/targets/v3_pseudo_full_opus/full.parquet: v3_pseudo_full with ONLY the label part replaced (2026-10-15).

Label part = Qile Jiang's Opus Blend soft labels (.tmp/qile/in/labels_opus_blend.csv, team-internal; values in
[0.05, 0.95]) instead of our calibrated merged_v1_1 report targets: on gold-58 the tables alone score 0.936 vs 0.873
(qa/opus/label_gold.txt). Teacher (main_v3 OOF mean), 0.5 mix and gold exclusion are v3_pseudo_full's; v3 is rebuilt
here first and compared byte-for-byte. Weight = max(0.1, Opus confidence) on every cell (his README's rule): our
weight rule grades agreement between our own voters, which says nothing about his labels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS  # noqa: E402
from rsna_kad.cv import load_folds  # noqa: E402
from rsna_kad.data import load_targets  # noqa: E402
from rsna_kad.labels.pseudo import mix_with_teacher, silent_weight_targets, teacher_mean  # noqa: E402
from rsna_kad.labels.targets import Calibration, TargetsConfig, make_targets, validate_targets  # noqa: E402

OPUS = ROOT / ".tmp/qile/in/labels_opus_blend.csv"
V3 = ROOT / "data/targets/v3_pseudo_full"
OUT = ROOT / "data/targets/v3_pseudo_full_opus"
N_ROWS = 4349
CONF_FLOOR = 0.1


def main() -> int:
    man = json.loads((V3 / "manifest.json").read_text())
    cfg = TargetsConfig.load(ROOT / man["config"])
    labels = pd.read_parquet(ROOT / man["labels"])
    labels[ID_COL] = labels[ID_COL].astype(str)
    folds = load_folds(ROOT / "data/folds/folds_v1.csv")
    folds[ID_COL] = folds[ID_COL].astype(str)
    gold_ids = set(folds.loc[folds["is_gold"], ID_COL])
    calib = Calibration.from_json(V3 / "calibration_full.json")
    lt = make_targets(labels, calib, cfg, exclude_ids=gold_ids)
    lt[ID_COL] = lt[ID_COL].astype(str)
    rw = silent_weight_targets(labels, calib, cfg, exclude_ids=gold_ids)
    rw[ID_COL] = rw[ID_COL].astype(str)
    teacher = teacher_mean([pd.read_parquet(ROOT / t / "oof.parquet") for t in man["teachers"]])
    teacher = teacher[~teacher[ID_COL].isin(gold_ids)].reset_index(drop=True)
    v3, cnt = mix_with_teacher(lt, teacher, rw, set(lt[ID_COL]))
    assert cnt == {"rows": N_ROWS, "no_teacher": 0}, cnt
    ref = load_targets(V3 / "full.parquet")
    for c in ref.columns:
        assert np.array_equal(ref[c].to_numpy(), v3[c].astype(ref[c].dtype).to_numpy()), f"v3 rebuild drift: {c}"

    cols = list(pd.read_csv(Path(DATA_DIR) / "sample_submission.csv", nrows=1).columns[1:])
    assert len(cols) == len(LABEL_KEYS)
    op = pd.read_csv(OPUS, dtype={ID_COL: str}).set_index(ID_COL)
    assert len(op) == 4407 and op.index.is_unique
    op = op.reindex(lt[ID_COL])
    assert op[cols].notna().all().all(), "Opus table misses a training study"
    lo = lt.copy()
    for k, c in zip(LABEL_KEYS, cols):
        lo[f"{k}_target"] = op[c].to_numpy(np.float32)
        lo[f"{k}_weight"] = np.maximum(CONF_FLOOR, op[f"{c}_confidence"].to_numpy(np.float64)).astype(np.float32)
    # mix_with_teacher takes its weights from the re-weighted twin, which must share the targets: here it is the
    # same Opus table, so every weight is the confidence rule
    new, cnt = mix_with_teacher(lo, teacher, lo, set(lo[ID_COL]))
    assert cnt == {"rows": N_ROWS, "no_teacher": 0}, cnt
    new = new[ref.columns]
    for c in ref.columns:
        new[c] = new[c].astype(ref[c].dtype)
    validate_targets(new, n_rows=N_ROWS)
    assert not set(new[ID_COL].astype(str)) & gold_ids
    assert np.array_equal(ref[ID_COL].astype(str).to_numpy(), new[ID_COL].astype(str).to_numpy())
    stats = []
    for k in LABEL_KEYS:
        t0, t1 = ref[f"{k}_target"].to_numpy(np.float64), new[f"{k}_target"].to_numpy(np.float64)
        w0, w1 = ref[f"{k}_weight"].to_numpy(np.float64), new[f"{k}_weight"].to_numpy(np.float64)
        stats.append({"key": k, "target_before": t0.mean(), "target_after": t1.mean(),
                      "mean_abs_change": float(np.abs(t1 - t0).mean()),
                      "spearman_before_after": float(pd.Series(t0).corr(pd.Series(t1), method="spearman")),
                      "weight_before": w0.mean(), "weight_after": w1.mean()})
    OUT.mkdir(parents=True, exist_ok=True)
    new.to_parquet(OUT / "full.parquet", index=False)
    (OUT / "calibration_full.json").write_bytes((V3 / "calibration_full.json").read_bytes())
    (OUT / "manifest.json").write_text(json.dumps({
        **man, "version": "v3_pseudo_full_opus", "derived_from": "data/targets/v3_pseudo_full",
        "label_part": str(OPUS.relative_to(ROOT)), "weight": f"max({CONF_FLOOR}, opus confidence)",
        "builder": "qa/opus/make_targets_opus.py"}, indent=1))
    st = pd.DataFrame(stats)
    st.to_csv(ROOT / "qa/opus/target_stats.csv", index=False)
    print(st.round(3).to_string(index=False))
    print(f"OK {OUT / 'full.parquet'}: {new.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
