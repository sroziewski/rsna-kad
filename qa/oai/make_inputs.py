"""Training inputs for main_v5_C + OAI (qa/oai, 2026-10-12): targets v3_pseudo_full_oai, folds_v1_oai.csv and the
merged cache root /raid/sroziewski/cache/rsna_kad_oai/v1.

Label rule, fixed before any OAI model was trained (MOAKS grades via qa/oai/moaks_labels.py; area = MOAKS cartilage
area grade 0-3, full = full-thickness grade 0-3, max over the compartment's subregions):
  PF OA, Lateral OA: target 1 (weight 1) if full >= 1 or area == 3; 0.5 (weight 0.5) if area == 2; else 0 (weight 1).
  Synovitis: Hoffa synovitis 2-3 -> 1 (weight 1), 1 -> 0.5 (weight 0.5), 0 -> 0 (weight 1); ungraded -> weight 0.
  Every other finding: weight 0.
Radiology reports call OA on moderate change, so full-thickness loss or >75 % area is "OA", 50-75 % area without a
full-thickness defect is the uncertain band, and mild change counts as negative.

The merged cache keeps the cache version "v1" (bundles record the cache config) and points at the existing v1 series
through symlinks; train with RSNA_KAD_CACHE=/raid/sroziewski/cache/rsna_kad_oai.
"""
import argparse
import json
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/raid/sroziewski/sandbox/rsna_kad")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "qa/oai")]
import moaks_labels as m  # noqa: E402
from rsna_kad.constants import ID_COL, LABEL_KEYS  # noqa: E402

BASE_T = ROOT / "data/targets/v3_pseudo_full"
OUT_T = ROOT / "data/targets/v3_pseudo_full_oai"
FOLDS, OUT_F = ROOT / "data/folds/folds_v1.csv", ROOT / "data/folds/folds_v1_oai.csv"
V1, OAI_C, MERGED = (Path("/raid/sroziewski/cache/rsna_kad/v1"), Path("/raid/sroziewski/cache/rsna_kad/oai_v1"),
                     Path("/raid/sroziewski/cache/rsna_kad_oai/v1"))


def oai_targets(cached):
    d = m.load()
    d["study"] = ["oai_%s_%s" % (i, "R" if s == 1 else "L") for i, s in zip(d.ID, d.SIDE)]
    d = d[d.study.isin(cached)].reset_index(drop=True)
    base = pd.read_parquet(BASE_T / "full.parquet")
    keys = [c[:-7] for c in base.columns if c.endswith("_target")]
    out = pd.DataFrame({ID_COL: d.study})
    for k in keys:
        out[f"{k}_target"], out[f"{k}_weight"] = np.float32(0), np.float32(0)
    for key, regions in (("pf_oa", m.PF), ("lat_oa", m.LAT)):
        area, full = m.cart(d, regions), m.cart(d, regions, True)
        pos, mid, ok = (full >= 1) | (area >= 3), (area == 2) & ~((full >= 1) | (area >= 3)), area.notna()
        out[f"{key}_target"] = np.where(pos, 1.0, np.where(mid, 0.5, 0.0)).astype(np.float32)
        out[f"{key}_weight"] = np.where(~ok, 0.0, np.where(mid, 0.5, 1.0)).astype(np.float32)
    syn = d["V00MSYIC"]
    out["synovitis_target"] = np.where(syn >= 2, 1.0, np.where(syn == 1, 0.5, 0.0)).astype(np.float32)
    out["synovitis_weight"] = np.where(syn.isna(), 0.0, np.where(syn == 1, 0.5, 1.0)).astype(np.float32)
    assert len(keys) == len(LABEL_KEYS)
    return base, out[base.columns]


def merged_cache():
    if MERGED.exists():
        shutil.rmtree(MERGED)
    (MERGED / "train").mkdir(parents=True)
    a, b = pd.read_parquet(V1 / "index.parquet"), pd.read_parquet(OAI_C / "index.parquet")
    assert not set(a[ID_COL]) & set(b[ID_COL])
    for root, df in ((V1, a[a.split == "train"]), (OAI_C, b)):
        for s in df[ID_COL].unique():
            (MERGED / "train" / s).symlink_to(root / "train" / s)
    (MERGED / "test").symlink_to(V1 / "test")
    idx = pd.concat([a, b[[c for c in a.columns if c in b.columns]]], ignore_index=True)
    idx.to_parquet(MERGED / "index.parquet", index=False)
    man = json.loads((V1 / "manifest.json").read_text())
    man["merged_from"] = [str(V1), str(OAI_C)]
    (MERGED / "manifest.json").write_text(json.dumps(man, indent=1))
    return idx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="label stats only")
    a = ap.parse_args()
    oai_idx = pd.read_parquet(OAI_C / "index.parquet")
    cached = set(oai_idx.loc[oai_idx.ok, ID_COL])
    base, t = oai_targets(cached)
    print("OAI studies with cache + MOAKS", len(t))
    for k in ("pf_oa", "lat_oa", "synovitis"):
        w = t[f"{k}_weight"] > 0
        print(f"  {k}: graded {int(w.sum())}, mean target {t.loc[w, f'{k}_target'].mean():.3f}, "
              f"pos {(t[f'{k}_target'] == 1).mean():.3f} half {(t[f'{k}_target'] == 0.5).mean():.3f}; "
              f"competition mean target {base[f'{k}_target'].mean():.3f}")
    if a.report:
        sys.exit(0)
    OUT_T.mkdir(parents=True, exist_ok=True)
    pd.concat([base, t], ignore_index=True).to_parquet(OUT_T / "full.parquet", index=False)
    shutil.copy(BASE_T / "calibration_full.json", OUT_T / "calibration_full.json")
    man = json.loads((BASE_T / "manifest.json").read_text())
    man.update(version=OUT_T.name, n_rows=len(base) + len(t), oai_rows=len(t), builder="qa/oai/make_inputs.py",
               oai_rule=__doc__.split("Label rule")[1].split("The merged cache")[0].strip())
    (OUT_T / "manifest.json").write_text(json.dumps(man, indent=1))
    f = pd.read_csv(FOLDS, dtype=str)
    subj = t[ID_COL].str.split("_").str[1]
    oai_f = pd.DataFrame({ID_COL: t[ID_COL], "fold": [zlib.crc32(s.encode()) % 5 for s in subj],
                          "site_key": "OAI", "lang": "oai", "is_gold": "False"})
    pd.concat([f, oai_f.astype(str)], ignore_index=True).sort_values(ID_COL).to_csv(OUT_F, index=False, lineterminator="\n")
    idx = merged_cache()
    print("targets", len(base) + len(t), "folds", len(f) + len(oai_f), "merged index rows", len(idx),
          "train studies", idx[idx.split == "train"][ID_COL].nunique())
