"""Training inputs for the C leg + Andrey's external knees (qa/ext, 2026-10-16): targets <base>_ext, folds_v1_ext.csv
and the merged cache root <cache>/rsna_kad_ext/v1 (OAI recipe, qa/oai/make_inputs.py).

Label rule, fixed before any model was trained: competition rows stay exactly the base table (our labels; Andrey's
own competition labels are a different source and would pull the leg toward his model). External rows (mrnet-*,
kneemri-*) that have every series cached: target = his pseudo-label clipped to [0, 1], weight = his cell weight
(0/1; MCL and Baker's are 0 for all external studies in his file). Folds: crc32(exam) % 5, so the series of one
KneeMRI exam share a fold; full-data training ignores folds apart from membership.
    python qa/ext/make_inputs_ext.py --labels L.csv --cellw W.csv --ext-cache <convert_ext.py --out>
"""
import argparse
import json
import shutil
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rsna_kad.constants import ID_COL, LABEL_KEYS, LABEL_TO_KEY  # noqa: E402

EXT_PREFIXES = ("mrnet-", "kneemri-")


def exam_of(sid):
    return sid.rsplit("-", 1)[0] if sid.startswith("kneemri-") else sid


def ext_targets(labels, cellw, cached, base_cols):
    lab = pd.read_csv(labels, dtype={ID_COL: str}).set_index(ID_COL)
    w = pd.read_csv(cellw, dtype={ID_COL: str}).set_index(ID_COL)
    for name, df in (("labels", lab), ("cellw", w)):
        if df.index.duplicated().any():
            raise SystemExit(f"{name}: repeated StudyInstanceUID, e.g. {df.index[df.index.duplicated()][:3].tolist()}")
    ext = [s for s in lab.index if s.startswith(EXT_PREFIXES)]
    missing_w = sorted(set(ext) - set(w.index))
    if missing_w:
        raise SystemExit(f"{len(missing_w)} external IDs have no cell weights, e.g. {missing_w[:3]}")
    keep = [s for s in ext if s in cached]
    out = pd.DataFrame({ID_COL: keep})
    for col in lab.columns:
        key = LABEL_TO_KEY[col]
        t, ww = lab.loc[keep, col].to_numpy(np.float64), w.loc[keep, col].to_numpy(np.float64)
        if np.isnan(t).any() or np.isnan(ww).any() or ((ww < 0) | (ww > 1)).any():
            raise SystemExit(f"{col}: NaN or weight outside [0, 1]")
        out[f"{key}_target"] = np.clip(t, 0, 1).astype(np.float32)
        out[f"{key}_weight"] = ww.astype(np.float32)
    assert sorted(c[:-7] for c in out.columns if c.endswith("_target")) == sorted(LABEL_KEYS)
    return out[base_cols], len(ext)


def check_params(v1, ext_cache):
    man = json.loads((v1 / "manifest.json").read_text())
    ext_man = json.loads((ext_cache / "manifest.json").read_text())
    if man["params_sha256"] != ext_man["params_sha256"]:
        raise SystemExit("external cache built with other cache parameters")
    return man


def merged_cache(v1, ext_cache, merged, keep):
    # symlink targets must be absolute: a relative one resolves against the link's own folder
    v1, ext_cache, merged = v1.resolve(), ext_cache.resolve(), merged.resolve()
    man = check_params(v1, ext_cache)
    for src in (v1, ext_cache):
        if merged == src or merged in src.parents or src in merged.parents:
            raise SystemExit(f"--merged {merged} overlaps the source cache {src}")
    if merged.exists():
        # only ever delete a folder this script made: the merged root is replaced wholesale on each run
        mf = merged / "manifest.json"
        ours = mf.exists() and "merged_from" in json.loads(mf.read_text())
        if not ours and any(merged.iterdir()):
            raise SystemExit(f"--merged {merged} exists and is not a merged cache; refusing to delete it")
        shutil.rmtree(merged)
    (merged / "train").mkdir(parents=True)
    a, b = pd.read_parquet(v1 / "index.parquet"), pd.read_parquet(ext_cache / "index.parquet")
    if set(a[ID_COL]) & set(b[ID_COL]):
        raise SystemExit("external cache shares study IDs with the v1 cache")
    # only the studies that made it into the targets: an error row has NaN counts, which would turn the merged
    # index's int/bool columns into float/object, and a study without a folder would become a dangling link
    b = b[b[ID_COL].isin(keep)]
    b = b[[c for c in a.columns if c in b.columns]].astype({c: a[c].dtype for c in a.columns if c in b.columns})
    for root, df in ((v1, a[a.split == "train"]), (ext_cache, b)):
        for s in df[ID_COL].unique():
            (merged / "train" / s).symlink_to(root / "train" / s)
    (merged / "test").symlink_to(v1 / "test")
    idx = pd.concat([a, b], ignore_index=True)
    idx.to_parquet(merged / "index.parquet", index=False)
    man["merged_from"] = [str(v1), str(ext_cache)]
    (merged / "manifest.json").write_text(json.dumps(man, indent=1))
    return idx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cellw", required=True)
    ap.add_argument("--ext-cache", required=True)
    ap.add_argument("--base-targets", default=str(ROOT / "data/targets/v3_pseudo_full"))
    ap.add_argument("--out-targets", default=str(ROOT / "data/targets/v3_pseudo_full_ext"))
    ap.add_argument("--folds", default=str(ROOT / "data/folds/folds_v1.csv"))
    ap.add_argument("--out-folds", default=str(ROOT / "data/folds/folds_v1_ext.csv"))
    ap.add_argument("--v1-cache", default="/raid/sroziewski/cache/rsna_kad/v1")
    ap.add_argument("--merged", default="/raid/sroziewski/cache/rsna_kad_ext/v1")
    a = ap.parse_args(argv)
    base_t, out_t = Path(a.base_targets), Path(a.out_targets)
    check_params(Path(a.v1_cache), Path(a.ext_cache))
    base = pd.read_parquet(base_t / "full.parquet")
    ext_idx = pd.read_parquet(Path(a.ext_cache) / "index.parquet")
    per_study = ext_idx.groupby(ID_COL)["ok"].all()
    cached = set(per_study[per_study].index)
    t, n_ext = ext_targets(a.labels, a.cellw, cached, list(base.columns))
    folds = pd.read_csv(a.folds, dtype=str)
    gold = set(folds.loc[folds.is_gold.str.lower().isin(["true", "1"]), ID_COL])
    if not gold:
        raise SystemExit(f"{a.folds}: no gold rows found in is_gold")
    if set(t[ID_COL]) & (set(base[ID_COL]) | gold | set(folds[ID_COL])):
        raise SystemExit("an external ID collides with a competition/gold/fold ID")
    print(f"external IDs in labels {n_ext}, fully cached {len(cached)}, used {len(t)}")
    for src in EXT_PREFIXES:
        s = t[t[ID_COL].str.startswith(src)]
        print(f"  {src[:-1]}: {len(s)} studies; mean target " + " ".join(
            f"{k} {s[f'{k}_target'].mean():.2f}" for k in LABEL_KEYS) + "; weighted cells/study "
            f"{sum((s[f'{k}_weight'] > 0).sum() for k in LABEL_KEYS) / max(len(s), 1):.1f}")
    out_t.mkdir(parents=True, exist_ok=True)
    pd.concat([base, t], ignore_index=True).to_parquet(out_t / "full.parquet", index=False)
    for extra in ("calibration_full.json",):
        if (base_t / extra).exists():
            shutil.copy(base_t / extra, out_t / extra)
    man = json.loads((base_t / "manifest.json").read_text())
    man.update(version=out_t.name, n_rows=len(base) + len(t), ext_rows=len(t), builder="qa/ext/make_inputs_ext.py",
               ext_labels=str(a.labels), ext_cellw=str(a.cellw),
               ext_rule=__doc__.split("Label rule")[1].split("python qa")[0].strip())
    (out_t / "manifest.json").write_text(json.dumps(man, indent=1))
    src_key = t[ID_COL].str.split("-").str[0].str.upper()
    ext_f = pd.DataFrame({ID_COL: t[ID_COL], "fold": [zlib.crc32(exam_of(s).encode()) % 5 for s in t[ID_COL]],
                          "site_key": src_key, "lang": "ext", "is_gold": "False"})
    pd.concat([folds, ext_f.astype(str)], ignore_index=True).sort_values(ID_COL).to_csv(
        a.out_folds, index=False, lineterminator="\n")
    idx = merged_cache(Path(a.v1_cache), Path(a.ext_cache), Path(a.merged), set(t[ID_COL]))
    print("targets", len(base) + len(t), "folds", len(folds) + len(ext_f), "merged index rows", len(idx),
          "train studies", idx[idx.split == "train"][ID_COL].nunique())


if __name__ == "__main__":
    main()
