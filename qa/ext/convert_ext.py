"""MRNet .npy / KneeMRI .pck volumes -> cache v1 entries (qa/ext, 2026-10-16).

Neither dataset ships DICOM, so the competition cache build cannot read them. This writes the same cache layout
(<out>/train/<study>/<series>.npy + sidecar, index.parquet, manifest.json) through the competition's own
preprocess.series_to_volume, with a Geometry made up from an assumed field of view: no pixel spacing is published
for either set. Assumed FOV 160 mm (typical knee protocol), i.e. 0.625 mm/px for MRNet's 256 px and 0.5 mm/px for
KneeMRI's 320 px; the 130 mm centre crop then keeps 81 % of the image width.

Series typing (the cache's plane + fluid-sensitive flags; nothing else reaches the model):
  MRNet sagittal (T2 fat-sat) -> Sagittal/1 = sag_fs; coronal (T1) -> Coronal/0 = cor_t1 (lent to cor_fs by the slot
  fallback); axial (PD fat-sat) -> Axial/1 = ax_fs.  KneeMRI (sagittal PD fat-sat) -> Sagittal/1 = sag_fs.
Study IDs are the label file's: mrnet-NNNN (train 0000-1129, valid 1130-1249) and kneemri-<examId>-<seriesNo>.
    python qa/ext/convert_ext.py --ids <labels.csv> --mrnet <MRNet-v1.0 dir> --kneemri <dir> --out <cache root>
"""
import argparse
import io
import json
import multiprocessing
import os
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from rsna_kad.cache import INDEX_COLUMNS, MANIFEST_NAME, INDEX_NAME, series_path  # noqa: E402
from rsna_kad.config import load_cache_config, params_sha256, to_dict  # noqa: E402
from rsna_kad.constants import ID_COL, SERIES_COL  # noqa: E402
from rsna_kad.dicom import Geometry  # noqa: E402
from rsna_kad.preprocess import series_to_volume  # noqa: E402
from rsna_kad.series import assign_slots, slot_of  # noqa: E402

MRNET_PLANES = {"sagittal": ("Sagittal", 1), "coronal": ("Coronal", 0), "axial": ("Axial", 1)}
MRNET_TRAIN_LAST = 1129
Z_STEP_UNKNOWN = 3.0  # recorded only; 2D series are never z-resampled


class _ArrayUnpickler(pickle.Unpickler):
    """The .pck files are downloaded pickles; allow only what a plain numpy array needs."""
    # _codecs.encode: the archive's protocol-2 pickles wrap the array bytes as encode(str, 'latin1')
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype"), ("_codecs", "encode")}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"refusing {module}.{name}")
        return super().find_class(module, name)


def load_pck(path):
    arr = _ArrayUnpickler(io.BytesIO(Path(path).read_bytes()), encoding="latin1").load()
    if not isinstance(arr, np.ndarray) or arr.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D array, got {type(arr).__name__}")
    return arr


def check_layout(arr, path):
    # (s, rows, cols) with fewer slices than pixels; a (rows, cols, s) array would otherwise be cut into ~300 strip
    # "slices" that still pass the cache's shape check. Slices need not be square (5 KneeMRI volumes are 320 x 316).
    n, rows, cols = arr.shape
    if n >= min(rows, cols) or arr.dtype.kind not in "ui":
        raise ValueError(f"{path}: unexpected layout {arr.shape} {arr.dtype}")


def series_tasks(ids, mrnet, kneemri):
    """[(study, series, plane, fs, kind, path)] for every external ID; missing files raise."""
    tasks, pck = [], None
    for sid in ids:
        if sid.startswith("mrnet-"):
            n = int(sid[6:])
            split = "train" if n <= MRNET_TRAIN_LAST else "valid"
            for d, (plane, fs) in MRNET_PLANES.items():
                p = Path(mrnet) / split / d / f"{n:04d}.npy"
                tasks.append((sid, f"{sid}.{d}", plane, fs, "npy", p))
        elif sid.startswith("kneemri-"):
            if pck is None:
                meta = pd.read_csv(Path(kneemri) / "metadata.csv")
                found = list(Path(kneemri).rglob("*.pck"))
                files = {p.name: p for p in found}
                if len(files) != len(found):
                    raise SystemExit(f"{len(found) - len(files)} duplicate .pck names under {kneemri}")
                if meta.duplicated(["examId", "seriesNo"]).any():
                    raise SystemExit("metadata.csv repeats an (examId, seriesNo) pair")
                pck = {f"kneemri-{e}-{s}": files.get(f) for e, s, f in
                       zip(meta.examId, meta.seriesNo, meta.volumeFilename)}
            p = pck.get(sid)
            if p is None:
                raise FileNotFoundError(f"{sid}: no metadata row or .pck file")
            tasks.append((sid, f"{sid}.sagittal", "Sagittal", 1, "pck", p))
        else:
            raise ValueError(f"not an external ID: {sid}")
    for t in tasks:
        if not Path(t[5]).is_file():
            raise FileNotFoundError(f"{t[0]}: {t[5]}")
    return tasks


def convert(args):
    (study, series, plane, fs, kind, path), out, cfg, fov, sha = args
    base = {"split": "train", ID_COL: study, SERIES_COL: series, "plane": plane, "fs": fs,
            "slot": slot_of(plane, fs), "params_sha256": sha, "ok": False, "error": ""}
    try:
        arr = np.load(path, allow_pickle=False) if kind == "npy" else load_pck(path)
        check_layout(arr, path)
        n, rows, cols = arr.shape
        geom = Geometry(spacing=fov / cols, rows=rows, cols=cols, iop=(), ipp_centre=(0.0, 0.0, 0.0),
                        plane=plane, is_3d=False, z_step=Z_STEP_UNKNOWN, n_slices=n, obliquity=1.0)
        vol, meta = series_to_volume(list(arr), geom, cfg)
        npy = series_path(out, "train", study, series)
        npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy, vol)
        rec = {**base, **asdict(meta), "ok": True, "is_3d": False, "spacing_mm": geom.spacing,
               "fov_mm": geom.fov_mm, "z_step_mm": Z_STEP_UNKNOWN, "rows": rows, "cols": cols,
               "plane_iop": plane, "manufacturer": "GE" if kind == "npy" else "SIEMENS", "series_number": 0,
               "source_path": str(path), "assumed_fov_mm": fov}
        npy.with_suffix(".json").write_text(json.dumps(rec, sort_keys=True))
        return rec
    except Exception as e:  # noqa: BLE001 - one bad volume becomes an error row, as in the cache build
        return {**base, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def index_frame(recs):
    df = pd.DataFrame(recs)
    ok = df[df["ok"]]
    ranked = assign_slots(ok.rename(columns={"n_src": "n_slices"}).assign(
        Anatomical_Plane=ok["plane"], Fluid_Sensitive=ok["fs"]))
    native = ranked[~ranked["fallback"]][[ID_COL, SERIES_COL, "rank", "selected"]]
    df = df.merge(native, on=[ID_COL, SERIES_COL], how="left")
    df["rank"] = df["rank"].fillna(-1).astype(int)
    df["selected"] = df["selected"].eq(True)
    for col in INDEX_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    for col in ("side_tag", "side_desc", "side_geom", "side_est", "error", "manufacturer", "plane_iop"):
        df[col] = df[col].fillna("").astype(str)
    df["side_conf"] = df["side_conf"].astype(np.float32)
    return df[INDEX_COLUMNS].sort_values(["split", ID_COL, SERIES_COL]).reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True, help="label CSV; its mrnet-/kneemri- StudyInstanceUIDs are converted")
    ap.add_argument("--mrnet", help="MRNet-v1.0 folder (train/, valid/)")
    ap.add_argument("--kneemri", help="KneeMRI folder (metadata.csv + .pck files anywhere below)")
    ap.add_argument("--only", choices=["mrnet", "kneemri"], help="convert one source only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs/cache_v1.yaml"))
    ap.add_argument("--mrnet-fov-mm", type=float, default=160.0)
    ap.add_argument("--kneemri-fov-mm", type=float, default=160.0)
    ap.add_argument("--limit", type=int, help="first N studies per source (smoke test)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    cfg = load_cache_config(a.config)
    ids = pd.read_csv(a.ids, usecols=[ID_COL], dtype=str)[ID_COL]
    picked = []
    for src in ("mrnet", "kneemri"):
        sel = sorted(ids[ids.str.startswith(src + "-")])
        if a.only and a.only != src:
            continue
        if sel and getattr(a, src) is None:
            raise SystemExit(f"{len(sel)} {src} IDs but no --{src} folder (use --only to skip a source)")
        picked += sel[:a.limit] if a.limit else sel
    tasks = series_tasks(picked, a.mrnet, a.kneemri)
    out, sha = Path(a.out), params_sha256(cfg)
    fov = {"npy": a.mrnet_fov_mm, "pck": a.kneemri_fov_mm}
    t0 = time.time()
    with multiprocessing.get_context("spawn").Pool(a.workers) as pool:
        recs = pool.map(convert, [(t, out, cfg, fov[t[4]], sha) for t in tasks], chunksize=4)
    idx = index_frame(recs)
    out.mkdir(parents=True, exist_ok=True)
    if (out / INDEX_NAME).exists():
        # runs per source (--only) share one --out: keep the other studies' rows, as cache._write_index does per split
        old = pd.read_parquet(out / INDEX_NAME)
        idx = pd.concat([old[~old[ID_COL].isin(set(picked))], idx], ignore_index=True)
        idx = idx.sort_values(["split", ID_COL, SERIES_COL]).reset_index(drop=True)
    tmp = out / (INDEX_NAME + ".tmp")
    idx.to_parquet(tmp, index=False)
    os.replace(tmp, out / INDEX_NAME)
    manifest = {"version": cfg.version, "params": to_dict(cfg), "params_sha256": sha,
                "n_series": int(len(idx)), "n_ok": int(idx["ok"].sum()), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "splits": {"train": {"n_series": int(len(idx)), "n_ok": int(idx["ok"].sum())}}, "geom_rules": {},
                "builder": "qa/ext/convert_ext.py", "assumed_fov_mm": {"mrnet": a.mrnet_fov_mm, "kneemri": a.kneemri_fov_mm}}
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1))
    bad = idx[~idx["ok"]]
    print(f"{len(picked)} studies this run; index {idx[ID_COL].nunique()} studies, {len(idx)} series, "
          f"{int(idx['ok'].sum())} ok, {len(bad)} errors, "
          f"{time.time() - t0:.0f}s -> {out}")
    if len(bad):
        print(bad[[ID_COL, SERIES_COL, "error"]].head(10).to_string())
    return 1 if len(bad) else 0


if __name__ == "__main__":
    sys.exit(main())
