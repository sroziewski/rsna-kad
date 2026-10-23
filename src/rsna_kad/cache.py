"""Preprocessing cache: one uint8 .npy per series plus index.parquet and manifest.json.
Resumable: a series whose .npy and sidecar are valid for the current parameters is
skipped, so an interrupted build continues and a config change forces a rebuild."""
from __future__ import annotations

import json
import logging
import os
import multiprocessing
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from rsna_kad import laterality
from rsna_kad.config import CacheConfig, from_dict, params_sha256, to_dict
from rsna_kad.constants import ID_COL, MAIN_SLOTS, SERIES_COL, SLOTS
from rsna_kad.dicom import list_series_files, read_header, read_slice, series_geometry, sort_headers
from rsna_kad.preprocess import series_to_volume
from rsna_kad.series import assign_slots, slot_of, slot_table

log = logging.getLogger(__name__)

INDEX_NAME = "index.parquet"
MANIFEST_NAME = "manifest.json"
SIDECAR_SUFFIX = ".json"
PROGRESS_EVERY = 30.0
# The parent is multi-threaded (pyarrow, cv2), so fork() risks deadlocks in the workers.
_POOL_CONTEXT = "forkserver"

INDEX_COLUMNS = [
    "split", ID_COL, SERIES_COL, "plane", "fs", "slot", "rank", "selected", "is_3d", "n_src",
    "n_cached", "spacing_mm", "fov_mm", "z_step_mm", "crop_mm", "px", "p_lo", "p_hi",
    "truncated_slices", "side_tag", "side_desc", "side_geom", "side_est", "side_conf", "ok", "error",
    # extras beyond the §3.2 contract, cheap to keep and needed by verify/laterality
    "manufacturer", "series_number", "rows", "cols", "centre_x", "mean_u8", "z_step_cached_mm",
    "tag_conf", "plane_iop",
]


def series_path(cache_dir: str | os.PathLike[str], split: str, study: str, series: str) -> Path:
    return Path(cache_dir) / split / study / f"{series}.npy"


def _sidecar(npy: Path) -> Path:
    return npy.with_suffix(SIDECAR_SUFFIX)


def _valid_npy(npy: Path, px: int, max_slices: int) -> bool:
    try:
        arr = np.load(npy, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return (arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[1:] == (px, px)
            and 1 <= arr.shape[0] <= max_slices)


def _load_sidecar(npy: Path, params_hash: str) -> dict[str, Any] | None:
    try:
        rec = json.loads(_sidecar(npy).read_text())
    except (OSError, ValueError):
        return None
    if not rec.get("ok") or rec.get("params_sha256") != params_hash:
        return None
    return rec


def _write_atomic_npy(npy: Path, vol: np.ndarray) -> None:
    tmp = npy.with_name(npy.name + ".tmp.npy")
    np.save(tmp, vol)
    os.replace(tmp, npy)


def _write_atomic_json(path: Path, doc: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True))
    os.replace(tmp, path)


def process_series(task: Mapping[str, Any]) -> dict[str, Any]:
    """Worker entry: decode, preprocess and store one series; never raises."""
    cfg = from_dict(CacheConfig, task["cfg"])
    npy = Path(task["npy"])
    base = {
        "split": task["split"], ID_COL: task["study"], SERIES_COL: task["series"],
        "plane": task["plane"], "fs": int(task["fs"]), "slot": "",
        "params_sha256": task["params_sha256"], "ok": False, "error": "",
    }
    if task.get("resume", True) and _valid_npy(npy, cfg.px, cfg.max_slices):
        rec = _load_sidecar(npy, task["params_sha256"])
        if rec is not None:
            rec["skipped"] = True
            return rec
    try:
        # Inside the try (architecture §4.1 F1): an unknown plane/fs pair in the CSV is an
        # error row, not a crash of the pool.
        base["slot"] = slot_of(task["plane"], task["fs"])
        files = list_series_files(task["series_dir"])
        if not files:
            raise FileNotFoundError(f"no .dcm files in {task['series_dir']}")
        headers = sort_headers(read_header(p) for p in files)
        geom = series_geometry(headers, cfg.max_2d_slices)
        slices = [read_slice(h.path) for h in headers]
        vol, meta = series_to_volume(slices, geom, cfg)
        npy.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic_npy(npy, vol)
        first = headers[0]
        rec = {
            **base, **asdict(meta), "ok": True,
            "is_3d": bool(geom.is_3d), "spacing_mm": geom.spacing, "fov_mm": geom.fov_mm,
            "z_step_mm": geom.z_step, "rows": geom.rows, "cols": geom.cols, "plane_iop": geom.plane,
            "manufacturer": first.manufacturer, "series_number": first.series_number,
            "centre_x": laterality.series_centre_x(headers),
            "side_tag": laterality.parse_laterality_tag(first.laterality),
            "side_desc": laterality.parse_desc_side(first.series_description),
            "skipped": False,
        }
        _write_atomic_json(_sidecar(npy), rec)
        return rec
    except Exception as exc:  # noqa: BLE001 - one bad series must not kill the build
        log.exception("series %s failed", task["series"])
        return {**base, "error": f"{type(exc).__name__}: {str(exc)[:200]}", "skipped": False}


def _tasks(data_dir: Path, out_dir: Path, split: str, cfg: CacheConfig, resume: bool,
           limit: int | None) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    series_df = pd.read_csv(data_dir / f"{split}_series.csv", dtype=str)
    if limit:
        series_df = series_df.head(limit)
    params_hash = params_sha256(cfg)
    cfg_dict = to_dict(cfg)
    tasks = []
    for r in series_df.itertuples(index=False):
        study, series = getattr(r, ID_COL), getattr(r, SERIES_COL)
        tasks.append({
            "split": split, "study": study, "series": series,
            "plane": r.Anatomical_Plane, "fs": int(r.Fluid_Sensitive),
            "series_dir": str(data_dir / f"{split}_series" / study / series),
            "npy": str(series_path(out_dir, split, study, series)),
            "cfg": cfg_dict, "params_sha256": params_hash, "resume": resume,
        })
    return series_df, tasks


def _study_sides(records: pd.DataFrame, rules: Mapping[str, laterality.GeomRule] | None,
                 cfg: CacheConfig) -> tuple[pd.DataFrame, dict[str, laterality.GeomRule]]:
    ok = records[records["ok"]]
    rows = []
    for study, grp in ok.groupby(ID_COL):
        tag, tag_conf, desc, cx = laterality.aggregate_series(grp.to_dict("records"))
        rows.append({ID_COL: study, "side_tag": tag, "tag_conf": tag_conf, "side_desc": desc,
                     "centre_x": cx, "manufacturer": grp["manufacturer"].mode().iloc[0]})
    study_df = pd.DataFrame(rows, columns=[ID_COL, "side_tag", "tag_conf", "side_desc", "centre_x",
                                           "manufacturer"])
    if rules is None:
        rules = laterality.fit_geom_convention(study_df, cfg.geom_min_studies, cfg.geom_min_agreement)
    return laterality.estimate_sides(study_df, rules), dict(rules)


def _finalize(records: pd.DataFrame, rules: Mapping[str, laterality.GeomRule] | None,
              cfg: CacheConfig) -> tuple[pd.DataFrame, dict[str, laterality.GeomRule]]:
    sides, rules = _study_sides(records, rules, cfg)
    df = records.drop(columns=[c for c in ("side_tag", "side_desc") if c in records.columns]
                      ).merge(sides.drop(columns=["manufacturer", "centre_x"]), on=ID_COL, how="left")
    ok_df = df[df["ok"]]
    ranked = assign_slots(ok_df.rename(columns={"n_src": "n_slices"}).assign(
        Anatomical_Plane=ok_df["plane"], Fluid_Sensitive=ok_df["fs"]))
    native = ranked[~ranked["fallback"]][[ID_COL, SERIES_COL, "rank", "selected"]]
    df = df.merge(native, on=[ID_COL, SERIES_COL], how="left")
    df["rank"] = df["rank"].fillna(-1).astype(int)
    df["selected"] = df["selected"].eq(True)
    for col in INDEX_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["side_conf"] = df["side_conf"].astype(np.float32)
    for col in ("side_tag", "side_desc", "side_geom", "side_est", "error", "manufacturer", "plane_iop"):
        df[col] = df[col].fillna("").astype(str)
    return df[INDEX_COLUMNS].sort_values([ "split", ID_COL, SERIES_COL]).reset_index(drop=True), rules


def read_index(cache_dir: str | os.PathLike[str]) -> pd.DataFrame:
    return pd.read_parquet(Path(cache_dir) / INDEX_NAME)


def read_manifest(cache_dir: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads((Path(cache_dir) / MANIFEST_NAME).read_text())


def rules_from_manifest(manifest: Mapping[str, Any]) -> dict[str, laterality.GeomRule]:
    return {m: laterality.GeomRule(**r) for m, r in manifest.get("geom_rules", {}).items()}


def build_cache(data_dir: str | os.PathLike[str], out_dir: str | os.PathLike[str], split: str,
                cfg: CacheConfig, workers: int = 24, resume: bool = True,
                limit: int | None = None) -> pd.DataFrame:
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    series_df, tasks = _tasks(data_dir, out_dir, split, cfg, resume, limit)
    log.info("cache %s/%s: %d series, %d workers, params %s", out_dir, split, len(tasks), workers,
             params_sha256(cfg)[:12])
    started = last = time.monotonic()
    records: list[dict[str, Any]] = []
    n_skipped = n_err = 0
    with multiprocessing.get_context(_POOL_CONTEXT).Pool(processes=workers) as pool:
        for rec in pool.imap_unordered(process_series, tasks, chunksize=1):
            records.append(rec)
            n_skipped += bool(rec.get("skipped"))
            n_err += not rec["ok"]
            now = time.monotonic()
            if now - last >= PROGRESS_EVERY or len(records) == len(tasks):
                last = now
                rate = len(records) / max(now - started, 1e-6)
                log.info("%d/%d done (%d skipped, %d errors) %.1f series/s ETA %.0fs",
                         len(records), len(tasks), n_skipped, n_err, rate,
                         (len(tasks) - len(records)) / max(rate, 1e-6))
    rec_df = pd.DataFrame(records)
    rules = None
    if split != "train" and (out_dir / MANIFEST_NAME).exists():
        rules = rules_from_manifest(read_manifest(out_dir)) or None
    index_df, rules = _finalize(rec_df, rules, cfg)
    _write_index(out_dir, index_df, cfg, rules)
    log.info("split %s: %d series, %d ok, %d errors, %d skipped, %.0fs",
             split, len(index_df), int(index_df["ok"].sum()), n_err, n_skipped, time.monotonic() - started)
    return index_df


def _write_index(out_dir: Path, index_df: pd.DataFrame, cfg: CacheConfig,
                 rules: Mapping[str, laterality.GeomRule]) -> None:
    index_path = out_dir / INDEX_NAME
    splits = set(index_df["split"])
    if index_path.exists():
        old = pd.read_parquet(index_path)
        old = old[~old["split"].isin(splits)]
        if len(old):
            index_df = pd.concat([old, index_df], ignore_index=True)
    index_df = index_df.sort_values(["split", ID_COL, SERIES_COL]).reset_index(drop=True)
    tmp = index_path.with_name(index_path.name + ".tmp")
    index_df.to_parquet(tmp, index=False)
    os.replace(tmp, index_path)
    manifest = {
        "version": cfg.version, "params": to_dict(cfg), "params_sha256": params_sha256(cfg),
        "n_series": int(len(index_df)), "n_ok": int(index_df["ok"].sum()),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "splits": {s: {"n_series": int((index_df["split"] == s).sum()),
                       "n_ok": int(index_df.loc[index_df["split"] == s, "ok"].sum())}
                   for s in sorted(set(index_df["split"]))},
        "geom_rules": {m: asdict(r) for m, r in rules.items()},
    }
    _write_atomic_json(out_dir / MANIFEST_NAME, manifest)


# ------------------------------------------------------------------------ loading


class CacheIndex:
    """index.parquet plus the per-study slot table (fallbacks resolved)."""

    def __init__(self, cache_dir: str | os.PathLike[str], slots: Sequence[str] = SLOTS):
        self.cache_dir = Path(cache_dir)
        self.df = read_index(self.cache_dir)
        self.slots: dict[str, dict[str, dict[str, list[str]]]] = {}
        for split, ok in self.df[self.df["ok"]].groupby("split"):
            assigned = assign_slots(ok.rename(columns={"n_src": "n_slices"}).assign(
                Anatomical_Plane=ok["plane"], Fluid_Sensitive=ok["fs"]), slots)
            self.slots[str(split)] = slot_table(assigned)

    def studies(self, split: str) -> list[str]:
        return sorted(self.slots.get(split, {}))

    def series_for(self, split: str, study: str, slot: str) -> list[str]:
        return self.slots.get(split, {}).get(study, {}).get(slot, [])


_INDEX_CACHE: dict[tuple[str, float], CacheIndex] = {}


def get_index(cache_dir: str | os.PathLike[str]) -> CacheIndex:
    path = Path(cache_dir) / INDEX_NAME
    key = (str(path), path.stat().st_mtime)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = CacheIndex(cache_dir)
    return _INDEX_CACHE[key]


def load_series(cache_dir: str | os.PathLike[str], split: str, study: str, series: str) -> np.ndarray:
    return np.load(series_path(cache_dir, split, study, series), mmap_mode="r")


def load_study(cache_dir: str | os.PathLike[str], split: str, study: str,
               slots: Sequence[str] = MAIN_SLOTS, sibling_rng: np.random.Generator | None = None,
               sibling_p: float = 0.3, index: CacheIndex | None = None) -> dict[str, np.ndarray]:
    """Rank-0 series per slot (fallback slots included); with sibling_rng, a random
    same-slot sibling replaces it with probability sibling_p. Missing slots are absent."""
    index = index or get_index(cache_dir)
    out: dict[str, np.ndarray] = {}
    for slot in slots:
        candidates = index.series_for(split, study, slot)
        if not candidates:
            continue
        chosen = candidates[0]
        if sibling_rng is not None and len(candidates) > 1 and sibling_rng.random() < sibling_p:
            chosen = candidates[int(sibling_rng.integers(1, len(candidates)))]
        out[slot] = load_series(cache_dir, split, study, chosen)
    return out


# ------------------------------------------------------------------------- verify


def _check_file(args: tuple[str, int, int]) -> tuple[bool, str, int, float, int]:
    path, px, max_slices = args
    try:
        arr = np.load(path, mmap_mode="r")
    except (OSError, ValueError) as exc:
        return False, f"unreadable: {exc}", 0, float("nan"), 0
    shape_ok = (arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[1:] == (px, px)
                and 1 <= arr.shape[0] <= max_slices)
    if not shape_ok:
        return False, f"bad shape {arr.shape} {arr.dtype}", 0, float("nan"), int(os.path.getsize(path))
    return True, "", int(arr.shape[0]), float(np.asarray(arr).mean()), int(os.path.getsize(path))


def verify_cache(cache_dir: str | os.PathLike[str], workers: int = 8,
                 mean_range: tuple[float, float] = (20.0, 200.0)) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    manifest = read_manifest(cache_dir)
    cfg = from_dict(CacheConfig, manifest["params"])
    df = read_index(cache_dir)
    ok_rows = df[df["ok"]]
    args = [(str(series_path(cache_dir, r.split, getattr(r, ID_COL), getattr(r, SERIES_COL))),
             cfg.px, cfg.max_slices) for r in ok_rows.itertuples(index=False)]
    with multiprocessing.get_context(_POOL_CONTEXT).Pool(processes=workers) as pool:
        results = pool.map(_check_file, args, chunksize=64)
    file_ok = np.asarray([r[0] for r in results], dtype=bool)
    means = np.asarray([r[3] for r in results], dtype=np.float64)
    n_slices = np.asarray([r[2] for r in results])
    total_bytes = int(sum(r[4] for r in results))
    problems = [(a[0], r[1]) for a, r in zip(args, results) if not r[0]]
    in_range = np.isfinite(means) & (means >= mean_range[0]) & (means <= mean_range[1])

    index = CacheIndex(cache_dir)
    pairs = [(split, s) for split in index.slots for s in index.studies(split)]
    slot_cov = {slot: sum(bool(index.series_for(split, s, slot)) for split, s in pairs) / max(len(pairs), 1)
                for slot in SLOTS}
    rules = rules_from_manifest(manifest)
    report = {
        "cache_dir": str(cache_dir),
        "n_index_rows": int(len(df)),
        "n_ok": int(len(ok_rows)),
        "n_error": int((~df["ok"]).sum()),
        "errors": df.loc[~df["ok"], [ID_COL, SERIES_COL, "error"]].head(20).to_dict("records"),
        "per_split": {s: int((df["split"] == s).sum()) for s in sorted(set(df["split"]))},
        "n_studies": len(pairs),
        "n_files_checked": int(len(results)),
        "n_files_bad": int((~file_ok).sum()),
        "bad_files": problems[:20],
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / (1 << 30), 2),
        "n_slices_min": int(n_slices[file_ok].min()) if file_ok.any() else 0,
        "n_slices_max": int(n_slices[file_ok].max()) if file_ok.any() else 0,
        "n_slices_mean": float(n_slices[file_ok].mean()) if file_ok.any() else 0.0,
        "mean_in_range_frac": float(in_range[file_ok].mean()) if file_ok.any() else 0.0,
        "mean_u8_median": float(np.nanmedian(means[file_ok])) if file_ok.any() else 0.0,
        "n_truncated_series": int((ok_rows["truncated_slices"] > 0).sum()),
        "truncated_slices_total": int(ok_rows["truncated_slices"].sum()),
        "n_3d": int(ok_rows["is_3d"].sum()),
        "slot_coverage": slot_cov,
        "side_est_counts": ok_rows.drop_duplicates(ID_COL)["side_est"].value_counts().to_dict(),
        "geom_rules": {m: asdict(r) for m, r in rules.items()},
        "passed": bool(file_ok.all() and len(problems) == 0),
    }
    return report


def summarize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Compact per-series CSV view (for the repo copy under analysis/out)."""
    cols = ["split", ID_COL, SERIES_COL, "slot", "rank", "selected", "is_3d", "n_src", "n_cached",
            "spacing_mm", "fov_mm", "z_step_mm", "p_lo", "p_hi", "truncated_slices", "mean_u8",
            "side_tag", "side_desc", "side_geom", "side_est", "side_conf", "manufacturer", "ok", "error"]
    return df[[c for c in cols if c in df.columns]]


def iter_studies(index_df: pd.DataFrame, split: str) -> Iterable[str]:
    return index_df.loc[(index_df["split"] == split) & index_df["ok"], ID_COL].unique()
