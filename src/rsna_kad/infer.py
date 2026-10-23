"""Test-time inference from raw DICOM (architecture §2.7): the cache pipeline re-run per
study in a small process pool, one GPU consumer thread per device holding its share of the
ensemble, fp16 autocast, a wall-clock guard and a validated submission.csv.

This module is inlined into notebooks/kaggle_inference.py, so it may import only the
modules listed in kaggle_export.INLINE_MODULES plus stdlib/numpy/pandas/torch/safetensors."""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from rsna_kad.config import CacheConfig, ModelConfig, from_dict, to_dict
from rsna_kad.constants import ID_COL, LABELS, MAIN_SLOTS, SERIES_COL, SLOT_FALLBACK, SLOTS
from rsna_kad.data import read_triplets, sample_centres, triplet_indices, view_mm_for, view_params
from rsna_kad.dicom import list_series_files, read_header, read_slice, series_geometry, sort_headers
from rsna_kad.model import FINGERPRINT_RTOL, KneeMIL, build_model, fingerprint
from rsna_kad.preprocess import VolumeMeta, series_to_volume
from rsna_kad.series import FAT_COL, assign_slots, slot_of, slot_table

log = logging.getLogger(__name__)

ENSEMBLE_NAME = "ensemble.json"
WEIGHTS_SUFFIX = ".safetensors"
META_SUFFIX = ".json"
KAGGLE_INPUT = "/kaggle/input"
KAGGLE_DATA_DIR = "/kaggle/input/rsna-knee-abnormality-detection"
KAGGLE_WEIGHTS_DIR = "/kaggle/input/rsna-kad-weights"
# Study trees are never where an input root lives; skipping them keeps find_dir cheap.
SKIP_DIRS = frozenset({"test_series", "train_series", "models", "wheels"})
KAGGLE_OUT = "/kaggle/working/submission.csv"
KAGGLE_TEMP = "/kaggle/temp"
TTA_Z_OFFSET = 0.5
DEFAULT_PRIOR = 0.5
PROGRESS_EVERY = 30.0
QUEUE_DEPTH = 8
# A hung decode (e.g. a malformed J2K in gdcm/openjpeg) must cost one study, not the run: the
# slowest measured study prepares in about a minute on Kaggle CPUs.
STUDY_TIMEOUT_S = 600.0
DEADLINE_ERROR = "unfinished at deadline"
# More prior-only rows than this (not counting the wall guard's) means the pipeline itself is
# broken - e.g. decoder wheels missing on compressed test DICOMs - and the file would score ~0.5.
PRIOR_ALARM_FRACTION = 0.10

# (img_px, slots, K, z_offsets, view_mm): what one model needs per study. A plain tuple so it
# crosses the process boundary without any class lookup in __main__ (the notebook is one flat
# script). view_mm is one absolute field of view in mm per slot, resolved by data.view_mm_for -
# equal to the cached field for every slot of a model without view crops, shorter for a slot
# that the model was trained on as a centre crop (a slot name may repeat with two fields).
InputSpec = tuple[int, tuple[str, ...], int, tuple[float, ...], tuple[float, ...]]


# ------------------------------------------------------------------ preprocessing


def safe_read_slice(path: str) -> np.ndarray | None:
    """read_slice with a last-resort catch: on Kaggle one exotic file must cost one slice
    (neighbour-filled), never the study."""
    try:
        return read_slice(path)
    except Exception as exc:  # noqa: BLE001
        log.warning("slice %s: %s: %s", path, type(exc).__name__, str(exc)[:120])
        return None


@dataclass
class SeriesInfo:
    series_uid: str
    plane: str
    fs: int
    series_dir: str
    fat: Any = None
    n_files: int = 0
    headers: list = field(default_factory=list)
    geom: Any = None
    error: str = ""


class StudyPreprocessor:
    """Raw study directory + its series rows (test_series.csv) -> {slot: uint8 (N, px, px)}.

    Same call sequence as cache.process_series (InstanceNumber sort, series_geometry,
    read_slice per file, series_to_volume) and the same slot selection as the cache index
    (assign_slots ranking over readable series, FS -> T1 fallback), so a test-time volume
    equals the cached one. A series that fails entirely is skipped in favour of the next
    candidate of its slot; a slot with no readable candidate is missing."""

    def __init__(self, cfg: CacheConfig | None = None, slots: Sequence[str] = MAIN_SLOTS):
        self.cfg = cfg or CacheConfig()
        self.slots = tuple(slots)

    def series_volume(self, series_dir: str | os.PathLike[str],
                      headers: Sequence | None = None) -> tuple[np.ndarray, VolumeMeta]:
        """uint8 (N, px, px) of one series; raises when nothing is readable."""
        if headers is None:
            files = list_series_files(series_dir)
            if not files:
                raise FileNotFoundError(f"no .dcm files in {series_dir}")
            headers = sort_headers(read_header(p) for p in files)
        geom = series_geometry(headers, self.cfg.max_2d_slices)
        slices = [safe_read_slice(h.path) for h in headers]
        return series_to_volume(slices, geom, self.cfg)

    def scan(self, study_dir: str | os.PathLike[str], rows: pd.DataFrame) -> list[SeriesInfo]:
        """Headers + geometry of every series of the study (cheap: pixels are not read)."""
        infos = []
        for r in rows.itertuples(index=False):
            uid = str(getattr(r, SERIES_COL))
            info = SeriesInfo(uid, str(r.Anatomical_Plane), 0, str(Path(study_dir) / uid),
                              getattr(r, FAT_COL, None))
            try:
                info.fs = int(r.Fluid_Sensitive)
                slot_of(info.plane, info.fs)
                files = list_series_files(info.series_dir)
                if not files:
                    raise FileNotFoundError(f"no .dcm files in {info.series_dir}")
                info.n_files = len(files)
                info.headers = sort_headers(read_header(p) for p in files)
                info.geom = series_geometry(info.headers, self.cfg.max_2d_slices)
            except Exception as exc:  # noqa: BLE001 - one bad series is a missing candidate
                info.error = f"{type(exc).__name__}: {str(exc)[:160]}"
                log.warning("series %s skipped: %s", uid, info.error)
            infos.append(info)
        return infos

    def ranked(self, study: str, infos: Sequence[SeriesInfo]) -> dict[str, list[str]]:
        """{native slot: [series uid in cache rank order]} over the readable series."""
        good = [i for i in infos if not i.error]
        if not good:
            return {}
        df = pd.DataFrame({
            ID_COL: study, SERIES_COL: [i.series_uid for i in good],
            "Anatomical_Plane": [i.plane for i in good], "Fluid_Sensitive": [i.fs for i in good],
            FAT_COL: [i.fat for i in good],
            "is_3d": [bool(i.geom.is_3d) for i in good], "n_slices": [len(i.headers) for i in good],
            "series_number": [i.headers[0].series_number for i in good],
        })
        table = assign_slots(df, SLOTS)
        return slot_table(table[~table["fallback"]]).get(study, {})

    def study(self, study_dir: str | os.PathLike[str], rows: pd.DataFrame,
              study: str | None = None) -> dict[str, np.ndarray]:
        study = study or Path(study_dir).name
        infos = self.scan(study_dir, rows)
        by_uid = {i.series_uid: i for i in infos}
        native = self.ranked(study, infos)
        decoded: dict[str, np.ndarray | None] = {}

        def decode(uid: str) -> np.ndarray | None:
            if uid not in decoded:
                info = by_uid[uid]
                try:
                    decoded[uid] = self.series_volume(info.series_dir, info.headers)[0]
                except Exception as exc:  # noqa: BLE001 - whole-series failure -> next candidate
                    log.warning("series %s undecodable: %s: %s", uid, type(exc).__name__, str(exc)[:160])
                    decoded[uid] = None
            return decoded[uid]

        out: dict[str, np.ndarray] = {}
        for slot in self.slots:
            for source in (slot, SLOT_FALLBACK.get(slot)):
                if source is None or slot in out:
                    continue
                for uid in native.get(source, []):
                    vol = decode(uid)
                    if vol is not None:
                        out[slot] = vol
                        break
        return out


# ------------------------------------------------------------------ triplet inputs


def study_inputs(vols: Mapping[str, np.ndarray], spec: InputSpec,
                 crop_mm: float = CacheConfig.crop_mm) -> tuple[np.ndarray, np.ndarray]:
    """(V, S, K, 3, px, px) uint8 for the V TTA views of spec and present (S,) bool.

    `crop_mm` is the field the volumes of `vols` cover (the cache geometry, §2.2); a slot whose
    spec field of view is shorter is centre-cropped to it exactly as data.StudyDataset does at
    evaluation time, which is what makes a repeated slot two different views instead of two
    copies of one."""
    px, slots, k, z_offsets, view_mm = spec
    present = np.array([s in vols for s in slots], dtype=bool)
    images = np.zeros((len(z_offsets), len(slots), k, 3, px, px), dtype=np.uint8)
    for j, slot in enumerate(slots):
        if not present[j]:
            continue
        vol = vols[slot]
        n = vol.shape[0]
        params = view_params(crop_mm, view_mm[j]) if view_mm else None
        for v, z in enumerate(z_offsets):
            idx = triplet_indices(sample_centres(n, k, None, z), n)
            images[v, j] = read_triplets(vol, idx, px, params)
    return images, present


# ------------------------------------------------------------------ members


@dataclass
class Member:
    """One ensemble member of the export layout: <prefix>.safetensors + <prefix>.json."""
    prefix: str
    weight: float = 1.0
    tta: bool = True
    meta: dict[str, Any] = field(default_factory=dict)
    model: KneeMIL | None = None
    device: str = "cpu"

    @property
    def spec(self) -> InputSpec:
        arch = self.meta["arch"]
        slots = tuple(arch["slots"])
        # Bundles exported before view crops existed have no "view_crop_mm"; view_mm_for then
        # returns the cached field for every slot, which is what those models were trained on.
        return (int(arch["img_px"]), slots, int(arch["K_infer"]),
                (0.0, TTA_Z_OFFSET) if self.tta else (0.0,),
                view_mm_for(slots, arch.get("view_crop_mm") or (), self.crop_mm))

    @property
    def crop_mm(self) -> float:
        """The cached field of view this member was trained against (§2.2)."""
        return float(self.meta.get("config", {}).get("cache", {}).get("crop_mm", CacheConfig.crop_mm))


def read_member_meta(prefix: str | os.PathLike[str]) -> dict[str, Any]:
    return json.loads(Path(str(prefix) + META_SUFFIX).read_text(encoding="utf-8"))


def load_member(prefix: str | os.PathLike[str], device: str | torch.device = "cpu",
                meta: Mapping[str, Any] | None = None,
                views: bool = False) -> tuple[KneeMIL, dict[str, Any]]:
    """Export-layout twin of model.load_bundle: pretrained=False, strict keys, fingerprint
    compared against the json (a tampered or foreign safetensors aborts here, §2.7).

    `views=True` states that the caller builds this member's inputs from `arch.view_crop_mm`
    (Member.spec + study_inputs do). Without it a view-crop member is refused instead of being
    fed an uncropped duplicate of its repeated slot, which looks like a valid input and would
    only show up as a quietly worse score."""
    prefix = str(prefix)
    meta = dict(meta or read_member_meta(prefix))
    mcfg = from_dict(ModelConfig, meta["config"]["model"])
    crop_mm = float(meta.get("config", {}).get("cache", {}).get("crop_mm", CacheConfig.crop_mm))
    # Raises when the slots repeat without crops to tell them apart - the shape of a bundle
    # whose view_crop_mm was lost between training and export.
    view_mm_for(mcfg.slots, mcfg.view_crop_mm, crop_mm)
    if tuple(mcfg.view_crop_mm) and not views:
        raise RuntimeError(f"{prefix}: this member has per-slot view crops {list(mcfg.view_crop_mm)} "
                           f"for slots {list(mcfg.slots)}; the caller must build its inputs with "
                           "infer.study_inputs (Member.spec carries the crops) and pass views=True")
    mcfg.grad_checkpoint = False
    model = build_model(mcfg, pretrained=False)
    state = load_file(prefix + WEIGHTS_SUFFIX)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"{prefix}: missing {result.missing_keys[:5]} unexpected {result.unexpected_keys[:5]}")
    got, want = fingerprint(model.state_dict()), meta["fingerprint"]
    if len(got) != len(want):
        raise RuntimeError(f"{prefix}: fingerprint length {len(got)} != {len(want)}")
    for g, w in zip(got, want):
        if abs(g - w) > FINGERPRINT_RTOL * max(abs(w), 1.0):
            raise RuntimeError(f"{prefix}: weight fingerprint mismatch {got} != {want}")
    model = model.to(device).eval()
    if torch.device(device).type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model, meta


def read_ensemble(weights_dir: str | os.PathLike[str]) -> dict[str, Any]:
    doc = json.loads((Path(weights_dir) / ENSEMBLE_NAME).read_text(encoding="utf-8"))
    if not doc.get("members"):
        raise ValueError(f"{weights_dir}/{ENSEMBLE_NAME}: no members")
    return doc


def members_from_ensemble(weights_dir: str | os.PathLike[str], doc: Mapping[str, Any] | None = None,
                          tta: bool | None = None) -> list[Member]:
    doc = doc or read_ensemble(weights_dir)
    members = []
    for m in doc["members"]:
        prefix = str(Path(weights_dir) / m["path"])
        members.append(Member(prefix, float(m.get("weight", 1.0)),
                              bool(m.get("tta", True)) if tta is None else tta, read_member_meta(prefix)))
    return members


def available_devices(requested: Sequence[str] | None = None) -> list[str]:
    if requested:
        return [d for d in requested if d]
    if torch.cuda.is_available():
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


def assign_devices(members: Sequence[Member], devices: Sequence[str]) -> None:
    """Round-robin: each device holds its share of the ensemble list (§2.7)."""
    for i, m in enumerate(members):
        m.device = devices[i % len(devices)]


# ------------------------------------------------------------------ producer


_WORKER: dict[str, Any] = {}


def _init_worker(cfg_dict: dict[str, Any], slots: Sequence[str], specs: Sequence[InputSpec]) -> None:
    torch.set_num_threads(1)
    _WORKER["pre"] = StudyPreprocessor(from_dict(CacheConfig, cfg_dict), slots)
    _WORKER["specs"] = list(specs)


def prepare_study(task: Mapping[str, Any]) -> dict[str, Any]:
    """Pool entry: decode + preprocess + triplet sampling for every spec; never raises."""
    study, study_dir = task["study"], task["study_dir"]
    t0 = time.monotonic()
    out: dict[str, Any] = {"study": study, "inputs": {}, "n_slots": 0, "error": "", "seconds": 0.0}
    try:
        pre: StudyPreprocessor = _WORKER["pre"]
        rows = pd.DataFrame(task["rows"], columns=study_row_cols(task["rows"]))
        vols = pre.study(study_dir, rows, study)
        out["n_slots"] = len(vols)
        if vols:
            crop_mm = pre.cfg.crop_mm
            out["inputs"] = {spec: study_inputs(vols, spec, crop_mm) for spec in _WORKER["specs"]}
    except Exception as exc:  # noqa: BLE001 - a broken study gets the prior, never a crash
        out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        log.exception("study %s failed", study)
    out["seconds"] = time.monotonic() - t0
    return out


def count_files(study_dir: str | os.PathLike[str]) -> int:
    n = 0
    for _root, _dirs, files in os.walk(study_dir):
        n += sum(1 for f in files if f.lower().endswith(".dcm"))
    return n


def order_studies(study_dirs: Mapping[str, str]) -> list[str]:
    """Descending file count so the 320-slice tail never straggles at the end (§2.7)."""
    counts = {s: count_files(d) for s, d in study_dirs.items()}
    return sorted(study_dirs, key=lambda s: (-counts[s], s))


def study_row_cols(rows: Sequence[Mapping[str, Any]] | pd.DataFrame) -> list[str]:
    """The test_series.csv columns routing needs; Fat_Suppression only when the CSV has it."""
    cols = [SERIES_COL, "Anatomical_Plane", "Fluid_Sensitive"]
    have = rows.columns if isinstance(rows, pd.DataFrame) else (rows[0] if len(rows) else {})
    return cols + ([FAT_COL] if FAT_COL in have else [])


def study_rows(series_df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    cols = study_row_cols(series_df)
    out: dict[str, list[dict[str, Any]]] = {}
    for study, grp in series_df.groupby(ID_COL, sort=False):
        out[str(study)] = grp[cols].astype(str).to_dict("records")
    return out


def pool_context() -> multiprocessing.context.BaseContext:
    # fork (what torch's DataLoader uses) needs no pickling of __main__ functions, so the flat
    # notebook works whether Kaggle runs it as a script or through IPython; the pool is
    # created before any model touches CUDA.
    methods = multiprocessing.get_all_start_methods()
    return multiprocessing.get_context("fork" if "fork" in methods else "spawn")


# ------------------------------------------------------------------ consumer


class _Consumer(threading.Thread):
    """Runs the members of one device over every prepared study; partial sums per study."""

    def __init__(self, device: str, members: Sequence[Member], fp16: bool, stop: threading.Event):
        super().__init__(name=f"consumer-{device}", daemon=True)
        self.device = device
        self.members = list(members)
        self.fp16 = fp16 and torch.device(device).type == "cuda"
        self.queue: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
        self.stop = stop
        self.sums: dict[str, np.ndarray] = {}
        self.seconds: dict[str, float] = {}
        self.failed: dict[str, str] = {}
        self.error: BaseException | None = None

    def predict(self, inputs: Mapping[InputSpec, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        total = np.zeros(len(LABELS), dtype=np.float64)
        dev_type = torch.device(self.device).type
        with torch.inference_mode(), torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=self.fp16):
            for m in self.members:
                images, present = inputs[m.spec]
                pres = torch.from_numpy(present).to(self.device).unsqueeze(0)
                probs = np.zeros(len(LABELS), dtype=np.float64)
                for view in range(images.shape[0]):
                    x = torch.from_numpy(np.ascontiguousarray(images[view])).to(self.device).unsqueeze(0)
                    logits, _ = m.model(x, pres)
                    probs += logits.float().sigmoid().squeeze(0).cpu().numpy()
                total += m.weight * probs / images.shape[0]
        return total

    def put(self, item: Any) -> None:
        while True:
            try:
                self.queue.put(item, timeout=1.0)
                return
            except queue.Full:
                if self.error is not None:
                    return

    def run(self) -> None:
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    return
                study, inputs = item
                if self.stop.is_set():
                    continue
                t0 = time.monotonic()
                try:
                    self.sums[study] = self.predict(inputs)
                except Exception as exc:  # noqa: BLE001 - e.g. CUDA OOM on one study: its row gets the prior
                    self.failed[study] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    log.exception("consumer %s: study %s failed", self.device, study)
                    if torch.device(self.device).type == "cuda":
                        torch.cuda.empty_cache()
                self.seconds[study] = time.monotonic() - t0
        except BaseException as exc:  # noqa: BLE001 - surfaced by predict_studies
            self.error = exc
            log.exception("consumer %s failed", self.device)


# ------------------------------------------------------------------ driver


def max_rss_gb() -> float:
    """Peak resident set of this process (Linux/macOS); NaN where `resource` is missing."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    except (ImportError, OSError, AttributeError):
        return float("nan")


def study_record(rec: dict[str, Any]) -> tuple[dict[str, Any], dict[InputSpec, tuple[np.ndarray, np.ndarray]]]:
    """Split a prepare_study result into the metadata kept for the whole run and the ~28 MB
    of triplets, which must only live in the consumer queues (1,300 x 28 MB would not fit)."""
    inputs = rec.pop("inputs")
    return {**rec, "has_inputs": bool(inputs)}, inputs


def prior_vector(prior: Sequence[float] | float | None) -> np.ndarray:
    if prior is None:
        return np.full(len(LABELS), DEFAULT_PRIOR, dtype=np.float64)
    if np.isscalar(prior):
        return np.full(len(LABELS), float(prior), dtype=np.float64)
    arr = np.asarray(prior, dtype=np.float64)
    if arr.shape != (len(LABELS),):
        raise ValueError(f"prior must have {len(LABELS)} values, got {arr.shape}")
    return arr


def predict_studies(members: Sequence[Member], study_dirs: Mapping[str, str], series_df: pd.DataFrame,
                    devices: Sequence[str] | None = None, workers: int = 3, cache_cfg: CacheConfig | None = None,
                    fp16: bool = True, prior: Sequence[float] | float | None = None,
                    deadline_s: float | None = None,
                    study_timeout_s: float | None = STUDY_TIMEOUT_S) -> pd.DataFrame:
    """Ensemble mean of sigmoids for every study in study_dirs ({StudyInstanceUID: dir}).
    Members are loaded here (fingerprint-checked) onto `devices`; studies that fail to prepare,
    whose preprocessing exceeds study_timeout_s, that fail on a device, have no usable slot, or
    are unfinished at deadline_s (seconds after the call) get the prior row. A device thread
    that dies outright stops the run early; every study it did not finish gets the prior too.
    Returns StudyInstanceUID + one column per LABELS name + n_slots/prep_s/gpu_s/error."""
    if not members:
        raise ValueError("predict_studies: no members")
    started = time.monotonic()
    deadline = None if deadline_s is None else started + deadline_s
    devices = available_devices(devices)
    assign_devices(members, devices)
    for m in members:
        if not m.meta:
            m.meta = read_member_meta(m.prefix)
    prior_v = prior_vector(prior)
    weight_sum = float(sum(m.weight for m in members))
    specs = sorted({m.spec for m in members})
    slots = tuple(dict.fromkeys(s for spec in specs for s in spec[1]))
    cfg = cache_cfg
    if cfg is None:
        cache_dict = members[0].meta.get("config", {}).get("cache")
        cfg = from_dict(CacheConfig, cache_dict) if cache_dict else CacheConfig()

    rows = study_rows(series_df)
    order = order_studies(study_dirs)
    tasks = [{"study": s, "study_dir": study_dirs[s], "rows": rows.get(s, [])} for s in order]
    pool = pool_context().Pool(processes=max(1, workers), initializer=_init_worker,
                               initargs=(to_dict(cfg), slots, specs))
    stop = threading.Event()
    consumers: list[_Consumer] = []
    prep: dict[str, dict[str, Any]] = {}
    n_done = 0
    last_log = started
    timed_out = False
    try:
        for m in members:
            if m.model is None:
                m.model, m.meta = load_member(m.prefix, m.device, m.meta, views=True)
                log.info("loaded %s on %s (%s)", Path(m.prefix).name, m.device, m.meta["arch"]["backbone"])
        consumers = [_Consumer(d, [m for m in members if m.device == d], fp16, stop) for d in devices]
        for c in consumers:
            c.start()
        # Bounded dispatch instead of pool.imap: imap buffers every finished result in the
        # parent, so a GPU slower than the producers would pile up ~28 MB per study.
        pending: deque = deque()
        todo = iter(tasks)
        max_inflight = max(1, workers) + QUEUE_DEPTH

        def submit_more() -> None:
            while len(pending) < max_inflight:
                task = next(todo, None)
                if task is None:
                    return
                pending.append((task["study"], pool.apply_async(prepare_study, (task,))))

        submit_more()
        while pending:
            study, result = pending.popleft()
            wait = study_timeout_s
            if deadline is not None:
                left = max(deadline - time.monotonic(), 0.0)
                wait = left if wait is None else min(wait, left)
            try:
                rec = result.get(timeout=wait)
            except multiprocessing.TimeoutError:
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    log.warning("wall-clock guard hit after %d/%d studies; the rest get the prior", n_done,
                                len(tasks))
                    stop.set()
                    break
                # The hung worker keeps its pool slot until pool.terminate(); the others go on.
                rec = {"study": study, "inputs": {}, "n_slots": 0, "seconds": float(wait),
                       "error": f"preprocessing timeout after {wait:.0f}s"}
                log.error("study %s: %s", study, rec["error"])
            submit_more()
            prep[rec["study"]], inputs = study_record(rec)
            if inputs:
                for c in consumers:
                    c.put((rec["study"], inputs))
            del inputs, rec
            n_done += 1
            now = time.monotonic()
            if now - last_log >= PROGRESS_EVERY or n_done == len(tasks):
                last_log = now
                log.info("prepared %d/%d studies, %.1fs elapsed, peak RSS %.1f GB", n_done, len(tasks),
                         now - started, max_rss_gb())
            if deadline is not None and now > deadline:
                timed_out = True
                log.warning("wall-clock guard hit after %d/%d studies; the rest get the prior", n_done, len(tasks))
                stop.set()
                break
            if any(c.error is not None for c in consumers):
                break
    finally:
        pool.terminate()
        pool.join()
        for c in consumers:
            c.put(None)
        for c in consumers:
            c.join()
    dead = [f"consumer {c.device} failed: {c.error!r}"[:200] for c in consumers if c.error is not None]
    for msg in dead:
        log.error("%s; its unfinished studies get the prior", msg)
    unfinished = dead[0] if dead else DEADLINE_ERROR

    out_rows = []
    for s in order:
        rec = prep.get(s)
        sums = [c.sums.get(s) for c in consumers]
        failed = [f"{c.device}: {c.failed[s]}" for c in consumers if s in c.failed]
        if rec is None:
            err = unfinished
        elif rec["error"]:
            err = rec["error"]
        elif not rec["has_inputs"]:
            err = "no usable slot"
        elif failed:
            err = "inference failed on " + failed[0]
        elif any(v is None for v in sums):
            err = unfinished
        else:
            err = ""
        probs = prior_v if err else np.sum(sums, axis=0) / weight_sum
        out_rows.append({
            ID_COL: s, **{lab: float(np.clip(p, 0.0, 1.0)) for lab, p in zip(LABELS, probs)},
            "n_slots": int(rec["n_slots"]) if rec else 0,
            "prep_s": float(rec["seconds"]) if rec else float("nan"),
            "gpu_s": float(sum(c.seconds.get(s, 0.0) for c in consumers)), "error": err,
        })
    df = pd.DataFrame(out_rows, columns=[ID_COL, *LABELS, "n_slots", "prep_s", "gpu_s", "error"])
    log.info("predicted %d studies in %.1fs (%d prior rows%s)", len(df), time.monotonic() - started,
             int((df["error"] != "").sum()), ", wall guard hit" if timed_out else "")
    return df


def failure_fraction(pred_df: pd.DataFrame) -> float:
    """Share of prior-only rows other than the wall guard's (those are its intended output)."""
    if not len(pred_df):
        return 0.0
    err = pred_df["error"].fillna("")
    return float(((err != "") & (err != DEADLINE_ERROR)).mean())


# ------------------------------------------------------------------ submission


def write_submission(pred_df: pd.DataFrame, sample_csv: str | os.PathLike[str], out_csv: str | os.PathLike[str],
                     prior: Sequence[float] | float | None = None) -> pd.DataFrame:
    """Rows of sample_submission.csv in its order, the 12 label columns, floats in [0, 1], no
    NaN; a study missing from pred_df takes the prior when one is given, else it is an error."""
    sample = pd.read_csv(sample_csv, dtype={ID_COL: str})
    cols = [ID_COL, *LABELS]
    if list(sample.columns) != cols:
        raise ValueError(f"{sample_csv}: unexpected columns {list(sample.columns)}")
    pred = pred_df.copy()
    pred[ID_COL] = pred[ID_COL].astype(str)
    pred = pred.drop_duplicates(ID_COL, keep="first").set_index(ID_COL)[list(LABELS)]
    extra = sorted(set(pred.index) - set(sample[ID_COL]))
    if extra:
        log.warning("%d predicted studies are not in %s (dropped)", len(extra), sample_csv)
    absent = [s for s in sample[ID_COL] if s not in pred.index]
    if absent:
        if prior is None:
            raise ValueError(f"{len(absent)} studies of {sample_csv} have no prediction, e.g. {absent[:3]}")
        log.warning("%d studies without prediction get the prior", len(absent))
        fill = pd.DataFrame([prior_vector(prior)] * len(absent), index=absent, columns=list(LABELS))
        pred = pd.concat([pred, fill])
    values = pred.loc[sample[ID_COL], list(LABELS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("submission has NaN/inf values")
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError(f"submission values outside [0, 1]: {values.min()} .. {values.max()}")
    out = pd.DataFrame(values.astype(np.float32), columns=list(LABELS))
    out.insert(0, ID_COL, sample[ID_COL].to_numpy())
    if len(out) != len(sample):
        raise ValueError(f"submission has {len(out)} rows, sample has {len(sample)}")
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_name(out_csv.name + ".tmp")
    out.to_csv(tmp, index=False, float_format="%.6f")
    os.replace(tmp, out_csv)
    return out


# ------------------------------------------------------------------ notebook entry point


def find_dir(preferred: str | os.PathLike[str], marker: str, root: str = KAGGLE_INPUT, max_depth: int = 4) -> Path:
    """preferred when it holds marker, else the shallowest directory under root that does
    (Kaggle mounts inputs at /kaggle/input/<slug> or /kaggle/input/{datasets,competitions}/...)."""
    preferred = Path(preferred)
    if (preferred / marker).exists():
        return preferred
    root_p = Path(root)
    if root_p.is_dir():
        frontier = [root_p]
        for _depth in range(max_depth):
            nxt = []
            for d in frontier:
                try:
                    children = sorted(c for c in d.iterdir() if c.is_dir())
                except OSError:
                    continue
                for c in children:
                    if (c / marker).exists():
                        log.info("found %s under %s", marker, c)
                        return c
                    if c.name not in SKIP_DIRS:
                        nxt.append(c)
            frontier = nxt
    return preferred


def describe_inputs(root: str = KAGGLE_INPUT, max_depth: int = 3, limit: int = 40) -> list[str]:
    """Directory names under root down to max_depth (diagnostics in the kernel log)."""
    out: list[str] = []
    frontier = [Path(root)]
    for _depth in range(max_depth):
        nxt = []
        for d in frontier:
            try:
                for c in sorted(d.iterdir()):
                    if len(out) >= limit:
                        return out + ["..."]
                    out.append(str(c.relative_to(root)) + ("/" if c.is_dir() else ""))
                    if c.is_dir() and c.name not in SKIP_DIRS:
                        nxt.append(c)
            except OSError:
                continue
        frontier = nxt
    return out


def test_study_dirs(data_dir: str | os.PathLike[str], series_df: pd.DataFrame,
                    sample: pd.DataFrame | None = None) -> dict[str, str]:
    root = Path(data_dir) / "test_series"
    ids = list(sample[ID_COL].astype(str)) if sample is not None else sorted(series_df[ID_COL].astype(str).unique())
    return {s: str(root / s) for s in dict.fromkeys(ids)}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="RSNA KAD inference: raw DICOM test studies -> submission.csv")
    ap.add_argument("--data-dir", default=KAGGLE_DATA_DIR)
    ap.add_argument("--weights", default=KAGGLE_WEIGHTS_DIR, help="export dir holding ensemble.json")
    ap.add_argument("--out", default=KAGGLE_OUT)
    ap.add_argument("--device", action="append", default=None, help="cuda:0, cuda:1, cpu (repeatable)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N studies of sample_submission")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--wall-hours", type=float, default=None, help="prior rows for studies unfinished by then")
    ap.add_argument("--study-timeout", type=float, default=STUDY_TIMEOUT_S,
                    help="seconds one study may spend in preprocessing before it gets the prior row")
    ap.add_argument("--max-prior-fraction", type=float, default=PRIOR_ALARM_FRACTION,
                    help="exit non-zero (after writing the file) when more rows than this fall back to the prior")
    ap.add_argument("--install-wheels", action="store_true", help="(consumed by the notebook prelude)")
    return ap


def notebook_main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout, force=True)
    args = build_arg_parser().parse_args(argv)
    if Path("/kaggle").is_dir():
        os.makedirs(KAGGLE_TEMP, exist_ok=True)
    t0 = time.monotonic()
    if Path(KAGGLE_INPUT).is_dir():
        log.info("inputs: %s", describe_inputs(KAGGLE_INPUT))
    data_dir = find_dir(args.data_dir, "sample_submission.csv")
    weights = find_dir(args.weights, ENSEMBLE_NAME)
    doc = read_ensemble(weights)
    members = members_from_ensemble(weights, doc, tta=False if args.no_tta else None)
    devices = available_devices(args.device)
    log.info("ensemble %s: %d members, mode %s, devices %s, torch %s", weights, len(members), doc.get("mode"),
             devices, torch.__version__)
    sample = pd.read_csv(data_dir / "sample_submission.csv", dtype={ID_COL: str})
    series_df = pd.read_csv(data_dir / "test_series.csv", dtype=str)
    if args.limit:
        sample = sample.head(args.limit)
    study_dirs = test_study_dirs(data_dir, series_df, sample)
    prior = doc.get("prior")
    workers = args.workers if args.workers is not None else int(doc.get("workers", 3))
    wall_h = args.wall_hours if args.wall_hours is not None else doc.get("wall_guard_hours")
    deadline = None if wall_h is None else float(wall_h) * 3600.0 - (time.monotonic() - t0)
    t1 = time.monotonic()
    pred = predict_studies(members, study_dirs, series_df, devices=devices, workers=workers,
                           fp16=not args.fp32, prior=prior, deadline_s=deadline,
                           study_timeout_s=args.study_timeout)
    infer_s = time.monotonic() - t1
    write_submission(pred, data_dir / "sample_submission.csv", args.out, prior=prior)
    n = max(len(pred), 1)
    log.info("submission %s: %d rows | setup %.1fs | inference %.1fs = %.2fs/study (prep %.2fs, gpu %.2fs per "
             "study) | %d prior rows", args.out, len(pred), t1 - t0, infer_s, infer_s / n, pred["prep_s"].mean(),
             pred["gpu_s"].mean(), int((pred["error"] != "").sum()))
    frac = failure_fraction(pred)
    if frac > args.max_prior_fraction:
        errors = pred.loc[(pred["error"] != "") & (pred["error"] != DEADLINE_ERROR), "error"]
        log.error("PRIOR ALARM: %.1f%% of rows are prior-only fallbacks (limit %.0f%%); the file is written but "
                  "would score ~0.5 AUC. Most common causes: %s", 100 * frac, 100 * args.max_prior_fraction,
                  errors.str[:80].value_counts().head(3).to_dict())
        return 3
    return 0
