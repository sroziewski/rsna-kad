#!/usr/bin/env python3
"""Pipeline A of our RSNA Knee Abnormality Detection solution, in one self-contained file.

COMPETITION
-----------
Kaggle "rsna-knee-abnormality-detection": knee MRI studies (several DICOM series each) -> 12
independent binary findings: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA,
PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture. Metric: macro ROC AUC over the 12.
Train has 4,407 studies (~530 GiB of uncompressed DICOM) but only 58 of them carry image-level
("gold") labels; the rest come with a free-text, multilingual radiology report. The hidden test
set has images only.

WHAT THIS FILE DOES (end to end)
--------------------------------
1. build-cache   DICOM -> one uint8 volume per series: InstanceNumber order, unreadable slices
                 neighbour-filled, constant 130 mm centre crop at native resolution, 3-D series
                 slab-averaged to ~3 mm, at most 40 slices, resized to 384 px, robust 1-99 %
                 windowing. Series are routed to slots (plane x fluid-sensitive) with a ranking
                 rule; a missing fat-suppressed slot borrows the T1 series of the same plane.
2. make-folds    5 site-grouped folds (the hospital is the group; language ~ site), balanced for
                 size, gold count and language.
3. make-targets  Soft targets + per-cell weights from the report-label table: empirical-Bayes
                 calibration of each (label, report-state) cell against the gold rows, PAVA
                 monotone, voter-disagreement down-weighting; then the "v3 pseudo" step:
                 target = 0.5 x label + 0.5 x teacher OOF on every cell, and the weight of
                 "report silent on an under-reported finding" cells raised 0.1 -> 0.5.
4. train         KneeMIL: a 2-D ConvNeXt-V2-tiny backbone over 2.5-D triplets [i-1, i, i+1]
                 (12 per view in training, 24 at inference), gated-attention MIL per view, a
                 learned [MISSING] vector for absent views, concat fusion + presence bits ->
                 MLP -> 12 logits, plus per-view auxiliary heads. 5 views: sag_fs, cor_fs,
                 ax_fs, sag_t1 at 130 mm and a second sag_fs view centre-cropped to 90 mm (menisci/
                 ACL). Weighted soft BCE, AdamW (backbone 1e-4, heads 1e-3), cosine + warmup,
                 EMA 0.999, bf16, manifold mixup on the view embeddings, 8 epochs, last epoch kept.
5. predict       Raw test DICOM -> same preprocessing -> mean of the members' sigmoids.

THE LABEL STAGE (not in this file)
----------------------------------
The report-label table that make-targets consumes was produced offline: Qwen3-32B-AWQ served
locally with vLLM read every report (two prompt variants), a multilingual rule extractor, a
Gemma-3-27B voter and a public label table were merged into per-(study, finding) states
{affirmed, hedged, negated, absent} + severity, with evidence-quote checks. Columns used here:
StudyInstanceUID, {key}_state, {key}_severity, optional further voters {key}_state_v2/_v3 ...,
parse_ok, postoperative, {key}_evidence_ok. Prompts and the extraction code are left out.

RESULTS
-------
Shipped entry "main_v5_C" = 6 full-data models of this exact recipe (seeds 2026, 2126, 2226,
2326, 2426, 2526), equal weights, no TTA: public LB 0.948 (top of LB at the time ~0.961).
Gold-58 macro AUC (held out from every model and target): single seeds 0.918-0.922, the
6-seed mean 0.9254. History: MVP 0.865 -> report labels 0.931 -> 3-arm blend 0.933 ->
pseudo-label targets 0.943 -> full-data, arm-C only 0.948.

TRIED AND REJECTED (measured; please don't re-suggest without a new angle)
-------------------------------------------------------------------------
- Stacking (per-label LR / global weights over 10 models, nested folds): lost to equal weights.
- Pseudo-label round 2 (teacher = our own pseudo-label students): negative.
- Blending other arms (convnext_small, effnetv2_s) after pseudo-labelling: arms converged, the
  weaker ones dilute C (3-arm 0.945 vs C-only 0.948).
- z-offset TTA: +0.0005, dropped. Crop TTA (115/130 mm second view): gold +0.0027, LB 0.947 vs
  0.948 - tie at 2x inference, not adopted. Rotation TTA: nothing.
- Pairwise ranking loss (0.12 weight, memory bank): gold -0.0000 over 3 seeds.
- MRI intensity augmentation (bias field, gamma, blur, noise): gold -0.0001 over 3 seeds.
- Per-label attention heads, transformer head over triplet tokens (deeper, stochastic depth):
  no gain / divergence at peak LR.
- An independent "pipeline B" (DINOv3 ViT-B/16 @288, own teacher, aux findings) merged 50/50:
  +0.012 before pseudo-labelling, -0.0004 after (both pipelines converge, rho ~0.90).
- Claude-labelled reports as a further voter: +0.004 marginal on a 481-report pilot, not pursued.
Decision rule throughout: accept only paired gold-58 gains that clear a pre-declared bar; the
public LB resolves only ~0.002-0.003.

USAGE
-----
  python pipelineA.py build-cache  --data DIR --cache DIR --split train --workers 24
  python pipelineA.py make-folds   --study-summary CSV --out folds.csv
  python pipelineA.py make-targets --labels labels.parquet --teacher-oof a.parquet b.parquet ...
                                   --train-csv DIR/train.csv --folds folds.csv --fold full --out full.parquet
  python pipelineA.py train        --cache DIR --targets full.parquet --folds folds.csv
                                   --train-csv DIR/train.csv --fold full --seed 2026 --out runs/s2026
  python pipelineA.py predict      --data DIR --split test --models runs/s2026 runs/s2126 ... --out submission.csv
  python pipelineA.py score        --pred a.parquet [--base b.parquet] --train-csv DIR/train.csv

Requirements: numpy, pandas, pyarrow, opencv-python, pydicom (+ pylibjpeg plugins for compressed
DICOM), torch, timm >= 1.0, safetensors, scikit-learn (folds only), pyyaml (not needed).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import multiprocessing
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(1)
log = logging.getLogger("pipelineA")


# =====================================================================================
# 1. CONSTANTS AND CONFIG
# =====================================================================================

LABELS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
LABEL_KEYS = ("acl", "mcl", "med_men", "lat_men", "med_oa", "lat_oa",
              "pf_oa", "effusion", "synovitis", "baker", "contusion", "fracture")
LABEL_TO_KEY = dict(zip(LABELS, LABEL_KEYS))
# Findings whose report mention rate is far below their gold prevalence (synovitis: affirmed in
# 12.5 % of reports, positive in 46.6 % of gold studies): "not mentioned" is NOT a negative.
UNDER_REPORTED_KEYS = frozenset({"synovitis", "fracture", "contusion", "baker", "med_oa", "lat_oa", "pf_oa"})

ID_COL, SERIES_COL = "StudyInstanceUID", "SeriesInstanceUID"
TARGET_COLS = [f"{k}_target" for k in LABEL_KEYS]
WEIGHT_COLS = [f"{k}_weight" for k in LABEL_KEYS]
PRED_COLS = [f"{k}_pred" for k in LABEL_KEYS]

# Slot = (Anatomical_Plane, Fluid_Sensitive) from {split}_series.csv; Fluid_Sensitive equals
# Fat_Suppression on every training series.
SLOT_OF = {("Sagittal", 1): "sag_fs", ("Coronal", 1): "cor_fs", ("Axial", 1): "ax_fs",
           ("Sagittal", 0): "sag_t1", ("Coronal", 0): "cor_t1", ("Axial", 0): "ax_t1"}
SLOTS = ("sag_fs", "cor_fs", "ax_fs", "sag_t1", "cor_t1", "ax_t1")
SLOT_FALLBACK = {"sag_fs": "sag_t1", "cor_fs": "cor_t1", "ax_fs": "ax_t1"}
FAT_COL = "Fat_Suppression"
PLANES = ("Sagittal", "Coronal", "Axial")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# A full-data run uses this as its "fold index" in every seed offset (folds use 0-4).
FULL_SEED_OFFSET = 99


@dataclass
class CacheConfig:
    crop_mm: float = 130.0
    px: int = 384
    max_slices: int = 40
    z_target_mm: float = 3.0
    p_lo: float = 1.0
    p_hi: float = 99.0
    # Series with more slices are 3-D when MRAcquisitionType is absent (every 2-D series in the
    # data has <= 60 slices).
    max_2d_slices: int = 60
    geom_min_studies: int = 30
    geom_min_agreement: float = 0.95


@dataclass
class ModelConfig:
    backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"
    img_px: int = 384
    # One entry per VIEW. sag_fs appears twice: the second copy is a 90 mm centre crop of the
    # same cached volume with its own [MISSING] vector and aux head, not a second series.
    slots: tuple[str, ...] = ("sag_fs", "cor_fs", "ax_fs", "sag_t1", "sag_fs")
    view_crop_mm: tuple[float, ...] = (130.0, 130.0, 130.0, 130.0, 90.0)
    k_train: int = 12
    k_infer: int = 24
    attn_hidden: int = 256
    fusion_hidden: int = 512
    aux_weight: float = 0.3
    drop_rate: float = 0.1
    drop_path: float = 0.1
    grad_checkpoint: bool = True


@dataclass
class TrainConfig:
    """configs/main_armc_pseudo_full.yaml, the recipe of every main_v5_C member."""
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    epochs: int = 8
    batch_studies: int = 4
    lr_backbone: float = 1e-4
    lr_heads: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float = 0.999
    grad_clip: float = 2.0
    num_workers: int = 12
    seed: int = 2026
    sibling_p: float = 0.3          # use a lower-ranked series of the same slot instead
    slot_dropout_p: float = 0.1     # drop a present view (only while > 2 remain)
    mixup_p: float = 0.3            # manifold mixup on the filled view embeddings
    mixup_alpha: float = 0.4
    crop_scale_mm: tuple[float, float] = (115.0, 130.0)
    rot_deg: float = 8.0
    bc_jitter: float = 0.03
    amp_dtype: str = "bfloat16"
    eval_batch_studies: int = 2
    # Shapes vary with the number of present views; autotuning every new shape costs ~60 s.
    cudnn_benchmark: bool = False
    # torch.compile of the backbone forward: ~+45 % img/s for one ~4 min compile per run.
    compile_backbone: bool = True


# =====================================================================================
# 2. DICOM READING AND SERIES GEOMETRY
# =====================================================================================


@dataclass(frozen=True)
class Header:
    path: str
    instance_number: int
    series_number: int
    rows: int
    cols: int
    spacing: float
    iop: tuple[float, ...]
    ipp: tuple[float, ...]
    slice_thickness: float
    spacing_between_slices: float
    manufacturer: str
    laterality: str
    series_description: str
    mr_acquisition_type: str


@dataclass(frozen=True)
class Geometry:
    spacing: float
    rows: int
    cols: int
    plane: str
    is_3d: bool
    z_step: float
    n_slices: int

    @property
    def fov_mm(self) -> float:
        return self.rows * self.spacing


def _float(ds: pydicom.Dataset, key: str, default: float) -> float:
    value = ds.get(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _str(ds: pydicom.Dataset, key: str) -> str:
    value = ds.get(key)
    return "" if value is None else str(value).strip()


def read_header(path: str) -> Header:
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    spacing = ds.get("PixelSpacing")
    return Header(
        path=str(path), instance_number=int(_float(ds, "InstanceNumber", 0)),
        series_number=int(_float(ds, "SeriesNumber", 0)), rows=int(ds.Rows), cols=int(ds.Columns),
        spacing=float(spacing[0]) if spacing else 1.0,
        iop=tuple(float(v) for v in ds.get("ImageOrientationPatient", [1, 0, 0, 0, 1, 0])),
        ipp=tuple(float(v) for v in ds.get("ImagePositionPatient", [0, 0, 0])),
        slice_thickness=_float(ds, "SliceThickness", 0.0),
        spacing_between_slices=_float(ds, "SpacingBetweenSlices", 0.0),
        manufacturer=_str(ds, "Manufacturer"), laterality=_str(ds, "Laterality"),
        series_description=_str(ds, "SeriesDescription"), mr_acquisition_type=_str(ds, "MRAcquisitionType"))


def read_slice(path: str) -> np.ndarray | None:
    """float32 pixels with RescaleSlope/Intercept; None when the payload is unreadable (two train
    files are truncated; a compressed syntax without decoder plugin raises too) - the slice is
    then filled from its neighbours instead of losing the series."""
    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array
    except Exception as exc:  # noqa: BLE001
        log.warning("unreadable pixel data in %s: %s", path, str(exc)[:120])
        return None
    if arr.ndim != 2:
        return None
    out = arr.astype(np.float32)
    slope, intercept = _float(ds, "RescaleSlope", 1.0), _float(ds, "RescaleIntercept", 0.0)
    if slope != 1.0:
        out *= np.float32(slope)
    if intercept != 0.0:
        out += np.float32(intercept)
    return out


def list_series_files(series_dir: str | Path) -> list[str]:
    return sorted(str(p) for p in Path(series_dir).iterdir() if p.suffix.lower() == ".dcm")


def sort_headers(headers) -> list[Header]:
    # InstanceNumber is monotone along the slice normal for every series in the data.
    return sorted(headers, key=lambda h: (h.instance_number, h.path))


def slice_normal(iop: Sequence[float]) -> np.ndarray:
    return np.cross(np.asarray(iop[:3], np.float64), np.asarray(iop[3:6], np.float64))


def image_centre(h: Header) -> np.ndarray:
    """Patient-space (LPS, mm) position of the image centre."""
    row_dir, col_dir = np.asarray(h.iop[:3]), np.asarray(h.iop[3:6])
    return (np.asarray(h.ipp, np.float64) + row_dir * (h.cols - 1) / 2.0 * h.spacing
            + col_dir * (h.rows - 1) / 2.0 * h.spacing)


def series_geometry(headers: Sequence[Header], max_2d_slices: int = 60) -> Geometry:
    first, n = headers[0], len(headers)
    normal = slice_normal(first.iop)
    pos = np.asarray([np.dot(h.ipp, normal) for h in headers])
    steps = np.abs(np.diff(pos)) if n > 1 else np.asarray([first.slice_thickness or 1.0])
    z_step = float(np.median(steps)) if steps.size else 0.0
    if z_step <= 0:
        z_step = first.spacing_between_slices or first.slice_thickness or 1.0
    acq = first.mr_acquisition_type.upper()
    return Geometry(spacing=float(np.median([h.spacing for h in headers])), rows=first.rows, cols=first.cols,
                    plane=PLANES[int(np.argmax(np.abs(normal)))],
                    is_3d=acq == "3D" or (acq == "" and n > max_2d_slices), z_step=z_step, n_slices=n)


# =====================================================================================
# 3. SERIES -> SLOT ROUTING
# =====================================================================================


def slot_of(plane: str, fs) -> str:
    key = (str(plane).strip().capitalize(), int(fs))
    if key not in SLOT_OF:
        raise ValueError(f"no slot for plane={plane!r} fs={fs!r}")
    return SLOT_OF[key]


def flag_mismatch(fs, fat) -> int:
    """1 when Fat_Suppression is given and differs from Fluid_Sensitive. Every training series has
    equal flags; the host said test series may not (e.g. T2 without fat-sat). Such a series keeps
    its Fluid_Sensitive slot but ranks after every candidate whose flags match. Missing or
    unparseable Fat_Suppression counts as matching, which keeps routing identical to training."""
    try:
        if fat is None or pd.isna(fat) or str(fat).strip() == "":
            return 0
        return int(int(float(fat)) != int(fs))
    except (TypeError, ValueError):
        return 0


def _rank_key(row: Mapping) -> tuple[int, int, int, int]:
    # Matching flags first, then 2-D before 3-D, most slices, lowest SeriesNumber.
    return (int(row.get("flag_mismatch", 0) or 0), int(bool(row.get("is_3d", False))),
            -int(row.get("n_slices", 0) or 0), int(row.get("series_number", 0) or 0))


def assign_slots(series_df: pd.DataFrame, slots: Sequence[str] = SLOTS) -> pd.DataFrame:
    """One row per (study, slot, series) with its rank; plus fallback rows that lend the T1 series
    of a plane to that plane's missing fluid-sensitive slot. Cache index and inference both route
    through here, so a test volume is chosen exactly like a training one."""
    df = series_df.copy()
    df["native_slot"] = [slot_of(p, fs) for p, fs in zip(df["Anatomical_Plane"], df["Fluid_Sensitive"])]
    for col in ("is_3d", "n_slices", "series_number"):
        if col not in df.columns:
            df[col] = 0
    df["flag_mismatch"] = ([flag_mismatch(fs, fat) for fs, fat in zip(df["Fluid_Sensitive"], df[FAT_COL])]
                           if FAT_COL in df.columns else 0)
    if "ok" in df.columns:
        df = df[df["ok"].astype(bool)].copy()
    df["_key"] = [_rank_key(r) for r in df[["flag_mismatch", "is_3d", "n_slices", "series_number"]].to_dict("records")]
    df = df.sort_values([ID_COL, "native_slot", "_key", SERIES_COL], kind="stable")
    df["rank"] = df.groupby([ID_COL, "native_slot"]).cumcount()
    df["slot"], df["fallback"] = df["native_slot"], False
    native = df[[ID_COL, "slot", SERIES_COL, "rank", "fallback", "native_slot"]]
    present = native.groupby(ID_COL)["slot"].agg(set)
    extra = []
    for slot, source in SLOT_FALLBACK.items():
        if slot not in slots:
            continue
        missing = [s for s, have in present.items() if slot not in have and source in have]
        rows = native[native[ID_COL].isin(missing) & (native["slot"] == source)].copy()
        rows["slot"], rows["fallback"] = slot, True
        extra.append(rows)
    out = pd.concat([native, *extra], ignore_index=True) if extra else native.reset_index(drop=True)
    out = out[out["slot"].isin(list(slots))]
    out["selected"] = out["rank"] == 0
    return out.sort_values([ID_COL, "slot", "rank"]).reset_index(drop=True)


def slot_table(assigned: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """{study: {slot: [series in rank order]}}, fallback rows included."""
    table: dict[str, dict[str, list[str]]] = {}
    for study, slot, series in zip(assigned[ID_COL], assigned["slot"], assigned[SERIES_COL]):
        table.setdefault(study, {}).setdefault(slot, []).append(series)
    return table


# =====================================================================================
# 4. LATERALITY (recorded in the cache index; the model does not flip or condition on it)
# =====================================================================================

TAG_CONF_CONSISTENT, TAG_CONF_CONTRADICTORY, DESC_CONF = 0.95, 0.6, 0.8
_DESC_PATTERNS = (
    (re.compile(r"(^|[\s_\-])(RT|R|RIGHT|RECHTS|DROIT|DER)([\s_\-]|$)", re.I), "R"),
    (re.compile(r"(^|[\s_\-])(LT|L|LEFT|LINKS|GAUCHE|IZQ)([\s_\-]|$)", re.I), "L"),
)
_FAMILIES = (("siemens", "siemens"), ("philips", "philips"), ("gehc", "ge"), ("ge medical", "ge"),
             ("toshiba", "canon"), ("canon", "canon"), ("fujifilm", "hitachi"), ("hitachi", "hitachi"))


def manufacturer_family(name: str | None) -> str:
    low = (name or "").strip().lower()
    for needle, family in _FAMILIES:
        if needle in low:
            return family
    return low or "unknown"


def parse_laterality_tag(value: str | None) -> str:
    v = (value or "").strip().upper()
    return "L" if v in ("L", "LEFT") else "R" if v in ("R", "RIGHT") else ""


def parse_desc_side(desc: str | None) -> str:
    text = (desc or "").strip()
    if not text or text == "DummySeriesDesc!":
        return ""
    hits = {side for pat, side in _DESC_PATTERNS if pat.search(text)}
    return hits.pop() if len(hits) == 1 else ""


def estimate_sides(study_df: pd.DataFrame, min_studies: int = 30, min_agreement: float = 0.95) -> pd.DataFrame:
    """Side per study from the Laterality tag, else the SeriesDescription, else the sign of the FOV
    centre x (LPS) - the last only for a manufacturer family whose geometric convention agrees with
    the tags on >= min_agreement of >= min_studies consistently tagged studies."""
    cx = pd.to_numeric(study_df["centre_x"], errors="coerce")
    tagged = study_df[study_df["side_tag"].isin(["L", "R"]) & (study_df["tag_conf"] >= TAG_CONF_CONSISTENT)
                      & np.isfinite(cx)]
    rules = {}
    for fam, grp in tagged.groupby(tagged["manufacturer"].map(manufacturer_family)):
        left_pos = ((grp["centre_x"] > 0) == (grp["side_tag"] == "L")).mean()
        agreement = float(max(left_pos, 1.0 - left_pos))
        rules[fam] = (1 if left_pos >= 0.5 else -1, agreement, len(grp) >= min_studies and agreement >= min_agreement)
    est, conf = [], []
    for r in study_df.to_dict("records"):
        if r["side_tag"]:
            e, c = r["side_tag"], r["tag_conf"]
        elif r["side_desc"]:
            e, c = r["side_desc"], DESC_CONF
        else:
            rule = rules.get(manufacturer_family(r["manufacturer"]))
            if rule and rule[2] and np.isfinite(r["centre_x"]):
                e, c = ("L" if r["centre_x"] * rule[0] > 0 else "R"), rule[1]
            else:
                e, c = "unknown", 0.0
        est.append(e)
        conf.append(c)
    return study_df.assign(side_est=est, side_conf=np.asarray(conf, np.float32))


# =====================================================================================
# 5. SERIES -> CACHED VOLUME (shared by build-cache and predict)
# =====================================================================================


def fill_missing(slices: Sequence[np.ndarray | None]) -> list[np.ndarray]:
    good = [i for i, s in enumerate(slices) if s is not None]
    if not good:
        raise ValueError("no readable slice in series")
    out = []
    for i, s in enumerate(slices):
        if s is not None:
            out.append(s)
            continue
        prev = max((j for j in good if j < i), default=None)
        nxt = min((j for j in good if j > i), default=None)
        if prev is not None and nxt is not None:
            out.append((slices[prev].astype(np.float32) + slices[nxt].astype(np.float32)) / 2.0)
        else:
            out.append(slices[prev if prev is not None else nxt].astype(np.float32))
    return out


def crop_centre(vol: np.ndarray, size_px: int) -> np.ndarray:
    """Square centre crop of (N, H, W), zero-padded when the FOV is smaller than the crop."""
    n, h, w = vol.shape
    if h == size_px and w == size_px:
        return vol
    out = np.zeros((n, size_px, size_px), dtype=vol.dtype)
    y0, x0 = (h - size_px) // 2, (w - size_px) // 2
    sy0, sx0, dy0, dx0 = max(y0, 0), max(x0, 0), max(-y0, 0), max(-x0, 0)
    ny, nx = min(h - sy0, size_px - dy0), min(w - sx0, size_px - dx0)
    out[:, dy0:dy0 + ny, dx0:dx0 + nx] = vol[:, sy0:sy0 + ny, sx0:sx0 + nx]
    return out


def resize_stack(vol: np.ndarray, px: int) -> np.ndarray:
    n, h, w = vol.shape
    if h == px and w == px:
        return np.ascontiguousarray(vol, dtype=np.float32)
    # INTER_AREA is an anti-aliasing filter only when shrinking; enlarging with it degenerates to
    # nearest-neighbour, so coarse series (spacing > 0.34 mm at 130 mm / 384 px) use linear.
    interp = cv2.INTER_AREA if px < min(h, w) else cv2.INTER_LINEAR
    out = np.empty((n, px, px), dtype=np.float32)
    for i in range(n):
        out[i] = cv2.resize(np.ascontiguousarray(vol[i], dtype=np.float32), (px, px), interpolation=interp)
    return out


def resample_z(vol: np.ndarray, z_step: float, target_mm: float) -> tuple[np.ndarray, float]:
    """Average consecutive slabs so a thin 3-D series approaches target_mm; never interpolates."""
    if z_step <= 0 or z_step >= target_mm * 0.75:
        return vol, z_step
    factor = int(round(target_mm / z_step))
    if factor <= 1:
        return vol, z_step
    n_out = int(np.ceil(vol.shape[0] / factor))
    out = np.empty((n_out,) + vol.shape[1:], dtype=np.float32)
    for k in range(n_out):
        out[k] = vol[k * factor:(k + 1) * factor].mean(axis=0, dtype=np.float32)
    return out, z_step * factor


def series_to_volume(slices: Sequence[np.ndarray | None], geom: Geometry, cfg: CacheConfig) -> np.ndarray:
    """Slices in InstanceNumber order (None = unreadable) -> uint8 (N <= 40, 384, 384)."""
    vol = np.stack([s.astype(np.float32, copy=False) for s in fill_missing(slices)])
    # Crop at native resolution first: a 320-slice 640x1280 3-D series would otherwise cost
    # 1 GB of float32 per worker before any reduction.
    vol = crop_centre(vol, max(1, int(round(cfg.crop_mm / geom.spacing))))
    if geom.is_3d:
        vol, _ = resample_z(vol, geom.z_step, cfg.z_target_mm)
    if vol.shape[0] > cfg.max_slices:
        vol = vol[np.unique(np.linspace(0, vol.shape[0] - 1, cfg.max_slices).round().astype(int))]
    vol = resize_stack(vol, cfg.px)
    lo, hi = (float(v) for v in np.percentile(vol, [cfg.p_lo, cfg.p_hi]))
    if hi <= lo:
        hi = lo + 1.0
    scaled = (vol - lo) * (255.0 / (hi - lo))
    np.clip(scaled, 0.0, 255.0, out=scaled)
    return np.rint(scaled).astype(np.uint8)


def series_volume(series_dir: str | Path, cfg: CacheConfig,
                  headers: Sequence[Header] | None = None) -> tuple[np.ndarray, list[Header], Geometry]:
    if headers is None:
        files = list_series_files(series_dir)
        if not files:
            raise FileNotFoundError(f"no .dcm files in {series_dir}")
        headers = sort_headers(read_header(p) for p in files)
    geom = series_geometry(headers, cfg.max_2d_slices)
    return series_to_volume([read_slice(h.path) for h in headers], geom, cfg), list(headers), geom


# =====================================================================================
# 6. CACHE BUILD AND LOADING   layout: <cache>/<split>/<study>/<series>.npy + index.parquet
# =====================================================================================


def _cache_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Pool worker: one series -> .npy + index row; never raises."""
    cfg = CacheConfig(**task["cfg"])
    npy = Path(task["npy"])
    rec = {"split": task["split"], ID_COL: task["study"], SERIES_COL: task["series"], "plane": task["plane"],
           "fs": task["fs"], "fat": task["fat"], "ok": False, "error": ""}
    try:
        rec["slot"] = slot_of(task["plane"], task["fs"])
        vol, headers, geom = series_volume(task["series_dir"], cfg)
        npy.parent.mkdir(parents=True, exist_ok=True)
        tmp = npy.with_name(npy.name + ".tmp.npy")
        np.save(tmp, vol)
        os.replace(tmp, npy)
        first = headers[0]
        rec.update(ok=True, is_3d=bool(geom.is_3d), n_src=len(headers), n_cached=int(vol.shape[0]),
                   spacing_mm=geom.spacing, fov_mm=geom.fov_mm, series_number=first.series_number,
                   manufacturer=first.manufacturer,
                   centre_x=float(np.mean([image_centre(h)[0] for h in headers])),
                   side_tag=parse_laterality_tag(first.laterality),
                   side_desc=parse_desc_side(first.series_description))
    except Exception as exc:  # noqa: BLE001 - one bad series must not kill the build
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return rec


def _flag_or_none(value) -> int | None:
    try:
        return None if value is None or pd.isna(value) else int(float(value))
    except (TypeError, ValueError):
        return None


def build_cache(data_dir: str, cache_dir: str, split: str, cfg: CacheConfig, workers: int = 24,
                limit_studies: int | None = None, studies: Sequence[str] | None = None) -> pd.DataFrame:
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    series_df = pd.read_csv(data_dir / f"{split}_series.csv", dtype=str)
    if studies:
        series_df = series_df[series_df[ID_COL].isin(set(studies))]
    elif limit_studies:
        series_df = series_df[series_df[ID_COL].isin(series_df[ID_COL].drop_duplicates().head(limit_studies))]
    tasks = [{"split": split, "study": getattr(r, ID_COL), "series": getattr(r, SERIES_COL),
              "plane": r.Anatomical_Plane, "fs": int(r.Fluid_Sensitive), "fat": _flag_or_none(getattr(r, FAT_COL, None)),
              "series_dir": str(data_dir / f"{split}_series" / getattr(r, ID_COL) / getattr(r, SERIES_COL)),
              "npy": str(cache_dir / split / getattr(r, ID_COL) / f"{getattr(r, SERIES_COL)}.npy"),
              "cfg": asdict(cfg)} for r in series_df.itertuples(index=False)]
    log.info("cache %s/%s: %d series, %d workers", cache_dir, split, len(tasks), workers)
    # The parent is multi-threaded (pyarrow, cv2): fork() risks deadlocks in the workers.
    with multiprocessing.get_context("forkserver").Pool(processes=workers) as pool:
        records = list(pool.imap_unordered(_cache_task, tasks, chunksize=1))
    df = pd.DataFrame(records)
    ok = df[df["ok"]]
    rows = []
    for study, grp in ok.groupby(ID_COL):
        tag_votes = Counter(v for v in grp["side_tag"] if v)
        tag, share = (tag_votes.most_common(1)[0][0], tag_votes.most_common(1)[0][1] / sum(tag_votes.values())) \
            if tag_votes else ("", 0.0)
        desc_votes = Counter(v for v in grp["side_desc"] if v)
        cx = np.nan
        for plane in ("Axial", "Coronal", "Sagittal"):
            xs = grp.loc[grp["plane"] == plane, "centre_x"].dropna()
            if len(xs):
                cx = float(xs.mean())
                break
        rows.append({ID_COL: study, "study_side_tag": tag,
                     "tag_conf": 0.0 if not tag else (TAG_CONF_CONSISTENT if share == 1.0 else TAG_CONF_CONTRADICTORY),
                     "study_side_desc": desc_votes.most_common(1)[0][0] if desc_votes else "",
                     "study_centre_x": cx, "study_manufacturer": grp["manufacturer"].mode().iloc[0]})
    if rows:
        sides = estimate_sides(pd.DataFrame(rows).rename(columns={
            "study_side_tag": "side_tag", "study_side_desc": "side_desc", "study_centre_x": "centre_x",
            "study_manufacturer": "manufacturer"}), cfg.geom_min_studies, cfg.geom_min_agreement)
        df = df.merge(sides[[ID_COL, "side_est", "side_conf"]], on=ID_COL, how="left")
    index_path = cache_dir / "index.parquet"
    if index_path.exists():
        old = pd.read_parquet(index_path)
        df = pd.concat([old[old["split"] != split], df], ignore_index=True)
    df.sort_values(["split", ID_COL, SERIES_COL]).reset_index(drop=True).to_parquet(index_path, index=False)
    log.info("split %s: %d series, %d ok", split, len(records), int(ok.shape[0]))
    return df


class CacheIndex:
    """index.parquet -> per split {study: {slot: [series in rank order]}} with fallbacks resolved."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.df = pd.read_parquet(self.cache_dir / "index.parquet")
        self.slots: dict[str, dict[str, dict[str, list[str]]]] = {}
        for split, ok in self.df[self.df["ok"].astype(bool)].groupby("split"):
            routed = ok.rename(columns={"n_src": "n_slices"}).assign(
                Anatomical_Plane=ok["plane"], Fluid_Sensitive=ok["fs"], **{FAT_COL: ok["fat"]})
            self.slots[str(split)] = slot_table(assign_slots(routed))

    def studies(self, split: str) -> list[str]:
        return sorted(self.slots.get(split, {}))

    def series_for(self, split: str, study: str, slot: str) -> list[str]:
        return self.slots.get(split, {}).get(study, {}).get(slot, [])

    def load(self, split: str, study: str, slots: Sequence[str], rng: np.random.Generator | None = None,
             sibling_p: float = 0.3) -> dict[str, np.ndarray]:
        """Rank-0 series per slot; in training a random same-slot sibling replaces it with
        probability sibling_p. Missing slots are absent from the dict."""
        out = {}
        for slot in slots:
            cand = self.series_for(split, study, slot)
            if not cand:
                continue
            chosen = cand[0]
            if rng is not None and len(cand) > 1 and rng.random() < sibling_p:
                chosen = cand[int(rng.integers(1, len(cand)))]
            out[slot] = np.load(self.cache_dir / split / study / f"{chosen}.npy", mmap_mode="r")
        return out


# =====================================================================================
# 7. FOLDS (site-grouped)
# =====================================================================================


def make_folds(study_summary_csv: str, n_splits: int = 5, seed: int = 2026, n_trials: int = 200,
               w_size: float = 1.0, w_gold: float = 1.0, w_lang: float = 0.5) -> pd.DataFrame:
    """Groups = site_key. sklearn's StratifiedGroupKFold only balances the stratum, and with
    language ~ site it lands at 14-25 % fold sizes; this seeded greedy search (best of n_trials)
    balances fold size, gold count and language shares together.
    Input CSV columns: StudyInstanceUID, site_key, lang, is_gold."""
    df = pd.read_csv(study_summary_csv, dtype=str, keep_default_na=False)
    df = pd.DataFrame({ID_COL: df[ID_COL], "site_key": df["site_key"], "lang": df["lang"],
                       "is_gold": df["is_gold"].str.lower().isin(["true", "1", "yes"])}
                      ).sort_values(ID_COL).reset_index(drop=True)
    langs = sorted(df["lang"].unique())
    groups = df.groupby("site_key")
    names = list(groups.groups)
    n = groups.size().reindex(names).to_numpy(float)
    gold = groups["is_gold"].sum().reindex(names).to_numpy(float)
    lang = pd.crosstab(df["site_key"], df["lang"]).reindex(index=names, columns=langs).to_numpy(float)
    n_tot, g_tot, l_tot = n.sum(), max(gold.sum(), 1.0), np.maximum(lang.sum(axis=0), 1.0)
    target, k = 1.0 / n_splits, n_splits
    rng = np.random.default_rng(seed)

    def cost(fn, fg, fl):
        return (w_size * ((fn / n_tot - target) ** 2).sum() + w_gold * ((fg / g_tot - target) ** 2).sum()
                + w_lang * ((fl / l_tot - target) ** 2).sum(axis=0).mean())

    best_cost, best = np.inf, None
    for _ in range(n_trials):
        order = np.argsort(-n * rng.uniform(0.6, 1.4, size=len(n)), kind="stable")
        fn, fg, fl = np.zeros(k), np.zeros(k), np.zeros((k, len(langs)))
        assign = np.empty(len(n), dtype=int)
        for gi in order:
            costs = []
            for f in range(k):
                fn[f] += n[gi]; fg[f] += gold[gi]; fl[f] += lang[gi]
                costs.append(cost(fn, fg, fl))
                fn[f] -= n[gi]; fg[f] -= gold[gi]; fl[f] -= lang[gi]
            f = int(np.argmin(costs))
            assign[gi] = f
            fn[f] += n[gi]; fg[f] += gold[gi]; fl[f] += lang[gi]
        total = cost(fn, fg, fl)
        if total < best_cost:
            best_cost, best = total, assign.copy()
    df["fold"] = df["site_key"].map(dict(zip(names, best))).astype(int)
    return df[[ID_COL, "fold", "site_key", "lang", "is_gold"]]


def load_folds(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={ID_COL: str, "fold": int, "site_key": str, "lang": str})
    df["is_gold"] = df["is_gold"].astype(str).str.lower().isin(["true", "1"])
    return df


# =====================================================================================
# 8. REPORT LABELS -> SOFT TARGETS (+ v3 pseudo-label mix)
# =====================================================================================

CELLS = ("affirmed", "affirmed_mild", "hedged", "absent", "negated")
MONOTONE_CHAIN = ("affirmed", "hedged", "absent", "negated")


@dataclass
class TargetsConfig:
    """configs/targets_v1_1.yaml. Base value = prior probability that a cell is gold-positive."""
    prior_strength: float = 8.0
    base: dict[str, Any] = field(default_factory=lambda: {
        "affirmed": 0.92, "hedged": 0.50, "negated": 0.04,
        "absent_well_reported": 0.06, "absent_under_reported": 0.15,
        # The host's thresholds make a "mild" report finding gold-negative for several labels.
        "affirmed_mild": {"default": 0.35, "med_men": 0.25, "lat_men": 0.25, "effusion": 0.25, "baker": 0.25,
                          "mcl": 0.20, "acl": 0.35, "med_oa": 0.35, "lat_oa": 0.35, "pf_oa": 0.35,
                          "contusion": 0.70, "fracture": 0.70}})
    weights: dict[str, float] = field(default_factory=lambda: {
        "affirmed": 1.0, "negated": 1.0, "hedged": 0.7, "absent_well_reported": 0.5,
        "absent_under_reported": 0.1, "parse_fail": 0.2, "postoperative": 0.2,
        "hallucinated_evidence": 0.3, "disagreement": 0.5})


# v3 pseudo-labels: every cell becomes 0.5 x label + 0.5 x teacher (the RSNA-2024 lumbar 2nd place
# mix, fixed a priori). The teacher OOF of a fold-j study comes from models that trained on other
# folds' weak rows, so a student's weak OOF becomes optimistic - only gold-58 and the LB stay clean.
V3_MIX = 0.5
# "Report silent" on an under-reported finding had weight 0.1 ("don't learn from it"); with half
# the target now an image-model reading it is about as informative as a well-reported absent.
V3_SILENT_WEIGHT = 0.5


def voter_suffixes(table: pd.DataFrame) -> tuple[str, ...]:
    key = LABEL_KEYS[0]
    out = [""] if f"{key}_state" in table.columns else []
    out += [c[len(key) + 6:] for c in table.columns if c.startswith(f"{key}_state_v")]
    return tuple(out)


def cell_of(state: str, severity: str) -> str:
    return "affirmed_mild" if state == "affirmed" and severity == "mild" else state


def pava_decreasing(values: list[float], weights: list[float]) -> list[float]:
    blocks: list[list[float]] = []
    for v, w in zip(values, weights):
        blocks.append([v, w, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v1, w1, c1 = blocks[-2]
            v2, w2, c2 = blocks[-1]
            wt = w1 + w2
            blocks[-2:] = [[(v1 * w1 + v2 * w2) / wt if wt > 0 else (v1 + v2) / 2, wt, c1 + c2]]
    out: list[float] = []
    for v, _, c in blocks:
        out.extend([v] * c)
    return out


def calibrate(labels: pd.DataFrame, gold: pd.DataFrame, gold_ids: set[str],
              cfg: TargetsConfig) -> dict[str, dict[str, float]]:
    """Empirical-Bayes posterior mean P(gold positive | label, cell) with a Beta prior of strength
    8 around the base value, from the gold rows in gold_ids (all 58 for a full-data run, the other
    folds' gold for fold k); then PAVA enforces affirmed >= hedged >= absent >= negated."""
    mild = cfg.base["affirmed_mild"]
    g = gold[gold[ID_COL].isin(gold_ids)].merge(labels, on=ID_COL, how="inner", suffixes=("", "_lt"))
    s = float(cfg.prior_strength)
    values: dict[str, dict[str, float]] = {}
    for key in LABEL_KEYS:
        base = {"affirmed": cfg.base["affirmed"], "affirmed_mild": mild.get(key, mild["default"]),
                "hedged": cfg.base["hedged"], "negated": cfg.base["negated"],
                "absent": cfg.base["absent_under_reported" if key in UNDER_REPORTED_KEYS else "absent_well_reported"]}
        counts = {c: [0, 0] for c in CELLS}
        for st, sv, y in zip(g[f"{key}_state"].astype(str), g[f"{key}_severity"].astype(str), g[key]):
            c = cell_of(st, sv)
            counts[c][0] += 1
            counts[c][1] += int(float(y) > 0.5)
        v = {c: (s * base[c] + counts[c][1]) / (s + counts[c][0]) for c in CELLS}
        chain = pava_decreasing([v[c] for c in MONOTONE_CHAIN], [s + counts[c][0] for c in MONOTONE_CHAIN])
        v.update(dict(zip(MONOTONE_CHAIN, (float(x) for x in chain))))
        v["affirmed_mild"] = float(min(max(v["affirmed_mild"], v["negated"]), v["affirmed"]))
        values[key] = v
    return values


def make_label_targets(labels: pd.DataFrame, calib: Mapping[str, Mapping[str, float]], cfg: TargetsConfig,
                       exclude_ids: set[str]) -> pd.DataFrame:
    """Target = mean over voters of the calibrated cell value. Weight by cell, times a disagreement
    factor m = d + (1 - d)(2f - 1) (f = modal voter share, d = 0.5: 1.0 when unanimous, d at an
    even split); hallucinated evidence caps it at 0.3; a parse failure or postoperative report
    overrides every weight of the row with 0.2."""
    df = labels[~labels[ID_COL].isin(exclude_ids)].reset_index(drop=True)
    out = pd.DataFrame({ID_COL: df[ID_COL].to_numpy()})
    parse_ok = df["parse_ok"].to_numpy(bool) if "parse_ok" in df.columns else np.ones(len(df), bool)
    postop = df["postoperative"].to_numpy(bool) if "postoperative" in df.columns else np.zeros(len(df), bool)
    w = cfg.weights
    row_override = np.where(~parse_ok, w["parse_fail"], np.where(postop, w["postoperative"], np.nan))
    d = float(w["disagreement"])
    for key in LABEL_KEYS:
        per_voter = [[cell_of(a, b) for a, b in zip(df[f"{key}_state{sfx}"].astype(str),
                                                     df[f"{key}_severity{sfx}"].astype(str))]
                     for sfx in voter_suffixes(df)]
        target = np.mean([[calib[key][c] for c in cells] for cells in per_voter], axis=0)

        def weight_of(c: str) -> float:
            if c in ("affirmed", "affirmed_mild"):
                return w["affirmed"]
            if c in ("negated", "hedged"):
                return w[c]
            return w["absent_under_reported" if key in UNDER_REPORTED_KEYS else "absent_well_reported"]

        weight = np.array([weight_of(c) for c in per_voter[0]], np.float64)
        if len(per_voter) > 1:
            votes = np.array(per_voter, dtype=object).T
            share = np.array([max(np.unique(r, return_counts=True)[1]) for r in votes]) / len(per_voter)
            weight = weight * (d + (1.0 - d) * (2.0 * share - 1.0))
        if f"{key}_evidence_ok" in df.columns:
            weight = np.where(~df[f"{key}_evidence_ok"].to_numpy(bool),
                              np.minimum(weight, w["hallucinated_evidence"]), weight)
        weight = np.where(np.isnan(row_override), weight, row_override)
        out[f"{key}_target"] = np.clip(target, 0, 1).astype(np.float32)
        out[f"{key}_weight"] = np.clip(weight, 0, 1).astype(np.float32)
    return out


def teacher_mean(oofs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight mean of several OOF probability tables on their common studies."""
    idx = [o.assign(**{ID_COL: o[ID_COL].astype(str)}).set_index(ID_COL) for o in oofs]
    common = sorted(set.intersection(*(set(o.index) for o in idx)))
    arr = np.mean([o.loc[common, PRED_COLS].to_numpy(np.float64) for o in idx], axis=0)
    out = pd.DataFrame(arr, columns=PRED_COLS)
    out.insert(0, ID_COL, common)
    return out


def make_pseudo_targets(labels: pd.DataFrame, teacher_oofs: Sequence[pd.DataFrame], gold: pd.DataFrame,
                        folds: pd.DataFrame, fold: str, cfg: TargetsConfig | None = None) -> pd.DataFrame:
    """fold "full": calibration on all 58 gold, every non-gold row mixed with the teacher
    (= data/targets/v3_pseudo_full, the main_v5_C targets). fold k: calibration on the gold of the
    other folds and only rows outside fold k mixed, so fold k's validation stays label-only."""
    cfg = cfg or TargetsConfig()
    labels = labels.assign(**{ID_COL: labels[ID_COL].astype(str)})
    gold_ids = set(folds.loc[folds["is_gold"], ID_COL].astype(str))
    if fold == "full":
        calib_ids, mix_ids = gold_ids, None
    else:
        k = int(fold)
        fold_of = dict(zip(folds[ID_COL].astype(str), folds["fold"]))
        calib_ids = {s for s in gold_ids if fold_of.get(s) != k}
        mix_ids = {s for s, f in fold_of.items() if f != k}
    calib = calibrate(labels, gold, calib_ids, cfg)
    lt = make_label_targets(labels, calib, cfg, exclude_ids=gold_ids)
    rw = make_label_targets(labels, calib, replace(cfg, weights={**cfg.weights, "absent_under_reported": V3_SILENT_WEIGHT}),
                            exclude_ids=gold_ids)
    teacher = teacher_mean(teacher_oofs).set_index(ID_COL)
    ids = lt[ID_COL].astype(str)
    m = ids.isin(teacher.index).to_numpy()
    if mix_ids is not None:
        m &= ids.isin(mix_ids).to_numpy()
    aligned = teacher.reindex(ids)
    out = lt.copy()
    for k in LABEL_KEYS:
        lab = lt[f"{k}_target"].to_numpy(np.float64)
        mixed = (1.0 - V3_MIX) * lab + V3_MIX * np.nan_to_num(aligned[f"{k}_pred"].to_numpy(np.float64))
        out[f"{k}_target"] = np.where(m, np.clip(mixed, 0, 1), lab).astype(np.float32)
        out[f"{k}_weight"] = np.where(m, rw[f"{k}_weight"], lt[f"{k}_weight"]).astype(np.float32)
    log.info("targets: %d rows, %d mixed with the teacher", len(out), int(m.sum()))
    return out


def gold_frame(train_csv: str) -> pd.DataFrame:
    """The 58 gold rows of train.csv with snake-case 0/1 columns."""
    df = pd.read_csv(train_csv)
    g = df[df[LABELS[0]].notna()].rename(columns=LABEL_TO_KEY)
    g = g[[ID_COL, *LABEL_KEYS]].reset_index(drop=True)
    g[ID_COL] = g[ID_COL].astype(str)
    return g


# =====================================================================================
# 9. DATASET: K triplets per view, augmentation on uint8
# =====================================================================================


def sample_centres(n: int, k: int, rng: np.random.Generator | None, z_offset: float = 0.0) -> np.ndarray:
    """K slice centres in [0, n): one per equal stratum - uniformly random inside it in training,
    the stratum middle (shifted by z_offset strata) at evaluation."""
    pos = np.arange(k, dtype=np.float64) + (rng.random(k) if rng is not None else 0.5 + z_offset)
    return np.clip(np.floor(pos * n / k), 0, n - 1).astype(np.int64)


def triplet_indices(centres: np.ndarray, n: int) -> np.ndarray:
    return np.clip(centres[:, None] + np.array([-1, 0, 1])[None, :], 0, n - 1)


def warp_slices(slices: np.ndarray, px: int, params: Mapping[str, float] | None) -> np.ndarray:
    """(N, H, W) uint8 -> (N, px, px). With params: rotate + zoom about the centre at cache
    resolution, INTER_AREA resize, then contrast/brightness; without: a plain resize."""
    n, h, w = slices.shape
    out = np.empty((n, px, px), dtype=np.uint8)
    if params is None:
        for i in range(n):
            out[i] = cv2.resize(slices[i], (px, px), interpolation=cv2.INTER_AREA) if (h, w) != (px, px) else slices[i]
        return out
    m = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), params["rot_deg"], params["scale"])
    for i in range(n):
        img = cv2.warpAffine(slices[i], m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        out[i] = cv2.resize(img, (px, px), interpolation=cv2.INTER_AREA) if (h, w) != (px, px) else img
    if params["contrast"] != 1.0 or params["brightness"] != 0.0:
        out = cv2.convertScaleAbs(out.reshape(-1, px), alpha=params["contrast"],
                                  beta=params["brightness"]).reshape(n, px, px)
    return out


def read_triplets(vol: np.ndarray, idx: np.ndarray, px: int, params: Mapping[str, float] | None) -> np.ndarray:
    """(K, 3, px, px) uint8; each needed slice is read from the memmap and warped once."""
    uniq, inverse = np.unique(idx, return_inverse=True)
    return warp_slices(np.asarray(vol[uniq]), px, params)[inverse.reshape(idx.shape)]


def augment_params(rng: np.random.Generator, cfg: TrainConfig, crop_mm: float, view_mm: float) -> dict[str, float]:
    """One random crop/rotation/brightness per series. The crop window is scaled by view/crop, so
    the 90 mm view jitters over [79.6, 90] mm exactly as the 130 mm view over [115, 130]."""
    lo, hi = cfg.crop_scale_mm
    r = float(view_mm) / float(crop_mm)
    return {"scale": crop_mm / float(rng.uniform(lo * r, hi * r)),
            "rot_deg": float(rng.uniform(-cfg.rot_deg, cfg.rot_deg)),
            "contrast": float(1.0 + rng.uniform(-cfg.bc_jitter, cfg.bc_jitter)),
            "brightness": float(rng.uniform(-cfg.bc_jitter, cfg.bc_jitter) * 255.0)}


def view_params(crop_mm: float, view_mm: float) -> dict[str, float] | None:
    """Deterministic centre crop to view_mm; None (plain resize) for the full cached field."""
    if float(view_mm) >= float(crop_mm):
        return None
    return {"scale": float(crop_mm) / float(view_mm), "rot_deg": 0.0, "contrast": 1.0, "brightness": 0.0}


def view_inputs(vols: Mapping[str, np.ndarray], mcfg: ModelConfig, crop_mm: float, k: int,
                rng: np.random.Generator | None = None, cfg: TrainConfig | None = None,
                present: np.ndarray | None = None, z_offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(S, K, 3, px, px) uint8 + present (S,). Training passes rng + cfg (random strata, random
    crop/rotation/brightness); evaluation and test passes neither (stratum middles, centre crops)."""
    slots, px = tuple(mcfg.slots), mcfg.img_px
    if present is None:
        present = np.array([s in vols for s in slots], dtype=bool)
    images = np.zeros((len(slots), k, 3, px, px), dtype=np.uint8)
    for j, slot in enumerate(slots):
        if not present[j]:
            continue
        vol = vols[slot]
        n = vol.shape[0]
        idx = triplet_indices(sample_centres(n, k, rng, z_offset), n)
        params = (augment_params(rng, cfg, crop_mm, mcfg.view_crop_mm[j]) if rng is not None
                  else view_params(crop_mm, mcfg.view_crop_mm[j]))
        images[j] = read_triplets(vol, idx, px, params)
    return images, present


class StudyDataset(Dataset):
    """One item = one study: uint8 images (S, K, 3, px, px), present (S,), target/weight (12,).
    Studies without a targets row get zero weights (gold rows can sit in a validation set)."""

    def __init__(self, cache_dir: str, studies: Sequence[str], targets: pd.DataFrame | None, cfg: TrainConfig,
                 train: bool, split: str = "train", seed: int = 0):
        self.cache_dir, self.studies, self.cfg, self.train = str(cache_dir), list(studies), cfg, train
        self.split, self.seed, self.epoch = split, seed, 0
        self.k = cfg.model.k_train if train else cfg.model.k_infer
        self.load_slots = tuple(dict.fromkeys(cfg.model.slots))  # a repeated slot is loaded once
        self._index: CacheIndex | None = None
        self.targets = {}
        if targets is not None:
            t, w = targets[TARGET_COLS].to_numpy(np.float32), targets[WEIGHT_COLS].to_numpy(np.float32)
            self.targets = {s: (t[i], w[i]) for i, s in enumerate(targets[ID_COL].astype(str))}

    @property
    def index(self) -> CacheIndex:
        # Built lazily so every DataLoader worker builds its own instead of pickling the parent's.
        if self._index is None:
            self._index = CacheIndex(self.cache_dir)
        return self._index

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, i: int) -> dict[str, Any]:
        study, cfg = self.studies[i], self.cfg
        rng = np.random.default_rng([self.seed, self.epoch, i]) if self.train else None
        vols = self.index.load(self.split, study, self.load_slots, rng, cfg.sibling_p)
        present = np.array([s in vols for s in cfg.model.slots], dtype=bool)
        if self.train and cfg.slot_dropout_p > 0:
            for j in range(len(present)):
                if present[j] and present.sum() > 2 and rng.random() < cfg.slot_dropout_p:
                    present[j] = False
        images, present = view_inputs(vols, cfg.model, cfg.cache.crop_mm, self.k,
                                      rng if self.train else None, cfg, present)
        zero = np.zeros(len(LABEL_KEYS), np.float32)
        target, weight = self.targets.get(study, (zero, zero))
        return {"study": study, "images": torch.from_numpy(images), "present": torch.from_numpy(present),
                "target": torch.from_numpy(np.array(target, np.float32)),
                "weight": torch.from_numpy(np.array(weight, np.float32))}


def collate_studies(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"study": [b["study"] for b in batch],
            **{k: torch.stack([b[k] for b in batch]) for k in ("images", "present", "target", "weight")}}


# =====================================================================================
# 10. MODEL: backbone over triplets -> gated attention MIL per view -> fusion
# =====================================================================================


def normalize(x: torch.Tensor) -> torch.Tensor:
    """uint8 (..., 3, H, W) -> ImageNet-normalised float, on the GPU (the loader moves 4x fewer bytes)."""
    if x.dtype != torch.uint8:
        return x
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    return (x.float().div_(255.0) - mean) / std


class GatedAttention(nn.Module):
    """Ilse et al. (2018) gated attention pooling over the K triplets of one view."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.v, self.u, self.w = nn.Linear(dim, hidden), nn.Linear(dim, hidden), nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.w(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))).squeeze(-1).float().softmax(dim=1)
        return torch.einsum("mk,mkd->md", a, h.float()), a


class KneeMIL(nn.Module):
    """Parameter names match the shipped checkpoints (backbone.*, attn.*, missing, aux_w, aux_b,
    fusion.*), so a main_v5_C model.safetensors loads strictly into this class."""

    def __init__(self, cfg: ModelConfig, pretrained: bool = False):
        super().__init__()
        import timm
        self.cfg, n_labels = cfg, len(LABELS)
        self.backbone = timm.create_model(cfg.backbone, pretrained=pretrained, num_classes=0,
                                          global_pool="avg", drop_path_rate=cfg.drop_path)
        if cfg.grad_checkpoint:
            self.backbone.set_grad_checkpointing(True)
        d, s = self.backbone.num_features, len(cfg.slots)
        self.dim, self.n_slots = d, s
        self.attn = GatedAttention(d, cfg.attn_hidden)
        self.missing = nn.Parameter(torch.randn(s, d) * 0.02)
        self.aux_w = nn.Parameter(torch.randn(s, d, n_labels) * (d ** -0.5))
        self.aux_b = nn.Parameter(torch.zeros(s, n_labels))
        # Presence bits enter the fusion input so the MLP knows which views were real.
        self.fusion = nn.Sequential(nn.Linear(s * d + s, cfg.fusion_hidden), nn.GELU(), nn.Dropout(cfg.drop_rate),
                                    nn.Linear(cfg.fusion_hidden, n_labels))

    def head_parameters(self) -> list[nn.Parameter]:
        backbone = {id(p) for p in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in backbone]

    def embed(self, images: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """(B, S, K, 3, H, W), (B, S) -> (B, S, D); only present views pass through the backbone."""
        b, s, k = images.shape[:3]
        emb = images.new_zeros((b * s, self.dim), dtype=torch.float32)
        sel = present.reshape(-1).nonzero().squeeze(1)
        if sel.numel():
            x = images.view(b * s, k, *images.shape[3:])[sel].reshape(-1, *images.shape[3:])
            x = normalize(x).contiguous(memory_format=torch.channels_last)
            feats = self.backbone(x).view(sel.numel(), k, self.dim)
            emb = emb.index_put((sel,), self.attn(feats)[0])
        return emb.view(b, s, self.dim)

    def fill(self, emb: torch.Tensor, present_f: torch.Tensor) -> torch.Tensor:
        """Absent views -> their learned [MISSING] vector; a mixup-blended presence blends both."""
        p = present_f.unsqueeze(-1)
        return emb * p + self.missing.unsqueeze(0) * (1.0 - p)

    def fuse(self, filled: torch.Tensor, present_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        aux = torch.einsum("bsd,sdl->bsl", filled, self.aux_w) + self.aux_b
        return self.fusion(torch.cat([filled.flatten(1), present_f], dim=1)), aux

    def forward(self, images: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        present_f = present.float()
        return self.fuse(self.fill(self.embed(images, present), present_f), present_f)


def masked_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted mean of soft-target BCE over cells with weight > 0."""
    loss = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none") * weight
    return loss.sum() / weight.sum().clamp_min(1e-6)


class Ema:
    """Per-step EMA of all float state. The decay ramps as (1+t)/(10+t) up to 0.999, so short runs
    are not stuck near the ImageNet init (0.999^300 would keep 74 % of it)."""

    def __init__(self, model: nn.Module, decay: float):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.step = decay, 0
        self._ema = [v for v in self.module.state_dict().values() if v.is_floating_point()]
        self._src = [v for v in model.state_dict().values() if v.is_floating_point()]

    @torch.no_grad()
    def update(self) -> None:
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        torch._foreach_lerp_(self._ema, self._src, 1.0 - d)


def save_model(state: Mapping[str, torch.Tensor], cfg: TrainConfig, out_dir: Path, extra: Mapping[str, Any]) -> None:
    from safetensors.torch import save_file
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(out_dir / "model.safetensors"))
    doc = {"model": asdict(cfg.model), "cache": asdict(cfg.cache), **extra}
    (out_dir / "pipelineA.json").write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")


def load_model(model_dir: str | Path, device: str = "cpu") -> tuple[KneeMIL, ModelConfig, CacheConfig]:
    """A run directory of this script (pipelineA.json) or of the original repo (bundle.json)."""
    from safetensors.torch import load_file
    model_dir = Path(model_dir)
    if (model_dir / "pipelineA.json").exists():
        doc = json.loads((model_dir / "pipelineA.json").read_text(encoding="utf-8"))
        m, c = doc["model"], doc["cache"]
    else:
        doc = json.loads((model_dir / "bundle.json").read_text(encoding="utf-8"))["config"]
        m, c = doc["model"], doc["cache"]
    mcfg = ModelConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in m.items()
                          if k in ModelConfig.__dataclass_fields__})
    if not mcfg.view_crop_mm:
        mcfg.view_crop_mm = (float(c["crop_mm"]),) * len(mcfg.slots)
    mcfg.grad_checkpoint = False
    ccfg = CacheConfig(**{k: v for k, v in c.items() if k in CacheConfig.__dataclass_fields__})
    model = KneeMIL(mcfg, pretrained=False)
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    return model.to(device).eval(), mcfg, ccfg


# =====================================================================================
# 11. METRICS
# =====================================================================================

# Soft weak targets are binarised for AUC: positive >= 0.6, negative <= 0.2 with weight >= 0.5,
# the middle band and low-weight negatives are dropped.
POS_THR, NEG_THR, NEG_MIN_WEIGHT, THR_EPS = 0.6, 0.2, 0.5, 1e-6


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties; NaN if a class is missing."""
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, np.float64)
    n_pos, n_neg = int(y_true.sum()), int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def macro_auc(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None, hard: bool = False) -> tuple[float, dict]:
    """hard=True: y is 0/1 (gold). Otherwise soft weak targets binarised as above."""
    per = {}
    for j, key in enumerate(LABEL_KEYS):
        if hard:
            keep, pos = np.ones(len(y), bool), y[:, j] > 0.5
        else:
            pos = y[:, j] >= POS_THR - THR_EPS
            # float32 targets: 0.2f32 > 0.2, hence the tolerance.
            keep = pos | ((y[:, j] <= NEG_THR + THR_EPS) & (w[:, j] >= NEG_MIN_WEIGHT - THR_EPS))
        per[key] = auc(pos[keep], p[keep, j])
    vals = [v for v in per.values() if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), per


def gold_report(pred: pd.DataFrame, gold: pd.DataFrame, base: pd.DataFrame | None = None,
                n_boot: int = 2000, seed: int = 2026) -> dict[str, Any]:
    """Gold-58 macro AUC with a bootstrap CI; with `base`, the paired bootstrap of pred - base over
    the same resampled studies (how every decision in this project was made)."""
    gold = gold[gold[ID_COL].isin(set(pred[ID_COL].astype(str)))].reset_index(drop=True)

    def aligned(df):
        d = df.assign(**{ID_COL: df[ID_COL].astype(str)}).set_index(ID_COL).loc[gold[ID_COL]]
        return d[PRED_COLS].to_numpy(np.float64)
    y = gold[list(LABEL_KEYS)].to_numpy(np.float64)
    p = aligned(pred)
    q = aligned(base) if base is not None else None
    point, per = macro_auc(y, p, hard=True)
    rng = np.random.default_rng(seed)
    boots, deltas = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        m = macro_auc(y[idx], p[idx], hard=True)[0]
        boots.append(m)
        if q is not None:
            deltas.append(m - macro_auc(y[idx], q[idx], hard=True)[0])
    out = {"n": len(y), "macro_auc": point, "ci95": [float(np.nanpercentile(boots, 2.5)),
                                                     float(np.nanpercentile(boots, 97.5))], "per_label": per}
    if q is not None:
        d = np.asarray(deltas)
        out["paired_delta"] = {"point": point - macro_auc(y, q, hard=True)[0],
                               "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
                               "p_gt_0": float(np.nanmean(d > 0))}
    return out


# =====================================================================================
# 12. TRAINING
# =====================================================================================


def cosine_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


@torch.inference_mode()
def predict_cached(model: nn.Module, ds: StudyDataset, batch_size: int, workers: int, device: str,
                   dtype: torch.dtype) -> pd.DataFrame:
    model.eval()
    rows = []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate_studies)
    for batch in loader:
        with torch.autocast(device_type=torch.device(device).type, dtype=dtype, enabled=dtype != torch.float32):
            logits, _ = model(batch["images"].to(device), batch["present"].to(device))
        for study, p in zip(batch["study"], logits.float().sigmoid().cpu().numpy()):
            rows.append({ID_COL: study, **dict(zip(PRED_COLS, p))})
    return pd.DataFrame(rows, columns=[ID_COL, *PRED_COLS])


def train(cfg: TrainConfig, cache_dir: str, targets: pd.DataFrame, folds: pd.DataFrame, fold: str,
          out_dir: str, gold: pd.DataFrame | None = None, device: str = "cuda", limit: int | None = None,
          max_steps: int | None = None, pretrained: bool = True) -> dict[str, Any]:
    """fold "full": train on every non-gold study with a target, validate on the gold studies only
    (never trained on); fold k: train on the other folds, validate on fold k. The EMA weights of
    the LAST epoch are kept - selecting the "best" epoch on the validation fold measured +0.008
    optimism, and on a full-data run it would select on gold."""
    full = fold == "full"
    fk = FULL_SEED_OFFSET if full else int(fold)
    out = Path(out_dir)
    torch.manual_seed(cfg.seed + fk)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    index = CacheIndex(cache_dir)
    cached = set(index.studies("train"))
    folds = folds.assign(**{ID_COL: folds[ID_COL].astype(str)})
    gold_ids = set(folds.loc[folds["is_gold"], ID_COL])
    targets = targets.assign(**{ID_COL: targets[ID_COL].astype(str)})
    targets = targets[~targets[ID_COL].isin(gold_ids)]
    has_t = set(targets[ID_COL])
    in_train = (folds["fold"] != int(fold)) if not full else pd.Series(True, index=folds.index)
    train_ids = sorted(s for s, g, t in zip(folds[ID_COL], folds["is_gold"], in_train) if t and not g
                       and s in has_t and s in cached)
    val_ids = sorted(s for s, g, f in zip(folds[ID_COL], folds["is_gold"], folds["fold"])
                     if s in cached and (g if full else f == int(fold)))
    if limit:
        train_ids = sorted(np.random.default_rng(cfg.seed).choice(train_ids, min(limit, len(train_ids)), replace=False))
    assert not gold_ids & set(train_ids), "gold rows must never be trained on"
    log.info("fold %s: %d train, %d val studies", fold, len(train_ids), len(val_ids))

    train_ds = StudyDataset(cache_dir, train_ids, targets, cfg, train=True, seed=cfg.seed + fk)
    val_ds = StudyDataset(cache_dir, val_ids, targets, cfg, train=False)
    workers = min(cfg.num_workers, os.cpu_count() or 1)
    loader = DataLoader(train_ds, batch_size=cfg.batch_studies, shuffle=True,
                        generator=torch.Generator().manual_seed(cfg.seed + fk), num_workers=workers,
                        collate_fn=collate_studies, pin_memory=device.startswith("cuda"), drop_last=True,
                        persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None)
    model = KneeMIL(cfg.model, pretrained=pretrained).to(device).to(memory_format=torch.channels_last)
    ema = Ema(model, cfg.ema_decay)
    if cfg.compile_backbone:
        # Patched on the instance after the EMA copy: state_dict keys keep no _orig_mod prefix
        # and the EMA (evaluation) model stays eager.
        model.backbone.forward = torch.compile(model.backbone.forward, dynamic=True)
    opt = torch.optim.AdamW([{"params": list(model.backbone.parameters()), "lr": cfg.lr_backbone},
                             {"params": model.head_parameters(), "lr": cfg.lr_heads}],
                            weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    steps_per_epoch = len(loader)
    total, warmup = cfg.epochs * steps_per_epoch, int(cfg.warmup_epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_warmup(s, total, warmup))
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.amp_dtype]
    dev_type = torch.device(device).type
    if dev_type == "cpu":
        # CPU autocast casts the attention einsum to bf16 and index_put then refuses the mix;
        # CPU runs are smoke tests only, so they run in fp32.
        dtype = torch.float32
    mix_rng = np.random.default_rng(cfg.seed + 1000 * fk)
    history, step_count = [], 0
    for epoch in range(cfg.epochs):
        model.train()
        train_ds.set_epoch(epoch)
        t0, loss_sum, loss_n = time.monotonic(), 0.0, 0
        for step, batch in enumerate(loader):
            images, present = batch["images"].to(device, non_blocking=True), batch["present"].to(device, non_blocking=True)
            target, weight = batch["target"].to(device), batch["weight"].to(device)
            with torch.autocast(device_type=dev_type, dtype=dtype, enabled=dtype != torch.float32):
                present_f = present.float()
                filled = model.fill(model.embed(images, present), present_f)
                if cfg.mixup_p > 0 and images.shape[0] > 1 and mix_rng.random() < cfg.mixup_p:
                    lam = float(mix_rng.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                    perm = torch.randperm(images.shape[0], device=device)
                    filled = lam * filled + (1.0 - lam) * filled[perm]
                    present_f = lam * present_f + (1.0 - lam) * present_f[perm]
                    target = lam * target + (1.0 - lam) * target[perm]
                    weight = lam * weight + (1.0 - lam) * weight[perm]
                logits, aux = model.fuse(filled, present_f)
            loss = masked_bce(logits, target, weight)
            if cfg.model.aux_weight > 0:
                # Each view's aux head is trained on the study target, only where the view was present.
                aux_w = weight.unsqueeze(1) * present_f.unsqueeze(2)
                loss = loss + cfg.model.aux_weight * masked_bce(aux, target.unsqueeze(1).expand_as(aux), aux_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            ema.update()
            loss_sum, loss_n = loss_sum + float(loss.item()) * images.shape[0], loss_n + images.shape[0]
            step_count += 1
            if (step + 1) % 50 == 0:
                log.info("epoch %d step %d/%d loss %.4f lr %.2e", epoch, step + 1, steps_per_epoch,
                         loss_sum / loss_n, sched.get_last_lr()[0])
            if max_steps and step_count >= max_steps:
                break
        pred = predict_cached(ema.module, val_ds, cfg.eval_batch_studies, workers, device, dtype)
        rec = {"epoch": epoch, "train_loss": loss_sum / max(loss_n, 1), "train_s": time.monotonic() - t0}
        if not full:
            weak = targets.merge(pred, on=ID_COL)
            rec["oof_weak_auc"] = macro_auc(weak[TARGET_COLS].to_numpy(), weak[PRED_COLS].to_numpy(),
                                            weak[WEIGHT_COLS].to_numpy())[0]
        if gold is not None and gold[ID_COL].isin(pred[ID_COL]).sum() >= 2:
            rec["gold_auc"] = gold_report(pred, gold, n_boot=200)["macro_auc"]
        history.append(rec)
        log.info("epoch %d done: %s", epoch, {k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()})
        save_model(ema.module.state_dict(), cfg, out, {"fold": fold, "seed": cfg.seed, "history": history})
        pred.assign(fold=fold).to_parquet(out / "oof.parquet", index=False)
        if max_steps and step_count >= max_steps:
            break
    return {"history": history, "n_train": len(train_ids), "n_val": len(val_ids)}


# =====================================================================================
# 13. TEST-TIME INFERENCE FROM RAW DICOM
# =====================================================================================


def study_volumes(study_dir: str | Path, rows: pd.DataFrame, cfg: CacheConfig,
                  slots: Sequence[str]) -> dict[str, np.ndarray]:
    """Raw study -> {slot: uint8 volume}, choosing series exactly as the cache index does (same
    ranking, FS -> T1 fallback). A series that fails to decode yields to the next candidate."""
    study = str(rows[ID_COL].iloc[0]) if ID_COL in rows.columns else Path(study_dir).name
    infos = []
    for r in rows.itertuples(index=False):
        uid = str(getattr(r, SERIES_COL))
        try:
            fs = int(r.Fluid_Sensitive)
            slot_of(r.Anatomical_Plane, fs)
            files = list_series_files(Path(study_dir) / uid)
            if not files:
                raise FileNotFoundError(uid)
            headers = sort_headers(read_header(p) for p in files)
            geom = series_geometry(headers, cfg.max_2d_slices)
            infos.append({ID_COL: study, SERIES_COL: uid, "Anatomical_Plane": r.Anatomical_Plane,
                          "Fluid_Sensitive": fs, FAT_COL: getattr(r, FAT_COL, None), "is_3d": geom.is_3d,
                          "n_slices": len(headers), "series_number": headers[0].series_number,
                          "_headers": headers})
        except Exception as exc:  # noqa: BLE001 - an unreadable series is a missing candidate
            log.warning("series %s skipped: %s", uid, exc)
    if not infos:
        return {}
    df = pd.DataFrame(infos)
    native = slot_table(assign_slots(df.drop(columns="_headers"))[lambda d: ~d["fallback"]]).get(study, {})
    headers_of = dict(zip(df[SERIES_COL], df["_headers"]))
    decoded: dict[str, np.ndarray | None] = {}
    out = {}
    for slot in dict.fromkeys(slots):
        for source in (slot, SLOT_FALLBACK.get(slot)):
            if source is None or slot in out:
                continue
            for uid in native.get(source, []):
                if uid not in decoded:
                    try:
                        decoded[uid] = series_volume(Path(study_dir) / uid, cfg, headers_of[uid])[0]
                    except Exception as exc:  # noqa: BLE001
                        log.warning("series %s undecodable: %s", uid, exc)
                        decoded[uid] = None
                if decoded[uid] is not None:
                    out[slot] = decoded[uid]
                    break
    return out


@torch.inference_mode()
def predict(data_dir: str, split: str, model_dirs: Sequence[str], device: str = "cuda",
            prior: Sequence[float] | None = None) -> pd.DataFrame:
    """Equal-weight mean of the members' sigmoids (main_v5_C: 6 seeds, K_infer 24, no TTA, fp16
    on GPU). A study that fails entirely gets `prior` (main_v5_C shipped the mean OOF prevalence)."""
    data_dir = Path(data_dir)
    series_df = pd.read_csv(data_dir / f"{split}_series.csv", dtype=str)
    members = [load_model(d, device) for d in model_dirs]
    cfg = members[0][2]
    slots = tuple(dict.fromkeys(s for _, m, _ in members for s in m.slots))
    prior_v = np.full(len(LABELS), 0.5) if prior is None else np.asarray(prior, np.float64)
    fp16 = torch.device(device).type == "cuda"
    rows = []
    for study, grp in series_df.groupby(ID_COL, sort=False):
        try:
            vols = study_volumes(data_dir / f"{split}_series" / study, grp, cfg, slots)
            if not vols:
                raise RuntimeError("no readable series")
            total = np.zeros(len(LABELS))
            for model, mcfg, ccfg in members:
                images, present = view_inputs(vols, mcfg, ccfg.crop_mm, mcfg.k_infer)
                with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=fp16):
                    logits, _ = model(torch.from_numpy(images).unsqueeze(0).to(device),
                                      torch.from_numpy(present).unsqueeze(0).to(device))
                total += logits.float().sigmoid().squeeze(0).cpu().numpy() / len(members)
        except Exception as exc:  # noqa: BLE001 - a broken study gets the prior, never a crash
            log.warning("study %s -> prior: %s", study, exc)
            total = prior_v
        rows.append({ID_COL: study, **{lab: float(np.clip(v, 0, 1)) for lab, v in zip(LABELS, total)}})
    return pd.DataFrame(rows, columns=[ID_COL, *LABELS])


# =====================================================================================
# 14. CLI
# =====================================================================================


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("build-cache", help="DICOM -> uint8 volume cache + index.parquet")
    a.add_argument("--data", required=True, help="competition data dir ({split}_series.csv, {split}_series/)")
    a.add_argument("--cache", required=True)
    a.add_argument("--split", default="train")
    a.add_argument("--workers", type=int, default=24)
    a.add_argument("--limit-studies", type=int)
    a.add_argument("--studies", nargs="*", help="only these StudyInstanceUIDs")

    a = sub.add_parser("make-folds", help="site-grouped balanced folds")
    a.add_argument("--study-summary", required=True, help="CSV: StudyInstanceUID, site_key, lang, is_gold")
    a.add_argument("--out", required=True)

    a = sub.add_parser("make-targets", help="label table + teacher OOF -> soft targets/weights")
    a.add_argument("--labels", required=True, help="report-label parquet ({key}_state/_severity, voters _v2 ...)")
    a.add_argument("--teacher-oof", nargs="+", required=True, help="OOF parquet(s) with {key}_pred columns")
    a.add_argument("--train-csv", required=True)
    a.add_argument("--folds", required=True)
    a.add_argument("--fold", default="full", help="'full' or 0-4")
    a.add_argument("--out", required=True)

    a = sub.add_parser("train", help="train one model (fold k or full data)")
    a.add_argument("--cache", required=True)
    a.add_argument("--targets", required=True)
    a.add_argument("--folds", required=True)
    a.add_argument("--train-csv", help="for gold-58 validation metrics")
    a.add_argument("--fold", default="full")
    a.add_argument("--seed", type=int, default=2026)
    a.add_argument("--out", required=True)
    a.add_argument("--device", default="cuda")
    a.add_argument("--epochs", type=int)
    a.add_argument("--batch-studies", type=int)
    a.add_argument("--workers", type=int)
    a.add_argument("--limit", type=int, help="random subset of N training studies (smoke tests)")
    a.add_argument("--max-steps", type=int, help="stop after N optimiser steps (smoke tests)")
    a.add_argument("--no-compile", action="store_true")
    a.add_argument("--no-pretrained", action="store_true")

    a = sub.add_parser("predict", help="raw DICOM -> submission CSV")
    a.add_argument("--data", required=True)
    a.add_argument("--split", default="test")
    a.add_argument("--models", nargs="+", required=True, help="run dirs with model.safetensors")
    a.add_argument("--out", required=True)
    a.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a.add_argument("--prior", type=float, nargs=12, help="fallback row for an unreadable study")

    a = sub.add_parser("score", help="gold-58 macro AUC (+ paired bootstrap vs --base)")
    a.add_argument("--pred", required=True, help="parquet with StudyInstanceUID + {key}_pred")
    a.add_argument("--base")
    a.add_argument("--train-csv", required=True)

    args = ap.parse_args(argv)
    if args.cmd == "build-cache":
        build_cache(args.data, args.cache, args.split, CacheConfig(), args.workers, args.limit_studies, args.studies)
    elif args.cmd == "make-folds":
        df = make_folds(args.study_summary)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_bytes(df.sort_values(ID_COL).to_csv(index=False, lineterminator="\n").encode())
    elif args.cmd == "make-targets":
        out = make_pseudo_targets(pd.read_parquet(args.labels), [pd.read_parquet(p) for p in args.teacher_oof],
                                  gold_frame(args.train_csv), load_folds(args.folds), args.fold)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(args.out, index=False)
    elif args.cmd == "train":
        cfg = TrainConfig(seed=args.seed)
        if args.epochs:
            cfg.epochs = args.epochs
        if args.batch_studies:
            cfg.batch_studies = args.batch_studies
        if args.workers is not None:
            cfg.num_workers = args.workers
        if args.no_compile:
            cfg.compile_backbone = False
        res = train(cfg, args.cache, pd.read_parquet(args.targets), load_folds(args.folds), args.fold, args.out,
                    gold_frame(args.train_csv) if args.train_csv else None, args.device, args.limit,
                    args.max_steps, pretrained=not args.no_pretrained)
        print(json.dumps(res, indent=1, default=float))
    elif args.cmd == "predict":
        df = predict(args.data, args.split, args.models, args.device, args.prior)
        df.to_csv(args.out, index=False)
    elif args.cmd == "score":
        rep = gold_report(pd.read_parquet(args.pred), gold_frame(args.train_csv),
                          pd.read_parquet(args.base) if args.base else None)
        print(json.dumps(rep, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
