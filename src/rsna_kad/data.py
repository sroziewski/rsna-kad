"""Study-level dataset over the uint8 cache: per slot K triplets [i-1, i, i+1] of slices
(architecture §2.4), augmented on the uint8 volume, returned as uint8 so the dataloader
moves 4x fewer bytes; normalisation happens on the GPU (model.normalize)."""
from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rsna_kad.cache import CacheIndex, get_index, load_study
from rsna_kad.config import TrainConfig
from rsna_kad.constants import ID_COL, LABEL_KEYS

TARGET_COLS = [f"{k}_target" for k in LABEL_KEYS]
WEIGHT_COLS = [f"{k}_weight" for k in LABEL_KEYS]

cv2.setNumThreads(1)


# ------------------------------------------------------------------ slice sampling


def sample_centres(n: int, k: int, rng: np.random.Generator | None, z_offset: float = 0.0) -> np.ndarray:
    """K centre indices in [0, n). Uniform (rng None) takes the middle of K equal strata,
    shifted by z_offset strata (TTA, §2.6); random takes a uniform point per stratum, which
    is the stratified-random sampling with z-jitter of §2.4."""
    strata = np.arange(k, dtype=np.float64)
    pos = strata + (rng.random(k) if rng is not None else 0.5 + z_offset)
    return np.clip(np.floor(pos * n / k), 0, n - 1).astype(np.int64)


def triplet_indices(centres: np.ndarray, n: int) -> np.ndarray:
    """(K, 3) slice indices [i-1, i, i+1], edge-clamped."""
    return np.clip(centres[:, None] + np.array([-1, 0, 1])[None, :], 0, n - 1)


def read_triplets(vol: np.ndarray, idx: np.ndarray, px: int,
                  params: Mapping[str, float] | None = None) -> np.ndarray:
    """(K, 3, px, px) uint8 from a memmap: each needed slice is read and warped once."""
    uniq, inverse = np.unique(idx, return_inverse=True)
    warped = warp_slices(np.asarray(vol[uniq]), px, params)
    return warped[inverse.reshape(idx.shape)]


# ------------------------------------------------------------------ augmentation


def augment_params(rng: np.random.Generator, cfg: TrainConfig, crop_mm: float,
                   view_mm: float | None = None) -> dict[str, float]:
    """Random crop/rotation/brightness for one series. `crop_mm` is the field the cached volume
    covers; `view_mm` (default: the whole field) is the field this view should end up with, and
    the random window cfg.crop_scale_mm is scaled by view_mm/crop_mm so a 90 mm view jitters over
    [79.6, 90] mm exactly as the 130 mm view jitters over [115, 130]."""
    lo, hi = cfg.crop_scale_mm
    r = 1.0 if view_mm is None else float(view_mm) / float(crop_mm)
    return {
        "scale": crop_mm / float(rng.uniform(lo * r, hi * r)),
        "rot_deg": float(rng.uniform(-cfg.rot_deg, cfg.rot_deg)),
        "contrast": float(1.0 + rng.uniform(-cfg.bc_jitter, cfg.bc_jitter)),
        "brightness": float(rng.uniform(-cfg.bc_jitter, cfg.bc_jitter) * 255.0),
    }


def warp_slices(slices: np.ndarray, px: int, params: Mapping[str, float] | None) -> np.ndarray:
    """(N, H, W) uint8 -> (N, px, px) uint8. With params: rotate + zoom about the centre at
    the cache resolution, then INTER_AREA down-scaling (the same final resize as inference),
    then brightness/contrast; without params a plain resize."""
    n, h, w = slices.shape
    out = np.empty((n, px, px), dtype=np.uint8)
    if params is None:
        for i in range(n):
            out[i] = cv2.resize(slices[i], (px, px), interpolation=cv2.INTER_AREA) if (h, w) != (px, px) else slices[i]
        return out
    m = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), params["rot_deg"], params["scale"])
    for i in range(n):
        img = cv2.warpAffine(slices[i], m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if (h, w) != (px, px):
            img = cv2.resize(img, (px, px), interpolation=cv2.INTER_AREA)
        out[i] = img
    if params["contrast"] != 1.0 or params["brightness"] != 0.0:
        out = cv2.convertScaleAbs(out.reshape(-1, px), alpha=params["contrast"],
                                  beta=params["brightness"]).reshape(n, px, px)
    return out


def view_params(crop_mm: float, view_mm: float) -> dict[str, float] | None:
    """Deterministic centre crop of a `crop_mm` volume down to `view_mm` (the evaluation
    counterpart of augment_params). None when the view is the whole cached field, which keeps
    the no-crop path on the plain-resize branch of warp_slices."""
    if float(view_mm) >= float(crop_mm):
        return None
    return {"scale": float(crop_mm) / float(view_mm), "rot_deg": 0.0, "contrast": 1.0, "brightness": 0.0}


def slot_view_mm(cfg: TrainConfig) -> tuple[float, ...]:
    """One field of view in mm per entry of cfg.model.slots (§2.2 / ModelConfig.view_crop_mm)."""
    return view_mm_for(cfg.model.slots, cfg.model.view_crop_mm, cfg.cache.crop_mm)


def view_mm_for(slots: Sequence[str], view_crop_mm: Sequence[float], crop_mm: float) -> tuple[float, ...]:
    """Resolve (slots, view_crop_mm) against the cached field `crop_mm` into one absolute field
    of view per slot. Training (data.StudyDataset) and inference (infer.study_inputs) both go
    through this, so a model cannot be fed one set of views and trained on another."""
    slots, view = tuple(slots), tuple(view_crop_mm)
    crop_mm = float(crop_mm)
    if not view:
        if len(set(slots)) != len(slots):
            raise ValueError(f"slots {list(slots)} repeat but view_crop_mm is empty: a repeated slot "
                             "is only meaningful as a second view with its own crop")
        return (crop_mm,) * len(slots)
    if len(view) != len(slots):
        raise ValueError(f"view_crop_mm has {len(view)} entries for {len(slots)} slots")
    bad = [v for v in view if not 0.0 < float(v) <= crop_mm]
    if bad:
        raise ValueError(f"view_crop_mm entries must be in (0, {crop_mm}], got {bad}")
    pairs = list(zip(slots, (float(v) for v in view)))
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"duplicate (slot, view_crop_mm) pairs in {pairs}")
    return tuple(float(v) for v in view)


def augment_volume(vol_u8: np.ndarray, rng: np.random.Generator, cfg: TrainConfig,
                   px: int | None = None) -> np.ndarray:
    """Whole-volume variant of warp_slices with one random parameter set (all slices of a
    series share it, §2.4)."""
    px = px or cfg.model.img_px
    return warp_slices(np.asarray(vol_u8), px, augment_params(rng, cfg, cfg.cache.crop_mm))


# ------------------------------------------------------------------ dataset


class StudyDataset(Dataset):
    """One item = one study: uint8 images (S, K, 3, px, px), presence (S,), target/weight (12,).

    Missing slots give zero images and present=False. targets_df may be None (prediction only);
    studies without a targets row get zero weights, so gold rows can sit in a validation set."""

    def __init__(self, cache_dir: str | os.PathLike[str], studies: Sequence[str],
                 targets_df: pd.DataFrame | None, cfg: TrainConfig, train: bool,
                 split: str = "train", index: CacheIndex | None = None, seed: int = 0,
                 z_offset: float = 0.0):
        self.cache_dir = str(cache_dir)
        self.studies = list(studies)
        self.cfg = cfg
        self.train = train
        self.split = split
        self.seed = seed
        self.z_offset = z_offset
        self.slots = tuple(cfg.model.slots)
        self.view_mm = slot_view_mm(cfg)
        # A slot repeated with a second crop is loaded once and cropped twice.
        self.load_slots = tuple(dict.fromkeys(self.slots))
        self.k = cfg.model.k_train if train else cfg.model.k_infer
        self.px = cfg.model.img_px
        self.epoch = 0
        self._index = index
        self.targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if targets_df is not None:
            t = targets_df[TARGET_COLS].to_numpy(dtype=np.float32)
            w = targets_df[WEIGHT_COLS].to_numpy(dtype=np.float32)
            self.targets = {s: (t[i], w[i]) for i, s in enumerate(targets_df[ID_COL].astype(str))}

    @property
    def index(self) -> CacheIndex:
        # Resolved lazily so worker processes build their own instead of pickling the parent's.
        if self._index is None:
            self._index = get_index(self.cache_dir)
        return self._index

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.studies)

    def _rng(self, i: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, self.epoch, i])

    def __getitem__(self, i: int) -> dict[str, Any]:
        study = self.studies[i]
        rng = self._rng(i) if self.train else None
        cfg = self.cfg
        vols = load_study(self.cache_dir, self.split, study, self.load_slots, sibling_rng=rng,
                          sibling_p=cfg.sibling_p, index=self.index)
        present = np.array([s in vols for s in self.slots], dtype=bool)
        if self.train and cfg.slot_dropout_p > 0:
            for j, slot in enumerate(self.slots):
                if present[j] and present.sum() > 2 and rng.random() < cfg.slot_dropout_p:
                    present[j] = False
        images = np.zeros((len(self.slots), self.k, 3, self.px, self.px), dtype=np.uint8)
        for j, slot in enumerate(self.slots):
            if not present[j]:
                continue
            vol = vols[slot]
            n = vol.shape[0]
            idx = triplet_indices(sample_centres(n, self.k, rng, self.z_offset), n)
            params = (augment_params(rng, cfg, cfg.cache.crop_mm, self.view_mm[j]) if self.train
                      else view_params(cfg.cache.crop_mm, self.view_mm[j]))
            images[j] = read_triplets(vol, idx, self.px, params)
            if self.train and cfg.mri_aug:
                from rsna_kad.mri_aug import mri_augment
                images[j] = mri_augment(images[j], rng)
        target, weight = self.targets.get(study, (np.zeros(len(LABEL_KEYS), np.float32),
                                                   np.zeros(len(LABEL_KEYS), np.float32)))
        return {"study": study, "images": torch.from_numpy(images), "present": torch.from_numpy(present),
                "target": torch.from_numpy(np.array(target, dtype=np.float32)),
                "weight": torch.from_numpy(np.array(weight, dtype=np.float32))}


def collate_studies(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "study": [b["study"] for b in batch],
        "images": torch.stack([b["images"] for b in batch]),
        "present": torch.stack([b["present"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "weight": torch.stack([b["weight"] for b in batch]),
    }


def load_targets(path: str | os.PathLike[str]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df[ID_COL] = df[ID_COL].astype(str)
    missing = [c for c in TARGET_COLS + WEIGHT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def synthetic_targets(studies: Sequence[str], seed: int = 0) -> pd.DataFrame:
    """Random soft targets in the v0 value range, for plumbing tests only."""
    rng = np.random.default_rng(seed)
    n = len(studies)
    t = rng.choice([0.04, 0.06, 0.25, 0.5, 0.92], size=(n, len(LABEL_KEYS))).astype(np.float32)
    w = rng.choice([0.1, 0.5, 0.7, 1.0], size=(n, len(LABEL_KEYS))).astype(np.float32)
    df = pd.DataFrame({ID_COL: list(studies)})
    for j, k in enumerate(LABEL_KEYS):
        df[f"{k}_target"], df[f"{k}_weight"] = t[:, j], w[:, j]
    return df
