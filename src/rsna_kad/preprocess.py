"""Series → cached volume: neighbour fill, z-resampling, mm crop, resize, robust
windowing. Shared by the cache build and the Kaggle notebook; must stay pure numpy/cv2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from rsna_kad.config import CacheConfig
from rsna_kad.dicom import Geometry

cv2.setNumThreads(1)


@dataclass
class VolumeMeta:
    n_src: int
    n_cached: int
    truncated_slices: int
    p_lo: float
    p_hi: float
    spacing_mm: float
    fov_mm: float
    z_step_mm: float
    z_step_cached_mm: float
    crop_mm: float
    px: int
    is_3d: bool
    mean_u8: float


def fill_missing(slices: Sequence[np.ndarray | None]) -> tuple[list[np.ndarray], int]:
    """Replace None entries by the mean of the nearest readable neighbours."""
    n = len(slices)
    good = [i for i, s in enumerate(slices) if s is not None]
    if not good:
        raise ValueError("no readable slice in series")
    out: list[np.ndarray] = []
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
    return out, n - len(good)


def crop_px_size(spacing_mm: float, crop_mm: float) -> int:
    return max(1, int(round(crop_mm / spacing_mm)))


def crop_centre(vol: np.ndarray, size_px: int) -> np.ndarray:
    """Square centre crop of (N, H, W) with zero padding when the FOV is smaller."""
    n, h, w = vol.shape
    if h == size_px and w == size_px:
        return vol
    out = np.zeros((n, size_px, size_px), dtype=vol.dtype)
    y0, x0 = (h - size_px) // 2, (w - size_px) // 2
    sy0, sx0 = max(y0, 0), max(x0, 0)
    dy0, dx0 = max(-y0, 0), max(-x0, 0)
    ny, nx = min(h - sy0, size_px - dy0), min(w - sx0, size_px - dx0)
    out[:, dy0:dy0 + ny, dx0:dx0 + nx] = vol[:, sy0:sy0 + ny, sx0:sx0 + nx]
    return out


def resize_stack(vol: np.ndarray, px: int) -> np.ndarray:
    n, h, w = vol.shape
    if h == px and w == px:
        return np.ascontiguousarray(vol, dtype=np.float32)
    # INTER_AREA is only an anti-aliasing filter when shrinking; when enlarging it
    # degenerates to nearest-neighbour, so coarse series (spacing > 0.34 mm) use linear.
    interp = cv2.INTER_AREA if px < min(h, w) else cv2.INTER_LINEAR
    out = np.empty((n, px, px), dtype=np.float32)
    for i in range(n):
        out[i] = cv2.resize(np.ascontiguousarray(vol[i], dtype=np.float32), (px, px), interpolation=interp)
    return out


def crop_resample(img: np.ndarray, spacing_mm: float, crop_mm: float, px: int) -> np.ndarray:
    """One image (H, W) or stack (N, H, W) → (px, px) / (N, px, px) float32 covering
    crop_mm × crop_mm around the image centre."""
    single = img.ndim == 2
    vol = img[None] if single else img
    out = resize_stack(crop_centre(vol, crop_px_size(spacing_mm, crop_mm)), px)
    return out[0] if single else out


def resample_z(vol: np.ndarray, z_step: float, target_mm: float) -> tuple[np.ndarray, float]:
    """Average consecutive slabs so the slice step approaches target_mm; returns the
    new volume and its z step. Never interpolates finer than the source."""
    if z_step <= 0 or z_step >= target_mm * 0.75:
        return vol, z_step
    factor = int(round(target_mm / z_step))
    if factor <= 1:
        return vol, z_step
    n = vol.shape[0]
    n_out = int(np.ceil(n / factor))
    out = np.empty((n_out,) + vol.shape[1:], dtype=np.float32)
    for k in range(n_out):
        out[k] = vol[k * factor:(k + 1) * factor].mean(axis=0, dtype=np.float32)
    return out, z_step * factor


def subsample_indices(n: int, max_n: int) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_n).round().astype(int))


def robust_window(vol: np.ndarray, p_lo: float, p_hi: float) -> tuple[np.ndarray, float, float]:
    lo, hi = np.percentile(vol, [p_lo, p_hi])
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1.0
    scaled = (vol.astype(np.float32) - lo) * (255.0 / (hi - lo))
    np.clip(scaled, 0.0, 255.0, out=scaled)
    return np.rint(scaled).astype(np.uint8), lo, hi


def series_to_volume(slices: Sequence[np.ndarray | None], geom: Geometry,
                     cfg: CacheConfig) -> tuple[np.ndarray, VolumeMeta]:
    """slices in InstanceNumber order (None = unreadable) → uint8 (N, px, px) + metadata."""
    filled, n_trunc = fill_missing(slices)
    vol = np.stack([s.astype(np.float32, copy=False) for s in filled])
    n_src = vol.shape[0]
    # Crop at native resolution first: the 320-slice 640x1280 3D series would otherwise
    # cost 1 GB of float32 per worker before any reduction.
    vol = crop_centre(vol, crop_px_size(geom.spacing, cfg.crop_mm))
    z_step = geom.z_step
    if geom.is_3d:
        vol, z_step = resample_z(vol, geom.z_step, cfg.z_target_mm)
    idx = subsample_indices(vol.shape[0], cfg.max_slices)
    if idx.size != vol.shape[0]:
        z_step = z_step * (vol.shape[0] - 1) / max(idx.size - 1, 1)
        vol = vol[idx]
    vol = resize_stack(vol, cfg.px)
    u8, lo, hi = robust_window(vol, cfg.p_lo, cfg.p_hi)
    meta = VolumeMeta(
        n_src=n_src, n_cached=int(u8.shape[0]), truncated_slices=n_trunc, p_lo=lo, p_hi=hi,
        spacing_mm=geom.spacing, fov_mm=geom.fov_mm, z_step_mm=geom.z_step,
        z_step_cached_mm=float(z_step), crop_mm=cfg.crop_mm, px=cfg.px, is_3d=geom.is_3d,
        mean_u8=float(u8.mean()),
    )
    return u8, meta
