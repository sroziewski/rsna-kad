"""DICOM access: headers, pixel decoding with truncation tolerance, slice order
and series geometry. Everything the cache and the Kaggle notebook share."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pydicom

log = logging.getLogger(__name__)

PLANES = ("Sagittal", "Coronal", "Axial")


@dataclass(frozen=True)
class Header:
    path: str
    study_uid: str
    series_uid: str
    sop_uid: str
    instance_number: int
    series_number: int
    rows: int
    cols: int
    spacing: float
    iop: tuple[float, ...]
    ipp: tuple[float, ...]
    slice_thickness: float
    spacing_between_slices: float
    rescale_slope: float
    rescale_intercept: float
    pixel_representation: int
    bits_stored: int
    manufacturer: str
    model: str
    laterality: str
    series_description: str
    mr_acquisition_type: str


@dataclass(frozen=True)
class Geometry:
    spacing: float
    rows: int
    cols: int
    iop: tuple[float, ...]
    ipp_centre: tuple[float, float, float]
    plane: str
    is_3d: bool
    z_step: float
    n_slices: int
    # cosine of the angle between the slice normal and the nearest patient axis
    obliquity: float

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


def read_header(path: str | os.PathLike[str]) -> Header:
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    spacing = ds.get("PixelSpacing")
    spacing = float(spacing[0]) if spacing else 1.0
    iop = tuple(float(v) for v in ds.get("ImageOrientationPatient", [1, 0, 0, 0, 1, 0]))
    ipp = tuple(float(v) for v in ds.get("ImagePositionPatient", [0, 0, 0]))
    return Header(
        path=str(path),
        study_uid=_str(ds, "StudyInstanceUID"),
        series_uid=_str(ds, "SeriesInstanceUID"),
        sop_uid=_str(ds, "SOPInstanceUID"),
        instance_number=int(_float(ds, "InstanceNumber", 0)),
        series_number=int(_float(ds, "SeriesNumber", 0)),
        rows=int(ds.Rows),
        cols=int(ds.Columns),
        spacing=spacing,
        iop=iop,
        ipp=ipp,
        slice_thickness=_float(ds, "SliceThickness", 0.0),
        spacing_between_slices=_float(ds, "SpacingBetweenSlices", 0.0),
        rescale_slope=_float(ds, "RescaleSlope", 1.0),
        rescale_intercept=_float(ds, "RescaleIntercept", 0.0),
        pixel_representation=int(_float(ds, "PixelRepresentation", 0)),
        bits_stored=int(_float(ds, "BitsStored", 16)),
        manufacturer=_str(ds, "Manufacturer"),
        model=_str(ds, "ManufacturerModelName"),
        laterality=_str(ds, "Laterality"),
        series_description=_str(ds, "SeriesDescription"),
        mr_acquisition_type=_str(ds, "MRAcquisitionType"),
    )


def read_slice(path: str | os.PathLike[str]) -> np.ndarray | None:
    """Decoded pixels as float32 with RescaleSlope/Intercept applied; None when
    the pixel payload is unreadable (the 2 truncated train files raise ValueError)."""
    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array
    except (ValueError, OSError, AttributeError, RuntimeError, NotImplementedError,
            pydicom.errors.InvalidDicomError) as exc:
        # RuntimeError/NotImplementedError: pydicom 3 with no decoder plugin for a
        # compressed transfer syntax (architecture §4.1 F5) - the slice is neighbour-filled.
        log.warning("unreadable pixel data in %s: %s", path, str(exc)[:120])
        return None
    if arr.ndim != 2:
        log.warning("%s: expected a single frame, got shape %s", path, arr.shape)
        return None
    out = arr.astype(np.float32)
    slope, intercept = _float(ds, "RescaleSlope", 1.0), _float(ds, "RescaleIntercept", 0.0)
    if slope != 1.0:
        out *= np.float32(slope)
    if intercept != 0.0:
        out += np.float32(intercept)
    return out


def list_series_files(series_dir: str | os.PathLike[str]) -> list[str]:
    return sorted(str(p) for p in Path(series_dir).iterdir() if p.suffix.lower() == ".dcm")


def sort_headers(headers: Iterable[Header]) -> list[Header]:
    return sorted(headers, key=lambda h: (h.instance_number, h.path))


def sort_series(paths: Iterable[str | os.PathLike[str]]) -> list[str]:
    """Paths in InstanceNumber order (monotone along the slice normal for every
    series in the data, data_analysis.md §3)."""
    return [h.path for h in sort_headers(read_header(p) for p in paths)]


def slice_normal(iop: Sequence[float]) -> np.ndarray:
    row, col = np.asarray(iop[:3], dtype=np.float64), np.asarray(iop[3:6], dtype=np.float64)
    return np.cross(row, col)


def plane_of_iop(iop: Sequence[float]) -> str:
    return PLANES[int(np.argmax(np.abs(slice_normal(iop))))]


def image_centre(h: Header) -> np.ndarray:
    """Patient-space (LPS, mm) position of the image centre."""
    row_dir, col_dir = np.asarray(h.iop[:3]), np.asarray(h.iop[3:6])
    return (np.asarray(h.ipp, dtype=np.float64)
            + row_dir * (h.cols - 1) / 2.0 * h.spacing
            + col_dir * (h.rows - 1) / 2.0 * h.spacing)


def slice_positions(headers: Sequence[Header]) -> np.ndarray:
    normal = slice_normal(headers[0].iop)
    return np.asarray([np.dot(h.ipp, normal) for h in headers], dtype=np.float64)


def series_geometry(headers: Sequence[Header], max_2d_slices: int = 60) -> Geometry:
    """Geometry of a series from its (InstanceNumber-sorted) headers."""
    if not headers:
        raise ValueError("series_geometry needs at least one header")
    first = headers[0]
    n = len(headers)
    pos = slice_positions(headers)
    steps = np.abs(np.diff(pos)) if n > 1 else np.asarray([first.slice_thickness or 1.0])
    z_step = float(np.median(steps)) if steps.size else 0.0
    if z_step <= 0:
        z_step = first.spacing_between_slices or first.slice_thickness or 1.0
    acq = first.mr_acquisition_type.upper()
    is_3d = acq == "3D" or (acq == "" and n > max_2d_slices)
    normal = slice_normal(first.iop)
    centre = np.mean([image_centre(h) for h in headers], axis=0)
    return Geometry(
        spacing=float(np.median([h.spacing for h in headers])),
        rows=first.rows,
        cols=first.cols,
        iop=first.iop,
        ipp_centre=(float(centre[0]), float(centre[1]), float(centre[2])),
        plane=plane_of_iop(first.iop),
        is_3d=is_3d,
        z_step=z_step,
        n_slices=n,
        obliquity=float(np.max(np.abs(normal)) / (np.linalg.norm(normal) or 1.0)),
    )


def is_monotone_order(headers: Sequence[Header]) -> bool:
    pos = slice_positions(headers)
    d = np.diff(pos)
    return bool(d.size == 0 or np.all(d > 0) or np.all(d < 0))
