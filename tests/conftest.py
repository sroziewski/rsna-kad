"""Synthetic DICOM data shared by the stage-0/1 tests: two studies mimicking the real
layout (<data_dir>/train_series/<study>/<series>/*.dcm + train_series.csv), covering
2D, 3D (0.9 mm x 120 slices), non-square 640x1280, a truncated file, Philips-style
RescaleSlope, signed Toshiba-style pixels and contradictory Laterality tags."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

UID_ROOT = "1.2.826.0.1.3680043.8.498."
STUDY_A = UID_ROOT + "100001"
STUDY_B = UID_ROOT + "100002"

# ImageOrientationPatient per plane: (row direction, column direction) in LPS.
IOP = {
    "Sagittal": (0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
    "Coronal": (1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    "Axial": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
}
NORMAL = {"Sagittal": np.array([-1.0, 0, 0]), "Coronal": np.array([0, 1.0, 0]), "Axial": np.array([0, 0, 1.0])}

DISC_MM = 40.0        # radius of the bright central disc in every phantom
OFFSET_DISC_MM = 30.0  # a second, smaller disc shifted +30 mm along the row direction


@dataclass
class SeriesSpec:
    uid: str
    plane: str
    fs: int
    n: int
    rows: int
    cols: int
    spacing: float
    z_step: float
    series_number: int
    manufacturer: str = "SIEMENS"
    model: str = "Aera"
    laterality: str = "L"
    description: str = "DummySeriesDesc!"
    acq_type: str = "2D"
    signed: bool = False
    bits_stored: int = 12
    rescale_slope: float | None = None
    truncated_slices: tuple[int, ...] = ()
    centre_x: float = 60.0      # patient-space x of the FOV centre (LPS: +x = left)
    amplitude: float = 3000.0
    seed: int = 0
    paths: list[str] = field(default_factory=list)

    @property
    def slot(self) -> str:
        return {"Sagittal": "sag", "Coronal": "cor", "Axial": "ax"}[self.plane] + ("_fs" if self.fs else "_t1")


def phantom(rows: int, cols: int, spacing: float, z_frac: float, amplitude: float,
            signed: bool, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float32)
    cy, cx = (rows - 1) / 2.0, (cols - 1) / 2.0
    r_mm = np.hypot((yy - cy) * spacing, (xx - cx) * spacing)
    img = np.full((rows, cols), 0.05 * amplitude, dtype=np.float32)
    img[r_mm <= DISC_MM] = amplitude * (0.6 + 0.3 * z_frac)
    r2 = np.hypot((yy - cy) * spacing, (xx - cx) * spacing - OFFSET_DISC_MM)
    img[r2 <= 8.0] = amplitude
    img += rng.normal(0, 0.01 * amplitude, img.shape).astype(np.float32)
    if signed:
        img -= 0.4 * amplitude
        return np.clip(img, -32768, 32767).astype(np.int16)
    return np.clip(img, 0, 4095 if amplitude <= 4095 else 65535).astype(np.uint16)


def _origin(spec: SeriesSpec, k: int) -> np.ndarray:
    # Place the FOV centre at (centre_x, 20, -10) + k * z_step along the normal, then move
    # back to the top-left pixel so image_centre() recovers the centre.
    row_dir, col_dir = np.array(IOP[spec.plane][:3]), np.array(IOP[spec.plane][3:])
    centre = np.array([spec.centre_x, 20.0, -10.0]) + NORMAL[spec.plane] * (k * spec.z_step)
    return centre - row_dir * (spec.cols - 1) / 2 * spec.spacing - col_dir * (spec.rows - 1) / 2 * spec.spacing


def write_series(root: Path, study: str, spec: SeriesSpec) -> None:
    rng = np.random.default_rng(spec.seed)
    series_dir = root / study / spec.uid
    series_dir.mkdir(parents=True, exist_ok=True)
    spec.paths = []
    for k in range(spec.n):
        img = phantom(spec.rows, spec.cols, spec.spacing, k / max(spec.n - 1, 1), spec.amplitude,
                      spec.signed, rng)
        sop = f"{spec.uid}.{k + 1}"
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = sop
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = sop
        ds.StudyInstanceUID = study
        ds.SeriesInstanceUID = spec.uid
        ds.Modality = "MR"
        ds.Manufacturer = spec.manufacturer
        ds.ManufacturerModelName = spec.model
        ds.SeriesDescription = spec.description
        ds.Laterality = spec.laterality
        ds.PatientID = "synthetic"
        ds.MRAcquisitionType = spec.acq_type
        ds.InstanceNumber = k + 1
        ds.SeriesNumber = spec.series_number
        ds.ImagePositionPatient = [float(v) for v in _origin(spec, k)]
        ds.ImageOrientationPatient = list(IOP[spec.plane])
        ds.PixelSpacing = [spec.spacing, spec.spacing]
        ds.SliceThickness = spec.z_step
        ds.SpacingBetweenSlices = spec.z_step
        ds.Rows, ds.Columns = spec.rows, spec.cols
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = spec.bits_stored
        ds.HighBit = spec.bits_stored - 1
        ds.PixelRepresentation = 1 if spec.signed else 0
        ds.WindowCenter, ds.WindowWidth = 1000, 2000
        if spec.rescale_slope is not None:
            ds.RescaleSlope = spec.rescale_slope + 0.01 * k
            ds.RescaleIntercept = 0
            ds.RescaleType = "normalized"
        payload = img.tobytes()
        if k in spec.truncated_slices:
            payload = payload[: len(payload) // 2]
        ds.PixelData = payload
        path = series_dir / f"{sop}.dcm"
        ds.save_as(path, enforce_file_format=True)
        spec.paths.append(str(path))
    # Handed to tests in reverse order so sort_series cannot pass by relying on file order.
    spec.paths.reverse()


def study_specs() -> dict[str, list[SeriesSpec]]:
    a = [
        SeriesSpec(UID_ROOT + "200001", "Sagittal", 1, 30, 512, 512, 0.4, 3.0, 4,
                   truncated_slices=(10,), seed=1),
        SeriesSpec(UID_ROOT + "200002", "Axial", 1, 120, 256, 256, 0.47, 0.9, 6, acq_type="3D",
                   seed=2),
        SeriesSpec(UID_ROOT + "200003", "Coronal", 0, 20, 640, 1280, 0.25, 3.0, 3,
                   manufacturer="Philips Medical Systems", model="Ingenia", laterality="",
                   description="COR T1 L", rescale_slope=4.5, amplitude=800.0, seed=3),
        SeriesSpec(UID_ROOT + "200004", "Axial", 0, 25, 320, 320, 0.5, 3.5, 7,
                   manufacturer="TOSHIBA", model="Vantage", laterality="R", signed=True,
                   bits_stored=16, seed=4),
    ]
    b = [
        SeriesSpec(UID_ROOT + "300001", "Sagittal", 1, 12, 128, 128, 1.0, 3.0, 5, laterality="",
                   centre_x=-55.0, seed=5),
        SeriesSpec(UID_ROOT + "300002", "Sagittal", 1, 8, 128, 128, 1.0, 3.0, 3, laterality="",
                   centre_x=-55.0, seed=6),
        SeriesSpec(UID_ROOT + "300003", "Axial", 1, 6, 128, 128, 1.0, 4.0, 2, laterality="",
                   centre_x=-55.0, seed=7),
        SeriesSpec(UID_ROOT + "300004", "Coronal", 1, 6, 128, 128, 1.0, 4.0, 8, laterality="",
                   centre_x=-55.0, seed=8),
        SeriesSpec(UID_ROOT + "300005", "Sagittal", 0, 6, 128, 128, 1.0, 4.0, 9, laterality="",
                   centre_x=-55.0, seed=9),
    ]
    return {STUDY_A: a, STUDY_B: b}


@dataclass
class SyntheticData:
    data_dir: Path
    specs: dict[str, list[SeriesSpec]]
    seconds: float

    def series_df(self, split: str = "train") -> pd.DataFrame:
        return pd.read_csv(self.data_dir / f"{split}_series.csv", dtype=str)

    def spec(self, uid: str) -> SeriesSpec:
        return next(s for specs in self.specs.values() for s in specs if s.uid == uid)

    def series_dir(self, study: str, uid: str, split: str = "train") -> Path:
        return self.data_dir / f"{split}_series" / study / uid


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> SyntheticData:
    data_dir = tmp_path_factory.mktemp("rsna_data")
    started = time.perf_counter()
    specs = study_specs()
    rows = []
    for study, series in specs.items():
        for s in series:
            write_series(data_dir / "train_series", study, s)
            rows.append((study, s.uid, s.fs, s.fs, s.plane))
    train_df = pd.DataFrame(rows, columns=["StudyInstanceUID", "SeriesInstanceUID", "Fluid_Sensitive",
                                           "Fat_Suppression", "Anatomical_Plane"])
    train_df.to_csv(data_dir / "train_series.csv", index=False)
    # Study B doubles as the "test" split so the two-split index path is exercised.
    for s in specs[STUDY_B]:
        write_series(data_dir / "test_series", STUDY_B, s)
    train_df[train_df["StudyInstanceUID"] == STUDY_B].to_csv(data_dir / "test_series.csv", index=False)
    return SyntheticData(data_dir, specs, time.perf_counter() - started)


@pytest.fixture(scope="session")
def study_summary_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """400 studies over 40 sites (site => language), 30 gold rows, like analysis/out/study_summary.csv."""
    rng = np.random.default_rng(7)
    langs = ["en", "de", "es", "tr", "el", "bg", "hr", "nl", "fr"]
    sites = [f"MANU{i % 4}|MODEL{i}|1.5|sw|{langs[i % len(langs)]}" for i in range(40)]
    weights = rng.dirichlet(np.ones(40) * 2)
    site_idx = rng.choice(40, size=400, p=weights)
    df = pd.DataFrame({
        "study_dir": [f"{UID_ROOT}{500000 + i}" for i in range(400)],
        "site_key": [sites[i] for i in site_idx],
        "lang": [sites[i].split("|")[-1] for i in site_idx],
        "labelled": False,
    })
    df.loc[rng.choice(400, size=30, replace=False), "labelled"] = True
    path = tmp_path_factory.mktemp("folds") / "study_summary.csv"
    df.to_csv(path, index=False)
    return path
