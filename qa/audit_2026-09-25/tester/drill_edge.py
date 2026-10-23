"""Tester drills (audit 2026-09-25), CPU only, writes only under qa/audit_2026-09-25/tester/.

  build   - synthetic Kaggle-layout data dir with edge-case studies (tests/conftest.py writers)
            + JPEG-LS / JPEG2000 / JPEG-lossless / RLE transcodes of one real public test study
            + a 3-member mini export (A f0, E f0, C f0 of export/main_v3, same files via symlink)
  parity  - lossless transcodes decode to the identical uint8 slot volumes as the original
  predict - rsna_kad.infer.predict_studies + write_submission on the edge-case dir; one valid row each
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/raid/sroziewski/sandbox/rsna_kad")
OUT = PROJ / "qa/audit_2026-09-25/tester"
sys.path.insert(0, str(PROJ / "tests"))
from conftest import UID_ROOT, SeriesSpec, write_series  # noqa: E402

from rsna_kad.constants import LABELS  # noqa: E402

REAL = Path("/raid/sroziewski/data/rsna_kad")
EDGE = OUT / "edge_data"
MINI = OUT / "mini_export"
ID = "StudyInstanceUID"
COLS = [ID, "SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]
SYNTAXES = {"jpegls": "JPEGLSLossless", "j2k": "JPEG2000Lossless", "jpeg14": "JPEGLosslessProcess14_1",
            "rle": "RLELossless"}


def sid(n: int) -> str:
    return UID_ROOT + str(900000 + n)


def edge_studies() -> dict[str, tuple[str, list[tuple[SeriesSpec, str]]]]:
    """{study: (case, [(spec, csv plane), ...])}; csv plane can differ from the DICOM geometry."""
    u = lambda n: UID_ROOT + str(910000 + n)  # noqa: E731
    return {
        sid(1): ("one_series", [(SeriesSpec(u(1), "Sagittal", 1, 20, 256, 256, 0.6, 3.0, 3, seed=11), "Sagittal")]),
        sid(2): ("missing_coronal", [(SeriesSpec(u(2), "Sagittal", 1, 20, 256, 256, 0.6, 3.0, 3, seed=12), "Sagittal"),
                                     (SeriesSpec(u(3), "Axial", 1, 20, 256, 256, 0.6, 3.5, 4, seed=13), "Axial")]),
        sid(3): ("3d_only", [(SeriesSpec(u(4), "Sagittal", 1, 120, 256, 256, 0.6, 0.6, 3, acq_type="3D", seed=14),
                              "Sagittal"),
                             (SeriesSpec(u(5), "Coronal", 1, 100, 256, 256, 0.6, 0.7, 4, acq_type="3D", seed=15),
                              "Coronal")]),
        sid(4): ("truncated_slices", [(SeriesSpec(u(6), "Sagittal", 1, 20, 256, 256, 0.6, 3.0, 3,
                                                  truncated_slices=(0, 9, 19), seed=16), "Sagittal"),
                                      (SeriesSpec(u(7), "Coronal", 1, 20, 256, 256, 0.6, 3.0, 4, seed=17), "Coronal")]),
        sid(5): ("nonsquare_640x1280", [(SeriesSpec(u(8), "Coronal", 1, 20, 640, 1280, 0.25, 3.0, 3, seed=18),
                                         "Coronal"),
                                        (SeriesSpec(u(9), "Sagittal", 1, 20, 1280, 640, 0.25, 3.0, 4, seed=19),
                                         "Sagittal")]),
        sid(6): ("all_slices_truncated", [(SeriesSpec(u(10), "Sagittal", 1, 8, 128, 128, 1.0, 3.0, 3,
                                                      truncated_slices=tuple(range(8)), seed=20), "Sagittal")]),
        sid(7): ("unknown_plane_in_csv", [(SeriesSpec(u(11), "Sagittal", 1, 20, 256, 256, 0.6, 3.0, 3, seed=21),
                                           "Oblique"),
                                          (SeriesSpec(u(12), "Axial", 1, 20, 256, 256, 0.6, 3.5, 4, seed=22), "Axial")]),
        sid(8): ("dir_missing", []),         # listed in test_series.csv, no files on disk
        sid(9): ("not_in_series_csv", []),   # only in sample_submission.csv
    }


def transcode(src: Path, dst: Path, syntax: str) -> None:
    import gdcm
    r = gdcm.ImageReader()
    r.SetFileName(str(src))
    if not r.Read():
        raise RuntimeError(f"gdcm cannot read {src}")
    c = gdcm.ImageChangeTransferSyntax()
    c.SetTransferSyntax(gdcm.TransferSyntax(getattr(gdcm.TransferSyntax, syntax)))
    c.SetInput(r.GetImage())
    if not c.Change():
        raise RuntimeError(f"gdcm cannot transcode {src} to {syntax}")
    w = gdcm.ImageWriter()
    w.SetFileName(str(dst))
    w.SetFile(r.GetFile())
    w.SetImage(c.GetOutput())
    if not w.Write():
        raise RuntimeError(f"gdcm cannot write {dst}")


def build() -> None:
    if EDGE.exists():
        shutil.rmtree(EDGE)
    (EDGE / "test_series").mkdir(parents=True)
    rows, sample_ids = [], []
    for study, (case, series) in edge_studies().items():
        sample_ids.append(study)
        for spec, csv_plane in series:
            write_series(EDGE / "test_series", study, spec)
            rows.append((study, spec.uid, spec.fs, spec.fs, csv_plane))
        if case == "dir_missing":
            rows.append((study, UID_ROOT + "919999", 1, 1, "Sagittal"))
    # real public test study transcoded to four compressed syntaxes (study uid suffixed per syntax)
    real_series = pd.read_csv(REAL / "test_series.csv", dtype=str)
    real_study = sorted(real_series[ID].unique(), key=lambda s: sum(1 for _ in (REAL / "test_series" / s).rglob("*.dcm")))[0]
    for i, (tag, syntax) in enumerate(SYNTAXES.items()):
        new = sid(20 + i)
        sample_ids.append(new)
        for r in real_series[real_series[ID] == real_study].itertuples(index=False):
            src_dir = REAL / "test_series" / real_study / r.SeriesInstanceUID
            dst_dir = EDGE / "test_series" / new / r.SeriesInstanceUID
            dst_dir.mkdir(parents=True)
            for f in sorted(src_dir.glob("*.dcm")):
                transcode(f, dst_dir / f.name, syntax)
            rows.append((new, r.SeriesInstanceUID, r.Fluid_Sensitive, r.Fat_Suppression, r.Anatomical_Plane))
    pd.DataFrame(rows, columns=COLS).to_csv(EDGE / "test_series.csv", index=False)
    pd.DataFrame({ID: sample_ids, **{lab: 0.5 for lab in LABELS}}).to_csv(EDGE / "sample_submission.csv", index=False)
    (EDGE / "SOURCE").write_text(f"real study transcoded: {real_study}\n" + json.dumps(
        {sid(20 + i): t for i, t in enumerate(SYNTAXES)}) + "\n")
    # mini export: one fold of each arm, same weight files as export/main_v3
    if MINI.exists():
        shutil.rmtree(MINI)
    MINI.mkdir()
    (MINI / "models").symlink_to(PROJ / "export/main_v3/models")
    doc = json.loads((PROJ / "export/main_v3/ensemble.json").read_text())
    doc["members"] = [m for m in doc["members"] if m["path"].endswith("_f0")]
    doc["version"] = "tester mini export (f0 of each arm of main_v3)"
    (MINI / "ensemble.json").write_text(json.dumps(doc, indent=1))
    print("built", EDGE, "real study", real_study, "members", [m["path"] for m in doc["members"]])


def parity() -> None:
    from rsna_kad.infer import StudyPreprocessor
    src = (EDGE / "SOURCE").read_text().splitlines()
    real_study = src[0].split(": ")[1]
    tags = json.loads(src[1])
    real_series = pd.read_csv(REAL / "test_series.csv", dtype=str)
    edge_series = pd.read_csv(EDGE / "test_series.csv", dtype=str)
    pre = StudyPreprocessor()
    ref = pre.study(REAL / "test_series" / real_study, real_series[real_series[ID] == real_study], real_study)
    print("reference", real_study, {k: v.shape for k, v in ref.items()})
    import pydicom
    for study, tag in tags.items():
        one = next((EDGE / "test_series" / study).rglob("*.dcm"))
        ts = pydicom.dcmread(one, stop_before_pixels=True).file_meta.TransferSyntaxUID
        got = pre.study(EDGE / "test_series" / study, edge_series[edge_series[ID] == study], study)
        same = set(got) == set(ref) and all(np.array_equal(got[k], ref[k]) for k in ref)
        diff = max((int(np.abs(got[k].astype(int) - ref[k].astype(int)).max()) for k in ref if k in got), default=-1)
        print(f"{tag:7s} {ts.name:45s} slots {sorted(got)} identical={same} max|diff|={diff}")


def predict() -> None:
    from rsna_kad import infer
    doc = json.loads((MINI / "ensemble.json").read_text())
    members = infer.members_from_ensemble(MINI, doc)
    series_df = pd.read_csv(EDGE / "test_series.csv", dtype=str)
    sample = pd.read_csv(EDGE / "sample_submission.csv", dtype={ID: str})
    dirs = infer.test_study_dirs(EDGE, series_df, sample)
    pred = infer.predict_studies(members, dirs, series_df, devices=["cpu"], workers=2, fp16=False,
                                 prior=doc["prior"])
    out = infer.write_submission(pred, EDGE / "sample_submission.csv", OUT / "edge_submission.csv", prior=doc["prior"])
    cases = {s: c for s, (c, _) in edge_studies().items()}
    cases.update({s: f"real_{t}" for s, t in json.loads((EDGE / "SOURCE").read_text().splitlines()[1]).items()})
    prior = np.asarray(doc["prior"])
    p = pred.set_index(ID)
    for s in sample[ID]:
        row = out.set_index(ID).loc[s, list(LABELS)].to_numpy(dtype=float)
        ok = np.isfinite(row).all() and row.min() >= 0 and row.max() <= 1
        is_prior = np.allclose(row, prior, atol=1e-6)
        info = p.loc[s] if s in p.index else None
        print(f"{cases[s]:22s} valid={ok} prior={is_prior} n_slots={None if info is None else int(info['n_slots'])} "
              f"error={'' if info is None else info['error']!r} mean_p={row.mean():.4f}")
    print("rows", len(out), "== sample", len(sample), "; columns ok", list(out.columns) == [ID, *LABELS])


if __name__ == "__main__":
    {"build": build, "parity": parity, "predict": predict}[sys.argv[1]]()
