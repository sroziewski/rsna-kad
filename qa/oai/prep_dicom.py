"""OAI baseline MRI -> the competition layout, so scripts/build_cache.py runs on it unchanged:
<out>/train_series.csv and <out>/train_series/<study>/<series>/<n>.dcm, one study per MOAKS knee.

Study id oai_<ID>_<R|L>; series id = OAI barcode.  The OAI tarballs hold extension-less DICOMs, and
dicom.list_series_files only takes *.dcm, hence the rename.  Plane/flags follow the competition convention, where
Fluid_Sensitive always equals Fat_Suppression:
  SAG_IW_TSE (fat-sat intermediate-weighted)        -> Sagittal 1/1  (sag_fs)
  COR_MPR, AX_MPR (water-excitation DESS reformats) -> Coronal / Axial 1/1  (cor_fs, ax_fs)
  COR_IW_TSE (no fat-sat)                           -> Coronal 0/0  (cor_t1: only a fallback for a missing cor_fs)
OAI has no sagittal T1, so sag_t1 is absent for every OAI study.
"""
import argparse
import os
import shutil
import tarfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pydicom

RAW = Path("/raid/sroziewski/data/oai/raw")
SEL = Path("/raid/sroziewski/data/oai/meta/selected_series.csv")
FLAGS = {"SAG_IW_TSE": ("Sagittal", 1), "COR_MPR": ("Coronal", 1), "AX_MPR": ("Axial", 1), "COR_IW_TSE": ("Coronal", 0)}


def study_id(subject, side):
    return f"oai_{subject}_{side[0]}"


def extract(job):
    tar, dest = job
    done = Path(dest) / ".done"
    if done.exists():
        return dest, len(list(Path(dest).glob("*.dcm"))), ""
    tmp = Path(dest + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        tmp.mkdir(parents=True)
        with tarfile.open(tar) as t:
            members = [m for m in t.getmembers() if m.isfile()]
            for i, m in enumerate(sorted(members, key=lambda m: m.name)):
                src = t.extractfile(m)
                with open(tmp / f"{i:04d}.dcm", "wb") as f:
                    shutil.copyfileobj(src, f)
        # OAI MPR series start with one image in another orientation; series_geometry takes the first slice's
        # orientation, which collapsed such stacks to one slice.  Keep the majority orientation only.
        iop = {}
        for f in tmp.iterdir():
            h = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=["ImageOrientationPatient"])
            iop[f] = tuple(round(float(x), 2) for x in h.ImageOrientationPatient)
        major = Counter(iop.values()).most_common(1)[0][0]
        for f, o in iop.items():
            if o != major:
                f.unlink()
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(tmp, dest)
        done.touch()
        return dest, sum(o == major for o in iop.values()), ""
    except Exception as e:  # noqa: BLE001 - reported per series
        shutil.rmtree(tmp, ignore_errors=True)
        return dest, 0, f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/raid/sroziewski/data/oai/kad_fmt")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    out = Path(a.out)
    sel = pd.read_csv(SEL, dtype=str)
    sel["StudyInstanceUID"] = [study_id(s, d) for s, d in zip(sel.src_subject_id, sel.side)]
    sel["SeriesInstanceUID"] = sel.image_file.str.extract(r"/(\d+)\.tar\.gz$")[0]
    sel["tar"] = [str(RAW / u.split("submission_60350/", 1)[1].replace("00m/", "image03/00m/", 1)) for u in sel.image_file]
    if a.limit:
        sel = sel[sel.StudyInstanceUID.isin(sel.StudyInstanceUID.drop_duplicates().head(a.limit))]
    sel["dest"] = [str(out / "train_series" / st / se) for st, se in zip(sel.StudyInstanceUID, sel.SeriesInstanceUID)]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(extract, zip(sel.tar, sel.dest), chunksize=8))
    bad = [(d, e) for d, n, e in res if e or n == 0]
    print("series", len(res), "failed", len(bad), *bad[:5], sep="\n")
    ok = sel[~sel.dest.isin({d for d, _ in bad})]
    plane, fs = zip(*(FLAGS[q] for q in ok.seq))
    csv = pd.DataFrame({"StudyInstanceUID": ok.StudyInstanceUID, "SeriesInstanceUID": ok.SeriesInstanceUID,
                        "Fluid_Sensitive": fs, "Fat_Suppression": fs, "Anatomical_Plane": plane})
    csv.to_csv(out / "train_series.csv", index=False)
    print("studies", csv.StudyInstanceUID.nunique(), "series", len(csv))
