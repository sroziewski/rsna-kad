#!/usr/bin/env python
"""Exploratory data analysis for the RSNA Knee Abnormality Detection dataset.

Runs on gpu-server2 against /raid/sroziewski/data/rsna_kad. Three stages:

  scan     read every DICOM header (no pixels) with N processes -> out/headers.parquet
  decode   decode a stratified random sample of files, time it   -> out/decode_sample.csv
  analyze  everything else (CSVs, series, headers, oddities)     -> out/*.csv + out/report_facts.txt

Every number in docs/data_analysis.md comes from `analyze` (or `decode`).
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import random
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("RSNA_DATA", "/raid/sroziewski/data/rsna_kad"))
OUT = Path(os.environ.get("RSNA_OUT", "/raid/sroziewski/sandbox/rsna_kad/analysis/out"))
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# Tags whose per-file values are kept verbatim in headers.parquet. Anything else present in
# a file is only counted (tag-presence table). Keeps the parquet small and the analysis explicit.
KEEP = [
    "SOPClassUID", "SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID",
    "Modality", "Manufacturer", "ManufacturerModelName", "SoftwareVersions", "StationName",
    "InstitutionName", "InstitutionalDepartmentName", "DeviceSerialNumber", "MagneticFieldStrength",
    "ImagingFrequency", "ReceiveCoilName", "TransmitCoilName", "PatientID", "PatientSex", "PatientAge",
    "PatientPosition", "BodyPartExamined", "Laterality", "ImageLaterality", "SeriesDescription",
    "ProtocolName", "StudyDescription", "SequenceName", "ScanningSequence", "SequenceVariant",
    "ScanOptions", "MRAcquisitionType", "ImageType", "SliceThickness", "SpacingBetweenSlices",
    "RepetitionTime", "EchoTime", "InversionTime", "FlipAngle", "EchoTrainLength", "NumberOfAverages",
    "PixelBandwidth", "AcquisitionMatrix", "InPlanePhaseEncodingDirection", "SeriesNumber",
    "AcquisitionNumber", "InstanceNumber", "ImagePositionPatient", "ImageOrientationPatient",
    "SliceLocation", "SamplesPerPixel", "PhotometricInterpretation", "PlanarConfiguration",
    "NumberOfFrames", "Rows", "Columns", "PixelSpacing", "BitsAllocated", "BitsStored", "HighBit",
    "PixelRepresentation", "WindowCenter", "WindowWidth", "RescaleIntercept", "RescaleSlope",
    "RescaleType", "SmallestImagePixelValue", "LargestImagePixelValue", "PixelPaddingValue",
    "LossyImageCompression", "LossyImageCompressionRatio", "BurnedInAnnotation",
    "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate", "StudyTime", "SeriesTime",
    "AcquisitionTime", "ContentTime", "PatientBirthDate", "AccessionNumber", "StudyID",
]


# --------------------------------------------------------------------------- scan
def _val(elem) -> str:
    if elem.VR == "SQ":
        return f"SQ[{len(elem.value)}]"
    v = elem.value
    if isinstance(v, bytes):
        return f"bytes[{len(v)}]"
    if elem.VM > 1 or isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
        return "\\".join(str(x) for x in v)
    return str(v)


def read_header(rel: str) -> dict:
    import pydicom

    path = DATA / rel
    rec = {"path": rel, "size": -1, "error": "", "pixel_offset": -1, "tags": ""}
    try:
        rec["size"] = path.stat().st_size
        with open(path, "rb") as f:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            rec["pixel_offset"] = f.tell()
        fm = ds.file_meta
        rec["TransferSyntaxUID"] = str(fm.get("TransferSyntaxUID", ""))
        rec["MediaStorageSOPClassUID"] = str(fm.get("MediaStorageSOPClassUID", ""))
        tags = []
        for elem in ds:
            kw = elem.keyword or f"{elem.tag.group:04X}{elem.tag.element:04X}"
            tags.append(kw)
            if kw in KEEP:
                rec[kw] = _val(elem)[:512]
        rec["tags"] = ",".join(tags)
    except Exception as e:  # report every failure, never skip
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    return rec


def list_files() -> list[str]:
    files = []
    for split in ("train_series", "test_series"):
        root = DATA / split
        for study in sorted(os.listdir(root)):
            sdir = root / study
            if not sdir.is_dir():
                files.append(f"{split}/{study}")  # stray non-directory entry, reported later
                continue
            for series in sorted(os.listdir(sdir)):
                rdir = sdir / series
                if not rdir.is_dir():
                    files.append(f"{split}/{study}/{series}")
                    continue
                for fn in sorted(os.listdir(rdir)):
                    files.append(f"{split}/{study}/{series}/{fn}")
    return files


def cmd_scan(args):
    t0 = time.time()
    files = list_files()
    print(f"listed {len(files)} files in {time.time()-t0:.0f}s", flush=True)
    recs = []
    with Pool(args.procs) as pool:
        for i, rec in enumerate(pool.imap_unordered(read_header, files, chunksize=256)):
            recs.append(rec)
            if i % 50000 == 0:
                print(f"{i}/{len(files)} {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(recs)
    parts = df["path"].str.split("/", expand=True)
    df["split"], df["study_dir"], df["series_dir"], df["file"] = parts[0], parts[1], parts[2], parts[3]
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "headers.parquet", index=False)
    print(f"wrote {len(df)} rows, {df['error'].ne('').sum()} errors, {time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- decode
def decode_one(rel: str) -> dict:
    import pydicom

    rec = {"path": rel, "error": "", "t_read_ms": np.nan, "t_decode_ms": np.nan}
    try:
        t = time.perf_counter()
        ds = pydicom.dcmread(DATA / rel)
        rec["t_read_ms"] = (time.perf_counter() - t) * 1000
        rec["TransferSyntaxUID"] = str(ds.file_meta.TransferSyntaxUID)
        t = time.perf_counter()
        arr = ds.pixel_array
        rec["t_decode_ms"] = (time.perf_counter() - t) * 1000
        rec["dtype"] = str(arr.dtype)
        rec["shape"] = "x".join(map(str, arr.shape))
        flat = arr.ravel().astype(np.float64)
        q = np.percentile(flat, [0, 0.1, 1, 50, 99, 99.9, 100])
        for name, v in zip(["min", "p0.1", "p1", "p50", "p99", "p99.9", "max"], q):
            rec[name] = v
        rec["mean"] = flat.mean()
        rec["frac_zero"] = float((flat == 0).mean())
        rec["n_unique"] = int(np.unique(arr).size)
        rec["max_gt_bitsstored"] = bool(q[-1] >= 2 ** int(ds.BitsStored))
        rec["RescaleSlope"] = str(ds.get("RescaleSlope", ""))
        rec["RescaleIntercept"] = str(ds.get("RescaleIntercept", ""))
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    return rec


def cmd_decode(args):
    cols = ["path", "TransferSyntaxUID", "error", "split", "Manufacturer", "BitsStored", "PixelRepresentation", "size"]
    h = pd.read_parquet(OUT / "headers.parquet", columns=cols)
    h = h[h["error"].eq("") & h["split"].eq("train_series")]
    rng = random.Random(args.seed)
    # stratify by (transfer syntax, manufacturer): one syntax exists, so the manufacturer axis is what varies
    h["stratum"] = h["TransferSyntaxUID"] + "|" + h["Manufacturer"].fillna("?")
    strata = h["stratum"].value_counts()
    per = max(args.n // len(strata), 20)
    sample = []
    for s in strata.index:
        paths = h.loc[h["stratum"].eq(s), "path"].tolist()
        sample += rng.sample(paths, min(per, len(paths)))
    extra = OUT / "pixel_size_mismatch.csv"
    if extra.exists():
        sample += pd.read_csv(extra)["path"].tolist()
    print(f"decoding {len(sample)} files across {len(strata)} strata", flush=True)
    # single process so the timings are per-file wall times, not contended
    recs = [decode_one(p) for p in sample]
    df = pd.DataFrame(recs).merge(h[["path", "Manufacturer", "BitsStored", "PixelRepresentation", "size"]], on="path", how="left")
    df.to_csv(OUT / "decode_sample.csv", index=False)


def _drop_cache(rel: str) -> int:
    fd = os.open(DATA / rel, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return 0


def _decode_bytes(rel: str) -> int:
    import pydicom

    ds = pydicom.dcmread(DATA / rel)
    return int(ds.pixel_array.nbytes)


def cmd_throughput(args):
    """Cold- vs warm-cache full-decode throughput; posix_fadvise(DONTNEED) evicts our own files without root."""
    h = pd.read_parquet(OUT / "headers.parquet", columns=["path", "size", "split", "error"])
    h = h[h["split"].eq("train_series") & h["error"].eq("")]
    rng = random.Random(args.seed)
    sample = rng.sample(h["path"].tolist(), args.n)
    total_mb = h.set_index("path").loc[sample, "size"].sum() / 1e6
    rows = []
    for procs in args.procs:
        for phase in ("cold", "warm"):
            if phase == "cold":
                with Pool(24) as pool:
                    list(pool.imap_unordered(_drop_cache, sample, chunksize=64))
            t = time.time()
            with Pool(procs) as pool:
                px = sum(pool.imap_unordered(_decode_bytes, sample, chunksize=16))
            dt = time.time() - t
            rows.append({"procs": procs, "cache": phase, "files": len(sample), "file_MB": round(total_mb), "seconds": round(dt, 1),
                         "files_per_s": round(len(sample) / dt), "MB_per_s": round(total_mb / dt), "pixel_MB": round(px / 1e6)})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "decode_throughput.csv", index=False)
    n_train = int((pd.read_parquet(OUT / "headers.parquet", columns=["split"])["split"] == "train_series").sum())
    for _, r in df.iterrows():
        print(f"procs={r.procs} {r.cache}: full train ({n_train} files) ~ {n_train / r.files_per_s / 60:.1f} min", flush=True)


# --------------------------------------------------------------------------- analyze
class Facts:
    """Collects human-readable facts + small tables; everything in the report comes from here."""

    def __init__(self):
        self.lines: list[str] = []

    def p(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.lines.append(s)

    def table(self, name: str, df: pd.DataFrame, show: int | None = 40):
        df.to_csv(OUT / f"{name}.csv")
        self.p(f"\n## {name}  (rows={len(df)})")
        with pd.option_context("display.max_rows", 200, "display.max_columns", 50, "display.width", 200):
            self.p(df.head(show).to_string() if show else df.to_string())


def analyze_train_csv(F: Facts, dirs_train: set[str]) -> pd.DataFrame:
    tr = pd.read_csv(DATA / "train.csv", dtype={"StudyInstanceUID": str})
    F.p("# 1. train.csv")
    F.p("rows", len(tr), "unique StudyInstanceUID", tr["StudyInstanceUID"].nunique(),
        "duplicate study rows", tr["StudyInstanceUID"].duplicated().sum())
    F.p("columns", list(tr.columns))
    lab = tr[LABELS]
    n_present = lab.notna().sum(axis=1)
    F.p("labels present per row (count of rows):", n_present.value_counts().sort_index().to_dict())
    tr["labelled"] = n_present.eq(12)
    F.p("rows with any label", (n_present > 0).sum(), "rows with all 12", tr["labelled"].sum(),
        "rows with none", (n_present == 0).sum())
    lab_val = lab[tr["labelled"]].astype(float)
    F.p("label value set", sorted(pd.unique(lab_val.values.ravel())))
    prev = pd.DataFrame({"positives": lab_val.sum().astype(int), "n": len(lab_val)})
    prev["prevalence"] = (prev["positives"] / prev["n"]).round(4)
    F.table("label_prevalence", prev, None)
    F.p("positives per labelled study:", lab_val.sum(axis=1).astype(int).value_counts().sort_index().to_dict())
    F.p("all-negative labelled studies:", int(lab_val.sum(axis=1).eq(0).sum()))
    co = lab_val.T.astype(int) @ lab_val.astype(int)
    F.table("label_cooccurrence_counts", co, None)
    # phi correlation
    F.table("label_phi_correlation", lab_val.corr().round(2), None)
    tr["has_dir"] = tr["StudyInstanceUID"].isin(dirs_train)
    F.p("train.csv studies with image dir", tr["has_dir"].sum(), "without", (~tr["has_dir"]).sum())
    F.p("labelled studies with image dir", (tr["labelled"] & tr["has_dir"]).sum(),
        "labelled without dir", (tr["labelled"] & ~tr["has_dir"]).sum())
    F.p("image dirs without train.csv row", len(dirs_train - set(tr["StudyInstanceUID"])))

    # ---- reports
    rep = tr["Report"].fillna("")
    tr["rep_chars"] = rep.str.len()
    tr["rep_words"] = rep.str.split().str.len()
    tr["rep_lines"] = rep.str.count("\n") + 1
    F.p("empty/NaN reports", int(rep.eq("").sum()), "NaN reports", int(tr["Report"].isna().sum()))
    F.table("report_length_stats", tr[["rep_chars", "rep_words", "rep_lines"]].describe(
        percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99]).round(1), None)
    F.p("reports with newline", int(rep.str.contains("\n").sum()))
    F.p("exact duplicate reports (rows sharing text with another row)", int(rep.duplicated(keep=False).sum()),
        "distinct texts", rep.nunique())
    norm = rep.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    F.p("normalized duplicate reports", int(norm.duplicated(keep=False).sum()))
    dup_top = norm[norm.duplicated(keep=False)].value_counts().head(10)
    F.table("report_top_duplicates", dup_top.rename("count").to_frame().assign(
        text=lambda d: d.index.str.slice(0, 120)).reset_index(drop=True), None)
    # placeholders / templating
    ph = collections.Counter()
    for t in rep:
        ph.update(re.findall(r"\[[A-Za-z _\-]{2,30}\]", t))
        ph.update(re.findall(r"<[A-Za-z _\-]{2,30}>", t))
        ph.update(re.findall(r"\b(?:X{3,}|_{3,}|\*{3,})\b", t))
    F.table("report_placeholders", pd.Series(dict(ph.most_common(40)), name="count").to_frame(), None)
    rx = {"date_dmy": r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", "date_iso": r"\b\d{4}-\d{2}-\d{2}\b",
          "bracket_any": r"\[[^\]]{1,40}\]", "angle_any": r"<[^>]{1,40}>", "digits4": r"\b(19|20)\d{2}\b",
          "impression_en": r"(?i)\bimpression\b", "conclusion": r"(?i)\b(conclusi[oó]n|conclusão|schlussfolgerung|wnioski|résumé|conclusione)\b",
          "findings_en": r"(?i)\bfindings\b", "technique": r"(?i)\b(t[eé]cnica|technique|technik|technika|tecnica)\b"}
    F.table("report_regex_hits", pd.Series({k: int(rep.str.contains(v, regex=True).sum()) for k, v in rx.items()},
                                           name="rows_with_match").to_frame(), None)

    # language
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0

        def lang(t):
            try:
                return detect(t) if len(t.strip()) >= 10 else "empty"
            except Exception:
                return "unk"
        tr["lang"] = rep.map(lang)
    except Exception as e:
        F.p("langdetect failed:", e)
        tr["lang"] = "n/a"
    lt = tr.groupby("lang").agg(n=("lang", "size"), labelled=("labelled", "sum"),
                                med_chars=("rep_chars", "median")).sort_values("n", ascending=False)
    lt["labelled_frac"] = (lt["labelled"] / lt["n"]).round(3)
    F.table("report_language", lt, None)
    ex = []
    for lg in lt.index[:8]:
        sub = tr[tr["lang"].eq(lg)].sort_values("rep_chars")
        for _, r in sub.iloc[[len(sub) // 4, len(sub) // 2, 3 * len(sub) // 4]].iterrows():
            ex.append({"lang": lg, "labelled": r["labelled"], "chars": r["rep_chars"],
                       "excerpt": r["Report"].replace("\n", " ⏎ ")[:220]})
    F.table("report_examples", pd.DataFrame(ex), None)
    # label prevalence by language (labelled rows only)
    bl = tr[tr["labelled"]].groupby("lang")[LABELS].mean().round(3)
    bl.insert(0, "n", tr[tr["labelled"]].groupby("lang").size())
    F.table("label_prevalence_by_language", bl.sort_values("n", ascending=False), None)
    # keyword sanity: does the report mention ACL-ish terms when label=1?
    kw = {"ACL": r"(?i)\b(acl|lca|lcaa|cruzado anterior|crociato anteriore|vorderes kreuzband|anterior cruciate|acl-|krzyżow|croisé antérieur|lcae)\b",
          "Fracture": r"(?i)(fractur|fractura|frattura|fraktur|złamani)",
          "Baker's": r"(?i)(baker)"}
    rows = []
    for L, pat in kw.items():
        m = rep.str.contains(pat, regex=True)
        sub = tr[tr["labelled"]]
        rows.append({"label": L, "mention_rate_pos": round(m[sub.index][sub[L].eq(1)].mean(), 3),
                     "mention_rate_neg": round(m[sub.index][sub[L].eq(0)].mean(), 3),
                     "mention_rate_unlabelled": round(m[~tr["labelled"]].mean(), 3)})
    F.table("report_keyword_vs_label", pd.DataFrame(rows).set_index("label"), None)
    tr[["StudyInstanceUID", "labelled", "has_dir", "rep_chars", "rep_words", "lang"]].to_csv(OUT / "train_meta.csv", index=False)
    return tr


def analyze_series(F: Facts, h: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    F.p("\n# 2. Series")
    ts = pd.read_csv(DATA / "train_series.csv", dtype=str)
    te = pd.read_csv(DATA / "test_series.csv", dtype=str)
    for name, s in (("train_series.csv", ts), ("test_series.csv", te)):
        F.p(name, "rows", len(s), "studies", s["StudyInstanceUID"].nunique(), "series", s["SeriesInstanceUID"].nunique(),
            "dup series", s["SeriesInstanceUID"].duplicated().sum(), "NaN cells", int(s.isna().sum().sum()))
        F.p("  values:", {c: s[c].value_counts(dropna=False).to_dict() for c in ["Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane"]})
    ts["combo"] = ts["Anatomical_Plane"] + "|FS=" + ts["Fluid_Sensitive"] + "|FatSup=" + ts["Fat_Suppression"]
    te["combo"] = te["Anatomical_Plane"] + "|FS=" + te["Fluid_Sensitive"] + "|FatSup=" + te["Fat_Suppression"]
    F.p("train rows where Fluid_Sensitive != Fat_Suppression:", int((ts["Fluid_Sensitive"] != ts["Fat_Suppression"]).sum()),
        "test:", int((te["Fluid_Sensitive"] != te["Fat_Suppression"]).sum()))
    sps = ts.groupby("StudyInstanceUID").size()
    F.p("series per study (train):", sps.value_counts().sort_index().to_dict(), "mean", round(sps.mean(), 2))
    combo = ts["combo"].value_counts().to_frame("series")
    combo["studies_with"] = ts.groupby("combo")["StudyInstanceUID"].nunique()
    combo["frac_studies"] = (combo["studies_with"] / ts["StudyInstanceUID"].nunique()).round(3)
    F.table("series_combo_counts", combo, None)
    per = ts.groupby("StudyInstanceUID")["combo"].agg(lambda x: "; ".join(sorted(x)))
    F.table("series_combo_per_study_top", per.value_counts().head(25).to_frame("studies"), None)
    ns = ts["StudyInstanceUID"].nunique()
    has = {}
    for plane in ("Sagittal", "Coronal", "Axial"):
        for fs in ("1", "0"):
            m = ts["Anatomical_Plane"].eq(plane) & ts["Fluid_Sensitive"].eq(fs)
            has[f"{plane} FS={fs}"] = ts.loc[m, "StudyInstanceUID"].nunique()
        has[f"{plane} any"] = ts.loc[ts["Anatomical_Plane"].eq(plane), "StudyInstanceUID"].nunique()
    has["Sagittal FS=1 & Coronal FS=1 & Axial FS=1"] = int(np.all([ts.loc[ts["Anatomical_Plane"].eq(p) & ts["Fluid_Sensitive"].eq("1")]
                                                              .groupby("StudyInstanceUID").size().reindex(sps.index).fillna(0) > 0 for p in ("Sagittal", "Coronal", "Axial")], axis=0).sum())
    ht = pd.DataFrame({"studies_with": has})
    ht["studies_without"] = ns - ht["studies_with"]
    ht["frac_with"] = (ht["studies_with"] / ns).round(3)
    F.table("series_type_availability", ht, None)
    # multiples of same combo per study
    dup_combo = ts.groupby(["StudyInstanceUID", "combo"]).size()
    F.p("studies with >1 series of the same combo:", int((dup_combo > 1).groupby(level=0).any().sum()),
        "max same-combo count", int(dup_combo.max()))

    # disk vs csv
    hd = h[h["split"].eq("train_series")]
    disk = set(zip(hd["study_dir"], hd["series_dir"]))
    csv = set(zip(ts["StudyInstanceUID"], ts["SeriesInstanceUID"]))
    F.p("train: (study,series) pairs on disk", len(disk), "in csv", len(csv), "disk-only", len(disk - csv), "csv-only", len(csv - disk))
    F.p("train: study dirs", hd["study_dir"].nunique(), "csv studies", ts["StudyInstanceUID"].nunique(),
        "dirs not in csv", len(set(hd["study_dir"]) - set(ts["StudyInstanceUID"])), "csv not on disk", len(set(ts["StudyInstanceUID"]) - set(hd["study_dir"])))
    hte = h[h["split"].eq("test_series")]
    disk_t = set(zip(hte["study_dir"], hte["series_dir"]))
    csv_t = set(zip(te["StudyInstanceUID"], te["SeriesInstanceUID"]))
    F.p("test: pairs on disk", len(disk_t), "in csv", len(csv_t), "disk-only", len(disk_t - csv_t), "csv-only", len(csv_t - disk_t))
    # header UIDs vs directory names
    F.p("files whose header StudyInstanceUID != dir:", int((hd["StudyInstanceUID"] != hd["study_dir"]).sum()),
        "SeriesInstanceUID != dir:", int((hd["SeriesInstanceUID"] != hd["series_dir"]).sum()),
        "SOPInstanceUID != filename:", int((hd["SOPInstanceUID"] + ".dcm" != hd["file"]).sum()))
    # slices per series
    fps = hd.groupby(["study_dir", "series_dir"]).size().rename("n_files")
    F.table("files_per_series_stats", fps.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(1).to_frame(), None)
    F.p("files per series histogram (binned):", pd.cut(fps, [0, 1, 5, 10, 15, 20, 25, 30, 40, 60, 100, 1000]).value_counts().sort_index().to_dict())
    sdf = fps.reset_index().merge(ts, left_on=["study_dir", "series_dir"], right_on=["StudyInstanceUID", "SeriesInstanceUID"], how="left")
    F.table("files_per_series_by_combo", sdf.groupby("combo")["n_files"].describe(percentiles=[.05, .5, .95]).round(1), None)
    return ts, te


def _first_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.split("\\").str[0], errors="coerce")


def analyze_headers(F: Facts, h: pd.DataFrame, ts: pd.DataFrame, tr: pd.DataFrame):
    F.p("\n# 3. DICOM headers")
    F.p("files scanned", len(h), "train", int(h["split"].eq("train_series").sum()), "test", int(h["split"].eq("test_series").sum()))
    err = h[h["error"].ne("")]
    F.p("read errors", len(err))
    if len(err):
        F.table("header_errors", err[["path", "size", "error"]], 50)
    F.p("zero-size files", int(h["size"].eq(0).sum()), "size stats (bytes):", h["size"].describe().round(0).to_dict())
    F.p("entries not ending in .dcm:", int((~h["path"].str.endswith(".dcm")).sum()))
    ok = h[h["error"].eq("")].copy()
    # tag presence
    cnt = collections.Counter()
    for tags, n in ok["tags"].value_counts().items():
        for t in tags.split(","):
            cnt[t] += n
    tp = pd.Series(cnt, name="files").sort_values(ascending=False).to_frame()
    tp["frac"] = (tp["files"] / len(ok)).round(4)
    F.table("tag_presence", tp, None)
    F.p("distinct tags", len(tp), "tags in 100% of files", int(tp["files"].eq(len(ok)).sum()),
        "distinct tag-set patterns", ok["tags"].nunique())
    for col in ["TransferSyntaxUID", "MediaStorageSOPClassUID", "SOPClassUID", "Modality", "PhotometricInterpretation",
                "SamplesPerPixel", "BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation", "NumberOfFrames",
                "PlanarConfiguration", "LossyImageCompression", "BurnedInAnnotation", "BodyPartExamined", "PatientPosition",
                "MRAcquisitionType", "RescaleType", "PixelPaddingValue", "Laterality", "ImageLaterality", "PatientSex"]:
        if col in ok:
            F.p(f"{col}:", ok[col].fillna("<absent>").value_counts().head(12).to_dict())
    ok["dims"] = ok["Rows"] + "x" + ok["Columns"]
    F.table("dims_counts", ok["dims"].value_counts().head(40).to_frame("files"), None)
    ok["rows_i"], ok["cols_i"] = pd.to_numeric(ok["Rows"]), pd.to_numeric(ok["Columns"])
    F.p("Rows stats", ok["rows_i"].describe().round(0).to_dict())
    F.p("Columns stats", ok["cols_i"].describe().round(0).to_dict())
    F.p("distinct dims", ok["dims"].nunique(), "square", int((ok["rows_i"] == ok["cols_i"]).mean() * 1000) / 10, "%")
    # pixel payload check for uncompressed
    unc = ok["TransferSyntaxUID"].isin(["1.2.840.10008.1.2", "1.2.840.10008.1.2.1", "1.2.840.10008.1.2.2"])
    exp = ok["rows_i"] * ok["cols_i"] * pd.to_numeric(ok["BitsAllocated"]) // 8 * pd.to_numeric(ok["SamplesPerPixel"]).fillna(1)
    trailing = ok["size"] - ok["pixel_offset"]
    bad = unc & ((trailing - 12) != exp) & ((trailing - 8) != exp)
    F.p("uncompressed files whose pixel payload size != Rows*Cols*Bytes:", int(bad.sum()))
    if bad.sum():
        badf = ok.loc[bad, ["path", "size", "pixel_offset", "dims", "BitsAllocated"]].copy()
        badf["expected_pixel_bytes"] = exp[bad]
        badf["actual_pixel_bytes"] = trailing[bad] - 12
        manifest = DATA / ".download" / "manifest.tsv"
        if manifest.exists():
            man = pd.read_csv(manifest, sep="\t", header=None, names=["crc", "size", "path"], dtype=str).set_index("path")
            badf["manifest_size"] = badf["path"].map(man["size"])  # equal to on-disk size => file is like this in the Kaggle zip
        F.table("pixel_size_mismatch", badf, 30)
        ctx = ok[ok["series_dir"].isin(ok.loc[bad, "series_dir"])].groupby(["study_dir", "series_dir"]).agg(
            n_files=("path", "size"), n_truncated=("path", lambda x: int(x.isin(badf["path"]).sum())), desc=("SeriesDescription", "first"),
            inst_of_bad=("InstanceNumber", lambda x: ",".join(ok.loc[x.index, "InstanceNumber"][ok.loc[x.index, "path"].isin(badf["path"])])))
        ctx["labelled"] = ctx.index.get_level_values(0).isin(tr.loc[tr["labelled"], "StudyInstanceUID"])
        F.table("pixel_size_mismatch_context", ctx, None)
    # spacing
    ps = ok["PixelSpacing"].str.split("\\", expand=True)
    ok["ps_r"], ok["ps_c"] = pd.to_numeric(ps[0], errors="coerce"), pd.to_numeric(ps[1], errors="coerce")
    ok["thk"] = pd.to_numeric(ok["SliceThickness"], errors="coerce")
    ok["sbs"] = pd.to_numeric(ok["SpacingBetweenSlices"], errors="coerce")
    F.table("spacing_stats", ok[["ps_r", "ps_c", "thk", "sbs"]].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(3), None)
    F.p("PixelSpacing missing", int(ok["ps_r"].isna().sum()), "anisotropic in-plane (|r-c|>1e-3)", int((ok["ps_r"] - ok["ps_c"]).abs().gt(1e-3).sum()),
        "SliceThickness missing", int(ok["thk"].isna().sum()), "SpacingBetweenSlices missing", int(ok["sbs"].isna().sum()))
    ok["fov_r"] = (ok["ps_r"] * ok["rows_i"]).round(0)
    F.p("field of view rows*spacing (mm) stats", ok["fov_r"].describe(percentiles=[.05, .5, .95]).round(0).to_dict())
    # orientation -> plane
    iop = ok["ImageOrientationPatient"].str.split("\\", expand=True).apply(pd.to_numeric, errors="coerce")

    def plane_from_iop(m: np.ndarray) -> np.ndarray:
        r, c = m[:, :3], m[:, 3:6]
        n = np.cross(r, c)
        ax = np.abs(n).argmax(axis=1)
        out = np.array(["Sagittal", "Coronal", "Axial"])[ax]
        out = np.where(np.isnan(n).any(axis=1), "missing", out)
        return out
    if iop.shape[1] >= 6:
        ok["iop_plane"] = plane_from_iop(iop.iloc[:, :6].to_numpy(dtype=float))
        n = np.cross(iop.iloc[:, :3].to_numpy(float), iop.iloc[:, 3:6].to_numpy(float))
        ok["obliquity_deg"] = np.degrees(np.arccos(np.clip(np.abs(n).max(axis=1) / np.linalg.norm(n, axis=1), 0, 1))).round(1)
    else:
        ok["iop_plane"], ok["obliquity_deg"] = "missing", np.nan
    F.p("ImageOrientationPatient missing", int(ok["ImageOrientationPatient"].isna().sum()))
    F.p("obliquity (deg from nearest axis) stats", ok["obliquity_deg"].describe(percentiles=[.5, .9, .99]).round(1).to_dict())
    both = pd.concat([ts, pd.read_csv(DATA / "test_series.csv", dtype=str)], ignore_index=True)
    m = ok.merge(both[["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]],
                 left_on=["study_dir", "series_dir"], right_on=["StudyInstanceUID", "SeriesInstanceUID"], how="left", suffixes=("", "_csv"))
    ct = pd.crosstab(m["Anatomical_Plane"].fillna("<no csv row>"), m["iop_plane"])
    F.table("plane_csv_vs_iop_files", ct, None)
    ser_plane = m.groupby(["study_dir", "series_dir"]).agg(csv=("Anatomical_Plane", "first"), iop=("iop_plane", lambda x: x.mode().iat[0]),
                                                           n_iop_planes=("iop_plane", "nunique"))
    F.p("series where IOP-plane != csv plane:", int((ser_plane["csv"] != ser_plane["iop"]).sum()),
        "series with mixed IOP planes inside:", int((ser_plane["n_iop_planes"] > 1).sum()))
    F.table("plane_mismatch_series", ser_plane[ser_plane["csv"] != ser_plane["iop"]].join(
        m.groupby(["study_dir", "series_dir"])["SeriesDescription"].first()), 30)
    # 3D acquisitions and slice counts per csv combo
    acq = m.groupby(["study_dir", "series_dir"]).agg(acq=("MRAcquisitionType", "first"), combo=("Anatomical_Plane", "first"),
                                                      fs=("Fluid_Sensitive", "first"), n=("path", "size"), thk=("thk", "first"),
                                                      rows=("rows_i", "first"), cols=("cols_i", "first"), desc=("SeriesDescription", "first"))
    acq["combo"] = acq["combo"] + "|FS=" + acq["fs"]
    F.table("acqtype_by_combo_series", pd.crosstab(acq["combo"].fillna("<test>"), acq["acq"].fillna("<absent>")), None)
    F.p("3D series: n", int(acq["acq"].eq("3D").sum()), "slices stats", acq.loc[acq["acq"].eq("3D"), "n"].describe(percentiles=[.05, .5, .95]).round(0).to_dict(),
        "thickness stats", acq.loc[acq["acq"].eq("3D"), "thk"].describe(percentiles=[.05, .5, .95]).round(2).to_dict())
    F.p("2D series: n", int(acq["acq"].eq("2D").sum()), "slices stats", acq.loc[acq["acq"].eq("2D"), "n"].describe(percentiles=[.05, .5, .95]).round(0).to_dict())
    F.p("series with >60 slices by acq type:", acq.loc[acq["n"] > 60, "acq"].fillna("<absent>").value_counts().to_dict())
    F.table("top_desc_3d", acq.loc[acq["acq"].eq("3D"), "desc"].fillna("<absent>").value_counts().head(15).to_frame("series"), None)
    F.p("non-square series by acq type:", acq.loc[acq["rows"] != acq["cols"], "acq"].fillna("<absent>").value_counts().to_dict(),
        "non-square series total", int((acq["rows"] != acq["cols"]).sum()))
    F.table("dims_by_combo_series", acq.groupby("combo")[["rows", "cols", "n", "thk"]].median().round(1), None)
    F.table("bits_by_manufacturer_files", pd.crosstab(ok["Manufacturer"], ok["BitsStored"] + "b/pr" + ok["PixelRepresentation"]), None)
    F.table("bodypart_by_study", ok.groupby("study_dir")["BodyPartExamined"].agg(lambda x: "|".join(sorted(set(x.fillna("<absent>"))))).value_counts().head(15).to_frame("studies"), None)

    # slice ordering
    g = ok.groupby(["study_dir", "series_dir"])
    ok["inst"] = pd.to_numeric(ok["InstanceNumber"], errors="coerce")
    ipp = ok["ImagePositionPatient"].str.split("\\", expand=True).apply(pd.to_numeric, errors="coerce")
    ok["ipp_str"] = ok["ImagePositionPatient"]
    ser = g.agg(n=("path", "size"), inst_missing=("inst", lambda x: x.isna().sum()), inst_unique=("inst", "nunique"),
                inst_min=("inst", "min"), inst_max=("inst", "max"), ipp_missing=("ipp_str", lambda x: x.isna().sum()),
                ipp_unique=("ipp_str", "nunique"), dims_unique=("dims", "nunique"), ts_unique=("TransferSyntaxUID", "nunique"),
                thk_unique=("thk", "nunique"), ps_unique=("ps_r", "nunique"), sbs=("sbs", "first"), thk=("thk", "first"),
                desc=("SeriesDescription", "first"), acq_unique=("AcquisitionNumber", "nunique"), echo_unique=("EchoTime", "nunique"),
                frame_unique=("FrameOfReferenceUID", "nunique"), tsyn=("TransferSyntaxUID", "first"))
    ser["inst_dup"] = ser["n"] - ser["inst_unique"]
    ser["inst_contiguous"] = (ser["inst_max"] - ser["inst_min"] + 1 == ser["n"]) & ser["inst_missing"].eq(0)
    ser["ipp_dup"] = ser["n"] - ser["ipp_unique"]
    F.p("series total", len(ser))
    F.p("series with missing InstanceNumber", int((ser["inst_missing"] > 0).sum()), "with duplicate InstanceNumber", int((ser["inst_dup"] > 0).sum()),
        "non-contiguous InstanceNumber range", int((~ser["inst_contiguous"]).sum()), "InstanceNumber min !=1", int(ser["inst_min"].ne(1).sum()))
    F.p("series with missing IPP", int((ser["ipp_missing"] > 0).sum()), "with duplicate IPP", int((ser["ipp_dup"] > 0).sum()))
    F.p("series with mixed Rows x Columns", int((ser["dims_unique"] > 1).sum()), "mixed transfer syntax", int((ser["ts_unique"] > 1).sum()),
        "mixed SliceThickness", int((ser["thk_unique"] > 1).sum()), "mixed PixelSpacing", int((ser["ps_unique"] > 1).sum()),
        ">1 AcquisitionNumber", int((ser["acq_unique"] > 1).sum()), ">1 EchoTime", int((ser["echo_unique"] > 1).sum()),
        ">1 FrameOfReferenceUID", int((ser["frame_unique"] > 1).sum()))
    F.p("series with 1 file", int(ser["n"].eq(1).sum()), "with <=3 files", int(ser["n"].le(3).sum()), "with >60 files", int(ser["n"].gt(60).sum()))
    # slice spacing from IPP along normal
    okn = ok.dropna(subset=["obliquity_deg"]).copy()
    ipp_n = ipp.loc[okn.index].iloc[:, :3].to_numpy(float)
    nrm = np.cross(iop.loc[okn.index].iloc[:, :3].to_numpy(float), iop.loc[okn.index].iloc[:, 3:6].to_numpy(float))
    okn["proj"] = (ipp_n * nrm).sum(axis=1)
    sp = okn.sort_values("proj").groupby(["study_dir", "series_dir"])["proj"].agg(
        lambda x: np.nan if len(x) < 2 else float(np.median(np.diff(x.to_numpy())))).rename("ipp_step")
    ser = ser.join(sp)
    ser["step_vs_sbs"] = (ser["ipp_step"] - ser["sbs"]).abs()
    F.p("series where median IPP step differs from SpacingBetweenSlices by >0.1mm:", int((ser["step_vs_sbs"] > 0.1).sum()),
        "SpacingBetweenSlices absent:", int(ser["sbs"].isna().sum()))
    F.p("IPP step (mm) stats:", ser["ipp_step"].describe(percentiles=[.05, .5, .95]).round(2).to_dict())
    # does InstanceNumber order == IPP order?
    def order_agree(df):
        if df["inst"].isna().any() or df["proj"].isna().any() or len(df) < 2:
            return np.nan
        a = df.sort_values("inst")["proj"].to_numpy()
        d = np.diff(a)
        return bool((d > 0).all() or (d < 0).all())
    oa = okn.groupby(["study_dir", "series_dir"])[["inst", "proj"]].apply(order_agree).rename("inst_order_monotone_in_space")
    ser = ser.join(oa)
    F.p("InstanceNumber order monotone along slice normal:", ser["inst_order_monotone_in_space"].value_counts(dropna=False).to_dict())
    ser.to_csv(OUT / "series_summary.csv")
    weird = ser[(ser["dims_unique"] > 1) | (ser["inst_dup"] > 0) | (ser["ipp_dup"] > 0) | (ser["n"].le(3)) | ser["inst_order_monotone_in_space"].eq(False)]
    F.table("series_oddities", weird[["n", "dims_unique", "inst_dup", "ipp_dup", "inst_contiguous", "acq_unique", "echo_unique", "inst_order_monotone_in_space", "desc"]], 40)

    # descriptions
    for col in ["SeriesDescription", "ProtocolName", "StudyDescription", "SequenceName", "ScanningSequence", "SequenceVariant", "ScanOptions", "ImageType", "ReceiveCoilName"]:
        if col in ok:
            vc = ok.groupby(["study_dir", "series_dir"])[col].first().fillna("<absent>").value_counts()
            F.p(f"{col}: distinct={len(vc)} absent_series={int(vc.get('<absent>', 0))}")
            F.table(f"top_{col}", vc.head(50).to_frame("series"), 50)
    # series description vs csv combo
    sd = m.groupby(["study_dir", "series_dir"]).agg(desc=("SeriesDescription", "first"), plane=("Anatomical_Plane", "first"),
                                                     fs=("Fluid_Sensitive", "first"), fat=("Fat_Suppression", "first"), te=("EchoTime", "first"),
                                                     tr=("RepetitionTime", "first"), scanopt=("ScanOptions", "first"), seq=("ScanningSequence", "first"))
    sd["te"], sd["tr"] = pd.to_numeric(sd["te"], errors="coerce"), pd.to_numeric(sd["tr"], errors="coerce")
    F.table("echo_time_by_fluid_sensitive", sd.groupby("fs")[["te", "tr"]].describe(percentiles=[.05, .5, .95]).round(0).T, None)
    F.table("scanoptions_vs_fatsup", pd.crosstab(sd["scanopt"].fillna("<absent>"), sd["fat"].fillna("<no csv>")).sort_values("1", ascending=False).head(20), None)

    # site / scanner
    F.p("\n## scanner / site")
    for col in ["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "SoftwareVersions", "StationName", "InstitutionName",
                "DeviceSerialNumber", "InstitutionalDepartmentName"]:
        if col in ok:
            vc = ok.groupby("study_dir")[col].first().fillna("<absent>").value_counts()
            F.p(f"{col}: distinct={len(vc)}")
            F.table(f"studies_by_{col}", vc.head(40).to_frame("studies"), 40)
        else:
            F.p(f"{col}: not present in any file")
    ok_all = ok
    ok = ok[ok["split"].eq("train_series")]  # site/patient facts are about the training set only
    st = ok.groupby("study_dir").agg(manu=("Manufacturer", "first"), model=("ManufacturerModelName", "first"), field=("MagneticFieldStrength", "first"),
                                     sw=("SoftwareVersions", "first"), freq=("ImagingFrequency", "first"), pid=("PatientID", "first"),
                                     n_pid=("PatientID", "nunique"), n_manu=("Manufacturer", "nunique"), lat=("Laterality", lambda x: "|".join(sorted(set(x.fillna("<absent>"))))),
                                     sex=("PatientSex", "first"), age=("PatientAge", "first"), n_files=("path", "size"), n_series=("series_dir", "nunique"),
                                     tsyn=("TransferSyntaxUID", "first"), coil=("ReceiveCoilName", "first"), desc0=("SeriesDescription", "first"))
    st["freq_r"] = pd.to_numeric(st["freq"], errors="coerce").round(2)
    st["scanner_key"] = st["manu"].fillna("?") + "|" + st["model"].fillna("?") + "|" + st["field"].fillna("?") + "|" + st["sw"].fillna("?")
    st["scanner_key_freq"] = st["scanner_key"] + "|" + st["freq_r"].astype(str)
    F.p("studies with >1 Manufacturer inside", int((st["n_manu"] > 1).sum()), "studies with >1 PatientID inside", int((st["n_pid"] > 1).sum()))
    mm = ok[ok["study_dir"].isin(st.index[st["n_manu"] > 1])].groupby(["study_dir", "series_dir"]).agg(
        manu=("Manufacturer", "first"), model=("ManufacturerModelName", "first"), desc=("SeriesDescription", "first"), n=("path", "size"), dims=("dims", "first"))
    F.table("multi_manufacturer_studies", mm, None)
    lr = st["lat"].str.split("|").map(lambda v: {x[:1] for x in v if x and x[0] in "LR"})
    F.p("Laterality: studies with both L and R inside:", int(lr.map(lambda s: s == {"L", "R"}).sum()),
        "with only L:", int(lr.map(lambda s: s == {"L"}).sum()), "only R:", int(lr.map(lambda s: s == {"R"}).sum()), "neither:", int(lr.map(len).eq(0).sum()))
    F.p("distinct (manu|model|field|sw)", st["scanner_key"].nunique(), "distinct (+ImagingFrequency 2dp)", st["scanner_key_freq"].nunique(),
        "distinct ImagingFrequency raw", st["freq"].nunique(), "distinct ImagingFrequency 2dp", st["freq_r"].nunique())
    F.table("studies_by_scanner_key", st["scanner_key"].value_counts().to_frame("studies").assign(cum_frac=lambda d: (d["studies"].cumsum() / len(st)).round(3)), 60)
    F.table("studies_by_scanner_key_freq", st["scanner_key_freq"].value_counts().to_frame("studies"), 60)
    F.table("studies_by_model_field", st.groupby(["manu", "model", "field"], dropna=False).size().sort_values(ascending=False).to_frame("studies"), 60)
    # patient
    F.p("PatientID present in files", int(ok["PatientID"].notna().sum()), "distinct PatientID (train studies)", st["pid"].nunique(), "studies", len(st))
    spp = st.groupby("pid").size()
    F.p("studies per patient:", spp.value_counts().sort_index().to_dict())
    F.p("PatientID format examples:", st["pid"].dropna().head(3).tolist(), "lengths:", st["pid"].dropna().str.len().value_counts().to_dict())
    multi = st[st["pid"].isin(spp[spp > 1].index)].sort_values("pid")
    F.table("multi_study_patients", multi[["pid", "manu", "model", "lat", "sex", "age", "n_series", "n_files"]], 40)
    # are multi-study patients labelled?
    trm = tr.set_index("StudyInstanceUID")
    multi_l = trm.reindex(multi.index)["labelled"]
    F.p("multi-study-patient studies: labelled", int(multi_l.fillna(False).sum()), "of", len(multi))
    F.p("Laterality per study:", st["lat"].value_counts().head(10).to_dict())
    F.p("PatientSex:", st["sex"].fillna("<absent>").value_counts().to_dict())
    F.p("PatientAge present studies:", int(st["age"].notna().sum()))
    F.p("PatientAge (top):", st["age"].fillna("<absent>").value_counts().head(8).to_dict())
    st = st.join(trm[["labelled", "lang"]])
    # candidate GroupKFold keys: how many groups, how lumpy, and how the 58 labelled studies spread over them
    F.p("\n## candidate CV group keys")
    st["site_key"] = st["scanner_key"] + "|" + st["lang"].fillna("?")
    st["site_key_freq"] = st["scanner_key_freq"] + "|" + st["lang"].fillna("?")
    rows = []
    for k in ["manu", "scanner_key", "site_key", "scanner_key_freq", "site_key_freq", "lang"]:
        g = st.groupby(k, dropna=False).size().sort_values(ascending=False)
        gl = st[st["labelled"].fillna(False)].groupby(k, dropna=False).size()
        rows.append({"key": k, "groups": len(g), "largest": int(g.iloc[0]), "largest_frac": round(g.iloc[0] / len(st), 3),
                     "top5_frac": round(g.iloc[:5].sum() / len(st), 3), "groups_with_lt10": int((g < 10).sum()),
                     "groups_with_labelled": len(gl), "max_labelled_in_group": int(gl.max())})
    F.table("cv_group_key_candidates", pd.DataFrame(rows).set_index("key"), None)
    F.table("site_key_counts", st["site_key"].value_counts().to_frame("studies").assign(cum_frac=lambda d: (d["studies"].cumsum() / len(st)).round(3)), 40)
    # bilateral studies: both L and R Laterality inside
    bil = ok[ok["study_dir"].isin(lr.index[lr.map(lambda s: s == {"L", "R"})])]
    bt = bil.groupby(["study_dir", "series_dir"]).agg(lat=("Laterality", "first"), desc=("SeriesDescription", "first"), n=("path", "size"), frame=("FrameOfReferenceUID", "nunique"))
    F.p("bilateral (L+R) studies:", bt.index.get_level_values(0).nunique(), "series", len(bt), "; per-study Laterality pattern:",
        bt.groupby(level=0)["lat"].agg(lambda x: "".join(sorted(x.fillna("_").replace("", "_")))).value_counts().head(8).to_dict())
    F.table("bilateral_studies_series", bt, 30)
    F.table("scanner_vs_language", pd.crosstab(st["scanner_key"].str.slice(0, 60), st["lang"]).assign(total=lambda d: d.sum(axis=1)).sort_values("total", ascending=False).head(30), 30)
    F.table("scanner_vs_labelled", pd.crosstab(st["scanner_key"].str.slice(0, 60), st["labelled"]).sort_values(True, ascending=False).head(30), 30)
    F.table("transfer_syntax_by_study", st["tsyn"].value_counts().to_frame("studies"), None)
    st.to_csv(OUT / "study_summary.csv")
    # windowing / rescale
    F.p("\n## rescale / window")
    for col in ["RescaleSlope", "RescaleIntercept", "RescaleType", "WindowCenter", "WindowWidth", "SmallestImagePixelValue", "LargestImagePixelValue"]:
        if col in ok:
            F.p(f"{col}: present {int(ok[col].notna().sum())} files; distinct {ok[col].nunique()}; top {ok[col].value_counts().head(5).to_dict()}")
    rs = pd.to_numeric(ok["RescaleSlope"], errors="coerce")
    F.p("RescaleSlope != 1 files:", int((rs.notna() & rs.ne(1)).sum()), "by manufacturer:", ok.loc[rs.notna() & rs.ne(1), "Manufacturer"].value_counts().to_dict())
    # dates
    for col in ["StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate", "PatientBirthDate", "StudyTime", "AccessionNumber", "StudyID"]:
        if col in ok:
            F.p(f"{col}: present {int(ok[col].notna().sum())} distinct {ok[col].nunique()} sample {ok[col].dropna().head(3).tolist()}")
        else:
            F.p(f"{col}: absent")
    # test set
    F.p("\n# 5. Test set")
    te_h = h[h["split"].eq("test_series") & h["error"].eq("")]
    tt = te_h.groupby(["study_dir", "series_dir"]).agg(n=("path", "size"), manu=("Manufacturer", "first"), model=("ManufacturerModelName", "first"),
                                                       dims=("Rows", lambda x: x.iloc[0] + "x" + te_h.loc[x.index[0], "Columns"]),
                                                       desc=("SeriesDescription", "first"), pid=("PatientID", "first"), lat=("Laterality", "first"),
                                                       tsyn=("TransferSyntaxUID", "first"), thk=("SliceThickness", "first"), ps=("PixelSpacing", "first"))
    te_csv = pd.read_csv(DATA / "test_series.csv", dtype=str).set_index(["StudyInstanceUID", "SeriesInstanceUID"])
    te_csv.index.names = ["study_dir", "series_dir"]
    tt = tt.join(te_csv, how="outer")
    F.table("test_series_detail", tt, None)
    F.p("test PatientIDs also in train:", int(tt["pid"].isin(set(st["pid"])).sum()), "test scanner keys:", te_h.groupby("study_dir").agg(
        k=("Manufacturer", "first"), m=("ManufacturerModelName", "first"), f=("MagneticFieldStrength", "first")).to_dict("records"))
    tcsv = pd.read_csv(DATA / "test.csv", dtype=str)
    F.p("test.csv rows", len(tcsv), "sample_submission rows", len(pd.read_csv(DATA / "sample_submission.csv")))
    return ok, ser, st


def analyze_decode(F: Facts):
    p = OUT / "decode_sample.csv"
    if not p.exists():
        F.p("\n# 4. decode sample: NOT RUN")
        return
    d = pd.read_csv(p)
    d["error"] = d["error"].fillna("")
    F.p("\n# 4. Pixel decode sample")
    F.p("files", len(d), "failures", int(d["error"].ne("").sum()))
    if d["error"].ne("").any():
        F.table("decode_failures", d[d["error"].ne("")][["path", "size", "error"]], 20)
    F.table("decode_timing_by_syntax", d.groupby("TransferSyntaxUID")[["t_read_ms", "t_decode_ms"]].describe(percentiles=[.5, .9]).round(2).T, None)
    F.p("dtype:", d["dtype"].value_counts().to_dict(), "shape distinct:", d["shape"].nunique())
    F.table("decode_intensity_stats", d[["min", "p0.1", "p1", "p50", "p99", "p99.9", "max", "mean", "frac_zero", "n_unique"]].describe(percentiles=[.05, .5, .95]).round(1), None)
    F.p("files with max >= 2^BitsStored:", int(d["max_gt_bitsstored"].fillna(False).sum()), "max>4095:", int((d["max"] > 4095).sum()), "max>65535/2:", int((d["max"] > 32767).sum()))
    F.p("min<0 files:", int((d["min"] < 0).sum()), "by manufacturer:", d.loc[d["min"] < 0, "Manufacturer"].value_counts().to_dict())
    F.table("decode_intensity_by_manufacturer", d[d["error"].eq("")].groupby("Manufacturer").agg(
        n=("path", "size"), dtype=("dtype", lambda x: "/".join(sorted(set(x)))), min_min=("min", "min"), med_p50=("p50", "median"),
        med_p99=("p99", "median"), med_max=("max", "median"), max_max=("max", "max"), med_nuniq=("n_unique", "median"),
        rescale_ne1=("RescaleSlope", lambda x: int(pd.to_numeric(x, errors="coerce").fillna(1).ne(1).sum()))).round(0), None)
    tp = OUT / "decode_throughput.csv"
    if tp.exists():
        F.table("decode_throughput", pd.read_csv(tp), None)


def cmd_analyze(args):
    OUT.mkdir(parents=True, exist_ok=True)
    F = Facts()
    h = pd.read_parquet(OUT / "headers.parquet")
    for c in KEEP + ["TransferSyntaxUID", "MediaStorageSOPClassUID"]:
        if c not in h:
            h[c] = np.nan
    h = h[h["file"].notna()]  # entries without 4 path parts are stray files; count them below
    stray = pd.read_parquet(OUT / "headers.parquet", columns=["path"])
    stray = stray[stray["path"].str.count("/") != 3]
    F.p("stray entries (not split/study/series/file):", len(stray), stray["path"].head(5).tolist())
    dirs_train = set(h.loc[h["split"].eq("train_series"), "study_dir"])
    tr = analyze_train_csv(F, dirs_train)
    ts, te = analyze_series(F, h)
    analyze_headers(F, h, ts, tr)
    analyze_decode(F)
    (OUT / "report_facts.txt").write_text("\n".join(F.lines), encoding="utf-8")
    print("\nwrote", OUT / "report_facts.txt")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan"); s.add_argument("--procs", type=int, default=24); s.set_defaults(fn=cmd_scan)
    s = sub.add_parser("decode"); s.add_argument("--n", type=int, default=300); s.add_argument("--seed", type=int, default=0); s.set_defaults(fn=cmd_decode)
    s = sub.add_parser("throughput"); s.add_argument("--n", type=int, default=12000); s.add_argument("--seed", type=int, default=1)
    s.add_argument("--procs", type=int, nargs="+", default=[24, 8, 1]); s.set_defaults(fn=cmd_throughput)
    s = sub.add_parser("analyze"); s.set_defaults(fn=cmd_analyze)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
