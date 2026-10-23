"""Montages for 12 hand-picked series + an independent re-derivation of the cached volume
from the raw DICOMs (my own crop/resize/window; only pydicom/numpy/cv2). Read-only on the cache."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom

cv2.setNumThreads(2)
CACHE = Path("/raid/sroziewski/cache/rsna_kad/v1")
DATA = Path("/raid/sroziewski/data/rsna_kad")
OUTDIR = Path("/raid/sroziewski/sandbox/rsna_kad/qa/stage1/png")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(CACHE / "index.parquet")
tr = df[df["split"] == "train"]
rng = np.random.default_rng(1)


def pick(mask, name):
    sub = tr[mask]
    r = sub.sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]
    return name, r


CASES = [
    pick((tr.rows == 512) & (tr.cols == 512) & (tr.slot == "sag_fs") & ~tr.is_3d
         & tr.manufacturer.str.contains("SIEMENS", case=False), "01_sag_fs_512_siemens_2d"),
    pick((tr.slot == "cor_fs") & ~tr.is_3d & (tr.rows == 640), "02_cor_fs_2d"),
    pick((tr.slot == "ax_fs") & ~tr.is_3d & (tr.rows == 384), "03_ax_fs_2d"),
    pick((tr.rows == 640) & (tr.cols == 1280) & tr.is_3d, "04_dess_3d_640x1280"),
    pick((tr.rows == 1024) & (tr.cols == 1024), "05_1024x1024"),
    pick((tr.rows == 256) & (tr.cols == 256) & ~tr.is_3d, "06_256x256"),
    ("07_philips_rescale", None),
    pick(tr.manufacturer.str.contains("TOSHIBA", case=False) & (tr.p_lo < 5) & ~tr.is_3d, "08_toshiba_signed"),
    ("09_truncated", tr[tr.SeriesInstanceUID == "1.2.826.0.1.3680043.8.498.39396636671446532796538574315802928348"].iloc[0]),
    pick((tr.fov_mm < 125) & (tr.fov_mm > 100) & ~tr.is_3d, "10_fov_lt130_padded"),
    pick((tr.fov_mm > 190) & (tr.fov_mm < 230) & ~tr.is_3d & (tr.slot == "cor_fs"), "11_fov_gt190"),
    pick(tr.manufacturer.str.contains("GE MEDICAL", case=False) & (tr.slot == "sag_t1") & ~tr.is_3d, "12_ge"),
]


def _first_slope(r):
    d = DATA / f"{r.split}_series" / r.StudyInstanceUID / r.SeriesInstanceUID
    f = sorted(d.glob("*.dcm"))[0]
    return float(getattr(pydicom.dcmread(f, stop_before_pixels=True), "RescaleSlope", 1.0))


for i, (name, r) in enumerate(CASES):
    if r is None:
        cands = tr[tr.manufacturer.str.contains("Philips", case=False) & (tr.slot == "sag_fs") & ~tr.is_3d].sample(
            40, random_state=3)
        r = next(c for c in cands.itertuples(index=False) if _first_slope(c) != 1.0)
        CASES[i] = (name, r)


def read_series(r):
    d = DATA / f"{r.split}_series" / r.StudyInstanceUID / r.SeriesInstanceUID
    files = sorted(d.glob("*.dcm"))
    hdrs = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
    order = np.argsort([int(h.InstanceNumber) for h in hdrs], kind="stable")
    files = [files[i] for i in order]
    hdrs = [hdrs[i] for i in order]
    slices, trunc = [], []
    for i, f in enumerate(files):
        ds = pydicom.dcmread(f)
        try:
            a = ds.pixel_array.astype(np.float32)
        except Exception:
            slices.append(None)
            trunc.append(i)
            continue
        a = a * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
        slices.append(a)
    for i in trunc:  # neighbour mean
        prev = max([j for j in range(i) if slices[j] is not None], default=None)
        nxt = min([j for j in range(i + 1, len(slices)) if slices[j] is not None], default=None)
        slices[i] = (slices[prev] + slices[nxt]) / 2 if prev is not None and nxt is not None else slices[prev if prev is not None else nxt]
    vol = np.stack(slices)
    spacing = float(hdrs[0].PixelSpacing[0])
    iop = np.array(hdrs[0].ImageOrientationPatient, float)
    normal = np.cross(iop[:3], iop[3:])
    pos = np.array([np.dot(np.array(h.ImagePositionPatient, float), normal) for h in hdrs])
    dpos = np.diff(pos)
    monotone = bool(np.all(dpos > 0) or np.all(dpos < 0))
    z_step = float(np.median(np.abs(dpos))) if len(dpos) else 0.0
    acq = str(getattr(hdrs[0], "MRAcquisitionType", "")).upper()
    is_3d = acq == "3D" or (acq == "" and len(hdrs) > 60)
    return vol, spacing, z_step, is_3d, monotone, trunc, hdrs


def my_preprocess(vol, spacing, z_step, is_3d):
    n, h, w = vol.shape
    s = max(1, int(round(130.0 / spacing)))
    out = np.zeros((n, s, s), np.float32)
    y0, x0 = (h - s) // 2, (w - s) // 2
    sy, sx = max(y0, 0), max(x0, 0)
    dy, dx = max(-y0, 0), max(-x0, 0)
    ny, nx = min(h - sy, s - dy), min(w - sx, s - dx)
    out[:, dy:dy + ny, dx:dx + nx] = vol[:, sy:sy + ny, sx:sx + nx]
    vol = out
    if is_3d and 0 < z_step < 2.25:
        f = int(round(3.0 / z_step))
        if f > 1:
            vol = np.stack([vol[k * f:(k + 1) * f].mean(0) for k in range(int(np.ceil(vol.shape[0] / f)))])
    if vol.shape[0] > 40:
        idx = np.unique(np.linspace(0, vol.shape[0] - 1, 40).round().astype(int))
        vol = vol[idx]
    interp = cv2.INTER_AREA if 384 < s else cv2.INTER_LINEAR
    vol = np.stack([cv2.resize(v, (384, 384), interpolation=interp) for v in vol]) if s != 384 else vol
    lo, hi = np.percentile(vol, [1, 99])
    u8 = np.clip((vol - lo) * 255.0 / max(hi - lo, 1.0), 0, 255)
    return np.rint(u8).astype(np.uint8), float(lo), float(hi)


def to_u8(img):
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)


def letterbox(img_u8, size=384):
    h, w = img_u8.shape
    sc = size / max(h, w)
    r = cv2.resize(img_u8, (max(1, int(round(w * sc))), max(1, int(round(h * sc)))), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), np.uint8)
    y, x = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out, sc, y, x


def label(img, text):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    cv2.rectangle(img, (0, 0), (383, 18), (0, 0, 0), -1)
    cv2.putText(img, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    return img


results = []
for name, r in CASES:
    cached = np.load(CACHE / r.split / r.StudyInstanceUID / f"{r.SeriesInstanceUID}.npy", mmap_mode="r")
    vol, spacing, z_step, is_3d, monotone, trunc, hdrs = read_series(r)
    mine, lo, hi = my_preprocess(vol, spacing, z_step, is_3d)
    n = cached.shape[0]
    same_shape = mine.shape == cached.shape
    diff = np.abs(mine.astype(np.int16) - np.asarray(cached).astype(np.int16)) if same_shape else None
    idx = np.linspace(0, n - 1, 8).round().astype(int)
    row1 = np.hstack([label(np.asarray(cached[i]).copy(), f"cached z={i}/{n}") for i in idx])
    mid_src = vol.shape[0] // 2
    orig, sc, oy, ox = letterbox(to_u8(vol[mid_src]))
    orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    s = int(round(130 / spacing))
    h, w = vol.shape[1:]
    y0, x0 = (h - s) // 2, (w - s) // 2
    cv2.rectangle(orig, (int(ox + x0 * sc), int(oy + y0 * sc)), (int(ox + (x0 + s) * sc), int(oy + (y0 + s) * sc)), (0, 0, 255), 1)
    orig = label(orig, f"orig mid {h}x{w} sp={spacing:.3f} fov={h * spacing:.0f}mm box=130mm")
    mid_c = n // 2
    row2 = [orig, label(np.asarray(cached[mid_c]).copy(), f"cached mid z={mid_c} p={r.p_lo:.0f}/{r.p_hi:.0f}"),
            label(mine[min(mid_c, mine.shape[0] - 1)].copy(), f"my recompute p={lo:.0f}/{hi:.0f}"),
            label((np.clip(diff[mid_c] * 8, 0, 255).astype(np.uint8) if same_shape else np.zeros((384, 384), np.uint8)),
                  f"|diff|x8 max={int(diff.max()) if same_shape else -1} mean={diff.mean():.2f}" if same_shape else "shape mismatch")]
    if trunc:
        t = trunc[0]
        nbs = [j for j in (t - 1, t + 1) if 0 <= j < n]
        for j in sorted(nbs + [t]):
            row2.append(label(np.asarray(cached[j]).copy(), f"cached z={j}{' TRUNC-FILLED' if j == t else ''}"))
        nb = np.mean([np.asarray(cached[j]).astype(np.float32) for j in nbs], axis=0)
        row2.append(label(np.clip(np.abs(nb - cached[t]) * 8, 0, 255).astype(np.uint8),
                          f"|nbmean-filled|x8 max={int(np.abs(nb - cached[t]).max())}"))
    else:
        # z-continuity: adjacent-slice diff vs skip-one diff
        pass
    while len(row2) < 8:
        row2.append(np.zeros((384, 384, 3), np.uint8))
    row2 = np.hstack(row2[:8])
    hist = np.zeros((120, row1.shape[1], 3), np.uint8)
    hcnt = np.bincount(np.asarray(cached).ravel(), minlength=256).astype(float)
    hcnt = hcnt / hcnt.max() * 110
    for b in range(256):
        cv2.line(hist, (b * 4, 119), (b * 4, 119 - int(hcnt[b])), (200, 200, 200), 3)
    cv2.putText(hist, f"{name} | {r.manufacturer} | slot={r.slot} 3d={bool(r.is_3d)} n_src={r.n_src} n_cached={n} "
                      f"z={r.z_step_mm:.2f}->{r.z_step_cached_mm:.2f}mm | u8 hist: 0={hcnt[0] / 110 * 100:.0f}% of max, "
                      f"255 frac={np.mean(np.asarray(cached) == 255):.4f} 0 frac={np.mean(np.asarray(cached) == 0):.4f} mean={r.mean_u8:.1f}",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUTDIR / f"{name}.png"), np.vstack([hist, row1, row2]))
    # z-continuity: mean abs diff between adjacent cached slices vs slices 2 apart
    c = np.asarray(cached).astype(np.float32)
    adj = float(np.abs(np.diff(c, axis=0)).mean()) if n > 1 else 0.0
    skip = float(np.abs(c[2:] - c[:-2]).mean()) if n > 2 else 0.0
    results.append({
        "name": name, "series": r.SeriesInstanceUID, "study": r.StudyInstanceUID, "manufacturer": r.manufacturer,
        "slot": r.slot, "rows": int(r.rows), "cols": int(r.cols), "spacing": spacing, "fov_mm": round(float(r.fov_mm), 1),
        "is_3d": bool(is_3d), "n_src": int(r.n_src), "n_cached": n, "instnum_order_monotone_ipp": monotone,
        "truncated_idx": trunc, "same_shape": same_shape,
        "diff_max": int(diff.max()) if same_shape else None, "diff_mean": round(float(diff.mean()), 3) if same_shape else None,
        "diff_frac_gt2": round(float((diff > 2).mean()), 5) if same_shape else None,
        "p_lo_hi_index": [round(float(r.p_lo), 2), round(float(r.p_hi), 2)], "p_lo_hi_mine": [round(lo, 2), round(hi, 2)],
        "u8_frac0": round(float((c == 0).mean()), 4), "u8_frac255": round(float((c == 255).mean()), 4),
        "adj_slice_mad": round(adj, 2), "skip_slice_mad": round(skip, 2),
        "signed_min_raw": float(vol.min()), "raw_max": float(vol.max()),
        "rescale_slope_first": float(getattr(hdrs[0], "RescaleSlope", 1.0)),
        "bits": int(hdrs[0].BitsStored), "pixrep": int(hdrs[0].PixelRepresentation),
    })
    print(json.dumps(results[-1]))

# ---- random parity sweep: 60 series re-derived independently
sweep = []
samp = pd.concat([tr[~tr.is_3d].sample(45, random_state=7), tr[tr.is_3d].sample(15, random_state=7)])
for r in samp.itertuples(index=False):
    cached = np.asarray(np.load(CACHE / r.split / r.StudyInstanceUID / f"{r.SeriesInstanceUID}.npy", mmap_mode="r"))
    vol, spacing, z_step, is_3d, monotone, trunc, hdrs = read_series(r)
    mine, lo, hi = my_preprocess(vol, spacing, z_step, is_3d)
    ok = mine.shape == cached.shape
    d = np.abs(mine.astype(np.int16) - cached.astype(np.int16)) if ok else None
    sweep.append({"series": r.SeriesInstanceUID, "is_3d": bool(r.is_3d), "shape_ok": ok, "monotone": monotone,
                  "diff_max": int(d.max()) if ok else None, "diff_mean": round(float(d.mean()), 3) if ok else None,
                  "frac_gt1": round(float((d > 1).mean()), 5) if ok else None, "n": int(r.n_cached),
                  "spacing": spacing, "z": round(z_step, 2), "z_cached": round(float(r.z_step_cached_mm), 2)})
sw = pd.DataFrame(sweep)
print("SWEEP shape_ok:", int(sw.shape_ok.sum()), "/", len(sw), " monotone:", int(sw.monotone.sum()),
      " diff_max max:", int(sw.diff_max.max()), " diff_mean median:", float(sw.diff_mean.median()),
      " frac>1 max:", float(sw.frac_gt1.max()))
print(sw.sort_values("diff_max", ascending=False).head(8).to_string())
json.dump({"cases": results, "sweep": sweep}, open(OUTDIR.parent / "qa_montage_out.json", "w"), indent=1, default=str)
