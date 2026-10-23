#!/usr/bin/env python3
"""How wide a centre crop must be to keep the collateral ligaments in the coronal slot.

Arm C's sweep method (architecture §4.1) decides a view on train loss, throughput and memory
because smoke AUC at 200 studies has a +-0.04 noise floor. That method cannot separate a 90 mm
from a 100 mm coronal crop: the two cost the same. This measures the thing that actually
differs - whether the crop still contains the structure the view is for.

MCL and the lateral compartment sit at the medial/lateral edge of the knee, a few mm inside the
skin surface, so the question is the tissue extent along the image x axis (medial-lateral in a
coronal series) measured from the crop centre, since `data.view_params` crops about the centre.

  python qa/stage8b/cor_extent.py --n 500 --out qa/stage8b/cor_extent.json

Read-only: loads the cache memmaps, writes only the report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from rsna_kad.cache import read_index, read_manifest, series_path  # noqa: E402
from rsna_kad.constants import ID_COL, SERIES_COL  # noqa: E402


def tissue_halfwidth_mm(vol: np.ndarray, mm_per_px: float, band_mm: float = 30.0,
                        frac: float = 0.25) -> tuple[float, float]:
    """(centred width, tissue width) in mm of the limb at the joint line, from one series.

    Only the rows within `band_mm` of the image centre are used: the cache is centred on the
    knee, and the limb is at its narrowest there, whereas thigh and calf fill the whole 130 mm
    field and would hide the joint. A column counts as tissue when more than `frac` of that
    band's rows are above a quarter of the series' 99.5th percentile - a per-series threshold,
    so it separates air from tissue on fat-suppressed and T1 series alike. The centred width is
    2x the larger distance from the image centre, because `data.view_params` crops centrally:
    that is the crop this series would need."""
    mid_z = vol.shape[0] // 2
    zlo, zhi = max(0, mid_z - 3), min(vol.shape[0], mid_z + 4)
    sl = np.asarray(vol[zlo:zhi], dtype=np.float32)
    half_rows = int(round(band_mm / 2.0 / mm_per_px))
    mid_r = sl.shape[1] // 2
    sl = sl[:, max(0, mid_r - half_rows):mid_r + half_rows, :]
    thr = 0.25 * float(np.percentile(sl, 99.5))
    cols = (sl > thr).mean(axis=(0, 1))          # fraction of joint-line rows that are tissue
    on = np.flatnonzero(cols > frac)
    if on.size == 0:
        return float("nan"), float("nan")
    centre = (vol.shape[-1] - 1) / 2.0
    half = max(abs(on[0] - centre), abs(on[-1] - centre)) + 0.5
    return float(2 * half * mm_per_px), float((on[-1] - on[0] + 1) * mm_per_px)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="/raid/sroziewski/cache/rsna_kad/v1")
    ap.add_argument("--slots", nargs="+", default=["cor_fs", "sag_fs"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--crops", type=float, nargs="+", default=[80.0, 90.0, 100.0, 110.0, 130.0])
    ap.add_argument("--insets", type=float, nargs="+", default=[0.0, 10.0, 20.0],
                    help="mm the target structure sits inside the skin surface, per side")
    ap.add_argument("--out", default="qa/stage8b/cor_extent.json")
    a = ap.parse_args(argv)

    cache = Path(a.cache_dir)
    params = read_manifest(cache)["params"]
    mm_per_px = float(params["crop_mm"]) / int(params["px"])
    idx = read_index(cache)
    idx = idx[(idx["split"] == "train") & idx["ok"] & (idx["rank"] == 0)]
    rng = np.random.default_rng(a.seed)

    report = {"cache": str(cache), "mm_per_px": mm_per_px, "crop_mm": float(params["crop_mm"]),
              "n_requested": a.n, "slots": {}}
    for slot in a.slots:
        rows = idx[idx["slot"] == slot]
        studies = sorted(rows[ID_COL].unique())
        pick = set(rng.choice(studies, size=min(a.n, len(studies)), replace=False).tolist())
        rows = rows[rows[ID_COL].isin(pick)].drop_duplicates(ID_COL)
        need, full, fov = [], [], []
        for r in rows.itertuples(index=False):
            p = series_path(cache, "train", getattr(r, ID_COL), getattr(r, SERIES_COL))
            try:
                vol = np.load(p, mmap_mode="r")
            except OSError:
                continue
            h, w = tissue_halfwidth_mm(vol, mm_per_px)
            if np.isfinite(h):
                need.append(h)
                full.append(w)
                fov.append(float(getattr(r, "fov_mm")))
        need_a, full_a = np.array(need), np.array(full)
        report["slots"][slot] = {
            "n_series": int(len(need_a)),
            "centred_width_mm": {q: float(np.percentile(need_a, q)) for q in (5, 25, 50, 75, 90, 95, 99)},
            "tissue_width_mm": {q: float(np.percentile(full_a, q)) for q in (5, 50, 95)},
            "fov_mm_median": float(np.median(fov)),
            # Fraction of series kept by a centred crop of each size, allowing that the
            # structure of interest sits `inset` mm inside the skin surface on each side:
            # inset 0 = the whole limb, 10 = the collateral ligaments and the joint capsule,
            # 20 = the tibiofemoral compartments. The crop must clear `need - 2*inset`.
            "covered_at_inset": {str(int(i)): {str(c): float((need_a - 2 * i <= c).mean()) for c in a.crops}
                                 for i in a.insets},
        }
    text = json.dumps(report, indent=1)
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
