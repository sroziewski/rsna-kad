#!/usr/bin/env python3
"""Inference-time 2D blur on main_v5_C, gold-58 (user-requested 2026-10-19, after Andrey's gold report: Gaussian
sigma 0.5-0.8 before normalisation helps hx48 / MT-hires / S6 / nartaa's CoAtNets).

Variants fixed before any score: none; Gaussian sigma 0.5 / 0.65 / 0.8; Student-t kernel (nu 3 and 5) at scale 0.65.
The blur is applied to each cached uint8 slice before warp_slices (img_px 384 = cache px, so before == after resize)
and before the model's own /255 + ImageNet normalisation; result rounded back to uint8 as the loader expects.
Members = export/main_v5_C (seeds 2026-2526, tta false, 1/6 each, probability mean). Information only: 58 studies.

  python qa/blur/blur_gold.py [--variants ...] [--workers 8]     # GPU; writes qa/blur/gold_<variant>.csv, report.txt
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "qa/full")]
OUT = ROOT / "qa/blur"
SEEDS = (2026, 2126, 2226, 2326, 2426, 2526)
RUN = "main_armc_convnextv2_t_384_p_full_s{}"
VARIANTS = {"none": None, "g0.5": ("g", 0.5, None), "g0.65": ("g", 0.65, None), "g0.8": ("g", 0.8, None),
            "t3_0.65": ("t", 0.65, 3.0), "t5_0.65": ("t", 0.65, 5.0)}


def t_kernel(scale: float, nu: float) -> np.ndarray:
    """2D Student-t kernel (1 + r^2 / (nu scale^2))^(-(nu + 1) / 2), radius covering the heavy tail, sum 1."""
    r = max(3, math.ceil(6 * scale))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    k = (1.0 + (xx ** 2 + yy ** 2) / (nu * scale ** 2)) ** (-(nu + 1) / 2)
    return (k / k.sum()).astype(np.float32)


def blur_fn(spec):
    kind, s, nu = spec
    kern = t_kernel(s, nu) if kind == "t" else None

    def f(slices: np.ndarray) -> np.ndarray:
        out = np.empty_like(slices)
        for i in range(slices.shape[0]):
            x = slices[i].astype(np.float32)
            y = (cv2.GaussianBlur(x, (0, 0), sigmaX=s, borderType=cv2.BORDER_REFLECT_101) if kind == "g"
                 else cv2.filter2D(x, -1, kern, borderType=cv2.BORDER_REFLECT_101))
            out[i] = np.clip(np.rint(y), 0, 255).astype(np.uint8)
        return out
    return f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)

    import torch
    import rsna_kad.data as D
    from rsna_kad.config import load_config
    from rsna_kad.constants import DATA_DIR, ID_COL
    from rsna_kad.model import load_bundle
    from rsna_kad.train import PRED_COLS, amp_dtype, cache_root, predict_studies_cached
    import compare_pw as c

    cfg = load_config(ROOT / "configs/main_armc_pseudo_full.yaml")
    gold = c.gold_frame(Path(DATA_DIR) / "train.csv")
    ids = sorted(gold[ID_COL].astype(str))
    assert len(ids) == 58
    orig = D.read_triplets
    for v in a.variants:
        spec = VARIANTS[v]
        f = None if spec is None else blur_fn(spec)

        def patched(vol, idx, px, params=None, _f=f):
            uniq, inverse = np.unique(idx, return_inverse=True)
            sl = np.asarray(vol[uniq])
            warped = D.warp_slices(sl if _f is None else _f(sl), px, params)
            return warped[inverse.reshape(idx.shape)]
        D.read_triplets = patched          # StudyDataset.__getitem__ resolves it from the module; workers fork after this
        acc = None
        for s in SEEDS:
            model, _ = load_bundle(ROOT / "runs" / RUN.format(s) / "full", "cuda", views=True)
            ds = D.StudyDataset(cache_root(cfg), ids, None, cfg, train=False)
            p = predict_studies_cached(model, ds, tta=False, batch_size=cfg.eval_batch_studies,
                                       num_workers=a.workers, device="cuda", dtype=amp_dtype(cfg))
            p = p.set_index(ID_COL).loc[ids, PRED_COLS]
            acc = p / len(SEEDS) if acc is None else acc + p / len(SEEDS)
            del model
            torch.cuda.empty_cache()
        acc.reset_index().to_csv(OUT / f"gold_{v}.csv", index=False)
        print(f"{v}: done", flush=True)
    D.read_triplets = orig
    return 0


if __name__ == "__main__":
    sys.exit(main())
