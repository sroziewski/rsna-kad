#!/usr/bin/env python3
"""OOF of the 5-fold arm-C pseudo-label models under four pre-declared evaluation views.

  python qa/tta2/tta2_oof.py [--views base crop rotp5 rotm5]   # GPU; writes qa/tta2/<view>/oof.parquet
  python qa/tta2/tta2_oof.py --check                           # base vs the run's own training OOF
  python qa/tta2/tta2_oof.py --smoke 16                         # 16 studies of fold 0 -> .tmp/tta2_smoke

Views (coordinator 2026-09-30, fixed before any score): base = plain view; crop = every view's
field x 115/130 (130 -> 115 mm, 90 -> 79.6 mm); rotp5 / rotm5 = +5 / -5 deg for every slice and view.
No flips. Same inference path as predict_oof.py --recompute (load_bundle(views=True), K_infer 24,
bf16), but through rsna_kad.tta_views.ViewTTADataset; nothing under runs/ is opened for writing.
Wall time per fold and view -> qa/tta2/timing.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

RUN = "main_armc_convnextv2_t_384_p"
CONFIG = "configs/main_armc_pseudo.yaml"
K_INFER = 24
VIEWS = {"base": (1.0, 0.0), "crop": (115.0 / 130.0, 0.0), "rotp5": (1.0, 5.0), "rotm5": (1.0, -5.0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", nargs="*", default=list(VIEWS), choices=list(VIEWS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", type=int, default=None, metavar="N")
    args = ap.parse_args(argv)

    import numpy as np
    import pandas as pd
    from rsna_kad.constants import ID_COL
    from rsna_kad.train import PRED_COLS

    out_root = ROOT / (".tmp/tta2_smoke" if args.smoke else "qa/tta2")
    if args.check:
        a = pd.read_parquet(ROOT / "runs" / RUN / "oof.parquet").set_index(ID_COL).sort_index()
        b = pd.read_parquet(out_root / "base" / "oof.parquet").set_index(ID_COL).loc[a.index]
        d = np.abs(a[PRED_COLS].to_numpy() - b[PRED_COLS].to_numpy())
        print(f"base vs {RUN} training OOF: max |d| {d.max():.2e}, mean {d.mean():.2e} over {len(a)} studies")
        return 0

    import torch
    from rsna_kad.cache import get_index
    from rsna_kad.config import load_config
    from rsna_kad.cv import load_folds
    from rsna_kad.model import load_bundle
    from rsna_kad.train import amp_dtype, cache_root, predict_studies_cached
    from rsna_kad.tta_views import ViewTTADataset

    cfg = load_config(ROOT / CONFIG)
    assert cfg.run_id == RUN, cfg.run_id
    cfg.model.k_infer = K_INFER
    folds = load_folds(ROOT / cfg.folds_csv)
    cached = set(get_index(cache_root(cfg)).studies("train"))
    timing_path = out_root / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    for view in args.views:
        crop_factor, rot = VIEWS[view]
        parts, secs = [], {}
        for k in (range(1) if args.smoke else range(5)):
            fold_dir = ROOT / "runs" / RUN / f"fold{k}"
            assert not (fold_dir / "last.pt").exists(), f"{fold_dir}: unfinished fold"
            model, _ = load_bundle(fold_dir, "cuda", views=True)
            ids = sorted(s for s in folds.loc[folds["fold"] == k, ID_COL].astype(str) if s in cached)
            if args.smoke:
                ids = ids[: args.smoke]
            ds = ViewTTADataset(cache_root(cfg), ids, None, cfg, train=False, crop_factor=crop_factor, rot_deg=rot)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = predict_studies_cached(model, ds, tta=False, batch_size=cfg.eval_batch_studies,
                                          num_workers=args.workers, device="cuda", dtype=amp_dtype(cfg))
            torch.cuda.synchronize()
            secs[f"fold{k}"] = round(time.perf_counter() - t0, 1)
            pred["fold"] = k
            parts.append(pred[[ID_COL, "fold"] + PRED_COLS])
            print(f"{view} fold {k}: {len(pred)} studies in {secs[f'fold{k}']} s", flush=True)
            del model
            torch.cuda.empty_cache()
        oof = pd.concat(parts, ignore_index=True)
        assert (args.smoke or len(oof) == 4407) and oof[ID_COL].is_unique, len(oof)
        (out_root / view).mkdir(parents=True, exist_ok=True)
        oof.to_parquet(out_root / view / "oof.parquet", index=False)
        timing[view] = {"crop_factor": crop_factor, "rot_deg": rot, "per_fold_s": secs,
                        "total_s": round(sum(secs.values()), 1), "n_studies": int(len(oof)),
                        "gpu": torch.cuda.get_device_name(0)}
        timing_path.write_text(json.dumps(timing, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
