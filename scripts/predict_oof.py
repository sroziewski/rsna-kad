#!/usr/bin/env python3
"""Assemble runs/<run_id>/oof.parquet + metrics.json from the finished folds (§3.2 OOF contract).

  python scripts/predict_oof.py --config configs/mvp_effv2s_256.yaml
  python scripts/predict_oof.py --config configs/mvp_effv2s_256.yaml --recompute --tta --gpu 1

By default the per-epoch oof.parquet written by training (EMA weights of the best epoch) is
used; --recompute re-runs inference from each fold's bundle, optionally with z-offset TTA.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(5)))
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    import torch  # noqa: E402
    from rsna_kad.cache import get_index  # noqa: E402
    from rsna_kad.config import load_config  # noqa: E402
    from rsna_kad.constants import ID_COL  # noqa: E402
    from rsna_kad.cv import load_folds  # noqa: E402
    from rsna_kad.data import StudyDataset  # noqa: E402
    from rsna_kad.model import load_bundle  # noqa: E402
    from rsna_kad.train import (OOF_NAME, PRED_COLS, amp_dtype, assemble_oof, cache_root,  # noqa: E402
                                predict_studies_cached, run_dir_for)

    from rsna_kad.runguard import RunGuardError, assert_not_live  # noqa: E402

    cfg = load_config(args.config)
    try:
        assert_not_live([Path(args.runs_dir) / cfg.run_id])
    except RunGuardError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.recompute:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        folds = load_folds(cfg.folds_csv)
        cached = set(get_index(cache_root(cfg)).studies("train"))
        for k in args.folds:
            run_dir = run_dir_for(cfg, k, args.runs_dir)
            if not (run_dir / "bundle.json").exists():
                print(f"fold {k}: no bundle in {run_dir}, skipped")
                continue
            # views=True: StudyDataset applies model.view_crop_mm per slot.
            model, _ = load_bundle(run_dir, device, views=True)
            ids = sorted(s for s in folds.loc[folds["fold"] == k, ID_COL].astype(str) if s in cached)
            ds = StudyDataset(cache_root(cfg), ids, None, cfg, train=False)
            pred = predict_studies_cached(model, ds, tta=args.tta, batch_size=cfg.eval_batch_studies,
                                          num_workers=args.workers, device=device, dtype=amp_dtype(cfg))
            pred["fold"] = k
            pred[[ID_COL, "fold"] + PRED_COLS].to_parquet(run_dir / OOF_NAME, index=False)
            print(f"fold {k}: {len(pred)} studies predicted (tta={args.tta})")
    m = assemble_oof(args.runs_dir, cfg, args.folds)
    print(json.dumps({k: v for k, v in m.items() if not k.startswith("per_label")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
