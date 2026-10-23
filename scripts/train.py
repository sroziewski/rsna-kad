#!/usr/bin/env python3
"""Train one fold of KneeMIL (architecture §4 stage 3).

  python scripts/train.py --config configs/mvp_effv2s_256.yaml --fold 0 --gpu 1
  python scripts/train.py --config configs/smoke_effv2s_256.yaml --fold 0 --gpu 1 --limit 200
  python scripts/train.py --config configs/main_armc_pseudo_full.yaml --fold full --seed 2126

--fold full trains on all non-gold studies and evaluates on the 58 gold only (runs/<run_id>/full/).
--seed S overrides the config seed and, without --run-id, writes to run_id <run_id>_s<S>.

Outputs runs/<run_id>/fold<k>/{model.safetensors,bundle.json,oof.parquet,metrics.json,train.log};
an interrupted run resumes from last.pt. --gpu N sets CUDA_VISIBLE_DEVICES=N (omit it to keep
the caller's). --synthetic-targets replaces the target table by random values: plumbing and
timing only, never a real result.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _fold_arg(v: str) -> int | str:
    return v if v == "full" else int(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=_fold_arg, required=True, help="0-4 or 'full'")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="random subset of N train and N val studies")
    ap.add_argument("--run-id", default=None, help="override run_id from the config")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--synthetic-targets", action="store_true")
    args = ap.parse_args(argv)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    import torch  # noqa: E402  (after CUDA_VISIBLE_DEVICES)
    from rsna_kad.config import load_config  # noqa: E402
    from rsna_kad.runguard import seeded_run_id  # noqa: E402
    from rsna_kad.train import run_dir_for, train_fold  # noqa: E402

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
        if not args.run_id:
            cfg.run_id = seeded_run_id(cfg.run_id, args.seed)
    if args.run_id:
        cfg.run_id = args.run_id
    if args.epochs:
        cfg.epochs = args.epochs
    if args.workers is not None:
        cfg.num_workers = args.workers
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = run_dir_for(cfg, args.fold, args.runs_dir)
    metrics = train_fold(cfg, args.fold, run_dir, limit=args.limit, resume=not args.no_resume,
                         device=device, synthetic=args.synthetic_targets)
    print(f"fold {args.fold}: selected epoch {metrics['epoch']} ({cfg.select_epoch}) "
          f"oof_weak_auc {metrics['oof_weak_auc']} oof_strict_auc {metrics['oof_strict_auc']} "
          f"gold_auc {metrics['gold_auc']} | best epoch {metrics['best_epoch']['epoch']} "
          f"oof_weak_auc {metrics['best_epoch']['oof_weak_auc']} -> {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
