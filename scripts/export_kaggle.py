#!/usr/bin/env python3
"""Package finished folds for Kaggle (architecture §4 stage 4, §2.8).

  python scripts/export_kaggle.py --run runs/mvp_effv2s_256 --folds 0 1 2 3 4 --name mvp_v1 --user sroziews
  python scripts/export_kaggle.py --run runs/eff_effv2s_256 --mode efficiency --k-infer 12 --name eff_v1 --user sroziews
  python scripts/export_kaggle.py --run runs/main_convnext_s_384 runs/eff_effv2s_256 --name main_v2 --user sroziews
  python scripts/export_kaggle.py --name mvp_v1 --download-wheels        # only refresh export/wheels
  python scripts/export_kaggle.py --name mvp_v1 --push-dataset create    # first upload
  python scripts/export_kaggle.py --name mvp_v1 --push-dataset version -m "folds 0-4"
  python scripts/export_kaggle.py --name mvp_v1 --push-kernel

Writes export/<name>/{models,ensemble.json,wheels,VERSION,dataset-metadata.json}, regenerates
notebooks/kaggle_inference.py and notebooks/kernel/kernel-metadata.json. Every step is
idempotent; the wheel download is shared in export/wheels and copied into each bundle.

Refuses (exit 2, nothing written) a run that a live process still names, a fold holding last.pt
or not at its configured final epoch, a requested fold that is missing, and sweep/smoke runs
(`*_sweep_*`, `*smoke*`, or trained with --limit) unless --allow-smoke. Every member json and
ensemble.json record the git commit of this tree (`git_sha`; "-dirty" = uncommitted changes).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", default=None,
                    help="runs/<run_id> directories holding fold{k}/; several = one ensemble bundle")
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(5)))
    ap.add_argument("--k-infer", type=int, default=None,
                    help="override the triplet count of every member (default: the run's K_infer)")
    ap.add_argument("--name", required=True, help="export/<name>")
    ap.add_argument("--mode", choices=["main", "efficiency"], default="main")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--wall-hours", type=float, default=8.25)
    ap.add_argument("--user", default="", help="Kaggle username (dataset/kernel ids)")
    ap.add_argument("--export-root", default="export")
    ap.add_argument("--wheels-dir", default="export/wheels")
    ap.add_argument("--download-wheels", action="store_true")
    ap.add_argument("--no-verify", action="store_true", help="skip loading every member on the CPU")
    ap.add_argument("--push-dataset", choices=["create", "version"], default=None)
    ap.add_argument("-m", "--message", default="")
    ap.add_argument("--push-kernel", action="store_true")
    ap.add_argument("--config-dir", default=None, help="KAGGLE_CONFIG_DIR (default: ./.kaggle)")
    ap.add_argument("--allow-smoke", action="store_true",
                    help="ship runs named *_sweep_*/*smoke* or trained with --limit")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from rsna_kad import runguard  # noqa: E402

    if args.run:
        problems = runguard.export_problems(args.run, args.folds, allow_smoke=args.allow_smoke)
        if problems:
            print("export refused:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 2
    from rsna_kad import kaggle_export as ke  # noqa: E402

    out_dir = Path(args.export_root) / args.name
    user = args.user
    if not user:
        meta_path = out_dir / ke.DATASET_META
        if meta_path.exists():
            user = json.loads(meta_path.read_text(encoding="utf-8"))["id"].split("/")[0]
    if args.download_wheels:
        wheels = ke.download_wheels(args.wheels_dir)
        print(f"wheels in {args.wheels_dir}: {len(wheels)}")
        for w in wheels:
            print("  ", w)
    if args.run:
        run_dirs = [md for run in args.run for md in runguard.member_dirs(run, args.folds)]
        if not run_dirs:
            print("no folds requested", file=sys.stderr)
            return 2
        if not user:
            print("--user is required for a new export", file=sys.stderr)
            return 2
        doc = ke.export_bundle(run_dirs, out_dir, mode=args.mode, wheels_dir=args.wheels_dir,
                               tta=False if args.no_tta else None, workers=args.workers,
                               wall_guard_hours=args.wall_hours, user=user, verify=not args.no_verify,
                               k_infer=args.k_infer, git_sha=runguard.git_sha(root))
        print(f"exported {len(doc['members'])} members to {out_dir}: {doc['version']} git {doc['git_sha'] or '?'}")
        print("prior:", [round(p, 3) for p in doc["prior"]])
        print("wheels:", len(doc["wheels"]))
    nb = ke.build_notebook(root / "notebooks" / ke.NOTEBOOK_NAME)
    print(f"notebook {nb} ({nb.stat().st_size} bytes)")
    if user:
        km = ke.write_kernel_metadata(root / "notebooks" / "kernel", user)
        print(f"kernel metadata {km}")
    if args.push_dataset:
        print(ke.push_dataset(out_dir, args.message or f"{args.name} {args.mode}",
                              new=args.push_dataset == "create", config_dir=args.config_dir))
    if args.push_kernel:
        print(ke.push_kernel(root / "notebooks" / "kernel", config_dir=args.config_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
