#!/usr/bin/env python3
"""Build (or resume) the preprocessing cache and verify it.

  build_cache.py build  --split train|test|all [--workers 24] [--limit N]
  build_cache.py verify [--workers 24] [--report path.json]
  build_cache.py export-index --csv analysis/out/cache_index_v1.csv

Resumable: series whose .npy and sidecar match the config are skipped.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

from rsna_kad import cache
from rsna_kad.config import load_cache_config
from rsna_kad.constants import CACHE_DIR, DATA_DIR

log = logging.getLogger("build_cache")


def _disk(path: Path) -> str:
    probe = path if path.exists() else path.parent
    usage = shutil.disk_usage(probe)
    return f"free {usage.free / (1 << 30):.0f} GiB of {usage.total / (1 << 30):.0f} GiB"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "verify", "export-index"])
    parser.add_argument("--config", default="configs/cache_v1.yaml")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--out", default=None, help="cache root (default: $RSNA_KAD_CACHE/<version>)")
    parser.add_argument("--split", default="all", choices=["train", "test", "all"])
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None, help="build only the first N series (smoke test)")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--report", default=None, help="verify: write the JSON report here")
    parser.add_argument("--csv", default="analysis/out/cache_index_v1.csv")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_cache_config(args.config)
    out = Path(args.out or Path(CACHE_DIR) / cfg.version)
    splits = ["train", "test"] if args.split == "all" else [args.split]

    if args.command == "build":
        log.info("before: %s", _disk(out))
        started = time.monotonic()
        for split in splits:
            df = cache.build_cache(args.data_dir, out, split, cfg, workers=args.workers,
                                   resume=not args.no_resume, limit=args.limit)
            bad = df[~df["ok"]]
            if len(bad):
                log.error("%d series failed in %s; first: %s", len(bad), split,
                          bad[["StudyInstanceUID", "SeriesInstanceUID", "error"]].head(5).to_dict("records"))
        log.info("build done in %.0fs; after: %s", time.monotonic() - started, _disk(out))
        return 0

    if args.command == "verify":
        started = time.monotonic()
        report = cache.verify_cache(out, workers=args.workers)
        report["verify_seconds"] = round(time.monotonic() - started, 1)
        text = json.dumps(report, indent=2, default=str)
        print(text)
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(text)
        return 0 if report["passed"] else 1

    df = cache.read_index(out)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    cache.summarize_index(df).to_csv(args.csv, index=False, lineterminator="\n")
    log.info("wrote %s (%d rows)", args.csv, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
