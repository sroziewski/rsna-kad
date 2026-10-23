#!/usr/bin/env python3
"""OOF recomputation of the main_v4 members with and without z-TTA, into qa/ztta only.

  python qa/ztta/ztta_oof.py            # all members, both passes
  python qa/ztta/ztta_oof.py --check    # compare notta against each run's training OOF

Same inference path as `predict_oof.py --recompute` (load_bundle(views=True) + StudyDataset +
predict_studies_cached, K_infer from the config = 24, bf16 autocast), but every output goes to
qa/ztta/<run>/{notta,tta}_oof.parquet: predict_oof.py --recompute overwrites runs/<run>/fold*/oof.parquet in
place, and those files are the scored record of each run. Nothing under runs/ is opened for writing.
Wall time per fold and pass goes to qa/ztta/timing.json. The TTA pass is the z-offset-0.5 view plus the
plain one, so its time includes a full second pass - exactly what the Kaggle notebook pays.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

MEMBERS = {  # run dir -> config it was trained with
    "main_convnext_s_384_e8_p": "configs/main_armA8_pseudo.yaml",
    "eff_pseudo_256": "configs/eff_pseudo_256.yaml",
    "main_armc_convnextv2_t_384_p": "configs/main_armc_pseudo.yaml",
}
K_INFER = 24


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs="*", default=list(MEMBERS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--smoke", type=int, default=None, metavar="N",
                    help="first N studies of fold 0 only, written under .tmp/ztta_smoke (plumbing check)")
    args = ap.parse_args(argv)

    import numpy as np
    import pandas as pd
    import torch
    from rsna_kad.cache import get_index
    from rsna_kad.config import load_config
    from rsna_kad.constants import ID_COL
    from rsna_kad.cv import load_folds
    from rsna_kad.data import StudyDataset
    from rsna_kad.model import load_bundle
    from rsna_kad.train import PRED_COLS, amp_dtype, cache_root, predict_studies_cached

    out_root = ROOT / (".tmp/ztta_smoke" if args.smoke else "qa/ztta")
    if args.check:
        for run in args.members:
            a = pd.read_parquet(ROOT / "runs" / run / "oof.parquet").set_index(ID_COL).sort_index()
            b = pd.read_parquet(out_root / run / "notta_oof.parquet").set_index(ID_COL).loc[a.index]
            d = np.abs(a[PRED_COLS].to_numpy() - b[PRED_COLS].to_numpy())
            print(f"{run}: notta vs training OOF max |d| {d.max():.2e} mean {d.mean():.2e} ({len(a)} studies)")
        return 0

    device = "cuda"
    timing_path = out_root / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    for run in args.members:
        cfg = load_config(ROOT / MEMBERS[run])
        assert cfg.run_id == run, (cfg.run_id, run)
        cfg.model.k_infer = K_INFER
        folds = load_folds(ROOT / cfg.folds_csv)
        cached = set(get_index(cache_root(cfg)).studies("train"))
        (out_root / run).mkdir(parents=True, exist_ok=True)
        for tta in (False, True):
            tag = "tta" if tta else "notta"
            parts, secs = [], {}
            for k in (range(1) if args.smoke else range(5)):
                fold_dir = ROOT / "runs" / run / f"fold{k}"
                assert not (fold_dir / "last.pt").exists(), f"{fold_dir}: unfinished fold"
                model, _ = load_bundle(fold_dir, device, views=True)
                ids = sorted(s for s in folds.loc[folds["fold"] == k, ID_COL].astype(str) if s in cached)
                if args.smoke:
                    ids = ids[: args.smoke]
                ds = StudyDataset(cache_root(cfg), ids, None, cfg, train=False)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                pred = predict_studies_cached(model, ds, tta=tta, batch_size=cfg.eval_batch_studies,
                                              num_workers=args.workers, device=device, dtype=amp_dtype(cfg))
                torch.cuda.synchronize()
                secs[f"fold{k}"] = round(time.perf_counter() - t0, 1)
                pred["fold"] = k
                parts.append(pred[[ID_COL, "fold"] + PRED_COLS])
                print(f"{run} {tag} fold {k}: {len(pred)} studies in {secs[f'fold{k}']} s", flush=True)
                del model
                torch.cuda.empty_cache()
            oof = pd.concat(parts, ignore_index=True)
            assert (args.smoke or len(oof) == 4407) and oof[ID_COL].is_unique, len(oof)
            oof.to_parquet(out_root / run / f"{tag}_oof.parquet", index=False)
            timing.setdefault(run, {})[tag] = {"per_fold_s": secs, "total_s": round(sum(secs.values()), 1),
                                               "n_studies": int(len(oof)), "k_infer": K_INFER,
                                               "gpu": torch.cuda.get_device_name(0)}
            timing_path.write_text(json.dumps(timing, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
