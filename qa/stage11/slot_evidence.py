#!/usr/bin/env python3
"""Per-(label, view) evidence from a trained arm's aux heads, to check model.SLOT_PRIOR.

The anatomical prior of §7 A2 must not be copied from the competitor's table: their slots are
six (plane x contrast) series, ours are the four MAIN_SLOTS plus arm C's 90 mm sagittal view.
This script measures the thing the prior claims - "can finding L be read off view j at all?" -
with no extra training and one evaluation pass:

  KneeMIL's per-slot aux head is a linear read-out of view j's embedding alone (§2.4 deep
  supervision), trained against the same 12 targets as the fused head. So AUC(aux[:, j, l]) over
  the studies where view j is PRESENT is exactly "how much of label l is decodable from view j
  on its own". Absent views are excluded: their embedding is the learned [MISSING] vector, whose
  aux logit is a constant and would score 0.5 and pollute the ranking.

  python qa/stage11/slot_evidence.py --run runs/main_armc_convnextv2_t_384/fold0 --fold 0 \
      --out qa/stage11/slot_evidence.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a fold directory holding bundle.json + model.safetensors")
    ap.add_argument("--fold", type=int, required=True, help="the validation fold of that model")
    ap.add_argument("--targets-version", default=None, help="default: the run's own")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-studies", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="qa/stage11/slot_evidence.md")
    args = ap.parse_args(argv)

    from rsna_kad.cache import get_index
    from rsna_kad.config import TrainConfig, from_dict
    from rsna_kad.constants import ID_COL, LABEL_KEYS
    from rsna_kad.cv import load_folds
    from rsna_kad.data import StudyDataset, collate_studies, load_targets
    from rsna_kad.metrics import auc, binarise
    from rsna_kad.model import NARROW_VIEW_KEYS, SLOT_PRIOR, load_bundle
    from rsna_kad.train import cache_root, split_studies, targets_path
    from torch.utils.data import DataLoader

    model, meta = load_bundle(args.run, args.device, views=True)
    cfg = from_dict(TrainConfig, meta["config"])
    if args.targets_version:
        cfg.targets_version = args.targets_version
    slots, views = tuple(cfg.model.slots), tuple(cfg.model.view_crop_mm) or (cfg.model.crop_mm,) * len(cfg.model.slots)
    names = [f"{s}@{v:g}" for s, v in zip(slots, views)]

    index = get_index(cache_root(cfg))
    folds = load_folds(cfg.folds_csv)
    targets = load_targets(targets_path(cfg, args.fold))
    gold_ids = set(folds.loc[folds["is_gold"], ID_COL].astype(str))
    targets = targets[~targets[ID_COL].isin(gold_ids)]
    _, val_ids = split_studies(folds, args.fold, targets, set(index.studies("train")))
    val_ids = [s for s in val_ids if s not in gold_ids]
    ds = StudyDataset(cache_root(cfg), val_ids, None, cfg, train=False, index=index)
    loader = DataLoader(ds, batch_size=args.batch_studies, shuffle=False, num_workers=args.workers,
                        collate_fn=collate_studies, pin_memory=True)

    ids: list[str] = []
    aux_all, present_all = [], []
    with torch.inference_mode():
        for b in loader:
            images = b["images"].to(args.device, non_blocking=True)
            present = b["present"].to(args.device, non_blocking=True)
            with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16):
                _, aux = model(images, present)
            aux_all.append(aux.float().cpu().numpy())
            present_all.append(b["present"].numpy())
            ids += list(b["study"])
    aux = np.concatenate(aux_all)              # (N, S, 12)
    pres = np.concatenate(present_all)         # (N, S)

    t = targets.set_index(ID_COL).loc[ids]
    y = t[[f"{k}_target" for k in LABEL_KEYS]].to_numpy(np.float64)
    w = t[[f"{k}_weight" for k in LABEL_KEYS]].to_numpy(np.float64)

    tab = np.full((len(LABEL_KEYS), len(slots)), np.nan)
    cnt = np.zeros((len(LABEL_KEYS), len(slots)), int)
    for li, key in enumerate(LABEL_KEYS):
        keep, ybin = binarise(y[:, li], w[:, li])
        for si in range(len(slots)):
            m = keep & pres[:, si]
            cnt[li, si] = int(m.sum())
            if m.sum() > 20 and 0 < ybin[m].sum() < m.sum():
                tab[li, si] = auc(ybin[m], aux[m, si, li])

    rank = np.argsort(-np.nan_to_num(tab, nan=-1.0), axis=1)
    L = [f"# Per-view evidence from the aux heads of `{args.run}` (fold {args.fold})", "",
         f"{len(ids)} validation studies, targets `{cfg.targets_version}`. Each cell is the AUC of "
         "that view's own aux logit, over the studies where the view is present; `n` is the number "
         "of scored studies. Bold-equivalent marker `*` = the view is in `model.SLOT_PRIOR` for "
         "that label (`+` = kept only for NARROW_VIEW_KEYS on the 90 mm view).", "",
         "| label | " + " | ".join(names) + " | best view | prior views |", "|---|" + "---|" * (len(slots) + 2)]
    for li, key in enumerate(LABEL_KEYS):
        cells = []
        for si, (slot, view) in enumerate(zip(slots, views)):
            narrow = float(view) < float(cfg.model.crop_mm)
            on = slot in SLOT_PRIOR[key] and (not narrow or key in NARROW_VIEW_KEYS)
            v = "-" if np.isnan(tab[li, si]) else f"{tab[li, si]:.3f}"
            cells.append(f"{v}{'*' if on else ''} (n={cnt[li, si]})")
        best = names[int(rank[li, 0])]
        pri = [names[si] for si, (s_, v_) in enumerate(zip(slots, views))
               if s_ in SLOT_PRIOR[key] and (float(v_) >= float(cfg.model.crop_mm) or key in NARROW_VIEW_KEYS)]
        L.append(f"| {key} | " + " | ".join(cells) + f" | **{best}** | {', '.join(pri)} |")

    # Agreement: how often the measured top-1/top-2 view is inside the table we wrote by anatomy.
    hits1 = hits2 = n = 0
    for li, key in enumerate(LABEL_KEYS):
        order = [names[si] for si in rank[li] if not np.isnan(tab[li, rank[li][0]])]
        pri = {names[si] for si, (s_, v_) in enumerate(zip(slots, views))
               if s_ in SLOT_PRIOR[key] and (float(v_) >= float(cfg.model.crop_mm) or key in NARROW_VIEW_KEYS)}
        if not order or not pri or len(pri) == len(slots):
            continue
        n += 1
        hits1 += order[0] in pri
        hits2 += any(o in pri for o in order[:2])
    L += ["", f"Top-1 measured view inside the anatomical prior: **{hits1}/{n}**; top-2: **{hits2}/{n}** "
          "(labels whose prior row is uniform, i.e. a no-op, are excluded).", ""]
    text = "\n".join(L) + "\n"
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    pd.DataFrame(tab, index=list(LABEL_KEYS), columns=names).to_csv(
        Path(args.out).with_suffix(".csv"), float_format="%.4f")
    return 0


if __name__ == "__main__":
    sys.exit(main())
