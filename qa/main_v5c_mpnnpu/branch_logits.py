#!/usr/bin/env python3
"""Logits of finished full-data members on fixed inputs, through whichever rsna_kad is on PYTHONPATH.

Run once with release/v7's src and once with branch maxpool-nnpu's (69a8b8e) src; compare_logits
then checks the port of the max-pool residual: the same bundle must give the same logits.

  PYTHONPATH=<tree>/src python qa/main_v5c_mpnnpu/branch_logits.py <out.npz> <run dir> ...
"""
import sys
from pathlib import Path

import numpy as np
import torch

from rsna_kad.model import load_bundle

torch.set_num_threads(8)
out, runs = Path(sys.argv[1]), sys.argv[2:]
res = {}
for rd in runs:
    model, meta = load_bundle(Path(rd) / "full", "cpu", views=True)
    s, k, px = len(meta["arch"]["slots"]), int(meta["arch"]["K_infer"]), int(meta["arch"]["img_px"])
    g = torch.Generator().manual_seed(2026)
    images = torch.randint(0, 256, (2, s, k, 3, px, px), generator=g, dtype=torch.uint8)
    present = torch.ones(2, s, dtype=torch.bool)
    present[1, 3] = False  # one absent slot exercises the [MISSING] path too
    with torch.inference_mode():
        logits = model(images, present)[0].float().numpy()
    res[Path(rd).name] = logits
    print(Path(rd).name, "maxpool_gain" in dict(model.named_parameters()), logits.shape, flush=True)
np.savez(out, **res)
