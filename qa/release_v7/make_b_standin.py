#!/usr/bin/env python3
"""A stand-in pipeline-B member for the release/v7 dry run before B's real gate-0 smoke exists.

Randomly initialised (pretrained=False, nothing downloaded) KneeMIL on B's own smoke config
(configs/b_smoke_vitb_288.yaml: vit_base_patch16_dinov3 @288, per_label_attn head, 5 views with
110/90 mm crops, K_infer 16), saved as a finished fold so the export guard treats it like one.
Its predictions are meaningless; it exercises the loader, the new config keys, the 288 px input
path, the group weights and the licence files.

  PYTHONPATH=.worktrees/release_v7/src python qa/release_v7/make_b_standin.py .tmp/v7_standin/runs
"""
import json
import sys
from pathlib import Path

import torch

from rsna_kad.config import load_config
from rsna_kad.model import build_model, bundle_meta, save_bundle

worktree = Path(__file__).resolve().parents[2] / ".worktrees" / "release_v7"
cfg = load_config(worktree / "configs" / "b_smoke_vitb_288.yaml")
cfg.run_id = "b_standin_random"
final = cfg.epochs - 1
torch.manual_seed(0)
fold = Path(sys.argv[1]) / cfg.run_id / "fold0"
save_bundle(build_model(cfg.model, pretrained=False), bundle_meta(cfg, 0, metrics={"epoch": final}), fold)
(fold / "metrics.json").write_text(json.dumps({"epoch": final, "last_epoch": {"epoch": final}}))
print(fold, cfg.model.backbone, cfg.model.img_px, cfg.model.head, list(cfg.model.view_crop_mm))
