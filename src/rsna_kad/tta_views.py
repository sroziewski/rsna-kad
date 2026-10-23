"""Deterministic evaluation views for test-time augmentation beyond z-TTA (qa/tta2).

ViewTTADataset is StudyDataset at evaluation with one fixed geometric change per study: every
view's field shrunk by `crop_factor` (130 mm -> 115 mm at 115/130, a 90 mm view -> ~79.6 mm) and/or
every slice of every view rotated by `rot_deg`, through the same warp_slices path training uses.
No flips of any kind: medial/lateral must stay where it is. Brightness and contrast stay neutral.
With crop_factor 1 and rot_deg 0 it returns exactly what StudyDataset(train=False) returns.

Not inlined into the notebook; adopting a view for submission means porting it to infer.py."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rsna_kad.constants import LABEL_KEYS
from rsna_kad.data import StudyDataset, load_study, read_triplets, sample_centres, triplet_indices


def eval_view_params(crop_mm: float, view_mm: float, crop_factor: float = 1.0,
                     rot_deg: float = 0.0) -> dict[str, float] | None:
    """data.view_params with the field scaled by crop_factor and a fixed rotation; None (the
    plain-resize path) exactly when the result is the unrotated whole cached field."""
    field = float(view_mm) * float(crop_factor)
    if field >= float(crop_mm) and rot_deg == 0.0:
        return None
    return {"scale": float(crop_mm) / min(field, float(crop_mm)), "rot_deg": float(rot_deg),
            "contrast": 1.0, "brightness": 0.0}


class ViewTTADataset(StudyDataset):
    def __init__(self, *args: Any, crop_factor: float = 1.0, rot_deg: float = 0.0, **kw: Any):
        super().__init__(*args, **kw)
        if self.train:
            raise ValueError("ViewTTADataset is for evaluation only (train=False)")
        if not 0.0 < crop_factor <= 1.0:
            raise ValueError(f"crop_factor must be in (0, 1], got {crop_factor}")
        self.crop_factor = float(crop_factor)
        self.rot_deg = float(rot_deg)

    def __getitem__(self, i: int) -> dict[str, Any]:
        study = self.studies[i]
        cfg = self.cfg
        vols = load_study(self.cache_dir, self.split, study, self.load_slots, sibling_rng=None,
                          sibling_p=cfg.sibling_p, index=self.index)
        present = np.array([s in vols for s in self.slots], dtype=bool)
        images = np.zeros((len(self.slots), self.k, 3, self.px, self.px), dtype=np.uint8)
        for j, slot in enumerate(self.slots):
            if not present[j]:
                continue
            vol = vols[slot]
            n = vol.shape[0]
            idx = triplet_indices(sample_centres(n, self.k, None, self.z_offset), n)
            params = eval_view_params(cfg.cache.crop_mm, self.view_mm[j], self.crop_factor, self.rot_deg)
            images[j] = read_triplets(vol, idx, self.px, params)
        target, weight = self.targets.get(study, (np.zeros(len(LABEL_KEYS), np.float32),
                                                   np.zeros(len(LABEL_KEYS), np.float32)))
        return {"study": study, "images": torch.from_numpy(images), "present": torch.from_numpy(present),
                "target": torch.from_numpy(np.array(target, dtype=np.float32)),
                "weight": torch.from_numpy(np.array(weight, dtype=np.float32))}
