"""Pipeline A, core ideas only - a discussion excerpt of pipelineA.py (not runnable on its own).

Left out on purpose: DICOM reading, series routing, caching, folds, metrics, CLI, logging. Those
are standard and verified; the full, runnable version is pipelineA.py.

PROBLEM
  Knee MRI study (several DICOM series) -> 12 binary findings (ACL, MCL, medial/lateral meniscus,
  medial/lateral/PF OA, effusion, synovitis, Baker's cyst, contusion, fracture). Metric: macro AUC.
  4,407 training studies, but only 58 have image-level ("gold") labels. The rest have only a
  free-text multilingual radiology report. Test = images only. ~123 hospitals; folds grouped by site.

THE HARD PART IS LABELS, NOT THE NETWORK
  Reports were turned into per-finding states {affirmed, mild, hedged, absent, negated} by a local
  LLM (Qwen3-32B) + rules + a second LLM + a public table. Those states become SOFT targets with
  per-cell WEIGHTS (section 2), then a pseudo-label step mixes in a teacher model's predictions.
  Biggest trap: findings radiologists rarely write down. Synovitis is affirmed in 12.5 % of
  reports but positive in 46.6 % of gold studies, so "report silent" is NOT a negative.

RESULTS (public LB; top of LB ~0.961)
  report labels 0.931 -> 3-model blend 0.933 -> pseudo-label targets 0.943 -> full-data,
  single architecture, 6 seeds 0.948. Gold-58 AUC: single seed ~0.920, 6-seed mean 0.925.

TRIED, NO GAIN (each measured on gold-58 and/or LB)
  stacking / learned blend weights; a 2nd pseudo-label round; blending other backbones; TTA (z, crop,
  rotation); pairwise ranking loss; MRI intensity augmentation; per-label attention heads and a
  transformer head over slice tokens; an independent DINOv3 ViT-B pipeline merged 50/50 (+0.012
  before pseudo-labelling, -0.0004 after: both models converge to the same targets).

WHAT WE'D LIKE IDEAS ON
  1. Better use of the 58 gold studies than calibrating the report labels (section 2).
  2. Escaping "every model converges to the pseudo targets" - the reason ensembling stopped helping.
  3. Label-specific gaps: weakest gold AUCs are synovitis ~0.81, lateral OA ~0.84, PF OA ~0.88,
     lateral meniscus ~0.88-0.90 (vs ACL/effusion ~0.98).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LABEL_KEYS = ("acl", "mcl", "med_men", "lat_men", "med_oa", "lat_oa",
              "pf_oa", "effusion", "synovitis", "baker", "contusion", "fracture")
UNDER_REPORTED = {"synovitis", "fracture", "contusion", "baker", "med_oa", "lat_oa", "pf_oa"}


# ---------------------------------------------------------------------------------------------
# 1. Recipe (the exact shipped values)
# ---------------------------------------------------------------------------------------------
@dataclass
class Recipe:
    backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"   # timm, ImageNet-22k
    img_px: int = 384
    # 5 views; sag_fs appears twice: full 130 mm field + a 90 mm centre crop (menisci / ACL detail).
    views: tuple = ("sag_fs", "cor_fs", "ax_fs", "sag_t1", "sag_fs")
    view_mm: tuple = (130, 130, 130, 130, 90)
    k_train: int = 12          # triplets sampled per view in training
    k_infer: int = 24
    epochs: int = 8            # last epoch kept (picking the best epoch was +0.008 optimistic)
    batch_studies: int = 4
    lr_backbone: float = 1e-4
    lr_heads: float = 1e-3
    weight_decay: float = 0.05
    ema_decay: float = 0.999
    aux_weight: float = 0.3    # per-view auxiliary heads
    mixup_p: float = 0.3       # manifold mixup on view embeddings
    mixup_alpha: float = 0.4
    view_dropout_p: float = 0.1
    # augmentation: random crop 115-130 mm, rotation +-8 deg, brightness/contrast +-3 %


# ---------------------------------------------------------------------------------------------
# 2. Report labels -> soft targets + weights
# ---------------------------------------------------------------------------------------------
# Prior P(gold positive | report cell); refined per label from the gold studies (empirical Bayes).
PRIOR = {"affirmed": 0.92, "hedged": 0.50, "negated": 0.04,
         "absent_well_reported": 0.06, "absent_under_reported": 0.15, "affirmed_mild": 0.35}
WEIGHT = {"affirmed": 1.0, "negated": 1.0, "hedged": 0.7,
          "absent_well_reported": 0.5, "absent_under_reported": 0.1}
PRIOR_STRENGTH = 8.0


def calibrate_cell(prior: float, n_gold: int, n_gold_pos: int) -> float:
    """Beta(8 * prior) posterior mean from the gold studies falling in this (label, cell).
    Afterwards the cells of one label are forced monotone (affirmed >= hedged >= absent >= negated)
    with PAVA, because with ~58 gold rows single cells are noisy."""
    return (PRIOR_STRENGTH * prior + n_gold_pos) / (PRIOR_STRENGTH + n_gold)


def cell_weight(cell: str, key: str, voter_agreement: float) -> float:
    """voter_agreement = share of label voters agreeing with the majority (1.0 = unanimous).
    Disagreement down-weights to 0.5 at an even split; hallucinated evidence caps at 0.3;
    unparseable or post-operative reports get 0.2 on every label."""
    if cell == "absent":
        cell = "absent_under_reported" if key in UNDER_REPORTED else "absent_well_reported"
    base = WEIGHT.get(cell, 1.0)
    return base * (0.5 + 0.5 * (2 * voter_agreement - 1))


def pseudo_targets(label_target: np.ndarray, label_weight: np.ndarray, teacher_prob: np.ndarray,
                   silent_under_reported: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The step that took LB 0.933 -> 0.943. Teacher = out-of-fold predictions of earlier models.
    Every cell becomes 0.5 x label + 0.5 x teacher (fixed, not tuned), and cells where the report
    is silent on an under-reported finding go from weight 0.1 to 0.5 - half the target is now an
    image reading. Iterating (teacher = these students) made things worse."""
    target = 0.5 * label_target + 0.5 * teacher_prob
    weight = np.where(silent_under_reported, 0.5, label_weight)
    return target, weight


# ---------------------------------------------------------------------------------------------
# 3. Inputs: 2.5-D triplets per view
# ---------------------------------------------------------------------------------------------
# Each series is cached as a (n_slices, 384, 384) uint8 volume: 130 mm physical centre crop
# (constant field of view across scanners), 3-D series slab-averaged to ~3 mm, <= 40 slices.
# A missing fat-suppressed series borrows the T1 of the same plane; still-missing views are
# handled by the model (section 4).
def triplet_indices(n_slices: int, k: int, rng: np.random.Generator | None) -> np.ndarray:
    """K slice centres, one per equal stratum of the volume (random within the stratum in training,
    the middle at inference); each becomes an RGB-like triplet [i-1, i, i+1] for a 2-D backbone."""
    pos = np.arange(k) + (rng.random(k) if rng is not None else 0.5)
    centres = np.clip(np.floor(pos * n_slices / k), 0, n_slices - 1).astype(int)
    return np.clip(centres[:, None] + np.array([-1, 0, 1]), 0, n_slices - 1)   # (K, 3)


# ---------------------------------------------------------------------------------------------
# 4. Model: 2-D backbone -> gated-attention MIL per view -> fusion
# ---------------------------------------------------------------------------------------------
class GatedAttention(nn.Module):
    """Ilse et al. 2018: attention-weighted mean over the K triplets of one view."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.v, self.u, self.w = nn.Linear(dim, hidden), nn.Linear(dim, hidden), nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:          # (M, K, D) -> (M, D)
        a = self.w(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))).squeeze(-1).softmax(dim=1)
        return torch.einsum("mk,mkd->md", a, h)


class KneeMIL(nn.Module):
    """One shared backbone and one shared attention pool for all views. An absent view is replaced
    by its own learned [MISSING] vector, and the presence bits are fed to the fusion MLP too."""

    def __init__(self, r: Recipe):
        super().__init__()
        import timm
        self.backbone = timm.create_model(r.backbone, pretrained=True, num_classes=0, drop_path_rate=0.1)
        d, s, n = self.backbone.num_features, len(r.views), len(LABEL_KEYS)
        self.dim = d
        self.attn = GatedAttention(d)
        self.missing = nn.Parameter(torch.randn(s, d) * 0.02)
        self.aux_w = nn.Parameter(torch.randn(s, d, n) * d ** -0.5)   # per-view auxiliary heads
        self.aux_b = nn.Parameter(torch.zeros(s, n))
        self.fusion = nn.Sequential(nn.Linear(s * d + s, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, n))

    def embed(self, images: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """(B, S, K, 3, H, W) normalised images, (B, S) bool -> (B, S, D); absent views skip the backbone."""
        b, s, k = images.shape[:3]
        emb = images.new_zeros(b * s, self.dim)
        sel = present.reshape(-1).nonzero().squeeze(1)
        x = images.reshape(b * s, k, *images.shape[3:])[sel].flatten(0, 1)
        emb[sel] = self.attn(self.backbone(x).view(sel.numel(), k, self.dim))
        return emb.view(b, s, self.dim)

    def fill_and_fuse(self, emb: torch.Tensor, present_f: torch.Tensor):
        p = present_f.unsqueeze(-1)
        filled = emb * p + self.missing * (1 - p)
        aux = torch.einsum("bsd,sdl->bsl", filled, self.aux_w) + self.aux_b
        return self.fusion(torch.cat([filled.flatten(1), present_f], 1)), aux


# ---------------------------------------------------------------------------------------------
# 5. Loss and one training step
# ---------------------------------------------------------------------------------------------
def weighted_soft_bce(logits, target, weight):
    loss = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none") * weight
    return loss.sum() / weight.sum().clamp_min(1e-6)


def train_step(model: KneeMIL, r: Recipe, batch: dict, rng: np.random.Generator):
    """Optimiser: AdamW, backbone 1e-4 / heads 1e-3, cosine schedule with 1 warmup epoch, grad clip 2,
    bf16 autocast, EMA 0.999 of the weights (the EMA model is the one shipped)."""
    present_f = batch["present"].float()
    target, weight = batch["target"], batch["weight"]
    emb = model.embed(batch["images"], batch["present"])
    if rng.random() < r.mixup_p:                      # manifold mixup at the view-embedding level
        lam = float(rng.beta(r.mixup_alpha, r.mixup_alpha))
        perm = torch.randperm(emb.shape[0])
        emb = lam * emb + (1 - lam) * emb[perm]
        present_f = lam * present_f + (1 - lam) * present_f[perm]
        target = lam * target + (1 - lam) * target[perm]
        weight = lam * weight + (1 - lam) * weight[perm]
    logits, aux = model.fill_and_fuse(emb, present_f)
    loss = weighted_soft_bce(logits, target, weight)
    # Each view's aux head learns the study target, only where that view was present.
    aux_weight = weight.unsqueeze(1) * present_f.unsqueeze(2)
    loss = loss + r.aux_weight * weighted_soft_bce(aux, target.unsqueeze(1).expand_as(aux), aux_weight)
    return loss


# ---------------------------------------------------------------------------------------------
# 6. Training regime and submission
# ---------------------------------------------------------------------------------------------
# - Full-data training (all ~4,350 non-gold studies), 8 epochs per model.
# - The 58 gold studies are NEVER trained on: they are used to calibrate targets (section 2) and
#   as the only clean validation (paired bootstrap; we accept a change only if it clears +0.004
#   or more there - the public LB resolves only ~0.002-0.003).
# - Submission = plain mean of 6 seeds' sigmoid outputs, K = 24 triplets per view, no TTA.
