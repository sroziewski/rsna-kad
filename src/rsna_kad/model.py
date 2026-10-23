"""KneeMIL: shared 2-D backbone over triplets -> per-slot gated attention MIL -> slot fusion
with presence mask (architecture §2.4), plus the checkpoint bundle format of §3.2."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from safetensors.torch import load_file, save_file

from rsna_kad.config import ModelConfig, TrainConfig, from_dict, to_dict
from rsna_kad.constants import LABEL_KEYS, LABELS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
WEIGHTS_NAME = "model.safetensors"
BUNDLE_NAME = "bundle.json"
N_FINGERPRINT = 8
FINGERPRINT_RTOL = 1e-4

HEAD_SHARED = "shared_attn"
HEAD_LABEL = "per_label_attn"
HEAD_RESIDUAL = "per_label_residual"
HEAD_XFMR = "per_label_xfmr"
HEADS = (HEAD_SHARED, HEAD_LABEL, HEAD_RESIDUAL, HEAD_XFMR)
# Heads without the §2.4 fusion MLP; their presence bits get per-label weights instead.
NO_FUSION_HEADS = (HEAD_LABEL, HEAD_XFMR)
# Fourier frequencies of the normalised depth position fed to the transformer head.
N_DEPTH_FREQ = 8
# An absent slot is masked out of the per-label softmax by adding log(present). The floor keeps
# the term finite (and the gradient defined) for a study whose slots are all absent.
PRESENCE_EPS = 1e-6

# Anatomical prior: for each finding, the slots in which a radiologist can actually assess it.
# The competitor's SLOT_PRIOR_TABLE is over their six (plane x contrast) slots; this one is
# re-derived for OUR slot names and is checked against our own per-slot evidence in
# qa/stage11/slot_evidence.md (per-slot aux-head AUC of arm C fold 0). It is a bias on the
# attention logits, not a mask: a slot outside the list is down-weighted, never blocked.
#
#   acl        sagittal - the ligament runs anterior-posterior in the notch and is read on
#              sagittal images; coronal shows only a fragment of it.
#   mcl        coronal only - a superficial medial band, the one plane that shows its full length.
#   med_men /  sagittal (bow-tie body and both horns) plus coronal (body, radial tears). Medial
#   lat_men    vs lateral is the sagittal SLICE position, so both keys get the same slots.
#   med_oa /   coronal carries tibiofemoral joint-space narrowing and osteophytes; sag_t1 carries
#   lat_oa     the marrow/osteophyte detail. Fluid-sensitive axial sees neither compartment.
#   pf_oa      axial is the patellofemoral plane; sagittal shows trochlear/patellar cartilage.
#   effusion / suprapatellar pouch and the recesses: fluid-sensitive sagittal and axial.
#   synovitis
#   baker      popliteal fossa, between medial gastrocnemius and semimembranosus: axial is the
#              diagnostic plane, sagittal the confirmatory one; never a T1-only slot.
#   contusion  marrow oedema is only visible fat-suppressed - all three FS slots, no sag_t1.
#   fracture   no anatomical restriction; every slot is listed, which makes the row uniform and
#              therefore a no-op after the softmax. That is deliberate: we do not invent a prior
#              where anatomy gives none.
SLOT_PRIOR: dict[str, tuple[str, ...]] = {
    "acl": ("sag_fs", "sag_t1"),
    "mcl": ("cor_fs", "cor_t1"),
    "med_men": ("sag_fs", "cor_fs", "sag_t1"),
    "lat_men": ("sag_fs", "cor_fs", "sag_t1"),
    "med_oa": ("cor_fs", "cor_t1", "sag_t1"),
    "lat_oa": ("cor_fs", "cor_t1", "sag_t1"),
    "pf_oa": ("ax_fs", "sag_fs", "sag_t1"),
    "effusion": ("sag_fs", "ax_fs"),
    "synovitis": ("sag_fs", "ax_fs"),
    "baker": ("ax_fs", "sag_fs"),
    "contusion": ("sag_fs", "cor_fs", "ax_fs"),
    "fracture": ("sag_fs", "cor_fs", "ax_fs", "sag_t1", "cor_t1", "ax_t1"),
}
# A view cropped below the cached field (arm C's 90 mm sagittal meniscus view) keeps only what
# sits near the image centre, so only central findings keep their prior there. The evidence is
# ours: qa/stage8b measured that a 90 mm *coronal* crop deletes the MCL in 78 % of studies
# because it is 10 mm inside the skin - the same argument disqualifies every peripheral or
# recess-bound finding (effusion's suprapatellar pouch, Baker's popliteal cyst, the patella)
# from a narrow view, while ACL and the menisci are exactly what the crop was built for.
NARROW_VIEW_KEYS: frozenset[str] = frozenset({"acl", "med_men", "lat_men"})


def normalize(x: torch.Tensor) -> torch.Tensor:
    """uint8 (..., 3, H, W) -> float in ImageNet units; float input is assumed normalised."""
    if x.dtype != torch.uint8:
        return x
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    return (x.float().div_(255.0) - mean) / std


def prior_matrix(slots: Sequence[str], view_crop_mm: Sequence[float], crop_mm: float,
                 strength: float, n_labels: int = len(LABEL_KEYS)) -> torch.Tensor:
    """(n_labels, len(slots)) of `strength` where SLOT_PRIOR says the finding is assessable.

    `slots` is the VIEW list, so a slot may appear twice with two fields of view; a view
    narrower than `crop_mm` only keeps the prior of NARROW_VIEW_KEYS. All-zero (strength 0) and
    row-uniform priors are both exact no-ops after the softmax."""
    p = torch.zeros(n_labels, len(slots))
    if not strength or n_labels != len(LABEL_KEYS):
        return p
    views = tuple(view_crop_mm) or (float(crop_mm),) * len(slots)
    if len(views) != len(slots):
        raise ValueError(f"view_crop_mm has {len(views)} entries for {len(slots)} slots")
    for i, key in enumerate(LABEL_KEYS):
        for j, (slot, view) in enumerate(zip(slots, views)):
            narrow = float(view) < float(crop_mm)
            if slot in SLOT_PRIOR[key] and (not narrow or key in NARROW_VIEW_KEYS):
                p[i, j] = float(strength)
    return p


class LabelAttentionHead(nn.Module):
    """One attention distribution per finding over the whole (slot x triplet) bag (§4 A2).

    The bag holds the S slot summaries first and then the S*K triplet tokens, so an absent slot
    still contributes its learned [MISSING] summary and every label has a well-defined
    distribution for every study. Two biases are added to the attention logits before the
    softmax: the anatomical prior (constant per slot) and log(present) for the triplet tokens -
    exact masking for a hard presence, and a smooth down-weighting for the blended presence that
    manifold mixup produces. Scoring MLP and read-out follow the competitor's RaptorClassifier
    (Linear-Tanh-Dropout-Linear over the normalised tokens, pooled with the same normalisation)."""

    def __init__(self, dim: int, n_slots: int, n_labels: int, hidden: int, dropout: float,
                 prior: torch.Tensor):
        super().__init__()
        self.n_slots, self.n_labels = n_slots, n_labels
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Dropout(dropout),
                                   nn.Linear(hidden, n_labels))
        self.cls_w = nn.Parameter(torch.randn(n_labels, dim) * (dim ** -0.5))
        self.cls_b = nn.Parameter(torch.zeros(n_labels))
        # Persistent so a bundle records the exact prior its weights were trained with.
        self.register_buffer("slot_prior", prior)

    def bag_bias(self, present_f: torch.Tensor, k: int) -> torch.Tensor:
        """(B, S + S*K, 1) presence mask + (1, S + S*K, n_labels) prior, already summed."""
        prior = torch.cat([self.slot_prior, self.slot_prior.repeat_interleave(k, dim=1)], dim=1)
        keep = torch.log(present_f.clamp_min(PRESENCE_EPS)).repeat_interleave(k, dim=1)
        mask = torch.cat([torch.zeros_like(present_f), keep], dim=1)
        return prior.t().unsqueeze(0).float() + mask.unsqueeze(-1).float()

    def forward(self, bag: torch.Tensor, present_f: torch.Tensor,
                token_valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """`token_valid` (B, S*K) bool marks real triplet tokens when slices are ragged; None (the
        training/inference path of every existing head) leaves the arithmetic untouched."""
        k = (bag.shape[1] - self.n_slots) // self.n_slots
        h = self.norm(bag)
        bias = self.bag_bias(present_f, k)
        if token_valid is not None:
            keep = torch.cat([torch.ones_like(present_f, dtype=torch.bool), token_valid], dim=1)
            bias = bias.masked_fill(~keep.unsqueeze(-1), math.log(PRESENCE_EPS))
        a = (self.score(h).float() + bias).softmax(dim=1)
        ctx = torch.einsum("bnl,bnd->bld", a, h.float())
        return (ctx * self.cls_w).sum(-1) + self.cls_b, a


class XfmrBlock(nn.Module):
    """Pre-norm transformer encoder layer with per-sample stochastic depth on both residuals."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, attn_drop: float, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(round(dim * mlp_ratio))
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, key_padding: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.drop_path(self.attn(h, h, h, key_padding_mask=key_padding, need_weights=False)[0])
        return x + self.drop_path(self.mlp(self.norm2(x)))


class TokenEncoder(nn.Module):
    """Contextualises the per-label bag before the read-out (head `per_label_xfmr`).

    Every token is projected from the backbone width to `dim` and gets a learned view embedding
    (one per entry of `slots`, so the 90 mm sagittal view is a view of its own) plus either a
    "summary" embedding (the S slot summaries) or a Fourier encoding of its normalised depth
    position. The depth position is the triplet's rank among the valid triplets of its slot,
    (j + 0.5) / n_valid: data.sample_centres takes one centre per equal stratum of the stack, so
    this equals centre / n_slices up to the within-stratum jitter whatever the stack length
    (11-40 slices), without threading slice indices through the data path. Absent slots' and
    padded triplets are key-padding-masked in every layer; summary tokens never are, so no row is
    ever fully masked. Stochastic depth rises linearly from 0 at the first layer to `drop_path`
    at the last (DeiT/ViT convention)."""

    def __init__(self, in_dim: int, n_slots: int, dim: int, layers: int, heads: int,
                 mlp_ratio: float, attn_drop: float, drop_path: float):
        super().__init__()
        self.n_slots = n_slots
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim))
        self.view_emb = nn.Parameter(torch.randn(n_slots, dim) * 0.02)
        self.summary_emb = nn.Parameter(torch.randn(dim) * 0.02)
        self.depth = nn.Linear(2 * N_DEPTH_FREQ, dim)
        self.drop_path_rates = torch.linspace(0.0, float(drop_path), layers).tolist() if layers > 1 else [0.0] * layers
        self.blocks = nn.ModuleList(XfmrBlock(dim, heads, mlp_ratio, attn_drop, r)
                                    for r in self.drop_path_rates)
        self.norm = nn.LayerNorm(dim)

    def depth_positions(self, b: int, k: int, token_valid: torch.Tensor | None,
                        device: torch.device) -> torch.Tensor:
        """(B, S*K) normalised position in (0, 1) of each triplet among its slot's valid ones."""
        s = self.n_slots
        if token_valid is None:
            pos = (torch.arange(k, device=device, dtype=torch.float32) + 0.5) / k
            return pos.repeat(s).unsqueeze(0).expand(b, -1)
        v = token_valid.view(b, s, k).float()
        rank = v.cumsum(dim=2) - 0.5
        return (rank / v.sum(dim=2, keepdim=True).clamp_min(1.0) * v).view(b, s * k)

    def forward(self, bag: torch.Tensor, present_f: torch.Tensor,
                token_valid: torch.Tensor | None = None) -> torch.Tensor:
        b, n, _ = bag.shape
        s = self.n_slots
        k = (n - s) // s
        x = self.proj(bag)
        pos = self.depth_positions(b, k, token_valid, bag.device)
        freq = (2.0 ** torch.arange(N_DEPTH_FREQ, device=bag.device, dtype=torch.float32)) * math.pi
        ang = pos.unsqueeze(-1) * freq
        depth = self.depth(torch.cat([ang.sin(), ang.cos()], dim=-1))
        view = self.view_emb.repeat_interleave(k, dim=0).unsqueeze(0)
        tri = x[:, s:] + view + depth
        summ = x[:, :s] + (self.view_emb + self.summary_emb).unsqueeze(0)
        x = torch.cat([summ, tri], dim=1)
        # Any (mixup-blended) presence keeps a slot's triplets visible; only a fully absent slot
        # or a padded triplet is removed from the keys.
        keep = (present_f > 0).repeat_interleave(k, dim=1)
        if token_valid is not None:
            keep = keep & token_valid
        pad = torch.cat([torch.zeros(b, s, dtype=torch.bool, device=bag.device), ~keep], dim=1)
        for blk in self.blocks:
            x = blk(x, pad)
        return self.norm(x)


class GatedAttention(nn.Module):
    """Ilse et al. gated attention pooling over the K triplets of one slot."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.v = nn.Linear(dim, hidden)
        self.u = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.w(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))).squeeze(-1).float().softmax(dim=1)
        return torch.einsum("mk,mkd->md", a, h.float()), a


class KneeMIL(nn.Module):
    def __init__(self, cfg: ModelConfig, pretrained: bool = False, n_labels: int = len(LABELS)):
        super().__init__()
        self.cfg = cfg
        self.slots = tuple(cfg.slots)
        self.n_labels = n_labels
        self.backbone = timm.create_model(cfg.backbone, pretrained=pretrained, num_classes=0,
                                          global_pool="avg", drop_path_rate=cfg.drop_path)
        if cfg.grad_checkpoint:
            self.backbone.set_grad_checkpointing(True)
        d, s = self.backbone.num_features, len(self.slots)
        self.dim = d
        self.n_slots = s
        self.head = cfg.head
        if self.head not in HEADS:
            raise ValueError(f"model.head must be one of {HEADS}, got {cfg.head!r}")
        self.attn = GatedAttention(d, cfg.attn_hidden)
        self.missing = nn.Parameter(torch.randn(s, d) * 0.02)
        self.aux_w = nn.Parameter(torch.randn(s, d, n_labels) * (d ** -0.5))
        self.aux_b = nn.Parameter(torch.zeros(s, n_labels))
        # Everything above, and this block, is built in the same order for every head, so
        # `shared_attn` keeps the parameter set and the RNG stream of the published arms.
        if self.head not in NO_FUSION_HEADS:
            self.fusion = nn.Sequential(
                nn.Linear(s * d + s, cfg.fusion_hidden), nn.GELU(), nn.Dropout(cfg.drop_rate),
                nn.Linear(cfg.fusion_hidden, n_labels))
        if self.head != HEAD_SHARED:
            self.label_head = LabelAttentionHead(
                cfg.xfmr_dim if self.head == HEAD_XFMR else d, s, n_labels, cfg.label_attn_hidden,
                cfg.label_attn_drop,
                prior_matrix(self.slots, cfg.view_crop_mm, cfg.crop_mm, cfg.slot_prior_strength,
                             n_labels))
        if self.head == HEAD_RESIDUAL:
            # Zero gate: at initialisation the arm IS shared_attn, and the label attention can
            # only ever perturb that baseline (the competitor's CodexResidualPool/_GatedDelta).
            self.gate = nn.Parameter(torch.zeros(n_labels))
        elif self.head in NO_FUSION_HEADS:
            # The fusion MLP is gone in this head, so the presence bits get their own per-label
            # weights - the one place §2.4's presence signal would otherwise be lost.
            self.pres_w = nn.Parameter(torch.zeros(n_labels, s))
        if self.head == HEAD_XFMR:
            self.encoder = TokenEncoder(d, s, cfg.xfmr_dim, cfg.xfmr_layers, cfg.xfmr_heads,
                                        cfg.xfmr_mlp_ratio, cfg.xfmr_attn_drop, cfg.xfmr_drop_path)

    def head_parameters(self) -> list[nn.Parameter]:
        backbone = set(id(p) for p in self.backbone.parameters())
        return [p for p in self.parameters() if id(p) not in backbone]

    def triplet_features(self, images: torch.Tensor, present: torch.Tensor):
        """Backbone features of the present slots only: (n_present, K, D) or None, plus the flat
        indices of those slots and the (B, S, K) shape. Split out of `embed` so the per-label
        heads can pool the same features a second time without a second backbone pass."""
        b, s, k = images.shape[:3]
        sel = present.reshape(-1).nonzero().squeeze(1)
        if sel.numel() == 0:
            return None, sel, (b, s, k)
        x = images.view(b * s, k, *images.shape[3:])[sel].reshape(-1, *images.shape[3:])
        x = normalize(x).contiguous(memory_format=torch.channels_last)
        return self.backbone(x).view(sel.numel(), k, self.dim), sel, (b, s, k)

    def embed(self, images: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """images (B, S, K, 3, H, W) uint8/float, present (B, S) bool -> slot embeddings
        (B, S, D); only present slots pass through the backbone, absent ones are zero.

        For the per-label heads the S*K triplet tokens are appended along dim 1, giving the bag
        (B, S + S*K, D). Carrying the bag in the same tensor is what keeps `train.train_fold`
        head-agnostic: its manifold mixup mixes whatever `fill` returns, so the bag is mixed
        with the slot embeddings and the two stay consistent."""
        feats, sel, (b, s, k) = self.triplet_features(images, present)
        emb = images.new_zeros((b * s, self.dim), dtype=torch.float32)
        if feats is not None:
            emb = emb.index_put((sel,), self.attn(feats)[0])
        emb = emb.view(b, s, self.dim)
        if self.head == HEAD_SHARED:
            return emb
        tok = images.new_zeros((b * s, k, self.dim), dtype=torch.float32)
        if feats is not None:
            tok = tok.index_put((sel,), feats.float())
        return torch.cat([emb, tok.view(b, s * k, self.dim)], dim=1)

    def fill(self, emb: torch.Tensor, present_f: torch.Tensor) -> torch.Tensor:
        """Replace absent slots by their learned [MISSING] vector; present_f (B, S) in [0, 1]
        so mixed (manifold-mixup) presence blends the two. Triplet tokens (rows S: of a
        per-label bag) pass through untouched - their absent slots are masked in the softmax."""
        p = present_f.unsqueeze(-1)
        if self.head == HEAD_SHARED:
            return emb * p + self.missing.unsqueeze(0) * (1.0 - p)
        slots = emb[:, :self.n_slots] * p + self.missing.unsqueeze(0) * (1.0 - p)
        return torch.cat([slots, emb[:, self.n_slots:]], dim=1)

    def fuse(self, filled: torch.Tensor, present_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.head == HEAD_SHARED:
            aux = torch.einsum("bsd,sdl->bsl", filled, self.aux_w) + self.aux_b
            return self.fusion(torch.cat([filled.flatten(1), present_f], dim=1)), aux
        slots = filled[:, :self.n_slots]
        aux = torch.einsum("bsd,sdl->bsl", slots, self.aux_w) + self.aux_b
        delta, _ = self.label_head(self.readout_tokens(filled, present_f), present_f)
        if self.head == HEAD_RESIDUAL:
            base = self.fusion(torch.cat([slots.flatten(1), present_f], dim=1))
            return base + self.gate * delta, aux
        return delta + torch.einsum("bs,ls->bl", present_f, self.pres_w), aux

    def label_attention(self, images: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """(B, S + S*K, n_labels) attention weights, one distribution per label over the bag."""
        if self.head == HEAD_SHARED:
            raise RuntimeError("shared_attn has one attention over each slot, not one per label")
        present_f = present.float()
        filled = self.fill(self.embed(images, present), present_f)
        return self.label_head(self.readout_tokens(filled, present_f), present_f)[1]

    def readout_tokens(self, filled: torch.Tensor, present_f: torch.Tensor) -> torch.Tensor:
        """The bag the per-label read-out attends over: as-is for the depth-1 heads, contextualised
        by the transformer encoder for `per_label_xfmr`."""
        return self.encoder(filled, present_f) if self.head == HEAD_XFMR else filled

    def forward(self, images: torch.Tensor, present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        present_f = present.float()
        return self.fuse(self.fill(self.embed(images, present), present_f), present_f)


def build_model(cfg: ModelConfig, pretrained: bool = False) -> KneeMIL:
    return KneeMIL(cfg, pretrained=pretrained)


# ------------------------------------------------------------------ bundle


def fingerprint(state: Mapping[str, torch.Tensor]) -> list[float]:
    """Sums of 8 float tensors at fixed positions of the sorted key list."""
    keys = sorted(k for k, v in state.items() if v.is_floating_point())
    picks = [keys[(i * len(keys)) // N_FINGERPRINT] for i in range(N_FINGERPRINT)]
    return [float(state[k].detach().double().sum().item()) for k in picks]


def bundle_meta(cfg: TrainConfig, fold: int, metrics: Mapping[str, Any] | None = None,
                git_sha: str = "") -> dict[str, Any]:
    m = cfg.model
    return {
        "run_id": cfg.run_id, "fold": int(fold),
        "arch": {"backbone": m.backbone, "img_px": m.img_px, "crop_mm": m.crop_mm, "slots": list(m.slots),
                 "view_crop_mm": list(m.view_crop_mm), "K_infer": m.k_infer, "attn_hidden": m.attn_hidden,
                 "fusion": m.fusion_hidden},
        "norm": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD),
                 "p_lo": cfg.cache.p_lo, "p_hi": cfg.cache.p_hi},
        "labels": list(LABELS),
        "config": to_dict(cfg),
        "metrics": dict(metrics or {}),
        "git_sha": git_sha,
    }


def save_bundle(model: nn.Module | Mapping[str, torch.Tensor], meta: Mapping[str, Any],
                out_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Weights (state_dict or module) as safetensors + bundle.json with the fingerprint."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = model.state_dict() if isinstance(model, nn.Module) else model
    state = {k: v.detach().cpu().contiguous() for k, v in state.items()}
    doc = {**meta, "fingerprint": fingerprint(state)}
    tmp = out_dir / (WEIGHTS_NAME + ".tmp")
    save_file(state, str(tmp))
    os.replace(tmp, out_dir / WEIGHTS_NAME)
    (out_dir / BUNDLE_NAME).write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    return doc


def load_bundle(bundle_dir: str | os.PathLike[str], device: str | torch.device = "cpu",
                grad_checkpoint: bool | None = False, views: bool = False) -> tuple[KneeMIL, dict[str, Any]]:
    """Rebuild the model from bundle.json (pretrained=False) and load the weights strictly;
    a fingerprint mismatch aborts (architecture §2.7).

    `views=True` states that the caller builds per-slot crops from `arch.view_crop_mm`. A caller
    that does not (it would feed the model an uncropped duplicate of a repeated slot, which looks
    like a valid input and silently costs AUC) is refused here rather than at the metric."""
    bundle_dir = Path(bundle_dir)
    meta = json.loads((bundle_dir / BUNDLE_NAME).read_text(encoding="utf-8"))
    mcfg = from_dict(ModelConfig, meta["config"]["model"])
    if tuple(mcfg.view_crop_mm) and not views:
        raise RuntimeError(f"{bundle_dir}: this bundle has per-slot view crops "
                           f"{list(mcfg.view_crop_mm)} for slots {list(mcfg.slots)}; the caller must "
                           "crop each slot accordingly (data.slot_view_mm/view_params) and pass views=True")
    if grad_checkpoint is not None:
        mcfg.grad_checkpoint = grad_checkpoint
    model = build_model(mcfg, pretrained=False)
    state = load_file(str(bundle_dir / WEIGHTS_NAME))
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"{bundle_dir}: missing {result.missing_keys[:5]} unexpected {result.unexpected_keys[:5]}")
    got, want = fingerprint(model.state_dict()), meta["fingerprint"]
    for g, w in zip(got, want):
        if abs(g - w) > FINGERPRINT_RTOL * max(abs(w), 1.0):
            raise RuntimeError(f"{bundle_dir}: weight fingerprint mismatch {got} != {want}")
    return model.to(device).eval(), meta


def masked_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted mean of BCE-with-logits over cells with weight > 0 (architecture §2.4 loss)."""
    loss = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none") * weight
    return loss.sum() / weight.sum().clamp_min(1e-6)
