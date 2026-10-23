"""Typed configuration: YAML files map 1:1 onto these dataclasses. Unknown keys
raise, so a typo in a config never silently falls back to a default."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from rsna_kad.constants import MAIN_SLOTS

T = TypeVar("T")


@dataclass
class CacheConfig:
    version: str = "v1"
    crop_mm: float = 130.0
    px: int = 384
    max_slices: int = 40
    z_target_mm: float = 3.0
    p_lo: float = 1.0
    p_hi: float = 99.0
    # Series with more slices than this are treated as 3D when MRAcquisitionType is absent
    # (data_analysis.md §3: every series above 60 slices is 3D, 2D tops out at 60).
    max_2d_slices: int = 60
    # Trust a per-manufacturer geometric side convention only on this much evidence.
    geom_min_studies: int = 30
    geom_min_agreement: float = 0.95


@dataclass
class FoldsConfig:
    version: str = "v1"
    n_splits: int = 5
    seed: int = 2026
    group_col: str = "site_key"
    strat_col: str = "lang"
    # "balanced" (seeded greedy: size + gold + language) or "sklearn" (StratifiedGroupKFold)
    method: str = "balanced"
    n_trials: int = 200
    study_summary: str = "analysis/out/study_summary.csv"
    out_csv: str = "data/folds/folds_v1.csv"


@dataclass
class TargetsConfig:
    version: str = "v1"
    prior_strength: float = 8.0
    base_affirmed: float = 0.92
    base_hedged: float = 0.50
    base_negated: float = 0.04
    base_absent_reported: float = 0.06
    base_mild: dict[str, float] = field(default_factory=lambda: {
        "med_men": 0.25, "lat_men": 0.25, "effusion": 0.25, "baker": 0.25, "mcl": 0.20,
        "acl": 0.35, "med_oa": 0.35, "lat_oa": 0.35, "pf_oa": 0.35,
        "contusion": 0.70, "fracture": 0.70})
    weight_hedged: float = 0.7
    weight_absent_reported: float = 0.5
    weight_absent_under_reported: float = 0.1
    weight_parse_fail: float = 0.2
    weight_hallucinated: float = 0.3
    disagreement_weight: float = 0.5


@dataclass
class ModelConfig:
    backbone: str = "tf_efficientnetv2_s.in21k_ft_in1k"
    img_px: int = 256
    crop_mm: float = 130.0
    slots: tuple[str, ...] = MAIN_SLOTS
    # Field of view in mm per entry of `slots`, parallel to it; () = every slot sees the whole
    # cached crop (cache.crop_mm) and the data path is byte-identical to a config without this
    # key. A shorter value is a centre crop derived from the same cached volume, so a slot name
    # may appear twice with two crops - that second entry is an extra *view*, with its own
    # [MISSING] vector and aux head, not a second series (§2.2: 90 mm sagittal meniscus view).
    view_crop_mm: tuple[float, ...] = ()
    k_train: int = 16
    k_infer: int = 24
    attn_hidden: int = 256
    fusion_hidden: int = 512
    # Read-out over the pooled evidence (model.HEADS):
    #   shared_attn        one shared gated attention per slot -> concat -> MLP -> 12 logits (§2.4)
    #   per_label_attn     12 attention distributions over the whole (slot x triplet) bag
    #   per_label_residual the same 12 distributions, but only as a gated residual on top of
    #                      shared_attn's logits (gate starts at 0, so it degrades to shared_attn)
    # Only `shared_attn` builds the parameter set the published arms were trained with.
    head: str = "shared_attn"
    label_attn_hidden: int = 256
    label_attn_drop: float = 0.2
    # Strength of the anatomical (label, slot) prior added to the per-label attention logits
    # before the softmax (model.SLOT_PRIOR). 0 disables it; ignored by `shared_attn`.
    slot_prior_strength: float = 0.55
    # `per_label_xfmr` only: a pre-norm transformer encoder over the same bag before the per-label
    # read-out. Stochastic depth rises linearly from 0 (first layer) to xfmr_drop_path (last).
    xfmr_layers: int = 4
    xfmr_dim: int = 384
    xfmr_heads: int = 6
    xfmr_mlp_ratio: float = 4.0
    xfmr_attn_drop: float = 0.1
    xfmr_drop_path: float = 0.15
    aux_weight: float = 0.3
    drop_rate: float = 0.1
    drop_path: float = 0.1
    grad_checkpoint: bool = True


@dataclass
class TrainConfig:
    run_id: str = "mvp"
    cache: CacheConfig = field(default_factory=CacheConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    targets_version: str = "v1"
    folds_csv: str = "data/folds/folds_v1.csv"
    epochs: int = 12
    batch_studies: int = 4
    lr_backbone: float = 1e-4
    lr_heads: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float = 0.999
    grad_clip: float = 2.0
    num_workers: int = 12
    seed: int = 2026
    sibling_p: float = 0.3
    slot_dropout_p: float = 0.1
    mixup_p: float = 0.3
    mixup_alpha: float = 0.4
    crop_scale_mm: tuple[float, float] = (115.0, 130.0)
    rot_deg: float = 8.0
    bc_jitter: float = 0.03
    amp_dtype: str = "bfloat16"
    targets_dir: str = "data/targets"
    # Label table with {key}_state columns for oof_strict_auc; "" = the one named in the
    # targets manifest, skipped when neither exists.
    labels_parquet: str = ""
    eval_batch_studies: int = 8
    # Which epoch's EMA weights become the bundle + OOF: "last" (fixed epoch count, a clean OOF
    # estimate - stage-3 finding F1 measured +0.008 optimism for "best") or "best" (highest
    # OOF-weak AUC on the validation fold). metrics.json records both epochs either way.
    select_epoch: str = "last"
    # The batch shape varies with the number of present slots; with cudnn autotune every new
    # shape costs ~60 s for convnext_small @384 (14 shapes per fold) and buys nothing there.
    cudnn_benchmark: bool = True
    # torch.compile(backbone.forward, dynamic=True): +45 % img/s for convnext_small @384 for
    # one ~4 min compile per fold. Only the training model is compiled (checkpoints unchanged).
    compile_backbone: bool = False
    # "bce" or "bce_pairwise" = (1 - pairwise_weight) x soft BCE + pairwise_weight x a pairwise
    # ranking term on the main logits (train.PairwiseRankLoss); aux heads stay plain BCE.
    # 0.12 is taken from a competitor's trainer and is not tuned here.
    loss: str = "bce"
    pairwise_weight: float = 0.12
    # Per-series MRI intensity augmentation in training (rsna_kad.mri_aug: bias field, gamma,
    # blur, noise; one parameter set per series). Off keeps the training inputs byte-identical.
    mri_aug: bool = False


@dataclass
class InferConfig:
    mode: str = "main"
    k_infer: int = 24
    workers: int = 3
    queue_depth: int = 8
    tta_z_offset: bool = True
    fp16: bool = True
    wall_guard_hours: float = 8.25


def from_dict(cls: type[T], data: dict[str, Any], where: str = "") -> T:
    if not isinstance(data, dict):
        raise TypeError(f"{where or cls.__name__}: expected a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise KeyError(f"{where or cls.__name__}: unknown keys {unknown}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        f = known[name]
        nested = f.type if isinstance(f.type, type) else None
        if nested is None and isinstance(f.default_factory, type) and is_dataclass(f.default_factory):
            nested = f.default_factory
        if nested is not None and is_dataclass(nested):
            value = from_dict(nested, value, f"{where + '.' if where else ''}{name}")
        elif isinstance(value, list) and _wants_tuple(f):
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)


def _wants_tuple(f: dataclasses.Field) -> bool:
    return str(f.type).startswith("tuple") or isinstance(f.default, tuple)


def to_dict(cfg: Any) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def params_sha256(cfg: Any) -> str:
    return hashlib.sha256(json.dumps(to_dict(cfg), sort_keys=True).encode()).hexdigest()


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    return {} if doc is None else doc


def load_config(path: str | Path) -> TrainConfig:
    return from_dict(TrainConfig, _read_yaml(path), Path(path).name)


def load_cache_config(path: str | Path) -> CacheConfig:
    return from_dict(CacheConfig, _read_yaml(path), Path(path).name)


def load_folds_config(path: str | Path) -> FoldsConfig:
    return from_dict(FoldsConfig, _read_yaml(path), Path(path).name)
