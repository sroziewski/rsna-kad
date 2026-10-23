"""One fold of KneeMIL training (architecture §2.4/§2.5): AdamW with two lr groups, cosine
schedule with warmup, per-step EMA, bf16 autocast, weighted soft-target BCE + aux heads,
manifold mixup on slot embeddings, per-epoch OOF evaluation, resumable from last.pt. The bundle
and OOF hold the epoch chosen by cfg.select_epoch ("last" or "best"); metrics.json reports both."""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from rsna_kad.cache import CacheIndex, get_index
from rsna_kad.config import TrainConfig, to_dict
from rsna_kad.constants import CACHE_DIR, DATA_DIR, ID_COL, LABEL_KEYS
from rsna_kad.cv import load_folds
from rsna_kad.data import StudyDataset, collate_studies, load_targets, synthetic_targets
from rsna_kad.metrics import NEG_MIN_WEIGHT, NEG_THR, POS_THR, THR_EPS, auc, gold_report, macro_auc
from rsna_kad.model import KneeMIL, build_model, bundle_meta, masked_bce, save_bundle

log = logging.getLogger(__name__)

PRED_COLS = [f"{k}_pred" for k in LABEL_KEYS]
LAST_NAME = "last.pt"
OOF_NAME = "oof.parquet"
METRICS_NAME = "metrics.json"
LOG_EVERY = 50
SELECT_POLICIES = ("last", "best")
FULL = "full"
# torch/numpy seed offset of a full-data run (folds use their index 0-4).
FULL_SEED_OFFSET = 99
LOSSES = ("bce", "bce_pairwise")
# Recent detached logits kept per label and class. Measured on targets v3_pseudo_full (100 random
# 4-study batches): mcl, fracture, synovitis and contusion have a valid in-batch pair in only
# 1-15 % of batches, so in-batch pairs alone would leave the rare labels without a ranking term.
PAIRWISE_BANK = 64


class PairwiseRankLoss:
    """Mean over labels of softplus(-(z_pos - z_neg)) over positive/negative pairs.

    Pos/neg follow metrics.binarise (target >= 0.6; target <= 0.2 with weight >= 0.5). Pairs
    are formed between the batch and itself plus a per-label bank of the last `bank` detached
    positive and negative logits; bank-bank pairs are left out because they carry no gradient.
    Labels without a pair contribute nothing; no pair at all gives an exact 0. The bank starts
    empty after a resume from last.pt."""

    def __init__(self, n_labels: int, bank: int = PAIRWISE_BANK):
        self.bank = bank
        self.pos: list[torch.Tensor | None] = [None] * n_labels
        self.neg: list[torch.Tensor | None] = [None] * n_labels

    @staticmethod
    def _masks(target: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = target >= POS_THR - THR_EPS
        neg = (target <= NEG_THR + THR_EPS) & (weight >= NEG_MIN_WEIGHT - THR_EPS)
        return pos, neg

    def _push(self, store: list, j: int, z: torch.Tensor) -> None:
        if z.numel() == 0:
            return
        old = store[j]
        new = z.detach() if old is None else torch.cat([old, z.detach()])
        store[j] = new[-self.bank:]

    def __call__(self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        z = logits.float()
        pos, neg = self._masks(target, weight)
        terms = []
        for j in range(z.shape[1]):
            zp, zn = z[pos[:, j], j], z[neg[:, j], j]
            bp = self.pos[j] if self.pos[j] is not None else z.new_zeros(0)
            bn = self.neg[j] if self.neg[j] is not None else z.new_zeros(0)
            diffs = [(zp[:, None] - torch.cat([zn, bn])[None, :]).reshape(-1),
                     (bp[:, None] - zn[None, :]).reshape(-1)]
            d = torch.cat(diffs)
            if d.numel():
                terms.append(torch.nn.functional.softplus(-d).mean())
            self._push(self.pos, j, zp)
            self._push(self.neg, j, zn)
        return torch.stack(terms).mean() if terms else z.sum() * 0.0


# ------------------------------------------------------------------ inputs


def cache_root(cfg: TrainConfig) -> Path:
    return Path(CACHE_DIR) / cfg.cache.version


def targets_path(cfg: TrainConfig, fold: int | str) -> Path:
    name = "full.parquet" if fold == FULL else f"fold{fold}.parquet"
    return Path(cfg.targets_dir) / cfg.targets_version / name


def labels_path(cfg: TrainConfig) -> Path | None:
    if cfg.labels_parquet:
        return Path(cfg.labels_parquet)
    manifest = Path(cfg.targets_dir) / cfg.targets_version / "manifest.json"
    if manifest.exists():
        labels = json.loads(manifest.read_text(encoding="utf-8")).get("labels")
        if labels and Path(labels).exists():
            return Path(labels)
    return None


def gold_frame_or_none() -> pd.DataFrame | None:
    train_csv = Path(DATA_DIR) / "train.csv"
    if not train_csv.exists():
        return None
    from rsna_kad.labels.targets import gold_frame
    g = gold_frame(train_csv)
    g[ID_COL] = g[ID_COL].astype(str)
    return g


def split_studies(folds: pd.DataFrame, fold: int, targets: pd.DataFrame, cached: set[str],
                  limit: int | None = None, seed: int = 0) -> tuple[list[str], list[str]]:
    """Train = other folds, never gold, with a targets row and a cache entry; val = fold k
    (weak + gold). With limit, a seeded random subset of each; the fold's gold rows stay."""
    folds = folds.assign(**{ID_COL: folds[ID_COL].astype(str)})
    has_target = set(targets[ID_COL].astype(str))
    tr = folds[(folds["fold"] != fold) & ~folds["is_gold"]]
    train_ids = sorted(s for s in tr[ID_COL] if s in has_target and s in cached)
    va = folds[folds["fold"] == fold]
    val_weak = sorted(s for s, g in zip(va[ID_COL], va["is_gold"]) if not g and s in cached)
    val_gold = sorted(s for s, g in zip(va[ID_COL], va["is_gold"]) if g and s in cached)
    if limit:
        rng = np.random.default_rng(seed)
        train_ids = sorted(rng.choice(train_ids, size=min(limit, len(train_ids)), replace=False))
        val_weak = sorted(rng.choice(val_weak, size=min(limit, len(val_weak)), replace=False))
    return train_ids, val_weak + val_gold


def split_full(folds: pd.DataFrame, targets: pd.DataFrame, cached: set[str],
               limit: int | None = None, seed: int = 0) -> tuple[list[str], list[str]]:
    """Full-data run: train = every non-gold study with a targets row and a cache entry, val =
    the gold studies only (never trained on, so their score is a held-out one)."""
    folds = folds.assign(**{ID_COL: folds[ID_COL].astype(str)})
    has_target = set(targets[ID_COL].astype(str))
    train_ids = sorted(s for s, g in zip(folds[ID_COL], folds["is_gold"]) if not g and s in has_target and s in cached)
    val_gold = sorted(s for s, g in zip(folds[ID_COL], folds["is_gold"]) if g and s in cached)
    if limit:
        rng = np.random.default_rng(seed)
        train_ids = sorted(rng.choice(train_ids, size=min(limit, len(train_ids)), replace=False))
    return train_ids, val_gold


def evaluate_gold(pred: pd.DataFrame, gold: pd.DataFrame | None, n_boot: int = 1000) -> dict[str, Any]:
    """`evaluate` for a full-data run, whose only held-out rows are the gold studies: no weak or
    strict metric exists (every weak row was trained on)."""
    out: dict[str, Any] = {"oof_weak_auc": float("nan"), "n_weak": 0, "per_label": {},
                           "oof_strict_auc": float("nan"), "gold_auc": float("nan"),
                           "gold_ci95": [float("nan")] * 2, "n_gold": 0}
    if gold is not None and gold[ID_COL].isin(pred[ID_COL]).any():
        rep = gold_report(pred, gold, n_boot=n_boot)
        out.update({"gold_auc": rep["macro_auc"], "gold_ci95": rep["ci95"], "n_gold": rep["n"],
                    "per_label_gold": rep["per_label"]})
    return out


# ------------------------------------------------------------------ optimisation


def cosine_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))


def build_optimizer(model: KneeMIL, cfg: TrainConfig) -> torch.optim.Optimizer:
    groups = [{"params": list(model.backbone.parameters()), "lr": cfg.lr_backbone},
              {"params": model.head_parameters(), "lr": cfg.lr_heads}]
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))


class Ema:
    """Per-step exponential moving average of all float state (weights and BN statistics).
    The decay ramps up as (1+t)/(10+t) so short runs (smoke tests) are not stuck near the
    initial weights: 0.999^300 would keep 74 % of the ImageNet init after 300 steps."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.step = 0
        self._ema = [v for v in self.module.state_dict().values() if v.is_floating_point()]
        self._src = [v for v in model.state_dict().values() if v.is_floating_point()]

    @torch.no_grad()
    def update(self) -> None:
        self.step += 1
        d = min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        torch._foreach_lerp_(self._ema, self._src, 1.0 - d)

    def state_dict(self) -> dict[str, Any]:
        return {"module": self.module.state_dict(), "step": self.step}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.module.load_state_dict(state["module"])
        self.step = int(state["step"])
        self._ema = [v for v in self.module.state_dict().values() if v.is_floating_point()]


def amp_dtype(cfg: TrainConfig) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.amp_dtype]


# ------------------------------------------------------------------ prediction / metrics


@torch.inference_mode()
def predict_studies_cached(model: torch.nn.Module, dataset: StudyDataset, tta: bool = False,
                           batch_size: int = 8, num_workers: int = 8, device: str | torch.device = "cuda",
                           dtype: torch.dtype = torch.bfloat16) -> pd.DataFrame:
    """Sigmoid predictions per study; tta averages the z-offset view (§2.6)."""
    datasets = [dataset]
    if tta:
        datasets.append(StudyDataset(dataset.cache_dir, dataset.studies, None, dataset.cfg, train=False,
                                     split=dataset.split, z_offset=0.5))
    model.eval()
    sums: dict[str, np.ndarray] = {}
    for ds in datasets:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=collate_studies, pin_memory=True)
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            present = batch["present"].to(device, non_blocking=True)
            with torch.autocast(device_type=torch.device(device).type, dtype=dtype, enabled=dtype != torch.float32):
                logits, _ = model(images, present)
            probs = logits.float().sigmoid().cpu().numpy()
            for study, p in zip(batch["study"], probs):
                sums[study] = sums.get(study, 0.0) + p / len(datasets)
    out = pd.DataFrame({ID_COL: list(sums)})
    preds = np.stack(list(sums.values())).astype(np.float32) if sums else np.zeros((0, len(LABEL_KEYS)), np.float32)
    for j, c in enumerate(PRED_COLS):
        out[c] = preds[:, j]
    return out


def strict_auc(pred: pd.DataFrame, labels: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """Affirmed vs negated rows only (§2.5 metric 2), macro over labels with both classes."""
    df = labels.assign(**{ID_COL: labels[ID_COL].astype(str)}).merge(pred, on=ID_COL, how="inner")
    per_label: dict[str, float] = {}
    for k in LABEL_KEYS:
        st = df[f"{k}_state"].astype(str).to_numpy()
        keep = (st == "affirmed") | (st == "negated")
        per_label[k] = auc(st[keep] == "affirmed", df.loc[keep, f"{k}_pred"].to_numpy()) if keep.sum() else float("nan")
    vals = [v for v in per_label.values() if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), per_label


def evaluate(pred: pd.DataFrame, targets: pd.DataFrame, gold: pd.DataFrame | None,
             labels: pd.DataFrame | None, n_boot: int = 1000) -> dict[str, Any]:
    weak = targets.merge(pred, on=ID_COL, how="inner")
    y = weak[[f"{k}_target" for k in LABEL_KEYS]].to_numpy()
    w = weak[[f"{k}_weight" for k in LABEL_KEYS]].to_numpy()
    p = weak[PRED_COLS].to_numpy()
    macro, per_label = macro_auc(y, p, w)
    out: dict[str, Any] = {"oof_weak_auc": macro, "n_weak": int(len(weak)), "per_label": per_label,
                           "oof_strict_auc": float("nan"), "gold_auc": float("nan"), "gold_ci95": [float("nan")] * 2,
                           "n_gold": 0}
    if labels is not None:
        weak_ids = set(targets[ID_COL])
        out["oof_strict_auc"], out["per_label_strict"] = strict_auc(pred[pred[ID_COL].isin(weak_ids)], labels)
    if gold is not None and gold[ID_COL].isin(pred[ID_COL]).any():
        rep = gold_report(pred, gold, n_boot=n_boot)
        out.update({"gold_auc": rep["macro_auc"], "gold_ci95": rep["ci95"], "n_gold": rep["n"],
                    "per_label_gold": rep["per_label"]})
    return out


def _plain(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _plain(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_plain(v) for v in d]
    if isinstance(d, (float, np.floating)):
        return None if not np.isfinite(d) else float(d)
    if isinstance(d, (np.integer,)):
        return int(d)
    return d


def write_json(path: Path, doc: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_plain(dict(doc)), indent=1), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ training


def _setup_logging(run_dir: Path) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    have = {getattr(h, "baseFilename", None) for h in root.handlers}
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    path = str((run_dir / "train.log").resolve())
    if path not in have:
        fh = logging.FileHandler(path)
        fh.setFormatter(fmt)
        root.addHandler(fh)


def train_fold(cfg: TrainConfig, fold: int | str, run_dir: str | os.PathLike[str], limit: int | None = None,
               resume: bool = True, device: str = "cuda", synthetic: bool = False,
               index: CacheIndex | None = None, pretrained: bool = True) -> dict[str, Any]:
    """`fold` = 0-4, or FULL: train on every non-gold study and evaluate on the gold ones only."""
    if cfg.select_epoch not in SELECT_POLICIES:
        raise ValueError(f"select_epoch must be one of {SELECT_POLICIES}, got {cfg.select_epoch!r}")
    full = fold == FULL
    if full and cfg.select_epoch != "last":
        # With gold as the only held-out rows, "best" would select the epoch on gold.
        raise ValueError("a full-data run needs select_epoch: last")
    fk = FULL_SEED_OFFSET if full else int(fold)
    if cfg.loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}, got {cfg.loss!r}")
    rank_loss = PairwiseRankLoss(len(LABEL_KEYS)) if cfg.loss == "bce_pairwise" else None
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(_plain(to_dict(cfg)), sort_keys=False), encoding="utf-8")
    torch.manual_seed(cfg.seed + fk)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    folds = load_folds(cfg.folds_csv)
    index = index or get_index(cache_root(cfg))
    cached = set(index.studies("train"))
    if synthetic:
        weak_ids = folds.loc[~folds["is_gold"], ID_COL].astype(str)
        targets = synthetic_targets(list(weak_ids), seed=cfg.seed)
        log.warning("SYNTHETIC random targets: plumbing/timing only, the metrics mean nothing")
    else:
        targets = load_targets(targets_path(cfg, fold))
    gold_ids = set(folds.loc[folds["is_gold"], ID_COL].astype(str))
    targets = targets[~targets[ID_COL].isin(gold_ids)]
    if full:
        train_ids, val_ids = split_full(folds, targets, cached, limit, cfg.seed)
    else:
        train_ids, val_ids = split_studies(folds, fold, targets, cached, limit, cfg.seed)
    assert not gold_ids & set(train_ids), "gold rows must never be trained on"
    gold = None if synthetic else gold_frame_or_none()
    lp = None if synthetic else labels_path(cfg)
    labels = pd.read_parquet(lp) if lp else None
    log.info("fold %s: %d train studies, %d val (%d gold), targets %s, labels %s, gold %s", fold,
             len(train_ids), len(val_ids), len(set(val_ids) & gold_ids), targets_path(cfg, fold), lp,
             "yes" if gold is not None else "none")

    train_ds = StudyDataset(cache_root(cfg), train_ids, targets, cfg, train=True, seed=cfg.seed + fk)
    val_ds = StudyDataset(cache_root(cfg), val_ids, targets, cfg, train=False)
    workers = min(cfg.num_workers, os.cpu_count() or 1)
    gen = torch.Generator().manual_seed(cfg.seed + fk)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_studies, shuffle=True, generator=gen,
                              num_workers=workers, collate_fn=collate_studies, pin_memory=True,
                              drop_last=True, persistent_workers=workers > 0,
                              prefetch_factor=4 if workers > 0 else None)

    model = build_model(cfg.model, pretrained=pretrained).to(device).to(memory_format=torch.channels_last)
    ema = Ema(model, cfg.ema_decay)
    if cfg.compile_backbone:
        # Patched on the instance after the EMA copy: state_dict keys stay unchanged (no
        # _orig_mod prefix) and the EMA/eval model stays eager.
        model.backbone.forward = torch.compile(model.backbone.forward, dynamic=True)
    opt = build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader)
    total_steps, warmup_steps = cfg.epochs * steps_per_epoch, int(cfg.warmup_epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_warmup(s, total_steps, warmup_steps))
    dtype = amp_dtype(cfg)
    dev_type = torch.device(device).type

    history: list[dict[str, Any]] = []
    best = {"epoch": -1, "oof_weak_auc": -1.0}
    start_epoch = 0
    last = run_dir / LAST_NAME
    if resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        history, best, start_epoch = ck["history"], ck["best"], ck["epoch"] + 1
        log.info("resumed from %s at epoch %d", last, start_epoch)

    mix_rng = np.random.default_rng(cfg.seed + 1000 * fk + start_epoch)
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        train_ds.set_epoch(epoch)
        t0 = time.monotonic()
        n_img = 0
        loss_sum = loss_n = 0
        if dev_type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step, batch in enumerate(train_loader):
            images = batch["images"].to(device, non_blocking=True)
            present = batch["present"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            weight = batch["weight"].to(device, non_blocking=True)
            n_img += int(present.sum().item()) * cfg.model.k_train
            with torch.autocast(device_type=dev_type, dtype=dtype, enabled=dtype != torch.float32):
                present_f = present.float()
                filled = model.fill(model.embed(images, present), present_f)
                if cfg.mixup_p > 0 and images.shape[0] > 1 and mix_rng.random() < cfg.mixup_p:
                    lam = float(mix_rng.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                    perm = torch.randperm(images.shape[0], device=device)
                    filled = lam * filled + (1.0 - lam) * filled[perm]
                    present_f = lam * present_f + (1.0 - lam) * present_f[perm]
                    target = lam * target + (1.0 - lam) * target[perm]
                    weight = lam * weight + (1.0 - lam) * weight[perm]
                logits, aux = model.fuse(filled, present_f)
            loss = masked_bce(logits, target, weight)
            if rank_loss is not None:
                w_pair = cfg.pairwise_weight
                loss = (1.0 - w_pair) * loss + w_pair * rank_loss(logits, target, weight)
            if cfg.model.aux_weight > 0:
                aux_w = weight.unsqueeze(1) * present_f.unsqueeze(2)
                loss = loss + cfg.model.aux_weight * masked_bce(aux, target.unsqueeze(1).expand_as(aux), aux_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            ema.update()
            loss_sum += float(loss.item()) * images.shape[0]
            loss_n += images.shape[0]
            if (step + 1) % LOG_EVERY == 0 or step + 1 == steps_per_epoch:
                el = time.monotonic() - t0
                log.info("epoch %d step %d/%d loss %.4f lr %.2e %.0f img/s", epoch, step + 1, steps_per_epoch,
                         loss_sum / max(loss_n, 1), sched.get_last_lr()[0], n_img / max(el, 1e-6))
        train_time = time.monotonic() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30 if dev_type == "cuda" else 0.0

        t1 = time.monotonic()
        pred = predict_studies_cached(ema.module, val_ds, batch_size=cfg.eval_batch_studies,
                                      num_workers=workers, device=device, dtype=dtype)
        pred["fold"] = fold
        m = evaluate_gold(pred, gold, n_boot=200) if full else evaluate(pred, targets, gold, labels, n_boot=200)
        rec = {"epoch": epoch, "train_loss": loss_sum / max(loss_n, 1), "train_s": train_time,
               "eval_s": time.monotonic() - t1, "img_per_s": n_img / max(train_time, 1e-6),
               "peak_mem_gb": peak_gb, "oof_weak_auc": m["oof_weak_auc"], "oof_strict_auc": m["oof_strict_auc"],
               "gold_auc": m["gold_auc"]}
        history.append(rec)
        log.info("epoch %d done: loss %.4f weak %.4f strict %.4f gold %.4f | train %.0fs eval %.0fs %.0f img/s peak %.1f GB",
                 epoch, rec["train_loss"], rec["oof_weak_auc"], rec["oof_strict_auc"], rec["gold_auc"],
                 train_time, rec["eval_s"], rec["img_per_s"], peak_gb)
        latest = {"epoch": epoch, **m}
        is_best = m["oof_weak_auc"] >= best["oof_weak_auc"] or np.isnan(m["oof_weak_auc"])
        if is_best:
            best = dict(latest)
        # The bundle and OOF always hold the epoch the policy selects; "last" rewrites them every
        # epoch so an interrupted run still has a consistent bundle/oof pair on disk.
        if cfg.select_epoch == "last" or is_best:
            meta = bundle_meta(cfg, fk, metrics={**m, "epoch": epoch, "select_epoch": cfg.select_epoch},
                               git_sha=git_sha())
            if full:
                meta["fold"] = FULL
            save_bundle(ema.module.state_dict(), meta, run_dir)
            pred[[ID_COL, "fold"] + PRED_COLS].to_parquet(run_dir / OOF_NAME, index=False)
        selected = latest if cfg.select_epoch == "last" else best
        write_json(run_dir / METRICS_NAME, {**selected, "select_epoch": cfg.select_epoch,
                                            "best_epoch": best, "last_epoch": latest, "history": history,
                                            "n_train": len(train_ids), "n_val": len(val_ids), "limit": limit,
                                            "synthetic": synthetic})
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "history": history, "best": best, "epoch": epoch}, last)
    if last.exists():
        last.unlink()
    metrics = json.loads((run_dir / METRICS_NAME).read_text(encoding="utf-8"))
    log.info("fold %s finished: selected epoch %s (%s) weak %.4f strict %s gold %s | best epoch %s weak %s",
             fold, metrics["epoch"], cfg.select_epoch, metrics["oof_weak_auc"] or float("nan"),
             metrics["oof_strict_auc"], metrics["gold_auc"], metrics["best_epoch"]["epoch"],
             metrics["best_epoch"]["oof_weak_auc"])
    return metrics


def git_sha() -> str:
    return os.environ.get("RSNA_KAD_GIT_SHA", "")


def run_dir_for(cfg: TrainConfig, fold: int | str, root: str | os.PathLike[str] = "runs") -> Path:
    return Path(root) / cfg.run_id / (FULL if fold == FULL else f"fold{fold}")


def assemble_oof(run_root: str | os.PathLike[str], cfg: TrainConfig, folds: Sequence[int] = range(5),
                 n_boot: int = 1000) -> dict[str, Any]:
    """Concatenate runs/<run_id>/fold{k}/oof.parquet -> runs/<run_id>/oof.parquet + metrics.json
    over every fold that has finished (§3.2 OOF contract)."""
    root = Path(run_root) / cfg.run_id
    parts, done = [], []
    for k in folds:
        p = root / f"fold{k}" / OOF_NAME
        if p.exists():
            parts.append(pd.read_parquet(p))
            done.append(int(k))
    if not parts:
        raise FileNotFoundError(f"no fold oof.parquet under {root}")
    oof = pd.concat(parts, ignore_index=True)
    oof[ID_COL] = oof[ID_COL].astype(str)
    oof.to_parquet(root / OOF_NAME, index=False)
    # Each fold's rows are scored against that fold's target table (calibrated without its gold).
    folds_df = load_folds(cfg.folds_csv)
    parts = []
    for k in done:
        ids = set(folds_df.loc[folds_df["fold"] == k, ID_COL].astype(str))
        t = load_targets(targets_path(cfg, k))
        parts.append(t[t[ID_COL].isin(ids)])
    targets = pd.concat(parts, ignore_index=True)
    lp = labels_path(cfg)
    m = evaluate(oof, targets, gold_frame_or_none(), pd.read_parquet(lp) if lp else None, n_boot=n_boot)
    m["folds"] = done
    m["n_rows"] = int(len(oof))
    shutil.copy(root / f"fold{done[0]}" / "config.yaml", root / "config.yaml")
    write_json(root / METRICS_NAME, m)
    return m
