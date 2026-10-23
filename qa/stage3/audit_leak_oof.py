"""QA stage 3: leakage audit + OOF/metrics contract + sklearn recomputation (read-only)."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
from rsna_kad.constants import DATA_DIR, ID_COL, LABEL_KEYS, LABELS  # noqa: E402
from rsna_kad.cv import load_folds  # noqa: E402
from rsna_kad.cache import get_index  # noqa: E402
from rsna_kad.config import load_config  # noqa: E402
from rsna_kad.data import load_targets  # noqa: E402
from rsna_kad.labels.targets import gold_frame  # noqa: E402
from rsna_kad.train import split_studies, cache_root  # noqa: E402

RUN = Path("runs/mvp_effv2s_256")
cfg = load_config("configs/mvp_effv2s_256.yaml")
folds = load_folds(cfg.folds_csv)
folds[ID_COL] = folds[ID_COL].astype(str)
print("folds_v1 md5", hashlib.md5(Path(cfg.folds_csv).read_bytes()).hexdigest(), "rows", len(folds))
gold_ids_csv = set(Path("data/labels/gold_ids.txt").read_text().split()) if Path("data/labels/gold_ids.txt").exists() else set()
gold_ids_folds = set(folds.loc[folds["is_gold"], ID_COL])
gold_train = gold_frame(Path(DATA_DIR) / "train.csv")
gold_train[ID_COL] = gold_train[ID_COL].astype(str)
gold_ids_train = set(gold_train[ID_COL])
print("gold ids: gold_ids.txt", len(gold_ids_csv), "folds is_gold", len(gold_ids_folds), "train.csv", len(gold_ids_train),
      "all equal:", (gold_ids_csv == gold_ids_folds == gold_ids_train) if gold_ids_csv else (gold_ids_folds == gold_ids_train))
gold_ids = gold_ids_folds | gold_ids_train | gold_ids_csv

index = get_index(cache_root(cfg))
cached = set(index.studies("train"))
print("cached train studies", len(cached))

# ---- (a)/(b) reconstruct the split for every fold and compare with config/log/oof
for k in range(5):
    fd = RUN / f"fold{k}"
    run_cfg = yaml.safe_load((fd / "config.yaml").read_text())
    assert run_cfg["run_id"] == "mvp_effv2s_256" and run_cfg["targets_version"] == "v0", run_cfg
    assert run_cfg["folds_csv"] == "data/folds/folds_v1.csv"
    targets = load_targets(Path(cfg.targets_dir) / "v0" / f"fold{k}.parquet")
    n_gold_in_targets = targets[ID_COL].isin(gold_ids).sum()
    targets_f = targets[~targets[ID_COL].isin(gold_ids)]
    train_ids, val_ids = split_studies(folds, k, targets_f, cached)
    log = (fd / "train.log").read_text()
    m = re.search(rf"fold {k}: (\d+) train studies, (\d+) val \((\d+) gold\)", log)
    oof = pd.read_parquet(fd / "oof.parquet")
    oof[ID_COL] = oof[ID_COL].astype(str)
    fold_ids = set(folds.loc[folds["fold"] == k, ID_COL])
    fold_gold = fold_ids & gold_ids
    print(f"fold{k}: reconstructed train={len(train_ids)} val={len(val_ids)} (gold in val {len(set(val_ids) & gold_ids)}) | "
          f"log says train={m.group(1)} val={m.group(2)} gold={m.group(3)} | "
          f"train∩gold={len(set(train_ids) & gold_ids)} train∩fold{k}={len(set(train_ids) & fold_ids)} | "
          f"fold{k} size={len(fold_ids)} gold={len(fold_gold)} | oof rows={len(oof)} oof==fold{k}: {set(oof[ID_COL]) == fold_ids} "
          f"oof folds={sorted(oof['fold'].unique())} | targets rows={len(targets)} gold rows in targets={n_gold_in_targets}")
    assert set(oof[ID_COL]) == fold_ids and len(oof) == len(fold_ids)
    assert not (set(train_ids) & gold_ids) and not (set(train_ids) & fold_ids)

# ---- (c) targets v0: all fold files identical? calibration sidecar
t0 = [pd.read_parquet(Path(cfg.targets_dir) / "v0" / f"fold{k}.parquet") for k in range(5)]
print("targets v0 fold files identical:", all(t0[0].equals(t) for t in t0[1:]), "rows", [len(t) for t in t0])
for k in range(5):
    p = Path(cfg.targets_dir) / "v0" / f"calibration_fold{k}.json"
    if p.exists():
        d = json.loads(p.read_text())
        print(f"calibration_fold{k}.json n_gold_train={d.get('n_gold_train')}")
tc = t0[0][[f"{k}_target" for k in LABEL_KEYS]].to_numpy()
wc = t0[0][[f"{k}_weight" for k in LABEL_KEYS]].to_numpy()
print("targets unique values", sorted(set(np.round(np.unique(tc), 3))), "weights", sorted(set(np.round(np.unique(wc), 3))))
print("target when weight==0:", np.unique(tc[wc == 0]) if (wc == 0).any() else "no weight-0 cells")
print("cells with weight<=0.1: %.3f" % (wc <= 0.1).mean())

# ---- OOF contract on the full assembled file
oof = pd.read_parquet(RUN / "oof.parquet")
oof[ID_COL] = oof[ID_COL].astype(str)
pred_cols = [f"{k}_pred" for k in LABEL_KEYS]
print("oof columns ok:", list(oof.columns) == [ID_COL, "fold"] + pred_cols, "rows", len(oof), "unique", oof[ID_COL].nunique(),
      "dtypes", set(str(d) for d in oof[pred_cols].dtypes), "min %.4f max %.4f" % (oof[pred_cols].min().min(), oof[pred_cols].max().max()),
      "nan", int(oof[pred_cols].isna().sum().sum()))
print("oof fold agrees with folds_v1:", (oof.merge(folds, on=ID_COL)[["fold_x", "fold_y"]].nunique(axis=1) == 1).all())
print("oof ∩ gold", len(set(oof[ID_COL]) & gold_ids), "oof == all folds ids", set(oof[ID_COL]) == set(folds[ID_COL]))
mj = json.loads((RUN / "metrics.json").read_text())
print("metrics.json keys", sorted(mj.keys()))

# ---- recompute oof_weak_auc with sklearn (per fold targets file, §2.5 binarisation)
parts = []
for k in range(5):
    t = load_targets(Path(cfg.targets_dir) / "v0" / f"fold{k}.parquet")
    ids = set(folds.loc[folds["fold"] == k, ID_COL])
    parts.append(t[t[ID_COL].isin(ids)])
targets = pd.concat(parts)
weak = targets.merge(oof, on=ID_COL, how="inner")
print("weak rows", len(weak))
per = {}
for k in LABEL_KEYS:
    y, w, p = weak[f"{k}_target"].to_numpy(), weak[f"{k}_weight"].to_numpy(), weak[f"{k}_pred"].to_numpy()
    pos = y >= 0.6
    neg = (y <= 0.2) & (w >= 0.5)
    keep = pos | neg
    per[k] = roc_auc_score(pos[keep].astype(int), p[keep])
    print(f"  {k:10s} n_pos={pos.sum():4d} n_neg={neg.sum():4d} sklearn={per[k]:.4f} reported={mj['per_label'][k]:.4f}")
macro = float(np.mean(list(per.values())))
print(f"RECOMPUTED oof_weak_auc={macro:.4f} reported={mj['oof_weak_auc']:.4f} diff={macro - mj['oof_weak_auc']:+.2e}")
# per-fold weak
for k in range(5):
    sub = weak[weak["fold"] == k]
    vals = []
    for kk in LABEL_KEYS:
        y, w, p = sub[f"{kk}_target"].to_numpy(), sub[f"{kk}_weight"].to_numpy(), sub[f"{kk}_pred"].to_numpy()
        keep = (y >= 0.6) | ((y <= 0.2) & (w >= 0.5))
        if (y[keep] >= 0.6).any() and (~(y[keep] >= 0.6)).any():
            vals.append(roc_auc_score((y[keep] >= 0.6).astype(int), p[keep]))
    fm = json.loads((RUN / f"fold{k}" / "metrics.json").read_text())
    print(f"  fold{k} weak sklearn={np.mean(vals):.4f} fold metrics.json={fm['oof_weak_auc']:.4f} best_epoch={fm['epoch']}")

# ---- gold AUC with sklearn
g = gold_train.merge(oof, on=ID_COL, how="inner")
print("gold rows joined", len(g))
gper = {}
for k in LABEL_KEYS:
    y = g[k].to_numpy(dtype=float)
    gper[k] = roc_auc_score((y > 0.5).astype(int), g[f"{k}_pred"].to_numpy()) if 0 < (y > 0.5).sum() < len(y) else float("nan")
gm = float(np.nanmean(list(gper.values())))
print(f"RECOMPUTED gold_auc={gm:.4f} reported={mj['gold_auc']:.4f} diff={gm - mj['gold_auc']:+.2e}; labels with NaN: {[k for k, v in gper.items() if np.isnan(v)]}")
print("per-label gold sklearn:", {k: round(v, 3) for k, v in gper.items()})

# ---- (d) optimism: best vs last epoch per fold
print("\n(d) best-epoch selection optimism (fold metrics.json history):")
best_w, last_w, best_g, last_g = [], [], [], []
for k in range(5):
    fm = json.loads((RUN / f"fold{k}" / "metrics.json").read_text())
    h = fm["history"]
    ws = [r["oof_weak_auc"] for r in h]
    gs = [r["gold_auc"] for r in h]
    print(f"  fold{k}: weak per epoch {[round(x, 4) for x in ws]} best ep {fm['epoch']} | gold {[round(x, 3) for x in gs]}")
    best_w.append(max(ws)); last_w.append(ws[-1]); best_g.append(gs[int(np.argmax(ws))]); last_g.append(gs[-1])
print(f"  mean weak best={np.mean(best_w):.4f} last={np.mean(last_w):.4f} optimism={np.mean(best_w) - np.mean(last_w):+.4f}; "
      f"gold at best-weak epoch={np.mean(best_g):.4f} last={np.mean(last_g):.4f}")
