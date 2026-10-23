"""QA stage 3: bundle contract, tamper test, data pipeline experiments on the real cache, model checks (CPU)."""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cv2
from safetensors.torch import load_file, save_file
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
torch.set_num_threads(4)
from rsna_kad.constants import ID_COL, LABEL_KEYS  # noqa: E402
from rsna_kad.config import load_config  # noqa: E402
from rsna_kad.cache import get_index, load_study  # noqa: E402
from rsna_kad.cv import load_folds  # noqa: E402
from rsna_kad.data import (StudyDataset, collate_studies, load_targets, sample_centres, triplet_indices,
                           read_triplets, warp_slices)  # noqa: E402
from rsna_kad.model import (IMAGENET_MEAN, IMAGENET_STD, build_model, fingerprint, load_bundle, normalize,
                            masked_bce)  # noqa: E402
from rsna_kad.train import cache_root, predict_studies_cached  # noqa: E402
from rsna_kad.metrics import macro_auc  # noqa: E402

RUN = Path("runs/mvp_effv2s_256")
QA = Path("qa/stage3")
cfg = load_config("configs/mvp_effv2s_256.yaml")

# ---------------- mcl boundary check (float32 0.2 vs float64 0.2)
oof = pd.read_parquet(RUN / "oof.parquet"); oof[ID_COL] = oof[ID_COL].astype(str)
t = load_targets(Path(cfg.targets_dir) / "v0" / "fold0.parquet")
w = t.merge(oof, on=ID_COL)
y32, ww, p = w["mcl_target"].to_numpy(), w["mcl_weight"].to_numpy(), w["mcl_pred"].to_numpy()
y64 = y32.astype(np.float64)
print("mcl: rows with target==0.2(f32):", int((y32 == np.float32(0.2)).sum()), "of which weight>=0.5:", int(((y32 == np.float32(0.2)) & (ww >= 0.5)).sum()))
for name, y in (("f32-compare", y32), ("f64-upcast (code)", y64)):
    pos = y >= 0.6; neg = (y <= 0.2) & (ww >= 0.5); keep = pos | neg
    print(f"  {name}: n_neg={neg.sum()} auc={roc_auc_score(pos[keep].astype(int), p[keep]):.4f}")
print("  code macro_auc mcl:", macro_auc(y32[:, None], p[:, None], ww[:, None], keys=("mcl",))[1])

# ---------------- 4. bundle contract
fd = RUN / "fold0"
meta = json.loads((fd / "bundle.json").read_text())
want = {"run_id", "fold", "arch", "norm", "labels", "fingerprint", "config", "metrics", "git_sha"}
print("bundle keys:", sorted(meta.keys()), "missing:", want - set(meta), "extra:", set(meta) - want)
print("arch keys:", sorted(meta["arch"].keys()), "norm keys:", sorted(meta["norm"].keys()), "n labels", len(meta["labels"]),
      "fingerprint len", len(meta["fingerprint"]), "git_sha", repr(meta["git_sha"]), "metrics epoch", meta["metrics"].get("epoch"),
      "metrics weak", meta["metrics"].get("oof_weak_auc"))
print("model.safetensors bytes", (fd / "model.safetensors").stat().st_size)
t0 = time.time()
model, meta2 = load_bundle(fd, "cpu")
print("load_bundle cpu ok in %.1fs, training=%s" % (time.time() - t0, model.training))
raw = load_file(str(fd / "model.safetensors"))
sd = model.state_dict()
print("bit-exact reload:", all(torch.equal(sd[k], raw[k]) for k in raw) and set(sd) == set(raw), "n tensors", len(raw),
      "n params", sum(p.numel() for p in model.parameters()), "backbone params", sum(p.numel() for p in model.backbone.parameters()))
keys = sorted(k for k, v in raw.items() if v.is_floating_point())
picks = [keys[(i * len(keys)) // 8] for i in range(8)]
print("fingerprint tensors:", picks)
# tamper A: zero a fingerprinted tensor
tdir = QA / "tamper"; tdir.mkdir(exist_ok=True)
shutil.copy(fd / "bundle.json", tdir / "bundle.json")
bad = {k: v.clone() for k, v in raw.items()}; bad[picks[3]] = torch.zeros_like(bad[picks[3]])
save_file(bad, str(tdir / "model.safetensors"))
try:
    load_bundle(tdir, "cpu"); print("TAMPER A (zero fingerprinted tensor %s): NOT DETECTED" % picks[3])
except RuntimeError as e:
    print("TAMPER A (zero fingerprinted tensor): detected ->", str(e)[:90])
# tamper B: zero a non-fingerprinted tensor
other = [k for k in keys if k not in picks and raw[k].numel() > 1000][len(keys) // 3]
bad = {k: v.clone() for k, v in raw.items()}; bad[other] = torch.zeros_like(bad[other])
save_file(bad, str(tdir / "model.safetensors"))
try:
    load_bundle(tdir, "cpu"); print("TAMPER B (zero non-fingerprinted tensor %s): NOT DETECTED (by design: 8-tensor fingerprint)" % other)
except RuntimeError as e:
    print("TAMPER B: detected ->", str(e)[:90])
# tamper C: remove a tensor
bad = {k: v.clone() for k, v in raw.items()}; bad.pop(other)
save_file(bad, str(tdir / "model.safetensors"))
try:
    load_bundle(tdir, "cpu"); print("TAMPER C (removed tensor): NOT DETECTED")
except RuntimeError as e:
    print("TAMPER C (removed tensor): detected ->", str(e)[:90])
shutil.rmtree(tdir)

# ---------------- 6. pretrained weights actually loaded
pre = build_model(cfg.model, pretrained=True)
key = "backbone.conv_stem.weight"
a, b, c = pre.state_dict()[key].flatten(), raw[key].flatten(), build_model(cfg.model, pretrained=False).state_dict()[key].flatten()
print(f"conv_stem corr(timm pretrained, bundle EMA) = {torch.corrcoef(torch.stack([a, b]))[0, 1]:.4f}; "
      f"corr(timm pretrained, random init) = {torch.corrcoef(torch.stack([a, c]))[0, 1]:.4f}; max|diff| pre-vs-bundle {(a - b).abs().max():.4f}")
key2 = "backbone.blocks.5.0.conv_pw.weight" if "backbone.blocks.5.0.conv_pw.weight" in raw else keys[-20]
a, b = pre.state_dict()[key2].flatten(), raw[key2].flatten()
print(f"{key2} corr(timm, bundle) = {torch.corrcoef(torch.stack([a, b]))[0, 1]:.4f}")
print("aux heads:", tuple(model.aux_w.shape), tuple(model.aux_b.shape), "missing vec:", tuple(model.missing.shape))

# ---------------- 5. data pipeline
print("\n(5a) centre sampling / triplets")
for n in (11, 30, 40):
    u = sample_centres(n, 24, None); r = sample_centres(n, 16, np.random.default_rng(0))
    print(f"  N={n}: K=24 uniform centres {u.tolist()}\n         K=16 random  centres {r.tolist()}; triplets of first/last: {triplet_indices(u, n)[[0, -1]].tolist()}")
    cover = np.zeros(n, int)
    for s in range(200):
        cover[np.unique(sample_centres(n, 16, np.random.default_rng(s)))] += 1
    assert triplet_indices(u, n).min() >= 0 and triplet_indices(u, n).max() <= n - 1
    print(f"         random coverage over 200 draws: min hits per slice {cover.min()}, slices never hit {int((cover == 0).sum())}")
# (b) normalisation
x = torch.randint(0, 256, (2, 3, 8, 8), dtype=torch.uint8)
ref = (x.float() / 255.0 - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
print("(5b) normalize(uint8) == (x/255-mean)/std:", torch.allclose(normalize(x), ref, atol=1e-6), "; float passthrough:", torch.equal(normalize(ref), ref))

index = get_index(cache_root(cfg))
folds = load_folds(cfg.folds_csv); folds[ID_COL] = folds[ID_COL].astype(str)
studies = sorted(index.studies("train"))[:40]
tgt = load_targets(Path(cfg.targets_dir) / "v0" / "fold0.parquet")
val_ds = StudyDataset(cache_root(cfg), studies, tgt, cfg, train=False)
tr_ds = StudyDataset(cache_root(cfg), studies, tgt, cfg, train=True, seed=7)
print("(5f) eval K =", val_ds.k, "train K =", tr_ds.k)
it0 = val_ds[0]; it1 = val_ds[0]
print("(5f) eval item deterministic:", torch.equal(it0["images"], it1["images"]), "shape", tuple(it0["images"].shape), it0["images"].dtype)
vols = load_study(cache_root(cfg), "train", studies[0], cfg.model.slots, index=index)
slot0 = cfg.model.slots[0]; vol = vols[slot0]; n = vol.shape[0]
c = sample_centres(n, 24, None); idx = triplet_indices(c, n)
plain = np.stack([np.stack([cv2.resize(np.asarray(vol[j]), (256, 256), interpolation=cv2.INTER_AREA) for j in row]) for row in idx])
print("(5c/f) eval images == plain INTER_AREA resize of cache slices (no aug):", np.array_equal(plain, it0["images"][0].numpy()),
      "uniform centres used:", c.tolist()[:6], "...")
# (c) augmentations identical across slices of a series: warp all slices with params and compare with per-triplet read
params = {"scale": 1.1, "rot_deg": 5.0, "contrast": 1.02, "brightness": 3.0}
full = warp_slices(np.asarray(vol), 256, params)
trip = read_triplets(vol, idx, 256, params)
print("(5c) read_triplets with params == whole-volume warp gathered:", np.array_equal(trip, full[idx]))
a0, a1, a2 = tr_ds[0]["images"], tr_ds[0]["images"], None
tr_ds.set_epoch(1); a2 = tr_ds[0]["images"]; tr_ds.set_epoch(0)
tr_ds2 = StudyDataset(cache_root(cfg), studies, tgt, cfg, train=True, seed=7)
print("(5g) train item same seed/epoch identical:", torch.equal(a0, a1), torch.equal(a0, tr_ds2[0]["images"]),
      "| different epoch differs:", not torch.equal(a0, a2), "| train differs from eval (aug on):", not np.array_equal(a0[0, :, :, :, :].numpy()[:16], it0["images"][0].numpy()[:16]))
# (d) slot dropout never < 2, sibling swap: run with p=1
from dataclasses import replace  # noqa: E402
cfg_drop = replace(cfg, slot_dropout_p=1.0, sibling_p=1.0)
ds_drop = StudyDataset(cache_root(cfg), sorted(index.studies("train"))[:300], tgt, cfg_drop, train=True, seed=1)
mins, base = [], []
for i in range(300):
    pres = ds_drop[i]["present"].sum().item()
    nat = sum(1 for s in cfg.model.slots if index.series_for("train", ds_drop.studies[i], s))
    mins.append(pres); base.append(nat)
mins, base = np.array(mins), np.array(base)
print("(5d) slot_dropout_p=1: natural present counts", np.bincount(base, minlength=4).tolist(), "-> after dropout", np.bincount(mins, minlength=4).tolist(),
      "; min present among studies with >=2 natural:", mins[base >= 2].min(), "; any study lost below 2 with 3 natural:", bool((mins[base == 3] < 2).any()))
n_sib = sum(1 for s in index.studies("train") for sl in cfg.model.slots if len(index.series_for("train", s, sl)) > 1)
n_missing = {sl: sum(1 for s in index.studies("train") if not index.series_for("train", s, sl)) for sl in cfg.model.slots}
print("(5d/e) (study,slot) pairs with >1 sibling series:", n_sib, "| studies missing a slot after fallback:", n_missing)
# (e) synthetic batch: missing slot -> presence 0, learned vector
m = model
imgs = torch.randint(0, 256, (2, 3, 4, 3, 64, 64), dtype=torch.uint8)
present = torch.tensor([[True, True, True], [True, False, True]])
with torch.no_grad():
    emb = m.embed(imgs, present)
    filled = m.fill(emb, present.float())
    print("(5e) absent slot embed is zero:", torch.equal(emb[1, 1], torch.zeros(m.dim)), "| filled absent == missing[1]:", torch.equal(filled[1, 1], m.missing[1]),
          "| present slot untouched:", torch.equal(filled[0, 1], emb[0, 1]))
    logits, aux = m(imgs, present)
    m.missing.data[1] += 1.0
    logits2, _ = m(imgs, present)
    print("(5e) changing missing[1] changes only study with absent slot 1: study0 same", torch.allclose(logits[0], logits2[0]), "study1 differs", not torch.allclose(logits[1], logits2[1]),
          "| output shapes", tuple(logits.shape), tuple(aux.shape), "finite", bool(torch.isfinite(logits).all() and torch.isfinite(aux).all()))
    m.missing.data[1] -= 1.0
    # attention sums to 1
    feats = torch.randn(5, 16, m.dim)
    _, a = m.attn(feats)
    print("(6) attention weights sum to 1 per slot:", torch.allclose(a.sum(1), torch.ones(5), atol=1e-5), "shape", tuple(a.shape))
    # loss weighted mean check
    lg, tg, wt = torch.randn(3, 12), torch.rand(3, 12), torch.tensor([[1.0] * 12, [0.0] * 12, [0.5] * 12])
    ref = (torch.nn.functional.binary_cross_entropy_with_logits(lg, tg, reduction="none") * wt).sum() / wt.sum()
    print("(7) masked_bce == weighted mean:", torch.allclose(masked_bce(lg, tg, wt), ref))
# (g) loader determinism, worker path
loader = torch.utils.data.DataLoader(tr_ds, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(3), num_workers=2, collate_fn=collate_studies)
b1 = next(iter(loader))
loader2 = torch.utils.data.DataLoader(tr_ds2, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(3), num_workers=2, collate_fn=collate_studies)
b2 = next(iter(loader2))
print("(5g) dataloader with workers, same seed -> same studies & images:", b1["study"] == b2["study"], torch.equal(b1["images"], b2["images"]))
# CPU prediction of 8 val studies from the bundle vs oof.parquet (fp32 CPU vs bf16 GPU)
ids = sorted(folds.loc[folds["fold"] == 0, ID_COL])[:8]
ds = StudyDataset(cache_root(cfg), ids, None, cfg, train=False)
pred = predict_studies_cached(model, ds, batch_size=4, num_workers=2, device="cpu", dtype=torch.float32)
cmp = pred.merge(oof, on=ID_COL, suffixes=("_cpu", "_oof"))
d = np.abs(cmp[[f"{k}_pred_cpu" for k in LABEL_KEYS]].to_numpy() - cmp[[f"{k}_pred_oof" for k in LABEL_KEYS]].to_numpy())
print("(4) CPU fp32 re-prediction of 8 fold-0 studies vs oof.parquet (bf16 GPU): max|diff| %.4f mean %.4f" % (d.max(), d.mean()))
