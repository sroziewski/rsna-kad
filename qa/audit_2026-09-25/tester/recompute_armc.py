"""Independent sklearn recomputation of arm C's assembled metrics (weak / strict / gold-58)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

KEYS = ["acl", "mcl", "med_men", "lat_men", "med_oa", "lat_oa", "pf_oa", "effusion", "synovitis", "baker",
        "contusion", "fracture"]
CSV = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
       "Synovitis", "Baker's", "Contusion", "Fracture"]
ID = "StudyInstanceUID"
RUN = Path("runs/main_armc_convnextv2_t_384")

oof = pd.read_parquet(RUN / "oof.parquet")
parts = pd.concat([pd.read_parquet(RUN / f"fold{k}/oof.parquet") for k in range(5)], ignore_index=True)
same = parts.sort_values(ID).reset_index(drop=True).equals(oof.sort_values(ID).reset_index(drop=True))
folds = pd.read_csv("data/folds/folds_v1.csv", dtype={ID: str})
print("oof rows", len(oof), "unique", oof[ID].nunique(), "== concat(fold oofs):", same,
      "fold col matches folds_v1:", bool((oof.merge(folds, on=ID, suffixes=("", "_f")).eval("fold == fold_f")).all()))

# weak: each fold's rows against that fold's target table; float32 targets compared in float32
tg = []
for k in range(5):
    t = pd.read_parquet(f"data/targets/v1/fold{k}.parquet")
    ids = set(folds.loc[folds.fold == k, ID])
    tg.append(t[t[ID].isin(ids)])
weak = pd.concat(tg).merge(oof, on=ID)
aucs = []
for k in KEYS:
    y = weak[f"{k}_target"].to_numpy(np.float32)
    w = weak[f"{k}_weight"].to_numpy(np.float32)
    pos = y >= np.float32(0.6)
    neg = (y <= np.float32(0.2)) & (w >= np.float32(0.5))
    m = pos | neg
    aucs.append(roc_auc_score(pos[m], weak.loc[m, f"{k}_pred"]))
print(f"weak   n={len(weak)} macro={np.mean(aucs):.6f}")

manifest = json.loads(Path("data/targets/v1/manifest.json").read_text())
lab = pd.read_parquet(manifest["labels"])
st = lab[[ID] + [f"{k}_state" for k in KEYS]].merge(oof[oof[ID].isin(weak[ID])], on=ID)
sa = []
for k in KEYS:
    s = st[f"{k}_state"].astype(str)
    m = s.isin(["affirmed", "negated"])
    sa.append(roc_auc_score(s[m] == "affirmed", st.loc[m, f"{k}_pred"]))
print(f"strict n={len(st)} labels={manifest['labels']} macro={np.mean(sa):.6f}")

tr = pd.read_csv("/raid/sroziewski/data/rsna_kad/train.csv", dtype={ID: str})
gold = tr.dropna(subset=CSV, how="all").merge(oof, on=ID)
ga = [roc_auc_score(gold[c] >= 0.5, gold[f"{k}_pred"]) for k, c in zip(KEYS, CSV)]
print(f"gold   n={len(gold)} macro={np.mean(ga):.6f}")

rep = json.loads((RUN / "metrics.json").read_text())
for name, mine, key in [("weak", np.mean(aucs), "oof_weak_auc"), ("strict", np.mean(sa), "oof_strict_auc"),
                        ("gold", np.mean(ga), "gold_auc")]:
    print(f"{name:6s} recomputed {mine:.6f} reported {rep[key]:.6f} diff {mine - rep[key]:+.2e}")
