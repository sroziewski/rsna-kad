"""Proof that no fold-0 study enters either pseudo-label source's train or val set, and that
each source only labels studies it did not train on. Uses train.py's own split_studies."""
import sys
import pandas as pd

sys.path.insert(0, "src")
from rsna_kad.cache import get_index
from rsna_kad.config import load_config
from rsna_kad.constants import ID_COL
from rsna_kad.cv import load_folds
from rsna_kad.data import load_targets
from rsna_kad.train import cache_root, split_studies, targets_path

orig = load_folds("data/folds/folds_v1.csv")
orig[ID_COL] = orig[ID_COL].astype(str)
of = dict(zip(orig[ID_COL], orig["fold"].astype(int)))
gold = set(orig.loc[orig["is_gold"], ID_COL])

cfg0 = load_config("configs/main_convnext_s_384_e8.yaml")
cached = set(get_index(cache_root(cfg0)).studies("train"))

for name in ("stage9_pseudo_srcA", "stage9_pseudo_srcB", "main_armA8_v2"):
    cfg = load_config(f"configs/{name}.yaml")
    folds = load_folds(cfg.folds_csv)
    tp = targets_path(cfg, 0)
    if not tp.exists():
        print(f"{name}: {tp} not built yet, skipped")
        continue
    t = load_targets(tp)
    t = t[~t[ID_COL].isin(gold)]
    tr, va = split_studies(folds, 0, t, cached)
    tro = pd.Series([of[s] for s in tr]).value_counts().sort_index().to_dict()
    vao = pd.Series([of[s] for s in va]).value_counts().sort_index().to_dict()
    print(f"\n{name} (--fold 0)")
    print(f"  train {len(tr):5d} studies, original folds: {tro}")
    print(f"  val   {len(va):5d} studies, original folds: {vao}")
    print(f"  fold-0 studies in train: {sum(1 for s in tr if of[s] == 0)}")
    print(f"  fold-0 studies in val  : {sum(1 for s in va if of[s] == 0)}")
    print(f"  train n val overlap    : {len(set(tr) & set(va))}")
    print(f"  gold in train          : {len(set(tr) & gold)}")
